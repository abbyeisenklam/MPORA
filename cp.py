import numpy as np
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Auto-detect best available device
def get_device(device: Literal["auto", "cuda", "cpu"] = "auto") -> torch.device:
    """Get torch device, with automatic CUDA detection."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        return torch.device("cpu")
    return torch.device(device)


class WeightNetwork(nn.Module):
    """Neural network w_θ: R^d → [0, 1] for learning calibration weights."""

    def __init__(self, input_dim: int, hidden_dims: tuple = (32,)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pinball_loss(
    residuals: torch.Tensor, q: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Pinball loss at level (1 - alpha).

    ρ_{1-α}(r - q) = (1-α)(r-q) if r >= q, else α(q-r)
    """
    diff = residuals - q
    return torch.where(diff >= 0, (1 - alpha) * diff, alpha * (-diff))


class WeightedConformalPredictor:
    """Weighted split conformal prediction for single-output regression.

    Implements the weighted conformal prediction algorithm from
    Barber et al. (2022) [arXiv:2202.13415].

    One instance per output dimension. For multi-output problems,
    create one predictor per target and orchestrate externally.

    Input space: R^d
    Output space: R (scalar)
    """

    def __init__(self, model, delta: float = 0.1):
        """
        Parameters
        ----------
        model : object
            A predictive model with a .predict(X) method that returns
            a 1D array of shape (n_samples,).
        delta : float
            Desired miscoverage level in (0, 1). The prediction set
            covers with probability at least 1 - delta.
        """
        assert 0 < delta < 1, "delta must be in (0, 1)"
        self.model = model
        self.delta = delta
        self.scores: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.q: Optional[float] = None
        self.q_star: Optional[float] = None  # (1-delta)-quantile of D_wt residuals
        self.weight_net: Optional[WeightNetwork] = None

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        weights: Optional[Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
        X_wt: Optional[np.ndarray] = None,
        y_wt: Optional[np.ndarray] = None,
        lambda_: float = 1.0,
        beta: float = 0.01,
        lr: float = 0.01,
        epochs: int = 100,
        hidden_dims: tuple = (32,),
        device: Literal["auto", "cuda", "cpu"] = "auto",
        verbose: bool = False,
        batch_size: int = 512,
    ):
        """Compute nonconformity scores on calibration data and the
        (weighted) conformal threshold.

        Parameters
        ----------
        X_cal : np.ndarray, shape (N_cal, d)
            Calibration inputs.
        y_cal : np.ndarray, shape (N_cal,)
            Calibration labels (single output dimension).
        weights : np.ndarray or Callable, optional
            Either:
            - np.ndarray, shape (N_cal,): Per-sample weights.
            - Callable: A function f(X_cal) -> weights array.
            If None and X_wt/y_wt provided, weights are learned.
            If None and no X_wt/y_wt, uniform weights (standard SCP).
        X_wt : np.ndarray, shape (N_wt, d), optional
            Weight-training inputs (D_wt). Required for learning weights.
        y_wt : np.ndarray, shape (N_wt,), optional
            Weight-training labels (single output dimension). Required for learning weights.
        lambda_ : float
            Weight for calibration term in loss (default 1.0).
        beta : float
            Regularization strength for weight network (default 0.01).
        lr : float
            Learning rate for weight network training (default 0.01).
        epochs : int
            Number of training epochs (default 100).
        hidden_dims : tuple
            Hidden layer dimensions for weight network (default (32,)).
        device : {"auto", "cuda", "cpu"}
            Device for training (default "auto" uses CUDA/MPS if available).
        verbose : bool
            Print training progress (default False).
        batch_size : int
            Batch size for D_wt DataLoader (default 512).
        """
        y_hat = self.model.predict(X_cal)
        # Nonconformity score: absolute residual (single output)
        self.scores = np.abs(y_hat - y_cal)
        N_cal = len(self.scores)

        # Resolve weights
        if weights is None:
            if X_wt is not None and y_wt is not None:
                # Learn weights from D_wt
                w = self._learn_weights(
                    X_cal,
                    self.scores,
                    X_wt,
                    y_wt,
                    lambda_=lambda_,
                    beta=beta,
                    lr=lr,
                    epochs=epochs,
                    hidden_dims=hidden_dims,
                    device=device,
                    verbose=verbose,
                    batch_size=batch_size,
                )
            else:
                w = None
        elif callable(weights):
            w = np.asarray(weights(X_cal))
        else:
            w = np.asarray(weights)

        if w is None:
            # Standard (unweighted) conformal threshold (Eq. 14)
            level = np.ceil((N_cal + 1) * (1 - self.delta)) / (N_cal + 1)
            self.q = np.quantile(self.scores, min(level, 1.0))
            self.weights = None
        else:
            assert len(w) == N_cal, f"weights length {len(w)} != N_cal {N_cal}"
            self.weights = w
            self.q = self._weighted_quantile(self.scores, w, self.delta)

    def _learn_weights(
        self,
        X_cal: np.ndarray,
        r_cal: np.ndarray,
        X_wt: np.ndarray,
        y_wt: np.ndarray,
        lambda_: float,
        beta: float,
        lr: float,
        epochs: int,
        hidden_dims: tuple,
        device: Literal["auto", "cuda", "cpu"],
        verbose: bool,
        batch_size: int = 512,  # this is 2^9
    ) -> np.ndarray:
        """Learn weights by minimizing Equation (15).

        Finds θ to minimize:
            (1/γ^cal) Σ_j w̃_j^θ(z_j) · (ρ_{1-δ}(S_j - q*) + ψ₁ log w̃_j^θ(z_j)) + ψ₂||θ||²

        where:
            q*   = empirical (1-δ)-th quantile of nonconformity scores in D_wt (fixed offline)
            w̃_j  = w_j / Σ_k w_k  (batch-normalized weights, sum to 1)
            ρ_{1-δ} = quantile (pinball) loss
            ψ₁   = lambda_  (categorical entropy regularization weight)
            ψ₂   = beta     (L2 regularization on network parameters θ)
        """
        dev = get_device(device)
        alpha = self.delta
        input_dim = X_cal.shape[1]
        if verbose:
            print(f"Training weight network on {dev}", flush=True)
            print(
                f"  N_cal={len(X_cal)}, N_wt={len(X_wt)}, input_dim={input_dim}",
                flush=True,
            )
            print(f"  hidden_dims={hidden_dims}, lr={lr}, epochs={epochs}", flush=True)
            print(
                f"  lambda={lambda_}, beta={beta}, batch_size={batch_size}", flush=True
            )

        # Step 1: Compute nonconformity scores on D_wt and fix q* offline
        if verbose:
            print("  Computing residuals on D_wt...", flush=True)
        y_hat_wt = self.model.predict(X_wt)
        r_wt = np.abs(y_hat_wt - y_wt)
        q_star = float(np.quantile(r_wt, 1.0 - alpha))
        self.q_star = q_star
        q_star_t = torch.tensor(q_star, dtype=torch.float32, device=dev)
        if verbose:
            print(
                f"  Fixed q* = {q_star:.4f} (empirical {1-alpha:.0%}-quantile of D_wt)",
                flush=True,
            )

        # Step 2: Initialize weight network
        self.weight_net = WeightNetwork(input_dim, hidden_dims).to(dev)

        # JIT compilation (PyTorch 2.0+) — reduces Python/launch overhead per batch
        if hasattr(torch, "compile"):
            try:
                self.weight_net = torch.compile(
                    self.weight_net, mode="reduce-overhead"
                )
                if verbose:
                    print("  Using torch.compile (reduce-overhead)", flush=True)
            except Exception:
                pass

        # Move all of D_cal to device once — eliminates per-batch CPU→GPU transfers
        try:
            X_cal_dev = torch.tensor(X_cal, dtype=torch.float32, device=dev)
            r_cal_dev = torch.tensor(r_cal, dtype=torch.float32, device=dev)
        except RuntimeError:
            # OOM fallback: keep on CPU, batches are moved to dev in the loop
            if verbose:
                print("  D_cal too large for device memory, keeping on CPU", flush=True)
            X_cal_dev = torch.tensor(X_cal, dtype=torch.float32)
            r_cal_dev = torch.tensor(r_cal, dtype=torch.float32)

        N_cal = X_cal_dev.shape[0]

        # Step 3: Optimizer with weight_decay replaces manual L2 term in loss
        optimizer = torch.optim.Adam(
            self.weight_net.parameters(), lr=lr, weight_decay=beta
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        eps = 1e-7
        patience = 20
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        # Training loop — manual GPU-side shuffled batching (no DataLoader overhead)
        epoch_bar = tqdm(range(epochs), desc="Epochs", disable=not verbose)
        for epoch in epoch_bar:
            # Shuffle indices on the same device as D_cal (GPU or CPU)
            perm = torch.randperm(N_cal, device=X_cal_dev.device)
            epoch_loss = 0.0
            n_batches = 0
            w_mean_last = 0.0

            batch_bar = tqdm(
                range(0, N_cal, batch_size),
                desc=f"  Epoch {epoch+1}",
                leave=False,
                disable=not verbose,
            )
            for start in batch_bar:
                idx = perm[start : start + batch_size]
                X_batch = X_cal_dev[idx].to(dev)
                r_batch = r_cal_dev[idx].to(dev)

                optimizer.zero_grad()

                # Raw weights in [0, 1] from sigmoid output
                w_raw = self.weight_net(X_batch)
                # Normalize within batch: w̃_j = w_j / Σ_k w_k
                w_tilde = w_raw / w_raw.sum().clamp(min=eps)

                # Term 1: Σ_j w̃_j · ρ_{1-δ}(S_j - q*)
                quant_loss = (w_tilde * pinball_loss(r_batch, q_star_t, alpha)).sum()

                # Term 2: ψ₁ · Σ_j w̃_j · log(w̃_j)  (categorical entropy)
                w_t = w_tilde.clamp(min=eps)
                entropy_reg = (w_t * w_t.log()).sum()

                # L2 regularization (ψ₂ · ||θ||²) handled by weight_decay in Adam
                loss = quant_loss + lambda_ * entropy_reg
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                w_mean_last = w_tilde.mean().item()
                batch_bar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.weight_net.state_dict().items()
                }
            else:
                patience_counter += 1

            cur_lr = optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix(
                loss=avg_loss, q_star=q_star, w_mean=w_mean_last, lr=cur_lr
            )

            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_state is not None:
            self.weight_net.load_state_dict(best_state)

        # Freeze and evaluate on full D_cal to get raw weights for weighted quantile
        self.weight_net.eval()
        with torch.no_grad():
            weights = self.weight_net(X_cal_dev.to(dev)).cpu().numpy()

        return weights

    def _weighted_quantile(
        self, scores: np.ndarray, weights: np.ndarray, delta: float
    ) -> float:
        """Compute weighted conformal threshold per Barber et al. (2022)

        q^(w)_delta = inf{ q in R : sum_j w_tilde_{j} * 1{S_j <= q} >= 1 - delta }

        where w_tilde_j = w_j / (sum(w) + 1) for calibration points, and
        w_tilde_{N+1} = 1 / (sum(w) + 1) for the new test point.
        """
        w_sum = weights.sum() + 1.0  # +1 for test point (Barber et al.)

        # If delta < w_tilde_{N+1}, threshold is +inf
        if delta < 1.0 / w_sum:
            return np.inf

        # Sort once and use vectorized cumsum
        order = np.argsort(scores)
        cumsum = np.cumsum(weights[order]) / w_sum

        # Find first index where cumulative weight >= 1 - delta
        idx = np.searchsorted(cumsum, 1.0 - delta, side="left")
        return scores[order[min(idx, len(scores) - 1)]]

    def predict_set(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Return point predictions and the conformal radius.

        The prediction set for each x is the interval:
            C(x) = [mu_hat(x) - q, mu_hat(x) + q]

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        y_hat : np.ndarray, shape (n,)
            Point predictions.
        q : float
            Conformal half-width (same for all points).
        """
        assert self.q is not None, "Must call calibrate() before predict_set()"
        return self.model.predict(X), self.q

    def predict_intervals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals.

            [mu_hat(x) - q, mu_hat(x) + q]

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        lower : np.ndarray, shape (n,)
        upper : np.ndarray, shape (n,)
        """
        y_hat, q = self.predict_set(X)
        return y_hat - q, y_hat + q

    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate empirical coverage probability on test data.

        Computes the fraction of test points where y_test falls within
        the prediction interval [mu_hat(x) - q, mu_hat(x) + q].

        Parameters
        ----------
        X_test : np.ndarray, shape (n, d)
            Test inputs.
        y_test : np.ndarray, shape (n,)
            True test labels (single output dimension).

        Returns
        -------
        dict with keys:
            - coverage: float, empirical coverage probability
            - target_coverage: float, 1 - delta (expected coverage)
            - n_covered: int, number of covered points
            - n_total: int, total number of test points
            - residuals: np.ndarray, |y_hat - y_test| for each point
        """
        assert self.q is not None, "Must call calibrate() before evaluate_coverage()"

        y_hat = self.model.predict(X_test)
        residuals = np.abs(y_hat - y_test)
        covered = residuals <= self.q

        return {
            "coverage": covered.mean(),
            "target_coverage": 1 - self.delta,
            "n_covered": covered.sum(),
            "n_total": len(y_test),
            "residuals": residuals,
        }
