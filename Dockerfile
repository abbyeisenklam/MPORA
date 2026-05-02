FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc make && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data first (large, rarely changes) so code edits don't invalidate this layer
COPY profiles/ profiles/
COPY splits/ splits/

# Copy code only
COPY *.py *.md *.txt *.sh ./
COPY c_implementation/ c_implementation/

RUN cd c_implementation && make xgb_userspace

CMD ["bash"]
