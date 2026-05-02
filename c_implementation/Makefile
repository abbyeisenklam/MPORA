# XGBoost userspace shared library
xgb_userspace:
	gcc -fPIC -shared -Ofast -march=native -mtune=native -mavx2 -mfma -funroll-loops -flto -DNDEBUG -o xgb_utils_userspace.so xgb_utils_userspace.c

clean:
	rm -f *.so