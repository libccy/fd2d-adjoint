nvcc externaltools\run_cuda.cu -arch=sm_50 -lcublas -lcusolver -Xcompiler "/wd4819" -o externaltools\run_cuda.exe && externaltools\run_cuda.exe
