export OMP_NUM_THREADS=48
numactl -C 0-47, 6-143 python inference_INT4.py
