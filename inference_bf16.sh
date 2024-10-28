export OMP_NUM_THREADS=48   #if you are using Intel 8468 (48 Cores)
#numactl -C 0-47, 6-143 
python inference_bf16_ipex.py
