import os
import torch
import time
import argparse
    
def track_process(uid):
    return len(os.popen(f'ps -p {uid}').read().strip().split("\n")) != 1

def occupy_memory(cuda_device, percentage):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    block_mem = int(int(total)*percentage - int(used))
    x = torch.cuda.FloatTensor(256, 1024, block_mem, device=torch.device(f"cuda:{options['gpu']}"))
    del x

if __name__ == '__main__':
    friendly = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--ratio', default=0.9, type=float)
    parser.add_argument('--uid', default='0', type=str)
    options = vars(parser.parse_args())
    occupy_memory(options['gpu'], options['ratio'])
    if options['uid'] != '0':
        while track_process(options['uid']):
            time.sleep(1000)
    else:
        while True:
            time.sleep(100)
