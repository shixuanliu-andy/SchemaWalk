"""
Created on Fri Jan  8 23:05:24 2021

@author: Shixuan Liu
"""

import numpy as np
import time
import json
import csv

# def load_json(input_dir, serial_key=False):
#     ret_dict = json.load(open(input_dir))
#     if serial_key:
#         ret_dict = {tuple([int(i) for i in k.split('_')]):[tuple(l) for l in v] for k,v in ret_dict.items()}
#     return ret_dict

# def load_txt(input_dir, metapath=False):
#     if metapath:
#         with open(input_dir, 'r') as f:
#             return [line for line in csv.reader(f, delimiter='\t')]
#     else:
#         with open(input_dir, 'r') as f:
#             return [line for line in csv.reader(f, delimiter='\t') if len(line)>1]

# def write_txt(info_list, out_dir):
#     with open(out_dir, 'w') as f:
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerows(info_list)

# def write_json(info_dict, out_dir, serial_key=False):
#     if serial_key:
#         info_dict = {'_'.join([str(i) for i in k]):v for k,v in info_dict.items()}
#     with open(out_dir, "w") as f:
#         json.dump(info_dict, f)