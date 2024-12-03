import json
import csv
import itertools

def load_txt(input_dir, filter=False, merge_list=False):
    if filter:
        ans = [line for line in csv.reader(open(input_dir), delimiter='\t') if len(line)>1]
    else:
        ans = [line for line in csv.reader(open(input_dir), delimiter='\t')]
    if merge_list:
        ans = list(itertools.chain.from_iterable(ans))
    return ans

def write_txt(info_list, out_dir):
    with open(out_dir, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(info_list)

def load_json(input_dir, serial_key=False):
    ret_dict = json.load(open(input_dir))
    if serial_key:
        ret_dict = {tuple([int(i) for i in k.split('_')]):[tuple(l) for l in v] for k,v in ret_dict.items()}
    return ret_dict

def write_json(info_dict, out_dir, serial_key=False):
    if serial_key:
        info_dict = {'_'.join([str(i) for i in k]):v for k,v in info_dict.items()}
    with open(out_dir, "w") as f:
        json.dump(info_dict, f)