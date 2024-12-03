"""
Created on Fri Jan  8 23:05:24 2021

@author: Shixuan Liu
"""

import numpy as np
import json
import csv
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict

def load_txt(input_dir, filter_=False):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        if filter_:
            return [line for line in csv_file if len(line)>1]
        else:
            return [line for line in csv_file]

def load_rels(input_dir):
    ans = []
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        for line in csv_file:
            ans.extend(line)
    return ans

def load_json(input_dir):
    return json.load(open(input_dir))

def load_ent_dict_txt(input_dir):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        try:
            dict_list = [[int(l[0]), int(l[1])] for l in csv_file if len(l)>=2]
        except:
            dict_list = [[int(l[0]), l[1]] for l in csv_file if len(l)>=2]
    return dict(dict_list)

def load_graph_yago(input_dir, ent_dict={}, rel_dict={},
                    num_mode=0, relation_mode=1, inv_mode=True,
                    nell=False, create_vocab=False, load_vocab=None):
    cols = ['e1','rel','e2']
    graph_triple = pd.read_csv(input_dir, sep='\t', header=0, names=cols, usecols=[1,2,3])
    graph_triple = graph_triple.apply(lambda x: x.str.strip('<>'))
    if load_vocab:
        graph_triple = graph_triple[graph_triple['e1'].isin(load_vocab.values()) & graph_triple['e2'].isin(load_vocab.values())]
    # graph_dict = {rel:info[['e1','e2']].values.tolist() for rel,info in graph_triple.groupby('rel')}
    if create_vocab:
        instances = set(graph_triple['e1'].unique()).union(graph_triple['e2'].unique())
        ins_vocab = {i:j for i,j in enumerate(instances)}
    if relation_mode == 0:
        returns = (graph_triple, ins_vocab) if create_vocab else graph_triple
    else:
        graph_dict = graph_triple.groupby('rel')[['e1', 'e2']].apply(lambda g:list(map(tuple, g.values.tolist()))).to_dict()
        if relation_mode == 1:
            returns = (graph_dict, ins_vocab) if create_vocab else graph_dict
        if relation_mode == 2:
            returns = (graph_dict, graph_triple, ins_vocab) if create_vocab else (graph_dict, graph_triple)
    return returns

def load_type_dict_txt_yago(input_dir, ins_ent_dict={}, sch_ent_dict={},
                       num_mode=0, collection_mode=1, rev_mode=1):
    ins_types = set()
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        if not collection_mode:
            dict_list = []
            for l in tqdm(csv_file):
                if len(l) <2:
                    continue
                if 'type' not in l[1]:
                    continue
                instance = l[0].strip('<>')
                ins_type = '_'.join(l[2].strip(' .<>0123456789').split('_')[1:])
                if instance in ins_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if not rev_mode:
                        dict_list.append([instance, ins_type])
                    else:
                        dict_list.append([ins_type, instance])
            return dict(dict_list)
        if collection_mode:
            ret = defaultdict(set)
            for l in tqdm(csv_file):
                if len(l) <2:
                    continue
                if 'type' not in l[1]:
                    continue
                instance = l[0].strip('<>')
                ins_type = l[2].strip('. ')
                ins_types.update(ins_type)
                if instance in ins_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if rev_mode: 
                        ret[ins_type].add(instance)
                    else:
                        ret[instance].add(ins_type)
            return dict(ret)

def load_graph(input_dir, ent_dict={}, rel_dict={},
               num_mode=0, relation_mode=1, inv_mode=True, nell=False):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        store = defaultdict(list)
        store_triple = []
        for line in csv_file:
            e1 = line[0]
            rel = line[1]
            if nell:
                rel = rel.split(':')[1]
            e2 = line[2]
            if not inv_mode and 'inv' in rel:
                # print (rel)
                continue
            if num_mode:
                e1 = ent_dict[e1]
                e2 = ent_dict[e2]
                rel = rel_dict[rel]
            if relation_mode:
                store[rel].append([e1, e2])
            else:
                store_triple.append([e1,rel,e2])
        if relation_mode:
            return dict(store)
        else:
            return store_triple

def load_biotype_dict_txt(input_dir, ins_ent_dict={}, sch_ent_dict={},
                       num_mode=0, collection_mode=1, rev_mode=1):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        if not collection_mode:
            dict_list = []
            for line in csv_file:
                instance = '_'.join(line[0].split('/')[-2:])
                ins_type = '_'.join(line[1].split(' '))
                ins_type = ins_type.replace('/', '_or_')
                if instance in ins_ent_dict.keys() and ins_type in sch_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if not rev_mode:
                        dict_list.append([instance, ins_type])
                    else:
                        dict_list.append([ins_type, instance])
            return dict(dict_list)
        if collection_mode:
            ret = defaultdict(set)
            for line in csv_file:
                instance = '_'.join(line[0].split('/')[-2:])
                ins_type = '_'.join(line[1].split(' '))
                ins_type = ins_type.replace('/', '_or_')
                if instance in ins_ent_dict.keys() and ins_type in sch_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if rev_mode: 
                        ret[ins_type].add(instance)
                    else:
                        ret[instance].add(ins_type)
            return dict(ret)

def load_type_dict_txt(input_dir, ins_ent_dict={}, sch_ent_dict={},
                       num_mode=0, collection_mode=1, rev_mode=1):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        if not collection_mode:
            dict_list = []
            for line in csv_file:
                instance = line[0]
                ins_type = '_'.join(line[2].split('_')[1:-1])
                if instance in ins_ent_dict.keys() and ins_type in sch_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if not rev_mode:
                        dict_list.append([instance, ins_type])
                    else:
                        dict_list.append([ins_type, instance])
            return dict(dict_list)
        if collection_mode:
            ret = defaultdict(set)
            for line in csv_file:
                instance = line[0]
                ins_type = '_'.join(line[2].split('_')[1:-1])
                if instance in ins_ent_dict.keys() and ins_type in sch_ent_dict.keys():
                    if num_mode:
                        instance = ins_ent_dict[instance]
                        ins_type = sch_ent_dict[ins_type]
                    if rev_mode: 
                        ret[ins_type].add(instance)
                    else:
                        ret[instance].add(ins_type)
            return dict(ret)
        
def write_file(dire, file_list):
    if type(file_list[0]) is str:
        with open(dire, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for line in file_list:
                writer.writerow([line])
    else:
        with open(dire, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(file_list)

def write_dict_file(dire, dict_list):
    with open(dire, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([len(dict_list)])
        for key, value in dict_list.items():
            writer.writerow([key, value])
        
def write_train2id(dire, ins_graph_new, entity_vocab, rel_vocab):
    with open(dire, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([len(ins_graph_new)])
        for line in ins_graph_new:
            writer.writerow([entity_vocab[line[0]], entity_vocab[line[2]], rel_vocab[line[1]]])
    
def write_json(dire, vocab_dict):
    with open(dire, 'w') as f:
        json.dump(vocab_dict, f)