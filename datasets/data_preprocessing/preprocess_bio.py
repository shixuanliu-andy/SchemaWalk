import utils
from collections import defaultdict
import random, csv, os
import numpy as np
import pandas as pd
    
def filter_type(vocab_dict):
    types = []
    for i in vocab_dict.values():
        types.extend(list(i))
    type_count = {i:types.count(i) for i in set(types)}
        
    new_vocab = {}
    count = 0
    for key, value in vocab_dict.items():
        if len(value) == 1:
            new_vocab[key] = tuple(value)[0]
            count += 1
        else:
            tmp = [(i, type_count[i]) for i in value]
            tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
            new_vocab[key] = tmp[0][0]
    return new_vocab

def filter_graph_with_type(ins_graph, types, ins_ent_vocab):
    ins_graph_new = []
    sch_graph_new = []
    relation_vocab_new = {"PAD": 0, "DUMMY_START_RELATION": 1, 
                          "NO_OP": 2, "UNK": 3}
    relation2id = {}
    rel_dict = defaultdict(list)
    counter = 4
    counter2 = 0
    for rel, entries in ins_graph.items():
        for e1, e2 in entries:
            if e1 in types and e2 in types:
                new_rel = "__".join([rel, types[e1], types[e2]])
                ins_graph_new.append([e1, new_rel, e2])
                if new_rel not in relation_vocab_new.keys():
                    relation_vocab_new[new_rel] = counter
                    relation2id[new_rel] = counter2
                    counter += 1
                    counter2 += 1
                rel_dict[relation_vocab_new[new_rel]].append([ins_ent_vocab[e1], ins_ent_vocab[e2]])
                new_list = [types[e1], new_rel, types[e2]]
                if new_list not in sch_graph_new:
                    sch_graph_new.append(new_list)
    return ins_graph_new, sch_graph_new, relation_vocab_new, dict(rel_dict), relation2id

#Step 1: Load and Generate Dataset
data_name = 'chembio'
# Load raw dataset
input_dir_base = '../data_preprocessed/{}/original/'.format(data_name)
output_dir_base = '../data_preprocessed/{}/'.format(data_name)
output_dir_vocab = '../data_preprocessed/{}/vocab/'.format(data_name)
output_dir_embedding = '../data_preprocessed/{}/embedding/'.format(data_name)

inv = True
if not os.path.exists(input_dir_base+'ins_graph.txt'):
    ins_graph_list = pd.read_csv(input_dir_base+'chem_bio_interactions.tsv', sep='\t',
                                 header=0, names=['e1','rel','e2'], usecols=[0,1,2])
    ins_graph_list['e1'] = ins_graph_list['e1'].apply(lambda x: '_'.join(x.split('/')[-2:]))
    ins_graph_list['e2'] = ins_graph_list['e2'].apply(lambda x: '_'.join(x.split('/')[-2:]))
    if inv:
        ins_graph_inv = ins_graph_list.copy()
        ins_graph_inv[['e1', 'e2']] = ins_graph_inv[['e2', 'e1']]
        ins_graph_inv['rel'] = ins_graph_inv['rel'].apply(lambda x: x+'_inv')
        ins_graph_list = pd.concat([ins_graph_list, ins_graph_inv])
        del ins_graph_inv
    ins_graph_list.to_csv(input_dir_base+'ins_graph.txt', sep='\t', index=False)
else:
    ins_graph_list = pd.read_csv(input_dir_base+'ins_graph.txt', sep='\t',
                                 header=0, names=['e1','rel','e2'], usecols=[0,1,2])

ins_graph = ins_graph_list.groupby('rel')[['e1', 'e2']].apply(lambda g:list(map(tuple, g.values.tolist()))).to_dict()

type_vocab_list = pd.read_csv(input_dir_base+'node2class.tsv', sep='\t',
                              header=0, names=['ent','type'], usecols=[0,1])
type_vocab_list['ent'] = type_vocab_list['ent'].apply(lambda x: '_'.join(x.split('/')[-2:]))
type_vocab_list['type'] = type_vocab_list['type'].apply(lambda x: x.split('/')[0].replace(' ', '_'))
ents = type_vocab_list['ent'].unique()
types = type_vocab_list['type'].unique()
ins_ent_vocab = {ent:id_ for id_, ent in enumerate(set(ents))}
sch_ent_vocab = {'PAD':0, 'UNK':1}
sch_ent_vocab.update({ent:id_+2 for id_, ent in enumerate(set(types))})
utils.write_json(input_dir_base+'entity_vocab.json', sch_ent_vocab)

type_vocab = dict(type_vocab_list.values.tolist())
type_vocab_list.insert(1, 'is', 'type')
type_vocab_list.to_csv(input_dir_base+'entity2type.txt', sep='\t', index=False)
# type_vocab = type_vocab_list.groupby('ent')['type'].apply(lambda g: map(tuple, g.values.tolist())).to_dict()
rev_type_vocab = {ty:ents.tolist() for ty, ents in type_vocab_list.groupby('type')['ent']}

# Generate Fixed Relation
ins_graph_new, sch_graph_new , relation_vocab_new, rel_dict_new, relation2id= filter_graph_with_type(ins_graph, type_vocab, ins_ent_vocab)
rel_to_reason = [i[1].split('__')[0] for i in sch_graph_new if 'inv' not in i[1]]

utils.write_file(output_dir_base+'graph.txt', sch_graph_new)
utils.write_file(output_dir_base+'ins_graph.txt', ins_graph_new)
with open(output_dir_base+'rel_reason.txt', 'w') as f:
    for i in rel_to_reason:
        f.write(i+'\n')
# Write Files for Vocab
utils.write_json(output_dir_vocab+'relation_vocab.json', relation_vocab_new)
utils.write_json(output_dir_vocab+'entity_vocab.json', sch_ent_vocab)
utils.write_json(output_dir_vocab+'rel_dict.json', rel_dict_new)
# Write Files for Training Embeddings
utils.write_dict_file(output_dir_embedding+'entity2id.txt', ins_ent_vocab)
utils.write_dict_file(output_dir_embedding+'relation2id.txt', relation2id)
utils.write_train2id(output_dir_embedding+'train2id.txt', ins_graph_new, ins_ent_vocab, relation2id)

test = False
if test:
    indice_train_ins = random.sample(range(len(ins_graph_new)), int(0.8*len(ins_graph_new)))
    indice_test_ins = random.sample(range(len(ins_graph_new)), int(0.1*len(ins_graph_new)))
    indice_val_ins = random.sample(range(len(ins_graph_new)), int(0.1*len(ins_graph_new)))
    ins_graph_train = [ins_graph_new[i] for i in indice_train_ins]
    ins_graph_test = [ins_graph_new[i] for i in indice_test_ins]
    ins_graph_val = [ins_graph_new[i] for i in indice_val_ins]
    utils.write_train2id(output_dir_embedding+'train2id.txt', ins_graph_train, ins_ent_vocab, relation2id)
    utils.write_train2id(output_dir_embedding+'test2id.txt', ins_graph_test, ins_ent_vocab, relation2id)
    utils.write_train2id(output_dir_embedding+'valid2id.txt', ins_graph_val, ins_ent_vocab, relation2id)

# raise Exception()
#Step 2: Training Embeddings for Instance Graph
import config
import models
import os

embedding_dim = 64
train_embed = False
if train_embed:
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    con = config.Config()
    con.set_in_path(output_dir_embedding)
    con.set_work_threads(8)
    con.set_train_times(1000)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(embedding_dim)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adam")
    #Models will be exported via tf.Saver() automatically
    # con.set_export_files(output_dir_base+"model.vec.tf", 0)
    #Model parameters will be exported to json files automatically
    con.set_out_files(output_dir_embedding+"embedding.json")
    #Initialize experimental settings
    con.init()
    #Set the knowledge embedding model
    con.set_model(models.TransE)
    #Train the model
    con.run()

# Step 3: Generate Embeddings for Schema Graph
embedding = utils.load_json(output_dir_embedding+"embedding.json")
ins_entity_embeddings = embedding['ent_embeddings']
rel_embeddings = embedding['rel_embeddings']

## Generate Embeddings with Mean Pooling
sch_entity_embbeding = []
for i, j in sch_ent_vocab.items():
    if j in [0,1]:
        sch_entity_embbeding.append(np.random.rand(embedding_dim))
    else:
        sch_entity_embbeding.append(np.mean([ins_entity_embeddings[ins_ent_vocab[ent]] for ent in rev_type_vocab[i]], axis=0))

rel_embbeding = [rel_embeddings[relation2id[i]] if j not in [0,1,2,3] else np.random.rand(embedding_dim) for i,j in relation_vocab_new.items()]

np.savetxt(output_dir_embedding+"pretrained_embeddings_entity.txt", np.matrix(sch_entity_embbeding))
np.savetxt(output_dir_embedding+"pretrained_embeddings_action.txt",np.matrix(rel_embbeding))