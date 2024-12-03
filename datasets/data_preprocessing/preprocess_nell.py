import utils
from collections import defaultdict
import random
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
from typetree import typetree, node

def gen_comb_rel_dict(ins_graph, types, ins_ent_vocab):
    comb_rel_vocab = {}
    rel_dict = defaultdict(list)
    counter = 0
    for rel in tqdm(ins_graph.keys()):
        ins_item = ins_graph[rel]
        for line in ins_item:
            [e1, e2] = line
            for type1 in types[e1]:
                for type2 in types[e2]:
                    new_rel = "__".join([rel, type1, type2])
                    if new_rel not in comb_rel_vocab.keys():
                        comb_rel_vocab[new_rel] = counter
                        counter += 1
                    rel_dict[comb_rel_vocab[new_rel]].append([ins_ent_vocab[e1], ins_ent_vocab[e2]])
    return comb_rel_vocab, rel_dict

def write_type(out_dir, type_vocab):
    to_write = []
    for k, vs in type_vocab.items():
        for v in vs:
            to_write.append([k,'type', v.strip('_')])
    utils.write_file(out_dir, to_write)
    return


def gen_sch_vocab(type_vocab):
    types = []
    for value in type_vocab.values():
        types.extend(value)
    sch_vocab = [(name.strip('_'), id_+2) for id_, name in enumerate(set(types))]
    sch_vocab.extend([('PAD', 0), ('UNK', 1)])
    return dict(sch_vocab)

def gen_rev_type_vocab(type_vocab):
    rev_type_vocab = defaultdict(list)
    for key, value in type_vocab.items():
        for val in value:
            rev_type_vocab[val.strip('_')].append(key)
    return rev_type_vocab

def filter_type(vocab_dict):
    types = []
    for i in vocab_dict.values():
        types.extend(list(i))
    type_count = {i.strip('_'):types.count(i) for i in set(types)}
        
    new_vocab = {}
    count = 0
    for key, value in vocab_dict.items():
        if len(value) == 1:
            new_vocab[key] = tuple(value)[0]
            count += 1
        else:
            tmp = [(i, type_count[i.strip('_')]) for i in value]
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
    for rel in tqdm(ins_graph.keys()):
        ins_item = ins_graph[rel]
        for line in ins_item:
            e1 = line[0]
            e2 = line[1]
            for type1 in types[e1]:
                for type2 in types[e2]:
                    new_rel = "__".join([rel, type1, type2])
                    ins_graph_new.append([e1, new_rel, e2])
                    if new_rel not in relation_vocab_new.keys():
                        relation_vocab_new[new_rel] = counter
                        relation2id[new_rel] = counter2
                        counter += 1
                        counter2 += 1
                    rel_dict[relation_vocab_new[new_rel]].append([ins_ent_vocab[e1], ins_ent_vocab[e2]])
                    if [type1, new_rel, type2] not in sch_graph_new:
                        sch_graph_new.append([type1, new_rel, type2])
    return ins_graph_new, sch_graph_new, relation_vocab_new, dict(rel_dict), relation2id

data_name = 'nell'
inv_mode = True
input_dir_base = '../original_data/{}/original/'.format(data_name)
output_dir_embedding = '../original_data/{}/embedding/'.format(data_name)

output_dir_base = '../data/{}/'.format(data_name)
output_dir_instance = '../data/{}/instance/'.format(data_name)
output_dir_schema = '../data/{}/schema/'.format(data_name)


ins_graph = utils.load_graph(input_dir_base+'NELL_DONE', relation_mode =1, inv_mode=inv_mode, nell=True)
ins_graph_list = utils.load_graph(input_dir_base+'NELL_DONE', relation_mode =0, inv_mode=inv_mode, nell=True)
type_vocab = utils.load_json(input_dir_base+'NELL_ent2type_DONE.json')
type_pool = set(itertools.chain.from_iterable(list(type_vocab.values())))
type_pool_list = list(itertools.chain.from_iterable(list(type_vocab.values())))
type_num = {i:type_pool_list.count(i) for i in type_pool}
rel_pool = set(ins_graph.keys())

ins_ent_vocab = dict([(name, id_) for id_, name in enumerate(type_vocab)])
# Generate Instance Graph
# utils.write_file(output_dir_instance+'ins_graph.txt', ins_graph_list)
## Generate Instance Vocab
# utils.write_json(output_dir_instance+'entity_vocab.json', ins_ent_vocab)

ont_nell = pd.read_csv(input_dir_base+'NELL.08m.1115.ontology.csv', sep='\t', error_bad_lines=False)
ont_nell = ont_nell[ont_nell['Entity'].str.contains('concept:')]
ont_nell = ont_nell[ont_nell['Value'].str.contains('concept:')]
ont_nell = ont_nell[ont_nell['Relation'].str.contains('generalizations')]
ont_nell['Entity'] = ont_nell['Entity'].str.split(':').str[1]
ont_nell['Value'] = ont_nell['Value'].str.split(':').str[1]
ont_nell = ont_nell[ont_nell["Entity"].isin(type_pool)]
ont_nell_list = ont_nell.values.tolist()
type_tree = typetree(ont_nell_list)

filtered_type_vocab = {}
MAX_TYPE = 2
for ent, type_list in tqdm(type_vocab.items()):
    res = type_tree.filter_typelist(type_list)
    candidate_set=set([])
    for pp in res:
        candidate_set.add(pp[-1])
    if len(candidate_set)>MAX_TYPE:
        sorted_ty=[(i, type_num[i]) for i in candidate_set]
        sorted_ty=sorted(sorted_ty,key=lambda x:x[1],reverse=True)
        candidate_set=[ge[0] for ge in sorted_ty[:MAX_TYPE]]
    filtered_type_vocab[ent] = list(candidate_set)

# Generate Comb_rel Dict
# comb_rel_vocab_full, rel_dict_full = gen_comb_rel_dict(ins_graph, type_vocab, ins_ent_vocab)
# utils.write_json(output_dir_instance+'comb_rel_vocab_full.json', comb_rel_vocab_full)
# utils.write_json(output_dir_instance+'comb_rel_dict_full.json', rel_dict_full)
comb_rel_vocab, rel_dict = gen_comb_rel_dict(ins_graph, filtered_type_vocab, ins_ent_vocab)
# utils.write_json(output_dir_instance+'comb_rel_vocab.json', comb_rel_vocab)
# utils.write_json(output_dir_instance+'comb_rel_dict.json', rel_dict)

# ----------------------------------------------------------------------------------------------------- #
# Generate Schema Graph
# sch_graph_list = [i.split('__') for i in comb_rel_vocab.keys()]
# sch_graph_list = [[j,i,k] for [i,j,k] in sch_graph_list]
# sch_rels = [i for i in rel_pool if not 'inverseof' in i]
# random.shuffle(sch_rels)
# sch_rel_to_train = sch_rels[:int(0.8*len(sch_rels))]
# sch_rel_to_test = sch_rels[int(0.8*len(sch_rels)):]
# utils.write_file(output_dir_schema+'sch_graph.txt', sch_graph_list)
# utils.write_file(output_dir_schema+'train_rels.txt', sch_rel_to_train)
# utils.write_file(output_dir_schema+'test_rels.txt', sch_rel_to_test)

## Generate Schema Vocab
sch_ent_vocab = gen_sch_vocab(filtered_type_vocab)
utils.write_json(output_dir_schema+'type_vocab.json', sch_ent_vocab)
sch_rels_all = list(rel_pool)
sch_rel_vocab = {"PAD": 0, "DUMMY_START_RELATION": 1, "NO_OP": 2, "UNK": 3}
sch_rel_vocab.update(dict(zip(sch_rels_all, range(4,4+len(sch_rels_all)))))
utils.write_json(output_dir_schema+'rel_vocab.json', sch_rel_vocab)
## Generate Schema Embedding
utils.write_dict_file(output_dir_embedding+'entity2id.txt', ins_ent_vocab)
relation2id = dict(zip(sch_rels_all, range(len(sch_rels_all))))
utils.write_dict_file(output_dir_embedding+'relation2id.txt', relation2id)
utils.write_train2id(output_dir_embedding+'train2id.txt', ins_graph_list, ins_ent_vocab, relation2id)
import config
import models
import os

# # Generate Fixed Relation
# # type_vocab_new = filter_type(type_vocab)
# write_type(input_dir_base+'entity2type.txt', type_vocab)
# ins_graph_new, sch_graph_new , relation_vocab_new, rel_dict_new, relation2id= filter_graph_with_type(ins_graph, type_vocab, ins_ent_vocab)
# sch_ent_vocab = gen_sch_vocab(type_vocab)
rev_type_vocab = gen_rev_type_vocab(type_vocab)

embedding_dim = 64
train_embed = True
if train_embed:
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    con = config.Config()
    con.set_in_path(output_dir_embedding)
    con.set_work_threads(16)
    con.set_train_times(3000)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(embedding_dim)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adam")
    con.set_out_files(output_dir_embedding+"embedding.json")
    con.init()
    con.set_model(models.TransE)
    con.run()
    
embedding = utils.load_json(output_dir_embedding+"embedding.json")
ins_entity_embeddings = embedding['ent_embeddings']
rel_embeddings = embedding['rel_embeddings']
sch_entity_embbeding = []
for i, j in sch_ent_vocab.items():
    if j in [0,1]:
        sch_entity_embbeding.append(np.random.rand(embedding_dim))
    else:
        sch_entity_embbeding.append(np.mean([ins_entity_embeddings[ins_ent_vocab[ent]] for ent in rev_type_vocab[i]], axis=0))
rel_embbeding = [rel_embeddings[relation2id[i]] if j not in [0,1,2,3] else np.random.rand(embedding_dim) for i,j in sch_rel_vocab.items()]

np.savetxt(output_dir_schema+"pretrained_type_embeddings.txt", np.matrix(sch_entity_embbeding))
np.savetxt(output_dir_schema+"pretrained_rel_embeddings.txt",np.matrix(rel_embbeding))

comb_rel_decompose = {}
for comb_rel, id_ in comb_rel_vocab.items():
    [rel, type1, type2] = comb_rel.split('__')
    type1, type2 = type1.strip('_'), type2.strip('_')
    rel_id, type1_id, type2_id = sch_rel_vocab[rel], sch_ent_vocab[type1], sch_ent_vocab[type2]
    comb_rel_decompose[id_] = (rel_id, type1_id, type2_id)
utils.write_json(output_dir_base+'comb_rel_decompose.json', comb_rel_decompose)