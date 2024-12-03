import utils
from collections import defaultdict
import random, csv
import numpy as np
from tqdm import tqdm

def load_ent_dict_txt(input_dir):
    with open(input_dir) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter='\t')
        dict_list = [[l[0], int(l[1])] for l in csv_file if len(l)>=2]
    return dict(dict_list)

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

#Step 1: Load and Generate Dataset
data_name = 'yago'
inv_mode = True

# Load raw dataset
input_dir_base = '../original_data/{}/original/'.format(data_name)
output_dir_embedding = '../original_data/{}/embedding/'.format(data_name)

output_dir_base = '../data/{}/'.format(data_name)
output_dir_instance = '../data/{}/instance/'.format(data_name)
output_dir_schema = '../data/{}/schema/'.format(data_name)


ins_ent_vocab = load_ent_dict_txt(input_dir_base+'entity2id.txt')
sch_ent_vocab = utils.load_json(input_dir_base+'entity_vocab.json')
ins_graph = utils.load_graph(input_dir_base+'ins_graph.txt', relation_mode =1, inv_mode=inv_mode)
ins_graph_list = utils.load_graph(input_dir_base+'ins_graph.txt', relation_mode =0, inv_mode=inv_mode)
sch_graph = utils.load_graph(input_dir_base+'graph.txt', relation_mode =1, inv_mode=inv_mode)
sch_graph_list = utils.load_graph(input_dir_base+'graph.txt', relation_mode =0, inv_mode=inv_mode)
type_vocab = utils.load_type_dict_txt(input_dir_base+'entity2type.txt', ins_ent_dict=ins_ent_vocab,
                                      sch_ent_dict = sch_ent_vocab, rev_mode = 0)
rev_type_vocab = utils.load_type_dict_txt(input_dir_base+'entity2type.txt', ins_ent_dict=ins_ent_vocab,
                                          sch_ent_dict = sch_ent_vocab, rev_mode = 1)

# Generate Instance Graph
# utils.write_file(output_dir_instance+'ins_graph.txt', ins_graph_list)
## Generate Instance Vocab
utils.write_json(output_dir_instance+'entity_vocab.json', ins_ent_vocab)
## Generate Comb_rel Dict
comb_rel_vocab, rel_dict = gen_comb_rel_dict(ins_graph, type_vocab, ins_ent_vocab)
utils.write_json(output_dir_instance+'comb_rel_vocab.json', comb_rel_vocab)
utils.write_json(output_dir_instance+'comb_rel_dict.json', rel_dict)

# ----------------------------------------------------------------------------------------------------- #
# Generate Schema Graph
sch_graph_noinv = {u:v for u,v in sch_graph.items() if 'inv' not in u}
sch_rels = list(sch_graph_noinv.keys())
# random.shuffle(sch_rels)
# sch_rel_to_train = sch_rels[:int(0.8*len(sch_rels))]
# sch_rel_to_test = sch_rels[int(0.8*len(sch_rels)):]
# utils.write_file(output_dir_schema+'sch_graph.txt', sch_graph_list)
# utils.write_file(output_dir_schema+'train_rels.txt', sch_rel_to_train)
# utils.write_file(output_dir_schema+'test_rels.txt', sch_rel_to_test)

## Generate Schema Vocab
utils.write_json(output_dir_schema+'type_vocab.json', sch_ent_vocab)
sch_rels_all = list(sch_graph.keys())
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
embedding_dim = 64
train_embed = True
if train_embed:
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    con = config.Config()
    con.set_in_path(output_dir_embedding)
    con.set_work_threads(8)
    con.set_train_times(3000)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(embedding_dim)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adam")
    # con.set_export_files(output_dir_base+"model.vec.tf", 0)
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

# Generate Comb_rel to (Rel, Type1, Type2) mapping:
comb_rel_decompose = {}
for comb_rel, id_ in comb_rel_vocab.items():
    [rel, type1, type2] = comb_rel.split('__')
    rel_id, type1_id, type2_id = sch_rel_vocab[rel], sch_ent_vocab[type1], sch_ent_vocab[type2]
    comb_rel_decompose[id_] = (rel_id, type1_id, type2_id)
utils.write_json(output_dir_base+'comb_rel_decompose.json', comb_rel_decompose)