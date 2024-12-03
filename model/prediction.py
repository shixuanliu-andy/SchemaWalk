import os, random
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix, identity
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import itertools
import numpy as np
from math import ceil
import torch
from utils.utils import load_txt, write_txt, write_json, load_json
from utils.node import gen_filtered_query, negative_sampling

class Prediction(object):
    def __init__(self, params, logger, device, target_rel=None):
        for key, val in params.items():
            setattr(self, key, val);
        self.logger = logger
        self.device = device
        self.rel_to_test = load_txt(self.rel_to_test, merge_list=True) if type(self.rel_to_test)==str else self.rel_to_test
        # Create Sparse Matrices Mapping
        self.N = len(self.entity_vocab)
        self.rev_entity_vocab = {j:i for i,j in self.entity_vocab.items()}
        if self.target_rel != 'multi':
            self.entity_type = {self.entity_vocab[i]:j for [i,j] in load_txt(self.entity_type) if i in self.entity_vocab}
        self.comb_rel_dict = {int(i):j for i,j in self.rel_dict.items()}
        self.comb_rel_decompose = {int(i):j for i,j in self.comb_rel_decompose.items()}
        self.gen_dataset()
        if self.target_rel == 'multi':
            write_json(self.dataset, params['output_dir']+'testset.json')

    def predict(self, kg_valid=False):
        if self.eval_mode == 'KGC': return self.KG_Completion(valid=kg_valid)
        if self.eval_mode == 'LP': return self.Link_Prediction()
    def gen_dataset(self):
        if self.eval_mode == 'KGC': self.gen_dataset_kgc()
        if self.eval_mode == 'LP': self.gen_dataset_lp()

    def KG_Completion(self, valid=False):
        self.load_metapaths()
        self.metapaths_connects = {u:self.cal_metapath_connect(v) for u,v in tqdm(self.metapaths.items())}
        hit_1s, hit_3s, hit_5s, hit_10s, MRRs = [], [], [], [], []
        for _ in tqdm(range(self.pred_times)):
            dataset_len = self.dataset_len
            self.dataset_cal = self.dataset if not valid else {i:random.sample(j, len(j)//self.valid_times) for i,j in self.evalset.items()}
            self.cal_scores()
            hit_1, hit_3, hit_5, hit_10, MRR = 0, 0, 0, 0, 0
            for rel, pairs in self.dataset_cal.items():
                rel_scores = self.scores[rel]
                if all([len(i)==i for i in rel_scores.values()]) == True:
                    dataset_len -= len(rel_scores)
                for e1, ans in pairs:
                    # if e1 not in rel_scores: continue
                    if len(rel_scores[e1]) == 0: continue
                    e2_scores = rel_scores[e1]
                    sorted_answers = sorted(e2_scores, key=e2_scores.get, reverse=True)
                    ans_pos = sorted_answers.index(ans) if ans in sorted_answers else None
                    if ans_pos != None:
                        MRR += 1.0/((ans_pos+1))
                        if ans_pos < 10:
                            hit_10 += 1
                            if ans_pos < 5:
                                hit_5 += 1
                                if ans_pos < 3:
                                    hit_3 += 1
                                    if ans_pos < 1:
                                        hit_1 += 1
            hit_1s.append(hit_1/dataset_len)
            hit_3s.append(hit_3/dataset_len)
            hit_5s.append(hit_5/dataset_len)
            hit_10s.append(hit_10/dataset_len)
            MRRs.append(MRR/dataset_len)
            # if valid: break
        return np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_5s), np.mean(hit_10s), np.mean(MRRs)

    def Link_Prediction(self, regress_split=0.6):
        self.load_metapaths()
        self.metapaths_connects = {u:self.cal_metapath_connect(v) for u,v in tqdm(self.metapaths.items())}
        metapaths_connects = {i:{(j,k):l for [j,k,l] in t} for i,t in self.metapaths_connects.items()}
        aps, AUCs = [], []
        for i in range(self.pred_times):
            features = []
            for e1, e2, y in self.dataset:
                feature_vec = [0]*len(self.metapaths)
                for mp, pair_info in metapaths_connects.items():
                    feature_vec[mp] = self.metapaths_conf[mp] if (e1,e2) in pair_info else 0
                features.append(feature_vec + [y])
            random.shuffle(features);
            features = np.array(features)
            features, y_vec = features[:,:-1], features[:,-1]
            #Prune Feature
            indice = np.sum(features, axis=0)!=0; features = features[:, indice]
            train_count = int(len(self.dataset)*regress_split)
            if self.lp_pool == 'max':
                features = np.expand_dims(np.max(features, axis=-1), axis=-1)
            elif self.lp_pool == 'sum':
                features = np.expand_dims(np.sum(features, axis=-1), axis=-1)
            train, test = features[:train_count, :], features[train_count:, :]
            train_y, test_y = y_vec[:train_count], y_vec[train_count:]
            model = Lasso(alpha=1e-5, max_iter=100000); model.fit(train, train_y)
            pre_y = model.predict(test)
            fpr, tpr, thresholds = roc_curve(test_y, pre_y)
            ap, auc_ = average_precision_score(test_y, pre_y), auc(fpr, tpr)
            self.logger.info(f'Round {i} - AP: {ap}, AUC: {auc_}')
            aps.append(ap); AUCs.append(auc_)
        return np.mean(aps), np.mean(AUCs)
    
    def load_metapaths(self):
        self.metapaths, self.metapaths_conf, self.metapaths_rel = [], [], []
        for f in os.listdir(self.metapath_dir):
            if '.txt' in f: continue
            metapaths_info = load_txt(os.path.join(self.metapath_dir, f))
            metapaths, conf = metapaths_info[::3], metapaths_info[1::3]
            self.metapaths.extend([[self.comb_rel_vocab[rel] for rel in path] for path in metapaths])
            self.metapaths_conf.extend([float(i[0]) for i in conf])
            self.metapaths_rel.extend([self.relation_vocab[f]]*len(metapaths))
        self.metapaths = {u:v for u,v in enumerate(self.metapaths)}
        self.metapaths_conf = {u:v for u,v in enumerate(self.metapaths_conf)}
        self.metapaths_rel = {u:v for u,v in enumerate(self.metapaths_rel)}
        self.logger.info(f'Loaded {len(self.metapaths)} Metapaths')
        
    def gen_dataset_kgc(self, index=None, valid_times=3):
        rel_to_test = [self.rel_to_test[index]] if index is not None else self.rel_to_test
        self.dataset, self.evalset, self.dataset_len = {}, {}, 0
        self.valid_times = valid_times
        all_triples = [i for i in load_txt(self.instance_graph) if i[1] in rel_to_test]
        for rel in rel_to_test:
            pairs = [(self.entity_vocab[e1],self.entity_vocab[e2]) for (e1,rel_,e2) in all_triples if rel_==rel]
            random.shuffle(pairs)
            data_len = ceil(self.pred_ratio*len(pairs))
            selected_pairs = pairs[:data_len]
            self.evalset[self.relation_vocab[rel]] = list(set(pairs)-set(selected_pairs))[:valid_times*data_len]
            self.dataset[self.relation_vocab[rel]] = selected_pairs
            self.dataset_len += len(selected_pairs)

    def gen_dataset_lp(self, test_split=0.2):
        self.graph, self.triples, self.query = defaultdict(set), [], []
        for comb_rel, value in self.comb_rel_dict.items():
            rel, _, _ = self.comb_rel_decompose[comb_rel]
            self.triples.extend([(e1, rel, e2) for e1, e2 in value])
            if self.relation_vocab[self.target_rel]==rel:
                self.query.extend([tuple(i) for i in value])
            else:
                for e1, e2 in value: self.graph[e1].add((rel, e2))
        self.query = list(set(self.query))
        # Generate Filtered Query
        filter_query_dir = self.link_prediction_dir+self.target_rel+'.filter_query'
        if not os.path.exists(filter_query_dir):
            self.filter_query = gen_filtered_query(self.graph, self.query, self.path_length+1)
            filter_query_str = [[self.rev_entity_vocab[e1], self.target_rel, self.rev_entity_vocab[e2]]
                                for [e1, e2] in self.filter_query]
            write_txt(filter_query_str, filter_query_dir)
        else:
            self.filter_query = [(self.entity_vocab[e1],self.entity_vocab[e2]) for (e1,rel,e2) in load_txt(filter_query_dir)]
        # Sample Testset
        testset_dir = self.link_prediction_dir+self.target_rel+'.testset'
        if not os.path.exists(testset_dir):
            random.shuffle(self.filter_query)
            self.testset = self.filter_query[:int(len(self.filter_query)*test_split)]
            testset_str = [[self.rev_entity_vocab[e1], self.target_rel, self.rev_entity_vocab[e2]]
                            for [e1, e2] in self.testset]
            write_txt(testset_str, testset_dir)
        else:
            self.testset = [(self.entity_vocab[e1], self.entity_vocab[e2]) for (e1,rel,e2) in load_txt(testset_dir)]
        self.gen_negset_lp()
        
    def gen_negset_lp(self):
        type_e2s = set([self.entity_type[e2] for e1,e2 in self.query])
        pairs, neg_pairs, existing_query = [tuple(l) for l in self.testset], set(), set(self.query)
        triples = self.triples.copy(); random.shuffle(triples)
        for e1, _, e2 in triples:
            neg_pair_sampled = set(negative_sampling(self.graph, e1, self.entity_type, type_e2s))
            neg_pairs = neg_pairs | (neg_pair_sampled - existing_query)
            if len(neg_pairs) >= int(0.5*len(pairs)): break
        self.dataset = [[l[0], l[1], 1] for l in pairs] + [[l[0], l[1], 0] for l in neg_pairs]

    def cal_metapath_connect(self, meta_path):
        csr_dict = defaultdict(csr_matrix)
        for rel in list(meta_path):
            indices, data = np.array(self.comb_rel_dict[rel]), np.ones(len(self.comb_rel_dict[rel]))
            csr_dict[rel] = csr_matrix((data, (indices[:,0], indices[:,1])), shape=(self.N, self.N))
        result = identity(self.N)
        for rel in meta_path:
            result *= csr_dict[rel]
        result = np.concatenate([result.nonzero(), result.data.reshape(1,-1)]).astype(int).T.tolist()
        return tuple(result)

    def cal_scores(self, valid=False):
        self.scores = defaultdict(dict)
        for rel, pairs in self.dataset_cal.items():
            valid_metapaths_ids = [u for u,v in self.metapaths_rel.items() if v==rel]
            if len(valid_metapaths_ids) == 0:
                score_pairs = {e1:{} for [e1,e2] in pairs}
            else:
                mp_connects = {u:v for u,v in self.metapaths_connects.items() if u in valid_metapaths_ids and len(v)>0}
                mp_connects = [[(e1, e2, count, mp_id) for (e1, e2, count) in pairs] for mp_id, pairs in mp_connects.items()]
                mp_connects = list(itertools.chain.from_iterable(mp_connects))
                score_pairs = defaultdict(dict)
                for e1, _ in pairs:
                    score_targets = defaultdict(list)
                    e1_connects = [(e2, count, mp_id) for (e1_, e2, count, mp_id) in mp_connects if e1_==e1]
                    if self.aggregation == 'bool':
                        for (e2_guess, _, mp_id) in e1_connects:
                            score_targets[e2_guess].append(self.metapaths_conf[mp_id])
                    if self.aggregation == 'count':
                        for (e2_guess, count, mp_id) in e1_connects:
                            score_targets[e2_guess].extend([self.metapaths_conf[mp_id]] * count)
                    score_pairs[e1] = self.similarity_function(score_targets)
            self.scores[rel] = score_pairs

    def similarity_function(self, score_targets):
        if self.pool == 'sum':
            return {e2:sum(confs) for e2, confs in score_targets.items()}
        if self.pool == 'max':
            return {e2:max(confs) for e2, confs in score_targets.items()}
        if self.pool == 'mean':
            return {e2:sum(confs)/len(confs) for e2, confs in score_targets.items()}