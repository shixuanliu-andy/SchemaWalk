from utils.configure import set_configure
from utils.utils import load_txt, write_txt, load_json, write_json
from utils.bfs import ins_BFS_bool_, gen_filter_query_sub, node
import os, csv, random, json
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from queue import Queue
from copy import deepcopy
from scipy.sparse import csr_matrix, identity
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class Downstream(object):
    def __init__(self, params):
        for key, val in params.items(): setattr(self, key, val);
        # Load schema-level dictionaries
        self.rev_relation_vocab = {u:v for v,u in self.relation_vocab.items()}
        self.rev_entity_vocab = {u:v for v,u in self.entity_vocab.items()}
        # Load instance-level dictionaries
        self.ins_rel_dict = {int(key):val for key, val in self.rel_dict.items()}
        self.ins_entity_vocab = {l[0]:int(l[1]) for l in load_txt(self.ins_entity_vocab)}
        self.load_type()
        self.init_data()
    
    def load_type(self):
        entity_type = []
        with open(self.entity_type, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if 'chembio' in self.data_name:
                    ent, type_ = line[0], line[2]
                if 'yago' in self.data_name:
                    ent, type_ = line[0], '_'.join(line[2].split('_')[1:-1])
                if 'nell' in self.data_name:
                    ent, type_ = line[0], line[2].strip('_')
                if ent in self.ins_entity_vocab and type_ in self.entity_vocab:
                    entity_type.append([self.ins_entity_vocab[ent], self.entity_vocab[type_]])
        self.entity_type = dict(entity_type)
        if 'nell' in self.data_name:
            self.graph_dict = defaultdict(dict)
            for key, value in self.graph.items():
                temp = defaultdict(list)
                for entry in value:
                    temp[entry[0]].append(entry[1])
                self.graph_dict[key] = temp
            nell_type = load_json(self.data_input_dir+'original/NELL_ent2type_DONE.json')
            self.nell_type = {self.ins_entity_vocab[i]:[k for k in j if k in self.entity_vocab] for i,j in nell_type.items()}

    def init_data_per_rel(self):
        for key, value in self.ins_rel_dict.items():
            entries = [[e1, key, e2] for e1, e2 in value]
            self.triples.extend(entries)
            if self.target_rel is not None:
                if self.target_rel==self.rev_relation_vocab[key].split('__')[0] and 'inv' not in self.rev_relation_vocab[key]:
                    self.query.extend([tuple(i) for i in value])
                else:
                    for e1, e2 in value:
                        self.graph[e1].append([key, e2])
        self.query = list(set(self.query))
        return
    
    def init_data_multi_rel(self, query_rate = 0.1):
        for key, value in self.ins_rel_dict.items():
            entries = [[e1, e2, key] for e1, e2 in value]
            self.triples.extend([[e1, key, e2] for e1, e2 in value])
            random.shuffle(entries)
            if 'inv' not in self.rev_relation_vocab[key]:
                queries, graph = entries[:int(query_rate*len(entries))], entries[int(query_rate*len(entries)):]
                self.query.extend([tuple(i) for i in queries])
            else:
                graph = entries
            for [e1, e2, key] in graph:
                self.graph[e1].append([key, e2])
        self.query = list(set(self.query))
        return

    def init_data(self):
        self.triples, self.query, self.graph = [], [], defaultdict(list)
        if self.target_rel is not None:
            self.init_data_per_rel()
        else:
            self.init_data_multi_rel()
    
    def gen_testset(self, test_split=0.2):
        # Generate/Load Filter Query (Generation would take long time, saved as file after generation)
        if self.target_rel:
            filter_query_dir = self.eval_data+self.target_rel+'.filter_query'
            testset_dir = self.eval_data+self.target_rel+'.testset'
        else:
            filter_query_dir = self.eval_data+'multi.filter_query'
            testset_dir = self.eval_data+'multi.testset'
        if not os.path.exists(filter_query_dir):
            self.gen_filter_query_mp(filter_query_dir)
        else:
            self.filter_query = load_txt(filter_query_dir)
            self.filter_query = [[int(line[0]), int(line[1])] for line in self.filter_query]
        # Generate/Load Testset
        if not os.path.exists(testset_dir):
            random.shuffle(self.filter_query)
            self.testset = self.filter_query[:int(len(self.filter_query)*test_split)]
            write_txt(self.testset, testset_dir)
        else:
            self.testset = load_txt(testset_dir)
            self.testset = [[int(line[0]), int(line[1])] for line in self.testset]
        return
    
    def link_prediction(self):
        self.gen_testset()
        self.gen_dataset()
        self.pca_threshold = 0.0001 if 'chembio' in self.data_name else 0.001
        self.load_metapaths()
        print('Feature Nums: {}'.format(len(self.metapaths)))
        feature_list, y_list = self.cal_feature()
        print('{}/{} Queries get feature'.format(sum([1 if sum(i)>0 else 0 for i in feature_list]), len(self.dataset)))
        conf_list = [[sum(i)] for i in feature_list]
        precision, recall, pr_auc, ap, fpr, tpr, AUC = self.regression(conf_list, y_list)
        print (f'AUC: {AUC}, ap: {ap}')
        return precision, recall, pr_auc, ap, fpr, tpr, AUC

    def regression(self, feature_list, y_list, regress_split=0.6):
        train = feature_list[:int(len(feature_list) * regress_split)]
        test = feature_list[int(len(feature_list) * regress_split):]
        train_y = y_list[:int(len(y_list) * regress_split)]
        test_y = y_list[int(len(y_list) * regress_split):]
        model = Lasso(alpha=0.00001, max_iter=100000)
        model.fit(train, train_y)
        pre_y = model.predict(test)
        # probs_y = model.predict_proba(test)[:, 1]
        fpr, tpr, thresholds = roc_curve(test_y, pre_y)
        precision, recall, _ = precision_recall_curve(test_y, pre_y)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(test_y, pre_y)
        AUC = auc(fpr, tpr)
        return precision, recall, pr_auc, ap, fpr, tpr, AUC

    def gen_dataset(self):
        # Load negative samples provided by chembio
        if 'chembio' in self.data_name:
            pos_pairs = self.testset.copy()
            pos_pairs = [l+[1] for l in pos_pairs]
            neg_set_dir = self.eval_data + self.target_rel + '.neg'
            negstr_set_dir = self.eval_data_str + self.target_rel + '.neg_str'
            if not os.path.exists(neg_set_dir):
                neg_pairs = load_txt(negstr_set_dir)
                neg_pairs = [[self.ins_entity_vocab[l[0]], self.ins_entity_vocab[l[1]]] for l in neg_pairs]
                write_txt(neg_pairs, neg_set_dir)
                neg_pairs = [l+[0] for l in neg_pairs]
            else:
                neg_pairs = load_txt(negstr_set_dir)
                neg_pairs = [l+[0] for l in neg_pairs]
            random.shuffle(neg_pairs)
            self.dataset = pos_pairs + neg_pairs[:int(0.5*len(pos_pairs))]
            return
        # Else, generate negative samples
        self.type_e2s = set([self.entity_type[i[1]] for i in self.query])
        pairs = [tuple(l) for l in self.testset]
        neg_pairs = []
        indice = list(range(len(self.triples)))
        random.shuffle(indice)
        for idx in indice:
            entry = self.triples[idx]
            e1, e2 = entry[0], entry[1]
            neg_pair_l = self.negative_sampling(e1)
            if neg_pair_l:
                for neg_pair in neg_pair_l:
                    if (neg_pair is not None) and (neg_pair not in self.query):
                        neg_pairs.append(neg_pair)
                    if len(neg_pairs) % 100 == 0 and len(neg_pairs) > 0:
                        print('len(neg pairs):', len(neg_pairs))
            if len(neg_pairs) >= int(0.5*len(pairs)):
                break
        self.dataset = [[line[0], line[1], 1] for line in pairs]
        self.dataset.extend([[line[0], line[1], 0] for line in neg_pairs])
        return
    
    def negative_sampling(self, e1, max_len=7, lowest_level=1):
        negative_pairs = []
        q = Queue()
        q.put(node(e1, level=1))
        l = 1
        while not q.empty() and len(negative_pairs) <= 3:
            cur = q.get()
            if self.entity_type[cur.ent] in self.type_e2s and cur.level>=lowest_level:
                negative_pairs.append((e1, cur.ent))
            if cur.ent in self.graph and cur.level <= max_len:
                if len(self.graph[cur.ent]) > 0:
                    l+=1
                    for i in self.graph[cur.ent]:
                        q.put(node(i[1], pre=cur, level=l, rel=i[0]))
        if len(negative_pairs) >= 3:
            random.shuffle(negative_pairs)
            return negative_pairs[:3]
        elif len(negative_pairs)>0:
            return negative_pairs
        else:
            return []
    
    def load_metapaths(self):
        self.metapaths = {}
        for f in os.listdir(self.metapath_dir):
            print ('Loading {}'.format(f))
            metapaths_ = load_txt(os.path.join(self.metapath_dir, f), metapath=True)
            metapaths, conf, hc = metapaths_[::3], metapaths_[1::3], metapaths_[2::3]
            metapaths = [self.relpath2id(i) for i in metapaths]
            scores = [float(i[0]) for i in conf]
            metapaths = [[u,v] for u,v in zip(metapaths, scores)]
            metapaths = filter(lambda x:x[1]>self.pca_threshold, metapaths)
            self.metapaths.update({tuple(line[0]):line[1] for line in metapaths})
        return
    
    def cal_arrival(self, metapath):
        graph_dict, N = defaultdict(csr_matrix), len(self.ins_entity_vocab)
        for rel in metapath:
            row, col, data = [], [], []
            for [src, tgt] in self.ins_rel_dict[int(rel)]:
                row.append(src)
                col.append(tgt)
                data.append(1.0)
            graph_dict[rel] = csr_matrix((deepcopy(data),
                                          (deepcopy(row),
                                           deepcopy(col))), shape=(N, N))
        result = identity(N)
        for i in range(len(metapath)):
            result *= graph_dict[int(metapath[i])]
        result = result.tocoo()
        indices = tuple(list(zip(result.row.tolist(), result.col.tolist())))
        return indices
        
    def cal_feature(self, nell=False):
        if not 'nell' in self.data_name:
            print ('Calculating Arrival for each metapath')
            self.feature_arrival = {i:self.cal_arrival(i) for i in tqdm(self.metapaths)}
        feature_vec = {i:0 for i in self.metapaths}
        feature_list, y_list = [], []
        print ('Calculating Feature Representation')
        for entry in tqdm(self.dataset):
            feature_vec_ = deepcopy(feature_vec)
            e1, e2, y = entry[0], entry[1], entry[2]
            for path in feature_vec_:
                if self.check_connection(path, e1, e2):
                    feature_vec_[path] = self.metapaths[path]
            feature_list.append(list(feature_vec_.values()))
            y_list.append(y)
        rn = list(range(len(feature_list)))
        random.shuffle(rn)
        feature_list = [feature_list[i] for i in rn]
        y_list = [y_list[i] for i in rn]
        #Prune Feature
        feature_list = np.array(feature_list)
        indice = np.sum(feature_list, axis=0)!=0
        return feature_list[:, indice], y_list
    
    def relpath2id(self, path):
        return [self.relation_vocab[rel] for rel in path]

    def check_connection(self, path, e1, e2):
        # For evaluation on NELL, we restore node types
        if 'nell' in self.data_name:
            return self.check_connection_nell(path, e1, e2)
        else:
            return (e1,e2) in self.feature_arrival[path]
    
    def check_connection_nell(self, path, e1, e2):
        q = Queue()
        q.put(node(e1, level=1))
        while not q.empty():
            cur = q.get()
            if cur.ent == e2 and cur.level == len(path)+1:
                return True
            if cur.level <= len(path):
                next_rel = path[cur.level-1]
                for next_rel in self.get_equivalent_rel(next_rel, cur.ent):
                    if next_rel in self.graph_dict[cur.ent]:
                        for i in self.graph_dict[cur.ent][next_rel]:
                            if self.rev_relation_vocab[next_rel].split('__')[2] in self.nell_type[i]:
                                q.put(node(i, pre=cur, level=cur.level+1, rel=next_rel))
        return False

    def get_equivalent_rel(self, next_rel, ent):
        next_rel = self.rev_relation_vocab[next_rel]
        type_ent = self.nell_type[ent]
        base_rel = next_rel.split('__')[0]
        start_ent = next_rel.split('__')[1]
        # This 'if' logic ensures connectivity is based on metapath, not logical rule
        if start_ent not in type_ent:
            return next_rel
        else:
            candidate_rel = [i for i in self.relation_vocab if base_rel in i]
            candidate_rel = [self.relation_vocab[i] for i in candidate_rel if i.split('__')[1] in type_ent]
            return candidate_rel
    
    def gen_filter_query_mp(self, filter_query_dir, worker_num = 15):
        # This is the multi-processing version of filter query
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        chunk_size = len(self.query)//worker_num
        sub_queries = list(chunks(list(self.query), chunk_size))
        jobs = []
        graphs = [deepcopy(self.graph) for _ in range(len(sub_queries))]
        write_dirs = [filter_query_dir+f'_{i}' for i in range(len(sub_queries))]
        for i in range(len(sub_queries)):
            p = mp.Process(target=gen_filter_query_sub, args=(graphs[i], sub_queries[i], write_dirs[i]))
            p.start()
            jobs.append(p)
        for p in jobs:
            p.join()
        files = [load_txt(i) for i in write_dirs]
        self.filter_query = []
        for file in files:
            self.filter_query.extend(file)
        if self.target_rel:
            self.filter_query = [[int(line[0]), int(line[1])] for line in self.filter_query]
        else:
            self.filter_query = [[int(line[0]), int(line[1]), int(line[2])] for line in self.filter_query]
        write_txt(self.filter_query, filter_query_dir)
    
    def gen_filter_query(self):
        print ('Filtering Query')
        self.filter_query = []
        count = 0
        for fact in tqdm(self.query):
            e1, e2 = fact[0], fact[1]
            res = self.ins_BFS_bool(e1, e2, max_len=5)
            if res:
                count += 1
                self.filter_query.append([e1,e2])
        return
    
    def ins_BFS_bool(self, e1, e2, max_len=5, attempt_thres=5e4):
        attempt = 0
        q = Queue()
        q.put(node(e1, level=1))
        l = 1
        while not q.empty() and attempt <= attempt_thres:
            cur = q.get()
            attempt += 1
            if cur.ent == e2:
                return True
            if cur.ent in self.graph and cur.level <= self.path_length:
                if len(self.graph[cur.ent])>0:
                    for i in self.graph[cur.ent]:
                        q.put(node(i[1], pre=cur, level=cur.level+1, rel=i[0]))
        return False

if __name__ == '__main__':
    # data_name = 'yago'
    # target_rel = 'isCitizenOf'
    # target_rel = 'diedIn'
    # target_rel = 'graduatedFrom'
    # data_name = 'nell'
    # target_rel = 'teamplaysagainstteam'
    # target_rel = 'competeswith'
    # target_rel = 'worksfor'
    data_name = 'chembio'
    target_rel = 'bind'
    options, logger, config = set_configure(data_name,target_rel)

    downstreamer = Downstream(options)
    path_logger_file = downstreamer.path_logger_file
        
    output_dir = downstreamer.output_dir
    metrics = defaultdict(list)
    # target_rel = 'isCitizenOf'
    for _ in range(5):
        precision, recall, pr_auc, ap, fpr, tpr, AUC = downstreamer.link_prediction(target_rel)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['pr_auc'].append(pr_auc)
        metrics['ap'].append(ap)
        metrics['fpr'].append(fpr)
        metrics['tpr'].append(tpr)
        metrics['AUC'].append(AUC)
    print (np.mean(metrics['AUC']), np.std(metrics['AUC']))
    print (np.mean(metrics['pr_auc']), np.std(metrics['pr_auc']))