import os
from tqdm import tqdm
import torch
from copy import deepcopy
from queue import Queue
from collections import defaultdict
from itertools import chain
import multiprocessing as MP
from utils.configure import set_configure
from utils.utils import write_txt, load_txt
from prediction import Prediction
import numpy as np
    
def cal_rewards_process(metapath_dir, chunk_rels, rev_relation_vocab, comb_rel_vocab, N,
                        device, csr_dict, rel_count, rev_comb_relation_vocab):
    for rel in chunk_rels:
        rel_str = rev_relation_vocab[rel]
        rewards = {}
        counter = 0
        mps = load_txt(os.path.join(metapath_dir, rel_str+'_raw'))
        for mp in mps:
            c1, c2 = mp[0].split('__')[1], mp[-1].split('__')[2]
            query_rel = comb_rel_vocab['__'.join([rel_str, c1, c2])]
            meta_path = tuple([comb_rel_vocab[i] for i in mp])
            
            result = torch.eye(N, device=device).to_sparse().coalesce()
            for comb_rel in meta_path:
                result = result.mm(csr_dict[comb_rel])
            result_masked = result.mul(csr_dict[query_rel])
            body = result._nnz() # number of pairs that have the meta-path
            head_body = result_masked._nnz() # number of pairs that have both query comb_relation and meta-path
            (conf, hc) = (0, head_body/rel_count) if body == 0 else (head_body/body, head_body/rel_count[rel])
            if conf > 0 or hc > 0:
                rewards[mp] = (conf, hc)
            counter += 1
            if len(rewards) == 3:
                break
            if counter % 100000 == 0:
                print (f'Finished {counter}/{len(mps)}')
        mp_list = [[[rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in rewards.items()]
        write_txt(list(chain.from_iterable(mp_list)), os.path.join(metapath_dir, rel_str))
    return

def BFS_process(graph, query, path_length, output_dir, rev_comb_rel_decompose, rev_comb_relation_vocab):
    results = set()
    for [c1, c2] in tqdm(query):
        q = Queue()
        q.put(node(c1, level=1))
        while not q.empty():
            cur = q.get()
            if cur.concept == c2:
                result = [i for i in chain.from_iterable(cur.back()) if i is not None]
                metapath = [rev_comb_rel_decompose[(result[2*i+1], result[2*i], result[2*i+2])] for i in range(len(result)//2)]
                results.add(tuple(metapath))
            if cur.concept in graph and cur.level < path_length+1:
                if len(graph[cur.concept])>0:
                    for i in graph[cur.concept]:
                        q.put(node(i[1], relation=i[0], pre_node=cur, level=cur.level+1))
    mp_list = [[rev_comb_relation_vocab[i] for i in mp] for mp in results]
    write_txt(mp_list, output_dir)
    
class node:
    def __init__(self, concept, relation=None, pre_node=None, level=1):
        self.concept = concept
        self.relation = relation
        self.pre_node = pre_node
        self.level = level

    def back(self):
        path = [(self.relation, self.concept)]
        t = self.pre_node
        while t != None:
            path.append((t.relation, t.concept))
            t = t.pre_node
        return path[::-1]
        
class BFS:
    def __init__(self, params, devices):
        for key, val in params.items():
            setattr(self, key, val);
        self.devices = devices
        self.rev_concept_vocab = {i:j for j,i in self.concept_vocab.items()}
        self.rev_relation_vocab = {i:j for j,i in self.relation_vocab.items()}
        self.rev_comb_relation_vocab = {i:j for j,i in self.comb_rel_vocab.items()}
        self.rev_comb_rel_decompose = {tuple(j):int(i) for i,j in self.comb_rel_decompose.items()}
        self.rel_to_test_txt = load_txt(self.rel_to_test, merge_list=True)
        self.metapath_dir = os.path.join(self.data_input_dir, 'metapaths_all')
        self.prepare_metapath_folder()
        # self.rel_to_test_txt = list(set(self.rel_to_test_txt) - set(os.listdir(self.metapath_dir)))
        self.rel_to_test = [self.relation_vocab[rel] for rel in self.rel_to_test_txt]
        self.full_graph, self.query_dict = defaultdict(list), defaultdict(list)
        for [c1, rel, c2] in load_txt(self.schema_graph):
            c1, c2, rel = self.concept_vocab[c1.strip('_')], self.concept_vocab[c2.strip('_')], self.relation_vocab[rel]
            self.full_graph[c1].append([rel, c2])
        for [c1, rel, c2] in load_txt(self.schema_graph_bfs):
            c1, c2, rel = self.concept_vocab[c1.strip('_')], self.concept_vocab[c2.strip('_')], self.relation_vocab[rel]
            self.query_dict[rel].append([c1, c2])

    def search(self):
        self.graph_dict = {}
        for rel in self.rel_to_test:
            graph = deepcopy(self.full_graph)
            for [c1, c2] in self.query_dict[rel]:
                graph[c1].remove([rel, c2])
            self.graph_dict[rel] = graph

        for rel in self.rel_to_test:
            results = set()
            for [c1, c2] in tqdm(self.query_dict[rel]):
                q = Queue()
                q.put(node(c1, level=1))
                while not q.empty():
                    cur = q.get()
                    if cur.concept == c2:
                        result = [i for i in chain.from_iterable(cur.back()) if i is not None]
                        metapath = [self.rev_comb_rel_decompose[(result[2*i+1], result[2*i], result[2*i+2])] for i in range(len(result)//2)]
                        print ([self.rev_comb_relation_vocab[i] for i in metapath])
                        results.add(tuple(metapath))
                    if cur.concept in graph and cur.level < self.path_length+1:
                        if len(graph[cur.concept])>0:
                            for i in graph[cur.concept]:
                                q.put(node(i[1], relation=i[0], pre_node=cur, level=cur.level+1))
            mp_list = [[self.rev_comb_relation_vocab[i] for i in mp] for mp in results]
            write_txt(mp_list, os.path.join(self.metapath_dir, self.rev_relation_vocab[rel]+'_raw'))
        return
    
    def search_mp(self):
        self.graph_dict = {}
        for rel in self.rel_to_test:
            graph = deepcopy(self.full_graph)
            for [c1, c2] in self.query_dict[rel]:
                graph[c1].remove([rel, c2])
            self.graph_dict[rel] = graph
        rev_comb_rel_decompose_dict = {rel:deepcopy(self.rev_comb_rel_decompose) for rel in self.rel_to_test}
        rev_comb_relation_vocab_dict = {rel:deepcopy(self.rev_comb_relation_vocab) for rel in self.rel_to_test}
        jobs = []
        for rel in self.rel_to_test:
            output_dir = os.path.join(self.metapath_dir, self.rev_relation_vocab[rel]+'_raw')
            p = MP.Process(target=BFS_process, args=(self.graph_dict[rel], self.query_dict[rel], self.path_length,
                                                     output_dir, rev_comb_rel_decompose_dict[rel], rev_comb_relation_vocab_dict[rel]))
            p.start()
            jobs.append(p)
        for p in jobs:
            p.join()
        return

    def cal_rewards_mp(self):
        N = len(self.ins_entity_vocab)
        self.comb_rel_mapping = {tuple(j):int(i) for i,j in self.comb_rel_decompose.items()}
        self.rel_count = defaultdict(int)
        for (c1, rel, c2), comb_rel in self.comb_rel_mapping.items():
            self.rel_count[rel] += len(self.rel_dict[str(comb_rel)])
        # worker_num = len(self.devices)
        worker_num = 2
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        chunk_size = len(self.rel_to_test)//worker_num
        chunk_rels = list(chunks(list(self.rel_to_test), chunk_size))
        csr_dict_list = [{int(i):torch.sparse_coo_tensor(torch.tensor(j).T, torch.ones(len(j)), [N,N]).coalesce().to(device) for i,j in self.rel_dict.items()} for device in self.devices]
        rev_relation_vocab_list = [deepcopy(self.rev_relation_vocab) for _ in range(worker_num)]
        comb_rel_vocab_list = [deepcopy(self.comb_rel_vocab) for _ in range(worker_num)]
        rel_count_list = [deepcopy(self.rel_count) for _ in range(worker_num)]
        rev_comb_relation_vocab_list = [deepcopy(self.rev_comb_relation_vocab) for _ in range(worker_num)]
        jobs = []
        for i in range(worker_num):
            p = MP.Process(target=cal_rewards_process, args=(self.metapath_dir, chunk_rels[i], rev_relation_vocab_list[i], 
                                                             comb_rel_vocab_list[i], N, self.devices[i], csr_dict_list[i],
                                                             rel_count_list[i], rev_comb_relation_vocab_list[i]))
            p.start()
            jobs.append(p)
        for p in jobs:
            p.join()
    
    def cal_rewards(self, part=0):
        self.N = len(self.ins_entity_vocab)
        self.csr_dict = {int(i):torch.sparse_coo_tensor(torch.tensor(j).T, torch.ones(len(j)), [self.N,self.N]).coalesce().to(self.devices) for i,j in self.rel_dict.items()}
        self.comb_rel_mapping = {tuple(j):int(i) for i,j in self.comb_rel_decompose.items()}
        self.rel_count = defaultdict(int)
        for (c1, rel, c2), comb_rel in self.comb_rel_mapping.items():
            self.rel_count[rel] += len(self.rel_dict[str(comb_rel)])
        for rel in self.rel_to_test:
            counter = 0
            rel_str = self.rev_relation_vocab[rel]
            rewards = {}
            mps = load_txt(os.path.join(self.metapath_dir, rel_str+'_raw'))
            for mp in mps:
                try:
                    c1, c2 = mp[0].split('__')[1], mp[-1].split('__')[2]
                    query_rel = self.comb_rel_vocab['__'.join([rel_str, c1, c2])]
                    mp = tuple([self.comb_rel_vocab[i] for i in mp])
                except:
                    print (mp)
                conf, hc = self.get_conf_hc(query_rel, mp, self.rel_count[rel])
                counter += 1
                if conf > 0 or hc > 0:
                    rewards[mp] = (conf, hc)
                if counter % 10000 == 0:
                    print (f'Finished {counter}/{len(mps)}')
            mp_list = [[[self.rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in rewards.items()]
            write_txt(list(chain.from_iterable(mp_list)), os.path.join(self.metapath_dir, self.rev_relation_vocab[rel]))
        return
    
    def rid_circle(self):
        for rel in self.rel_to_test:
            rel_str = self.rev_relation_vocab[rel]
            metapaths_info = load_txt(os.path.join(self.metapath_dir, rel_str))
            mps, confs, hcs = metapaths_info[::3], metapaths_info[1::3], metapaths_info[2::3]
            metapaths_info = list(zip(mps, confs, hcs))
            ans = {}
            for mp_str, conf, hc in tqdm(metapaths_info):
                mp = [i.split('__') for i in mp_str]
                ind = True
                for i in range(len(mp)-1):
                    if mp[i][0] == mp[i+1][0]+'_inv' or mp[i+1][0] == mp[i][0]+'_inv':
                        if mp[i][1] == mp[i+1][2] and mp[i][2] == mp[i+1][1]:
                            ind = False
                if ind:
                    conf, hc = float(conf[0]), float(hc[0])
                    ans[tuple(mp_str)] = (conf, hc)
            mp_list = [[list(mp), [conf], [hc]] for mp, (conf, hc) in ans.items()]
            write_txt(list(chain.from_iterable(mp_list)), os.path.join(self.metapath_dir, rel_str+'_acyclic'))
        return
        
    def get_conf_hc(self, query_rel, meta_path, rel_count):
        result = torch.eye(self.N, device=self.devices).to_sparse().coalesce()
        for rel in meta_path:
            result = result.mm(self.csr_dict[rel])
        result_masked = result.mul(self.csr_dict[query_rel])
        body = result._nnz() # number of pairs that have the meta-path
        head_body = result_masked._nnz() # number of pairs that have both query comb_relation and meta-path
        (conf, hc) = (0, head_body/rel_count) if body == 0 else (head_body/body, head_body/rel_count)
        # assert conf <= 1 and hc <= 1
        return conf, hc
    
    def prepare_metapath_folder(self):
        if not os.path.exists(self.metapath_dir):
            os.mkdir(self.metapath_dir)

    def store_metapaths(self, rel):
        rel_dict = {mp: (conf, hc) for mp, (conf, hc) in self.environment_test.repos[rel].items() if conf > 0 or hc > 0}
        mp_list = [[[self.rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in rel_dict.items()]
        write_txt(list(chain.from_iterable(mp_list)), self.metapath_dir+self.rev_relation_vocab[rel])

if __name__ == '__main__':
    train, test = False, False
    search, cal_rewards, rid_circle = False, False, True
    data_name = 'yago'
    # data_name = 'nell'
    target_rel = None
    logger = None
    options, logger = set_configure(data_name, target_rel, logger, train, test)
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{options['gpu']}" if torch.cuda.is_available() else "cpu")
    BFSearcher = BFS(options, device)
    if test:
        if search:
            BFSearcher.search()
        if cal_rewards:
            BFSearcher.cal_rewards(part=0)
        if rid_circle:
            BFSearcher.rid_circle()
            raise Exception()
    options['metapath_dir'] = BFSearcher.metapath_dir
    prediction = Prediction(options, logger, device)
    prediction.pred_times = 20
    # prediction.pool = 'sum'
    # hit_1, hit_3, hit_5, hit_10, mrr = prediction.predict()
    # logger.info(f'Hits@1: {hit_1}, Hits@3: {hit_3}, Hits@5: {hit_5}, Hits@10: {hit_10}, MRR: {mrr}')
    
    prediction.pool = 'mean'
    hit_1, hit_3, hit_5, hit_10, mrr = prediction.predict()
    logger.info(f'Hits@1: {hit_1}, Hits@3: {hit_3}, Hits@5: {hit_5}, Hits@10: {hit_10}, MRR: {mrr}')
    
    prediction.pool = 'max'
    hit_1, hit_3, hit_5, hit_10, mrr = prediction.predict()
    logger.info(f'Hits@1: {hit_1}, Hits@3: {hit_3}, Hits@5: {hit_5}, Hits@10: {hit_10}, MRR: {mrr}')