from queue import Queue
from tqdm import tqdm
import random

class node:
    def __init__(self, ent, rel=None, pre=None, next=None, level=1):
        self.ent=ent; self.rel=rel
        self.pre=pre; self.next=next
        self.level=level

    def retrace(self):
        path = [(self.rel, self.ent)]; t = self.pre
        while t != None:
            path.append((t.rel, t.ent)); t = t.pre
        return path[::-1]

def gen_filtered_query(graph, query, path_length):
    filter_query = []
    for e1, e2 in tqdm(query):
        if BFS_bool(graph, e1, e2, path_length):
            filter_query.append([e1,e2])
    return filter_query

def BFS_bool(graph, e1, e2, path_length, attempt_thres=5e4):
    attempt = 0; q = Queue(); q.put(node(e1, level=1))
    while not q.empty() and attempt <= attempt_thres:
        cur = q.get(); attempt += 1
        if cur.ent == e2: return True
        if cur.ent in graph and cur.level <= path_length:
            if len(graph[cur.ent])>0:
                for i in graph[cur.ent]:
                    q.put(node(i[1], pre=cur, level=cur.level+1, rel=i[0]))
    return False

def negative_sampling(graph, e1, entity_type, type_e2s, max_len=7, lowest_level=1, max_size=3):
    negative_pairs = []
    q = Queue(); q.put(node(e1, level=1)); l = 1
    while not q.empty() and len(negative_pairs) <= max_size:
        cur = q.get()
        if entity_type[cur.ent] in type_e2s and cur.level>=lowest_level:
            negative_pairs.append((e1, cur.ent))
        if cur.ent in graph and cur.level <= max_len:
            if len(graph[cur.ent]) > 0:
                l+=1
                for i in graph[cur.ent]:
                    q.put(node(i[1], pre=cur, level=l, rel=i[0]))
    negative_pairs = list(set(negative_pairs))
    if len(negative_pairs) >= max_size:
        random.shuffle(negative_pairs); negative_pairs = negative_pairs[:max_size]
    return negative_pairs