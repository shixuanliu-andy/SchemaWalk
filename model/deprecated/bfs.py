from utils.os import load_txt, write_txt, load_json, write_json
from queue import Queue
from tqdm import tqdm

def ins_BFS_bool_(graph, e1, e2, max_len=5, attempt_thres=5e4):
    attempt = 0
    q = Queue()
    q.put(node(e1, level=1))
    l = 1
    while not q.empty() and attempt <= attempt_thres:
        cur = q.get()
        attempt += 1
        if cur.ent == e2:
            return True
        if cur.ent in graph and cur.level <= 4:
            if len(graph[cur.ent])>0:
                for i in graph[cur.ent]:
                    q.put(node(i[1], pre=cur, level=cur.level+1, rel=i[0]))
    return False

def gen_filter_query_sub(graph, sub_query, write_dir):
    filter_query = []
    count = 0
    if len(sub_query[0]) == 2:
        for fact in tqdm(sub_query):
            e1, e2 = fact[0], fact[1]
            res = ins_BFS_bool_(graph, e1, e2, max_len=5)
            if res:
                count += 1
                filter_query.append([e1,e2])
    else:
        for fact in tqdm(sub_query):
            [e1, e2, rel] = fact
            res = ins_BFS_bool_(graph, e1, e2, max_len=5)
            if res:
                count += 1
                filter_query.append([e1,e2,rel])
    write_txt(filter_query, write_dir)
    return

class node:
    def __init__(self,ent,rel=None,pre=None,next=None,level=1):
        self.ent=ent
        self.rel=rel
        self.pre=pre
        self.next=next
        self.level=level

    def back(self, mode='comb'):
        if mode == 'comb':
            path = [(self.rel, self.ent)]
            t = self.pre
            while t != None:
                path.append((t.rel, t.ent))
                t = t.pre
            return path[::-1]
        if mode == 'sep':
            rel_path = [self.rel]
            ent_path = [self.ent]
            t = self.pre
            while t != None:
                if t.rel:
                    rel_path.append(t.rel)
                ent_path.append(t.ent)
                t=t.pre
            return ent_path[::-1], rel_path[::-1]