import numpy as np
import pandas as pd

class Replayer(object):
    def __init__(self, batch_size, memory_max, prioritized=True):
        self.memory = pd.DataFrame()
        self.insample_size = batch_size
        self.memory_max = memory_max
        self.alpha = 0.7
        self.prioritized = prioritized
    
    def __len__(self):
        return self.memory.shape[0]
    
    def store(self, data):
        if self.memory.shape[0] + self.insample_size >= self.memory_max:
            if self.memory.shape[0] > 0:
                to_remove = np.random.choice(self.memory.shape[0], self.insample_size, replace=False)
                self.memory = self.memory.drop(to_remove)
        self.memory = pd.concat([self.memory, data], ignore_index=True)
        if self.prioritized:
            self.memory = self.memory.sort_values(by=['advantage_abs'] , ascending=False)
    
    def sample(self, size=None):
        if self.prioritized:
            self.prob = np.arange(self.memory.shape[0])+1
            self.prob = np.power(self.prob, -self.alpha)
            self.prob /= np.sum(self.prob)
            indices = np.random.choice(self.memory.shape[0], size=self.insample_size, p=self.prob)
        else:
            indices = np.random.choice(self.memory.shape[0], size=self.insample_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)
        
class Replayer_rank(object):
    def __init__(self, config):
        self.memory = pd.DataFrame()
        self.insample_size = config.batch_size
        self.memory_max = self.insample_size*20
        self.alpha = 0.7
    
    def __len__(self):
        return self.memory.shape[0]
    
    def store(self, data):
        if self.memory.shape[0] + self.insample_size >= self.memory_max:
            to_remove = np.random.choice(self.memory.shape[0], self.insample_size, replace=False)
            self.memory = self.memory.drop(to_remove)
        self.memory = pd.concat([self.memory, data], ignore_index=True)
        self.memory = self.memory.sort_values(by=['advantage'] , ascending=False)
    
    def sample(self, size=None):
        self.prob = np.arange(self.memory.shape[0])+1
        self.prob = np.power(self.prob, -self.alpha)
        self.prob /= np.sum(self.prob)
        indices = np.random.choice(self.memory.shape[0], size=self.insample_size, p=self.prob)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)
    
if __name__ == '__main__':
    class config():
        def __init__(self):
            self.batch_size = 64
            self.max_length = 12
            self.input_dimension = 64
    config = config()
    replayer = Replayer(config)
    encoder_out = np.ones([config.batch_size, config.max_length, config.input_dimension])
    action = np.ones([config.batch_size, config.max_length, config.max_length])
    pis = np.ones([config.batch_size, 1])
    # reward = np.ones([config.batch_size, 1])
    advantage = np.random.randint(5, size=[config.batch_size, 1])
    package = list(zip(encoder_out, action, pis, advantage))
    df = pd.DataFrame(package, columns=['observation','action','pis','advantage'])
    for _ in range(21):
        replayer.store(df)
    print (len(replayer))
    observation, actions, pis, advantage = replayer.sample()
    print (observation)
    
    