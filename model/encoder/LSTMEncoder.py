import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, params, device, train):
        super(LSTMEncoder, self).__init__()
        self.device = device
        self.num_rollouts = params['num_rollouts'] if train else params['test_rollouts']
        self.batch_size = params['batch_size'] if train else params['test_batch_size']
        self.sample_size = self.batch_size * self.num_rollouts
        self.hidden_size = params['hidden_size']
        self.embedding_size = params['embedding_size']
        self.use_concept_embeddings = params['use_concept_embeddings']
        self.input_size = 2*self.embedding_size if self.use_concept_embeddings else self.embedding_size
        self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.init_state()

    def init_state(self):
        self.history_state = [torch.zeros([self.sample_size, self.hidden_size], device=self.device) for _ in range(2)]
        
    def beam_update(self, path_idx):
        self.history_state = [i[path_idx, :] for i in self.history_state]

    def forward(self, prev_action_embedding):
        hx, cx = self.lstm(prev_action_embedding, self.history_state)
        self.history_state = [hx, cx]
        return hx