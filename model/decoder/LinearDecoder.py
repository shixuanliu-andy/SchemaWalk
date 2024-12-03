import torch
import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, params):
        super(LinearDecoder, self).__init__()
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        if params['use_concept_embeddings']:
            self.input_size = self.hidden_size + 3*self.embedding_size
            self.output_size = 2*self.embedding_size
        else:
            self.input_size = self.hidden_size + self.embedding_size
            self.output_size = self.embedding_size
        
        self.rPAD = params['relation_vocab']['PAD']
        
        self.mlp1= nn.Linear(self.input_size, 4*self.hidden_size, bias=True)
        self.mlp2 = nn.Linear(4*self.hidden_size, self.output_size, bias=True)

    def forward(self, state_query_concat, candidate_action_embeddings, candidate_relations):
        hidden = torch.relu(self.mlp1(state_query_concat))
        output = torch.relu(self.mlp2(hidden)).unsqueeze(1)
        
        prelim_scores = torch.sum(torch.mul(candidate_action_embeddings, output), dim=-1)
        mask = torch.eq(candidate_relations, torch.ones_like(candidate_relations, dtype=torch.int64) * self.rPAD)
        scores = torch.where(mask, torch.ones_like(prelim_scores) * (-99999), prelim_scores)
        logits = torch.nn.functional.log_softmax(scores, dim=1)
        action = torch.multinomial(torch.softmax(scores, dim=1), 1)
        return scores, logits, action