import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
from models.state_embedding import StateEmbedding
import torch


class ActorCritic(nn.Module):
    def __init__(self,
                 n_p,
                 n_s,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_p = n_p
        # machine size for problems, no business with network
        self.n_s = n_s
        self.n_ops_perjob = n_s
        self.device = device
        self.feature_extract = StateEmbedding(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        n_p = self.n_p,
                                        n_s = self.n_s,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*6, hidden_dim_actor, 1).to(device)
        #self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*5, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim*4, hidden_dim_critic, 1).to(device)
#        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim*3, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                candidate,
                mask,
                ):

        h_pooled, h_nodes = self.feature_extract(x=x)
        # prepare policy feature: concat omega feature with global feature
#        print(candidate)
#        print(mask)
#        print(h_nodes)
#        print("adfasdfadsf")
        #print("h_nodes",h_nodes.shape)
#        print("h_pooled",h_pooled.shape)
#        exit(0)
        #print(h_nodes.shape)
        #print(candidate.shape)
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_s+1, h_nodes.size(-1))
 #       print(dummy)
 #       print(dummy.shape)
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand([candidate_feature.shape[0],candidate_feature.shape[1],h_pooled.unsqueeze(1).shape[2]])
  #      print(candidate_feature.shape)
   #     print(h_pooled_repeated.shape)
        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_p, self.n_s))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''

        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        #print("concateFea",concateFea.shape)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        #print("cand_scores",candidate_scores.shape)
        candidate_scores[mask_reshape==False] = float('-inf')
      #  print(candidate_scores)
        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        #print("v",v.shape)
        return pi, v


if __name__ == '__main__':
    print('Go home')
