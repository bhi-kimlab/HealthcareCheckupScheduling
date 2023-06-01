import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
# import sys
# sys.path.append("models/")

'''
class Attention(nn.Module):
    def __init__(self): super(Attention, self).__init__()

    def forward(self, g_fea, candidates_feas):
        attention_score = torch.mm(candidates_feas, g_fea.t())
        attention_weight = F.softmax(attention_score, dim=0)
        representation_weighted = torch.mm(attention_weight.t(), candidates_feas)
        feas_final = torch.cat((g_fea, representation_weighted), dim=1)
        return feas_final
'''


class StateEmbedding(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 n_p,
                 n_s,
                 # final_dropout,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(StateEmbedding, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.n_p = n_p
        self.n_s = n_s
        self.hidden_dim = hidden_dim
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        #print(n_p,n_s)
        # List of MLPs
        self.mlps_patient = MLP(num_mlp_layers, n_p, hidden_dim, hidden_dim)
        self.mlps_station =MLP(num_mlp_layers, 2, hidden_dim, hidden_dim)
        self.mlps_patient_area =MLP(num_mlp_layers, 5, hidden_dim, hidden_dim)
        self.current = nn.Linear(2+n_s,hidden_dim)
        #self.current = MLP(num_mlp_layers, 2+n_s, hidden_dim, hidden_dim)
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = nn.BatchNorm1d(hidden_dim)

    def forward(self,
                x):
        #p_num(0) // cur_time(1) // cur_patient_state(2~2+n_s+1) //
        #station_state(2+n_s+1~2+(n_s+1)*2) // station_expected_time
        #print("for",x)
        #print("x_shape",x.shape)
        x_p_num=x[:,0]
        x_time = x[:,1]
        x_current = x[:,1:2+self.n_s+1]
        #print("x_cur",x_current)
        #print("x_cur",x_current.shape)
        x_station_st = torch.transpose(x[:,2+self.n_s+1:2+(self.n_s+1)*3].unsqueeze(-1).reshape(-1,2,self.n_s+1),1,2).unsqueeze(-1).reshape(-1,2)
        #print("sta_st",x_station_st)
        #print("sta_st",x_station_st.shape)
        x_patient_st = torch.transpose(x[:,2+(self.n_s+1)*3:2+(self.n_s+1)*3+(self.n_s+1)*self.n_p].unsqueeze(-1).reshape(-1,self.n_p,self.n_s+1),1,2).unsqueeze(-1).reshape(-1,self.n_p)
        #print("pat_st",x_patient_st)
        #print("pat_st",x_patient_st.shape)

        x_patient_area_st = torch.transpose(x[:,2+(self.n_s+1)*3+(self.n_s+1)*self.n_p:].unsqueeze(-1).reshape(-1,self.n_p,5),1,2).unsqueeze(-1).reshape(-1,5)
        #print("pat_area_st",x_patient_area_st.shape)
        t1 = self.mlps_patient(x_patient_st)#.reshape(-1,self.n_s+1,self.hidden_dim)
        #print("t1",t1.shape)
        t1_1 = self.mlps_patient_area(x_patient_area_st)
        #print("t1_1",t1_1.shape)

        t2 = self.mlps_station(x_station_st)#.reshape(-1,self.n_s+1,self.hidden_dim)
        #print("t2",t2.shape)
      #  print(torch.cat((t1,t2),dim=1).shape)
        t3 = F.relu(self.current(x_current))
        #print("t3",t3.shape)
        t4 = torch.mean(t1.reshape(-1,self.n_s+1,self.hidden_dim),dim=1,keepdim=True).reshape(-1,self.hidden_dim)
        t4_1 = torch.mean(t1_1.reshape(-1,self.n_p,self.hidden_dim),dim=1,keepdim=True).reshape(-1,self.hidden_dim)
        t5 = torch.mean(t2.reshape(-1,self.n_s+1,self.hidden_dim),dim=1,keepdim=True).reshape(-1,self.hidden_dim)
        #print("t4",t4.shape)
        #print("t4_1",t4_1.shape)

        #t6 = torch.cat((t4,t5),dim=1)
        t6 = torch.cat((t4,t4_1,t5),dim=1)
        #print("t6",t6.shape)

        #exit(0)
        #print(torch.cat((t3,t6),dim=1).shape)
        #print(torch.cat((t1,t2),dim=1).shape)

        #exit(0)
        return torch.cat((t3,t6),dim=1), torch.cat((t1,t2),dim=1)


if __name__ == '__main__':

    ''' Test attention block
    attention = Attention()
    g = torch.tensor([[1., 2.]], requires_grad=True)
    candidates = torch.tensor([[3., 3.],
                               [2., 2.]], requires_grad=True)

    ret = attention(g, candidates)
    print(ret)
    loss = ret.sum()
    print(loss)

    grad = torch.autograd.grad(loss, g)

    print(grad)
    '''
