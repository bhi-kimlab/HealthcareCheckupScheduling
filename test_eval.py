from HEnv import HEnv
import sys
import os
import pickle
from uniform_instance_gen import uni_instance_gen
from uniform_instance_gen import uni_instance_gen2
from datetime import datetime
from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import pandas as pd
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
# from PPO_jssp_multiInstances_v2 import PPO
from PPO import PPO

log_dir="log/%s/%d_%d_%d"%(str(configs.log_time), configs.n_p, configs.n_s, configs.lr_name)
configs.log_dir="log/%s/%d_%d_%d"%("eval_test7", configs.n_p, configs.n_s, configs.lr_name)
os.makedirs(configs.log_dir)
os.makedirs(configs.log_dir+"/gif")
if(configs.reward=="sec"):
    snuh_data = np.load('snuh_data/snuh_process/snuh_train_'+str(configs.n_p)+'_'+str(configs.n_s)+'.npy',allow_pickle=True)
    vali_data = np.load('snuh_data/snuh_process/snuh_vali_'+str(configs.n_p)+'_'+str(configs.n_s)+'.npy',allow_pickle=True)
    test_data = np.load('snuh_data/snuh_process/snuh_test_'+str(configs.n_p)+'_'+str(configs.n_s)+'.npy',allow_pickle=True)
    if(configs.no_cap==False):
        snuh_station = pd.read_csv("snuh_data/snuh_time_estimate.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
    with open('snuh_data/snuh_process/snuh_train_%d_%d_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
        snuh_q_list = pickle.load(f)
    with open('snuh_data/snuh_process/snuh_vali_%d_%d_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
        vali_q = pickle.load(f)
    with open('snuh_data/snuh_process/snuh_test_%d_%d_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
        test_q = pickle.load(f)

else:
    snuh_data = np.load('snuh_2019_'+str(configs.n_p)+'_'+str(configs.n_s)+'_min.npy',allow_pickle=True)
    if(configs.no_cap==False):
        snuh_station = pd.read_csv("snuh_time_min.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
    else:
        snuh_station = pd.read_csv("snuh_time_min_nocap.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
    with open('snuh_2019_%d_%d_min_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
        snuh_q_list = pickle.load(f)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_p=configs.n_p,
              n_s=configs.n_s,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)


ppo.policy.load_state_dict(torch.load('%s/%d_%d_%d.pth'%(log_dir,configs.n_p, configs.n_s, configs.vali_model*50-1)))


log_file = open(configs.log_dir+'/reward_log_snuh_'+str(configs.n_p) + '_' + str(configs.n_s)+'_'+str(configs.lr)+'.txt','w')
log_file.writelines("greedy\tgreedy_sample\tmean_sample\tsnuh_exact\trandom_selection\trandom_without_leakage\tRL_greedy_without_leakage\tRL_sampling_without_leakage\tbest_sample\n")
log = []

for i in range(0,20):
    res = validate(test_data, ppo.policy,vali_q = test_q,snuh_station=snuh_station,n_tr=configs.n_tr,test=True,times=i)
    vali_result =  -res[0].mean()
    res[1].append(vali_result)
    log_file.writelines('\t'.join(map(str,res[1]))+'\n')
    log_file.flush()


