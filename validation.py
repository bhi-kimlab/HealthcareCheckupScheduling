import pickle
import copy
import imageio.v2 as imageio
from HEnv import HEnv
from mb_agg import g_pool_cal
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import time
from Params import configs
from dask.utils import SerializableLock
device = torch.device(configs.device)
def gothrough(env,action,env_tmp,depth=20):
    rewards = 0
    rng = np.random.RandomState(int(time.time()))
    st, reward, done = env_tmp.step(action)
    state = st[0]
    candidate = ([i for i in range(env_tmp.stations+1)])
    mask = env_tmp.get_legal_actions()
    rewards += reward
    while(1):
        depth-=1
        if(depth==0):
            return rewards
        station_state = np.copy(env_tmp.station_state_exp)
        station_state[mask==False]=1000000
        action = np.argmin(station_state)

        st, reward, done = env_tmp.step(action,estimate=True)
        state = st[0]
        candidate = ([i for i in range(env_tmp.stations+1)])
        mask = env_tmp.get_legal_actions()
        rewards += reward
        if done:
            return rewards

def gothrough_RL(model,env,action,env_tmp,depth=20,sample=False):
    rewards = 0
    rng = np.random.RandomState(int(time.time()))
    st, reward, done = env_tmp.step(action)
    state = st[0]
    candidate = ([i for i in range(env_tmp.stations+1)])
    mask = env_tmp.get_legal_actions()
    rewards += reward
    while(1):
        depth-=1
        if(depth==0):
            return rewards
        state_tensor = torch.from_numpy(np.copy(state)).to(device)
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        with torch.no_grad():
            pi, _ = model(x=state_tensor,
                            candidate=candidate_tensor.unsqueeze(0),
                            mask=mask_tensor.unsqueeze(0))
        if(sample):
            action = sample_select_action(pi, candidate)
        else:
            action = greedy_select_action(pi, candidate)
        st, reward, done = env_tmp.step(action,estimate=True)
        state = st[0]
        candidate = ([i for i in range(env_tmp.stations+1)])
        mask = env_tmp.get_legal_actions()
        rewards += reward
        if done:
            return rewards



def validate(vali_set, model,vali_q=[],snuh_station=[],n_tr=10,test=False,times=1):
    make_spans = []
    make_spans_mean = []
    RL_greedy = []
    SPT_res = []
    SNUH_res = []
    make_spans4 = []
    make_spans5 = []
    make_spans6 = []
    proposed_res = []

    rng = np.random.RandomState(int(time.time()))
    # rollout using model
    vali_set= vali_set[:]
    vali_q= vali_q[:]

    for data in vali_set:
        for _ in range(1):
            env = HEnv(data,snuh_station[:,2])
            st = env.reset()
            state = st[0]
            candidate = ([i for i in range(env.stations+1)])
            mask = env.get_legal_actions()
            rewards = 0
            reward_list=[]
            if(test==False):
                while True:
                    state_tensor = torch.from_numpy(np.copy(state)).to(device)
                    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                    with torch.no_grad():
                        pi, _ = model(x=state_tensor,
                                    candidate=candidate_tensor.unsqueeze(0),
                                    mask=mask_tensor.unsqueeze(0))
                    action = greedy_select_action(pi, candidate)
                    st, reward, done = env.step(action)
                    state = st[0]
                    candidate = ([i for i in range(env.stations+1)])
                    mask = env.get_legal_actions()
                    rewards += reward
                    if done:
                        break
                reward_list.append(rewards)
        RL_greedy.append(rewards)
    print("RL mean of greedy selection result is:",-sum(RL_greedy)/len(RL_greedy))
    ### greedy selection ###
    i=0
    for data in vali_set:
        env = HEnv(data,snuh_station[:,2])
        st = env.reset()
        state = st[0]
        candidate = ([i for i in range(env.stations+1)])
        mask = env.get_legal_actions()
        rewards = 0
        if(test==True):
            while True:
                station_state = np.copy(env.station_state_exp)
                station_state[mask==False]=1000000
                action = np.argmin(station_state)
                st, reward, done = env.step(action)
                state = st[0]
                candidate = ([i for i in range(env.stations+1)])
                mask = env.get_legal_actions()
                rewards += reward
                if done:
                    break
            im=[]
            im.append(imageio.imread(env.render()[0].to_image()))
            imageio.mimsave(configs.log_dir+"/gif/example_SPT_snuh"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".gif",im)
            env.render()[1].to_csv(configs.log_dir+"/example_SPT_snuh"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".csv")
        SPT_res.append(rewards)
        i+=1
    print("SPT result is:",-sum(SPT_res)/len(SPT_res))
    ##snuh exact solution
    snuh_exact_res =0
    if(len(vali_q)!=0):
        i=0
        for data,q in zip(vali_set,vali_q):
            env = HEnv(data,snuh_station[:,2])
            st = env.reset()
            state = st[0]
            candidate = ([i for i in range(env.stations+1)])
  #          mask = env.get_legal_actions()
            rewards = 0
            if(test==True):
                while True:

                    action = q[env.current_patient].pop(0)
                    st, reward, done = env.step(action)
                    state = st[0]
                    candidate = ([i for i in range(env.stations+1)])
    #               mask = env.get_legal_actions()
                    rewards += reward
                    if done:
                        break
                im=[]
                im.append(imageio.imread(env.render()[0].to_image()))
                imageio.mimsave(configs.log_dir+"/gif/example_exact_snuh"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".gif",im)
                env.render()[1].to_csv(configs.log_dir+"/example_exact_snuh"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".csv")
            SNUH_res.append(rewards)
            i+=1
        print("Exact snuh result is:",-sum(SNUH_res)/len(SNUH_res))
        snuh_exact_res = -sum(SNUH_res)/len(SNUH_res)

#### RL_greedy without data leakage ###
    for data in vali_set:
        re_min = -10000000000
        for _ in range(1):
            env = HEnv(data,snuh_station[:,2])
            st = env.reset()
            state = st[0]
            candidate = ([i for i in range(env.stations+1)])
            mask = env.get_legal_actions()
            rewards = 0
            if(test==True):
                while True:
                    indices_where_true = np.nonzero(mask)
                    random_int = rng.randint(len(indices_where_true[0]),size=1)
                    random_index = np.take(indices_where_true, random_int, axis=1)
                    if(len(indices_where_true[0])==1):
                        action = random_index[0][0]
                    else:
                        tmp_max=-9999999999
                        cand_action=0
                        state_tensor = torch.from_numpy(np.copy(state)).to(device)
                        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                        with torch.no_grad():
                            pi, _ = model(x=state_tensor,
                                    candidate=candidate_tensor.unsqueeze(0),
                                    mask=mask_tensor.unsqueeze(0))
                        for _ in range(n_tr):
                            tmp_action = sample_select_action(pi, candidate)
                            env_tmp = env.deep_copy()
                            tmp_reward = gothrough_RL(model,env, tmp_action,env_tmp,20)
                            if(tmp_reward>=tmp_max):
                                tmp_max = tmp_reward
                                cand_action = tmp_action
                        action = cand_action
                    st, reward, done = env.step(action)
                    state = st[0]
                    candidate = ([i for i in range(env.stations+1)])
                    mask = env.get_legal_actions()
                    rewards += reward
                    if done:
                        break
                im=[]
                im.append(imageio.imread(env.render()[0].to_image()))
                imageio.mimsave(configs.log_dir+"/gif/example_snuh_RL"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".gif",im)
                env.render()[1].to_csv(configs.log_dir+"/example_snuh_RL"+str(configs.n_p)+"_"+str(configs.n_s)+"_"+str(configs.lr)+"_"+str(i)+".csv")
            re_min = max(re_min,rewards)
        i+=1
        proposed_res.append(re_min)
    print("Random selection without leakage RL greedy result is:",-sum(proposed_res)/len(proposed_res))
    log_list = [-sum(SPT_res)/len(SPT_res), -sum(RL_greedy)/len(RL_greedy),snuh_exact_res,-sum(proposed_res)/len(proposed_res)]
    return [np.array(make_spans), log_list]
