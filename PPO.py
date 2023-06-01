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

device = torch.device(configs.device)


class Memory:
    def __init__(self):
        self.state_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.state_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_p,
                 n_s,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_p=n_p,
                                  n_s=n_s,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_p) + '_' + str(n_s) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        state_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            if(len(memories[i].candidate_mb)==1):
                continue
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # process each env data
            state_mb_t = torch.stack(memories[i].state_mb).to(device)
            state_mb_t = state_mb_t.reshape(-1, state_mb_t.size(-1))
            state_mb_t_all_env.append(state_mb_t)

            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        #mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=state_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i])
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def main():

    from HEnv import HEnv
    import sys
    import os
    import pickle
    from uniform_instance_gen import uni_instance_gen
    from uniform_instance_gen import uni_instance_gen2
    data_generator = uni_instance_gen2
    from datetime import datetime
    now = datetime.now()
    log_dir="log/%s/%d_%d_%d"%(now.strftime("%Y%m%d%H%M"), configs.n_p, configs.n_s, configs.lr_name)
    #arr_dist=np.load('arrival_time_dist.npy',allow_pickle=True)
    configs.log_dir = log_dir
    os.makedirs(log_dir)
    os.makedirs(log_dir+"/gif")
#    dataLoaded = np.load('./DataGen/generatedData' + str(configs.n_p) + '_' + str(configs.n_s) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    if(configs.reward=="sec"):
        snuh_data = np.load('snuh_data/snuh_process/snuh_train_'+str(configs.n_p)+'_'+str(configs.n_s)+'.npy',allow_pickle=True)
        vali_data = np.load('snuh_data/snuh_process/snuh_vali_'+str(configs.n_p)+'_'+str(configs.n_s)+'.npy',allow_pickle=True)
        if(configs.no_cap==False):
            snuh_station = pd.read_csv("snuh_data/snuh_time_estimate.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
        with open('snuh_data/snuh_process/snuh_train_%d_%d_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
            snuh_q_list = pickle.load(f)
        with open('snuh_data/snuh_process/snuh_vali_%d_%d_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
            vali_q = pickle.load(f)
    else:
        snuh_data = np.load('snuh_2019_'+str(configs.n_p)+'_'+str(configs.n_s)+'_min.npy',allow_pickle=True)
        if(configs.no_cap==False):
            snuh_station = pd.read_csv("snuh_time_min.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
        else:
            snuh_station = pd.read_csv("snuh_time_min_nocap.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
        with open('snuh_2019_%d_%d_min_q_list.pkl' %(configs.n_p, configs.n_s), 'rb') as f:
            snuh_q_list = pickle.load(f)

    train_data = snuh_data
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]
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
    log_file = open(log_dir+'/reward_log_snuh_'+str(configs.n_p) + '_' + str(configs.n_s)+'_'+str(configs.lr)+'.txt','w')
    log_file.writelines("SPT\tRL_greedy\tsnuh_exact\tproposed_model\n")
        # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000

    res = validate(vali_data, ppo.policy,vali_q,snuh_station,configs.n_tr)
    vali_result = res[1][1]
    validation_log.append(vali_result)

    snuh_exact_res = res[1][2]
    if vali_result < record:
        torch.save(ppo.policy.state_dict(), './{}.pth'.format(
            str(configs.n_p) + '_' + str(configs.n_s) + '_' + str(configs.low) + '_' + str(configs.high)))
        record = vali_result
    print('The validation quality is:', vali_result)
    file_writing_obj1 = open(log_dir+
        '/' + 'snuh_vali_' + str(configs.n_p) + '_' + str(configs.n_s) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
    file_writing_obj1.write(str(validation_log))
    log_file.writelines('\t'.join(map(str,res[1]))+'\n')
    log_file.flush()

    for i_update in range(configs.max_updates):
        t3 = time.time()
        ep_rewards = [0 for _ in range(configs.num_envs)]
        state_envs = []
        candidate_envs = []
        mask_envs = []
        if(configs.train=="random"):
            envs = [HEnv(data_generator(configs.n_p,configs.n_s,snuh_station,arr_dist),snuh_station[:,2]) for i in range(configs.num_envs)]
        else:
            envs = [HEnv(train_data[i_update%250],snuh_station[:,2]) for i in range(configs.num_envs)]
        for i, env in enumerate(envs):
            state = env.reset()
            state_envs.append(state[0])
            candidate_envs.append([i for i in range(env.stations+1)])
            mask_envs.append(env.get_legal_actions())
            ep_rewards[i] = - env.initQuality
        # rollout the env
        #print(candidate_envs)
        #print(state_envs)
        #print(mask_envs)
        while True:
            state_tensor_envs = [torch.from_numpy(np.copy(state)).to(device) for state in state_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                skip=0
                for i in range(configs.num_envs):
                    if(envs[i]._is_done()):
                        action_envs.append(-1)
                        a_idx_envs.append(-1)
                        skip+=1
                        continue
                    #print("i",i,len(state_tensor_envs))
                    pi, _ = ppo.policy_old(x=state_tensor_envs[i],
                                           candidate=candidate_tensor_envs[i].unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))
                    #print(candidate_envs[i])
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            state_envs = []
            candidate_envs = []
            mask_envs = []
            # Saving episode data
            done=True
            for i in range(configs.num_envs):
                if envs[i]._is_done() == False:
                    done=False
                    memories[i].state_mb.append(state_tensor_envs[i])
                    memories[i].candidate_mb.append(candidate_tensor_envs[i])
                    memories[i].mask_mb.append(mask_tensor_envs[i])
                    memories[i].a_mb.append(a_idx_envs[i])
                    state, reward, done = envs[i].step(action_envs[i])
                    #print("i",i,state,done)
                    state_envs.append(state[0])
                    candidate_envs.append([i for i in range(envs[i].stations+1)])
                    mask_envs.append(envs[i].get_legal_actions())
                    ep_rewards[i] += reward
                    memories[i].r_mb.append(reward)
                    memories[i].done_mb.append(done)
                else:
                    '''
                    memories[i].state_mb.append([])
                    memories[i].candidate_mb.append([])
                    memories[i].mask_mb.append([])
                    memories[i].a_mb.append([])
                    memories[i].r_mb.append([])
                    memories[i].done_mb.append([])
                    state_envs.append([])
                    candidate_envs.append([])
                    mask_envs.append([])
                    '''
            if done:
                break
#        for j in range(configs.num_envs):
#            ep_rewards[j] -= envs[j].posRewards
        loss, v_loss = ppo.update(memories, configs.n_p*configs.n_s, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open(log_dir+'/' + 'log_snuh_' + str(configs.n_p) + '_' + str(configs.n_s) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        if(configs.print_log==True):
            print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(i_update + 1, mean_rewards_all_env, v_loss))

        # validate and save use mean performance

        t4 = time.time()
        if (i_update + 1) % 50 == 0:
            res = validate(vali_data, ppo.policy,snuh_station=snuh_station,n_tr=configs.n_tr)
            vali_result =  res[1][1]
            res[1][2] = snuh_exact_res
            validation_log.append(vali_result)
            print(res[1])
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), log_dir+'/{}.pth'.format(
                    str(configs.n_p) + '_' + str(configs.n_s) + '_best'))
                record = vali_result
            print('The validation quality is:', vali_result)
            torch.save(ppo.policy.state_dict(), log_dir+'/{}.pth'.format(str(configs.n_p) + '_' + str(configs.n_s) + '_'+str(i_update)))
            file_writing_obj1 = open(
                log_dir+'/' + 'snuh_vali_' + str(configs.n_p) + '_' + str(configs.n_s) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))
            log_file.writelines('\t'.join(map(str,res[1]))+'\n')
            log_file.flush()
        t5 = time.time()

        # print('Training:', t4 - t3)
        # print('Validation:', t5 - t4)


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)
