import torch
import itertools
import sys
import random
import numpy as np
def _greedy_osspa(input_data,n_job,n_machine,is_real= False, mode = 'rand'):
    # input data는 batch 로 들어온다고 가정함

    #일단 graph 하나마다 처리하는 logic을 만들자..
    data = input_data[:,0]
    mask = torch.zeros(data.shape, dtype = torch.bool)
    arrival_time, arrival_job_index = torch.sort(data[-n_job:])

    machine_waiting_time = torch.zeros(n_machine, dtype = torch.float64)
    machine_procedure_time = torch.zeros(n_machine, dtype=torch.float64)
    current_selected_job = [arrival_job_index[0]]
    if is_real:
        #print(data[:-n_job].view(n_machine,n_job))
        fixed_machine_prc_time ,_= data[:-n_job].view(n_machine,n_job).max(1)
        total_prc_time = data[:-n_job].view(n_machine,n_job).sum(1)
        #print( fixed_machine_prc_time)
    i = 1
    mask[current_selected_job[0] + n_job*n_machine] = True
    mask[(data == 0)] = True
    current_time =0
    non_idle = torch.zeros(n_machine, dtype= torch.bool)
    machine_queue = [[] for _ in range(n_machine)]
    ongoing_machine_job = [-1 for _ in range(n_machine)]
    #job_time = [0 for _ in range(n_job)]
    job_time = torch.zeros(n_job)
    pi = []

    while mask.all() != True or (machine_procedure_time != 0).any():
        # select minimum waiting time machine
        for selected_job in current_selected_job:
            masked_waiting_time = machine_waiting_time.clone().detach()
            #print(selected_job)
            masked_waiting_time[mask.view(n_machine+1,n_job)[:-1,selected_job]] = 99999
            #print(masked_waiting_time)
            min_val, min_index = masked_waiting_time.min(dim = 0)
            if min_val > 1000:
                continue
            if min_val == 0 and is_real:
                if mode == 'max':
                    temp_fixed = fixed_machine_prc_time.clone().detach()
                    temp_fixed[masked_waiting_time != 0] = -9999
                    _, min_index = temp_fixed.max(0)
                elif mode == 'min':
                    temp_fixed = fixed_machine_prc_time.clone().detach()
                    temp_fixed[masked_waiting_time != 0] = 9999
                    _, min_index = temp_fixed.min(0)
                elif mode == 'total_max':
                    temp_fixed = total_prc_time.clone().detach()
                    temp_fixed[masked_waiting_time != 0] = -9999
                    _, min_index = temp_fixed.max(0)
                elif mode == 'rand':
                    candidate_idx = (masked_waiting_time==0).nonzero().squeeze(-1)
                    rnd_idx = random.randint(0, len(candidate_idx)-1)
                    min_index = candidate_idx[rnd_idx]


            non_idle[min_index ] = True
            # 현재 처리중인 job이 없을 경우 바로 실행
            if ongoing_machine_job[min_index] == -1:
                assert machine_procedure_time[min_index] == 0
                machine_procedure_time[min_index] += data[selected_job + min_index * n_job]
                ongoing_machine_job[min_index] = selected_job
            else : machine_queue[min_index].append(selected_job)
            mask[selected_job + min_index* n_job] = True
            machine_waiting_time[min_index] += data[selected_job + min_index * n_job]
            if selected_job + min_index* n_job < n_job*n_machine: pi.append((selected_job + min_index* n_job).item())


        current_selected_job = []
        # 현 시점에서 제일 빨리 일어나는 event 선택
        # procedure가 끝나거나 혹은 새로운 환자가 도착하거나
        min_time = 0
        if not non_idle.any() and i < n_job : min_time = arrival_time[i]-current_time
        else : min_time =  machine_procedure_time[non_idle].min().item()
        if  i < n_job:
           if min_time >= arrival_time[i]-current_time :
            #print(arrival_time[i])
            #print(current_time)
            min_time = arrival_time[i]-current_time
            #print(min_time)
            #print('arrival')

            current_selected_job.append(arrival_job_index[i])
            mask[arrival_job_index[i] + n_machine*n_job] = True
            i += 1
        ## 그만큼 시간이 흐르고,,
        assert min_time >= 0
        current_time += min_time

        machine_waiting_time[non_idle] -=  min_time
        machine_procedure_time[non_idle] -= min_time
        if is_real: total_prc_time[non_idle] -= min_time

        #print("waiting : {}".format(machine_waiting_time))
        #print("procedure : {}".format(machine_procedure_time))
        assert (machine_waiting_time >= machine_procedure_time).all()
        #print(current_time)

        # 시간이 흐른 뒤 event 결과로 새로 처리해야할 job들 선택
        for j in ((machine_procedure_time == 0) & (non_idle)).nonzero().squeeze(-1):
            ## 처리가 끝난 job

            job_time[ongoing_machine_job[j]] = current_time

            # 현재 처리가 끝난 job을 넣음

            current_selected_job.append(ongoing_machine_job[j])
            #print(ongoing_machine_job[j])
            ongoing_machine_job[j] = -1
            try :
                pop_job = machine_queue[j].pop(0)
                #print(pop_job)
                machine_procedure_time[j] = data[pop_job  + j* n_job]
                ongoing_machine_job[j] = pop_job

            except:
                non_idle[j] = False

    ## idle
    #print(machine_queue)
    #print('done')
    idle = (job_time - data.view(n_machine+1, n_job).sum(0))
    idle_mean = idle.mean()


    idle_ratio =  idle*100  /data.view(n_machine+1, n_job)[:-1,:].sum(0)
    #print(idle_ratio)
    #sys.exit()

    #return idle_ratio.mean()
    return idle_mean, pi

# 여러분

def human_policy(input, n_job, n_machine, stochastic = False,std = None, policy = 'LCW', station_rule = 'RAND'):
    duration = input[:-n_job,0].clone().detach().numpy()
    arrival = input[-n_job:,0].clone().detach().numpy()
    arrival_order = np.argsort(arrival).tolist()
    ##
    mask = np.zeros((n_machine, n_job), dtype = bool)
    mask[np.where(duration.reshape(n_machine,n_job) ==0  )] = True

    job_score = np.zeros(n_job)
    machine_score = duration.reshape(n_machine, n_job).sum(axis= 1)
    is_idle = np.ones(n_machine, dtype=bool)
    is_arrive = np.zeros(n_job, dtype = bool)
    machine_process_time = np.zeros(n_machine)
    machine_process_job = -np.ones(n_machine, dtype= np.int64)
    waiting_room = []
    #Event : 환자가 도착하거나, Station(환자)의 검진이 끝나거나

    # initialize job score:
    if policy == 'LCW' or policy == 'LAW': #Longest Current Waiting Time
        job_score = np.zeros(n_job)
    elif policy == 'LS': # Longest System Time
        job_score = np.zeros(n_job)
    elif policy == 'SERP' : # Shortest Expected Remaining Processing Time
        job_score = duration.reshape(n_machine,n_job).sum(axis = 0)
        #print(job_score)
    else:
        assert True
    time = 0
    i = 0
    pi = []

    ## cost 추적용
    job_ms = np.zeros(n_job)


    # 가장 가까운 event를 선택
    # machine이 검진이 끝나는 시간, 혹은 환자가 도착하는 시간
    def event_handler():
        # return값 : (시간, 환자, machine)
        _machine_event = machine_process_time[is_idle == False]
        temp_machine = machine_process_time.copy()
        temp_machine[is_idle] = 999
        if _machine_event.size == 0:
            if len(arrival_order) != 0:
                arv_job = arrival_order.pop(0)
                #print(arrival_order)
                #print(arrival[arv_job])
                #print(time)

                assert arrival[arv_job] - time >= 0
                is_arrive[arv_job] = True
                return (arrival[arv_job] - time , arv_job , None)
            else :
                return 0, None,None
        else:
            if len(arrival_order) == 0:
                return (temp_machine.min(),machine_process_job[temp_machine.argmin()],temp_machine.argmin())
            else:
                if temp_machine.min() <= arrival[arrival_order[0]] -time:
                    return (temp_machine.min(),machine_process_job[temp_machine.argmin()],temp_machine.argmin())
                else:
                    arv_job = arrival_order.pop(0)
                    is_arrive[arv_job] = True
                    return (arrival[arv_job] - time , arv_job , None)

    def update_state(job, station):
        is_idle[station] = False
        machine_process_job[station] = job

        if stochastic :
            cur_std = duration[job+station*n_job].item() * std
            machine_process_time[station] = duration[job+station*n_job] + (cur_std)*torch.randn(1)
        else:
            machine_process_time[station] = duration[job+station*n_job]
        #job_score[job] =0
        if policy == 'LCW':
            job_score[job] = 0
        mask[station ,job] = True
        if job in waiting_room:
            waiting_room.remove(job)
        pi.append(job+station*n_job)
        # cost 추적용
        job_ms[job] = time + duration[job+station*n_job]

    def score_tick_handler(event_time, policy):
        if policy == 'LCW' or policy == 'LAW': # longest current waiting time/longest accumulated waitung time
            job_score[waiting_room] += event_time
        elif policy == 'LS':
            #print(is_arrive)
            job_score[is_arrive] = time - arrival[is_arrive]
        elif policy == 'SERP':
            service_jobs = list(set(np.where(arrival ==True)[0].tolist()) - set(waiting_room))
            job_score[service_jobs] -= event_time


    while not mask.all():
        #print(mask)

        event_time , event_job, event_machine = event_handler()
        #print('event_time' + str(event_time))

        #tick handling을 먼저
        machine_process_time[is_idle == False] -= event_time
        time += event_time
        #job_score[waiting_room] += event_time
        score_tick_handler(event_time, policy)
        machine_score[is_idle == False] -= event_time

        # 환자 처리 : 비어있는 station이 존재할 경우, 높은 station score로, 아니면 waiting room으로
        if event_job is not None:
            #print(event_job)
            job_station = np.where(mask[:,event_job] != True)[0]
            # 현재 환자에게 남아있는 station 목록을 가져옴
            if job_station.size != 0:
                if is_idle[job_station].any():
                    #현재 환자에게 남아있는 station중 idle 한 station이 존재할 경우
                    candidate_station =  list(set(np.where(is_idle)[0].tolist() ) & set(job_station.tolist()))
                    if station_rule == 'MAX':
                        argmax = machine_score[candidate_station].argmax()
                    else:
                        argmax = random.randint(0, len(candidate_station)-1)
                    #print(argmax)
                    # 선택
                    selected_station = candidate_station[argmax]
                    update_state(event_job, selected_station)
                else:
                    # 없으면 waiting room으로
                    waiting_room.append(event_job)

        # Station 처리
        if event_machine is not None:
            # 현재 station에서 받아야 하는 환자 목록을 가져옴
            job_station = np.where(mask[event_machine,:] != True)[0]
            candidate_patient = list (set(job_station.tolist()) & set(waiting_room))
            if len(candidate_patient) == 0:
                #waiting room에 뭐가 없을 경우
                is_idle[event_machine] = True
                machine_process_job[event_machine] = -1
                #print('ttt')
            else:
                # 가장 높은 score를 갖는 job을 선택
                if policy == 'SERP':
                    job_score_ = np.reciprocal(job_score + 1e-10)
                else:
                    job_score_ = job_score.copy()
                #print(job_score)
                argmax = job_score_[candidate_patient].argmax()
                selected_patient = candidate_patient[argmax]
                update_state(selected_patient, event_machine)


        if event_machine is None and event_job is None:
            break
        i +=1

        ## tick handling
    job_pure = duration.reshape(n_machine,n_job).sum(axis = 0)
    cost = job_ms - job_pure - arrival


    return cost.mean(), pi





def greedy_osspa(input,n_job,n_machine,is_real= False):

    input = input.to('cpu')
    return torch.tensor([
        _greedy_osspa(graph,n_job,n_machine,is_real)[0]
        for graph
        in input
    ]).mean()

def optimal(data,n_job,n_machine):
    permutation = torch.tensor(list(itertools.permutations(range(n_job*n_machine),n_job*n_machine)))
    assert n_job*n_machine < 10, "Grpah size for optimal solution solver should be less than 10"
    arrival_time = data[-n_job:,0].expand(permutation.size(0), -1)
    #print(arrival_time)
    candidate_list = []
    machine_ms = torch.zeros(permutation.size(0), n_machine)
    job_ms = arrival_time.clone().detach()

    machine_pure = torch.zeros(permutation.size(0), n_machine)
    job_pure = arrival_time.clone().detach()
    data = data.expand(permutation.size(0),-1,-1)
    for i in range(n_job*n_machine):
        current_job_ms = job_ms.gather(1,
                torch.remainder(permutation[:,i], n_job)[:,None]
            )
        current_machine_ms = machine_ms.gather(1,
                torch.div(permutation[:,i], n_job, rounding_mode='trunc')[:,None]
            )
        max_v , _ = torch.stack((current_job_ms,current_machine_ms),dim = 0).max(0)
        duration = data.gather( 1 ,
              permutation[:,i,None,None].expand(data.size(0), -1, data.size(2))
            )[:,:,0].squeeze(-1)
        max_v = max_v.squeeze() + duration

        job_ms.scatter_(1, torch.remainder(permutation[:,i], n_job)[:,None], max_v[:,None])
        machine_ms.scatter_(1, torch.div(permutation[:,i], n_job, rounding_mode='trunc')[:,None], max_v[:,None])


        job_pure.scatter_add_(1, torch.remainder(permutation[:,i], n_job)[:,None], duration[:,None])
        machine_pure.scatter_add_(1, torch.div(permutation[:,i], n_job, rounding_mode='trunc')[:,None],duration[:,None])

    job_wait_time = job_ms - job_pure



    val, ind = job_wait_time.mean(1).min(dim = 0)
    return val, permutation[ind]
