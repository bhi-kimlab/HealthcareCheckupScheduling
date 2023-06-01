import bisect
import datetime
import random
from Code import Code
import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path
from queue import PriorityQueue
import copy
import threading
def exclude_lock(obj):
    if isinstance(obj, threading.LockType):
        return None
    return obj

def deep_copy_priority_queue(pq):
    new_pq = PriorityQueue()
    for value in pq.queue:
        new_pq.put(value)
    return new_pq

class HEnv(gym.Env):
    def __init__(self, data,mask):#env_config=None):
        """
        This environment model the patient shop scheduling problem as a single agent problem:

        -The actions correspond to a patient allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a patient, the end of the patient is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a patient),
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """
        """
        if env_config is None:
            env_config = {'instance_path': str(Path(__file__).parent.absolute()) + '/d01'}
        instance_path = env_config['instance_path']
        """
        # initial values for variables used for instance
        self._lock = threading.Lock()
        self.patients = data[1].shape[0]
        self.stations = data[0].shape[0]
        self.instance_matrix = None
        self.mask_arr = np.zeros((7,self.stations+1), dtype=np.int64)
        self.next_patients = PriorityQueue()
        self.initQuality = -10000000
#        self.patients_length = None
        self.max_time_op = 0
        self.max_time_patients = 0
        self.nb_legal_actions = 0
        self.nb_station_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_time_step = float('inf')
        self.current_time_step = float('inf')
        self.next_time_step = None
        self.legal_actions = None
        self.state = None
        self.patient_state = None
        self.patient_area_state = None
        self.station_state = None
        self.current_patient = None
        self.station_legal = None
        self.prev_min_idx = None
        self.prev_action = None
        self.current_area = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        self.station_expected_time  = np.zeros((self.stations+1), dtype=np.int64)
        self.station_capacity  = np.zeros((self.stations+1), dtype=np.int64)
        #print(data[0].shape)
        for i in range(self.stations):
            self.station_expected_time[i+1] = data[0][i]
            self.station_capacity[i+1] = data[2][i]
            self.mask_arr[mask[i]][i+1] = 1 # area 0~4
            self.mask_arr[5][i+1] = 1 #area 5 ll
            self.mask_arr[6][i+1] = mask[i] #area hashing
        self.instance_matrix = data[1]
        for i in range(self.patients):
            self.current_time_step = min(self.current_time_step, self.instance_matrix[i][0])
        self.action_space = gym.spaces.Discrete(self.stations + 1)
        # used for plotting
        self.colors = [
            #tuple([random.random() for _ in range(3)]) for _ in range(self.stations+1)
            tuple([float(i/(self.stations+1)) for _ in range(3)]) for i in range(self.stations+1)
        ]
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.stations+1,)),
            #"real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.patients, 7), dtype=np.float),
            "p_obs": gym.spaces.Box(0,1, shape=(self.patients, self.stations+1)),
        })
    def deep_copy(self):
        copy_obj = copy.deepcopy(self, memo={id(self.next_patients): None})
        copy_obj.next_patients = deep_copy_priority_queue(self.next_patients)
        return copy_obj

    def _get_current_state_representation(self):
        return [np.concatenate([np.array([self.current_patient,
            self.current_time_step]),
            self.patient_state.T[:,self.current_patient],
            self.station_state_exp,
            self.station_expected_time,
            #self.patient_state.flatten()],dtype=np.float32).reshape(1,-1),self.legal_actions,self.patient_area_state.flatten()]
            self.patient_state.flatten(),self.patient_area_state.flatten()],dtype=np.float32).reshape(1,-1),self.legal_actions,self.patient_area_state.flatten()]

    def get_legal_actions(self):
        legal_actions = self.legal_actions.copy()
        try:
#        '''
            if(legal_actions[Code.CT_pre]):
                legal_actions[Code.CT] = False
            if(legal_actions[Code.ENDO_pre]):
                legal_actions[Code.ENDO_A] = False
                legal_actions[Code.ENDO_B] = False
                legal_actions[Code.ENDO_C] = False
                legal_actions[Code.ENDO_re] = False
            if(legal_actions[Code.ENDO_A] or legal_actions[Code.ENDO_B] or legal_actions[Code.ENDO_C]):
                legal_actions[Code.ENDO_re] = False
            if(legal_actions[Code.DENT_pre]):
                legal_actions[Code.DENT]=False
            if(legal_actions[Code.MRI_pre]):
                legal_actions[Code.MRI_A] = False
                legal_actions[Code.MRI_B] = False
                legal_actions[Code.MRI_C] = False
        except IndexError:
            None
        legal_actions = legal_actions * self.mask_arr[self.current_area] == 1
        return legal_actions
    #@override
    def reset(self):
        self._lock=False
        self.current_time_step = 0
        self.next_time_step = 0
        self.next_patients = PriorityQueue()
        self.nb_legal_actions = self.patients
        self.nb_station_legal = 0
        self.initQuality = -10000000
        # represent all the legal actions
        self.legal_actions = np.zeros(self.stations+1 , dtype=np.bool_)
        self.prev_min_idx=-1
        self.prev_action=-1
        # used to represent the solution
        self.solution = np.full((self.patients, self.stations+1), -1, dtype=np.int64)
        self.station_legal = np.zeros(self.stations+1, dtype=np.bool_)
        self.state = np.zeros((self.patients, 7), dtype=np.float64)
        self.patient_state = np.zeros((self.patients, self.stations+1), dtype=np.float32)
        self.patient_area_state = np.zeros((self.patients, 5), dtype=np.bool_)
        self.station_state = np.zeros(self.stations+1, dtype=np.float32)
        self.station_state_all = np.full((self.stations+1,21),22388974, dtype=np.float32)
        self.station_state_exp = np.zeros(self.stations+1, dtype=np.float32)
        self.station_state_all_exp = np.full((self.stations+1,21),22388974, dtype=np.float32)
        self.station_state_all_cnt = np.zeros((self.stations+1,21),dtype=np.int64)

        for patient in range(self.patients):
            self.patient_state[patient] =  (self.instance_matrix[patient] != 0)
            self.patient_state[patient][0] = False
            if(self.patient_state[patient].sum() !=0):
                self.next_patients.put((self.instance_matrix[patient][0],patient,-1,-1,5))

        for i in range(1,self.stations+1):
            for j in range(0,self.station_capacity[i]):
                self.station_state_all[i][j] = 0
                self.station_state_all_exp[i][j] = 0

        self.current_time_step,self.current_patient,self.prev_action,self.prev_min_idx,self.current_area = self.next_patients.get()
        self.legal_actions = self.patient_state[self.current_patient]==True
        self.next_patients.put((1000000,-1,-1,-1,5))
        return self._get_current_state_representation()

#    @override
    def step(self, action: int,estimate=False):
        reward = 0.0
        if action == self.stations+2:
            scaled_reward = -reward
            return self._get_current_state_representation(), scaled_reward, self._is_done(), {}
        else:
            min_idx = self.station_state_all[action].argmin()
            #waiting_time = self.station_state[action]
            waiting_time = self.station_state_all[action][min_idx]
            if(estimate==False):
                time_needed = self.instance_matrix[self.current_patient][action]
            else:
                time_needed = self.station_expected_time[action]
            reward += waiting_time
            self.patient_state[self.current_patient][action] = False
            self.solution[self.current_patient][action] = self.current_time_step + waiting_time
            self.current_area = self.mask_arr[6][action]
            if(self.patient_state[self.current_patient].sum() !=0):
                if((self.patient_state[self.current_patient] * self.mask_arr[self.current_area]).sum() ==0):

                    self.patient_area_state[self.current_patient][self.current_area] = 0
                    self.next_patients.put((self.current_time_step+waiting_time+time_needed,self.current_patient,action,min_idx,5))
                else:
                    self.patient_area_state[self.current_patient][self.current_area] = 1
                    self.next_patients.put((self.current_time_step+waiting_time+time_needed,self.current_patient,action,min_idx,self.current_area))
            #self.station_state[action]+=time_needed
            self.station_state_all[action][min_idx]+=time_needed
            self.station_state_all_exp[action][min_idx]+=self.station_expected_time[action]
            self.station_state_all_cnt[action][min_idx]+=1
            self.next_time_step,self.next_patient,self.prev_action,self.prev_min_idx,self.current_area = self.next_patients.get()
#            for station in range(self.stations+1):
#                self.station_state[station] = max(0,self.station_state[station] -
#                (self.next_time_step - self.current_time_step))
            for station in range(self.stations+1):
                for capacity in range(self.station_capacity[station]):
                    self.station_state_all[station][capacity] = max(0,self.station_state_all[station][capacity] - (self.next_time_step - self.current_time_step))
                    self.station_state_all_exp[station][capacity] = max(0,self.station_state_all_exp[station][capacity] - (self.next_time_step - self.current_time_step))
                    if(self.prev_action == station and self.prev_min_idx == capacity):
                        self.station_state_all_cnt[station][capacity] -=1
                        self.station_state_all_exp[station][capacity] = self.station_state_all_cnt[station][capacity] * self.station_expected_time[station]
                self.station_state[station]  = self.station_state_all[station].min()
                self.station_state_exp[station]  = self.station_state_all_exp[station].min()
            self.current_time_step = self.next_time_step
            self.legal_actions = self.patient_state[self.next_patient,:]
            self.current_patient = self.next_patient
            self.current_time_step = self.next_time_step
            scaled_reward = -reward
            return self._get_current_state_representation(), scaled_reward, self._is_done()

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def _is_done(self):
        if self.current_patient == -1:
#            self.last_time_step = self.current_time_step
            return True
        return False

    def render(self, mode='human'):
        df = []
#        print(self.solution)
        for patient in range(self.patients):
            for i in range(self.stations+1):
                if(self.solution[patient][i] != -1):
                    dict_op = dict()
                    dict_op["Task"] = 'patient {}'.format(patient)
                    start_sec = self.solution[patient][i]
                    #start_sec = self.start_timestamp + self.solution[patient][i]
                    finish_sec = start_sec + self.instance_matrix[patient][i]
                    dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                    dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                    #dict_op["Start"] = start_sec
                    #dict_op["Finish"] = finish_sec

                    dict_op["Resource"] = "station {}".format(i)
                    df.append(dict_op)
 #       print(df)
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(df, index_col='Resource', colors=self.colors, show_colorbar=True,
                                  group_tasks=True)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        df = df.sort_values(by=["Task","Start"])
        return [fig,df]
