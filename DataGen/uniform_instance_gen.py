import numpy as np


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]

def uni_instance_gen(n_p, n_s, low=10, high=30):
    while True:
        station_time = np.random.randint(low=low,high=high,size=(n_s,1))
        arrival_time = np.random.randint(low=0,high=10, size=(n_p,1))
        patients_list = np.random.choice([0, 1], size=(n_p,n_s), p=[2./7, 5./7])
        patients_time = np.array([np.random.randint(low = station_time[i]-2,high = station_time[i]+2, size = (n_p)) for i in range(n_s)]).T
        patients = np.concatenate((arrival_time,(patients_list*patients_time)),axis=1)
        stations = station_time
        if(patients_list.sum()>=2):
            return np.array([stations, patients])

def uni_instance_gen2(n_p, n_s, snuh_station, arr_dist,low=10, high=30):
    while True:
        station_time = snuh_station[:,0][:n_s]
        #arrival_time = np.random.randint(low=0,high=10, size=(n_p,1))
        #sample = np.random.normal(10000,200,size=(n_p,1))
        sample = np.random.choice(arr_dist,size=(n_p,1))
        arrival_time = sample.astype(int)
        patients_list = np.random.choice([0, 1], size=(n_p,n_s), p=[3./4, 1./4])
        #patients_list = np.random.choice([0, 1], size=(n_p,n_s), p=[2./7, 5./7])
        patients_time = np.array([np.random.randint(low = station_time[i]-60,high = station_time[i]+60, size = (n_p)) for i in range(n_s)]).T
        patients_time = np.where(patients_time <1, 1, patients_time)
       # patients = np.concatenate((arrival_time,(patients_list*patients_time)),axis=1)
        patients = np.concatenate((arrival_time,(patients_list*patients_time)),axis=1)
        stations = station_time
        if(patients_list.sum()>=2):
            return np.array([stations, patients,snuh_station[:,1][:n_s]],dtype=object)



def override(fn):
    """
    override decorator
    """
    return fn


