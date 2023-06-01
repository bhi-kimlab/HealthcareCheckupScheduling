import numpy as np
from uniform_instance_gen import uni_instance_gen
from uniform_instance_gen import uni_instance_gen2
import sys
import pandas as pd
j = int(sys.argv[1])
m = int(sys.argv[2])
l = 50
h = 50
batch_size = 20
seed = 238

np.random.seed(seed)
arr_dist = np.load("arrival_time_dist.npy",allow_pickle=True)
snuh_station = pd.read_csv("../snuh_time.txt",sep="\s",header=None, names=["time","capacity","area"],index_col=0).values
data = [uni_instance_gen2(n_p=j, n_s=m, low=l, high=h, snuh_station=snuh_station,arr_dist=arr_dist) for _ in range(batch_size)]
data = np.array(data)
np.save('generatedData{}_{}_Seed{}.npy'.format(j, m, seed), data)
