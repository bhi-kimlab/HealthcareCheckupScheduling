#/bin/bash
n_s=44
nplist=("20 30 50 100 150")
#n_s = ("5 10 30 44 44")
ll=4
#lr=0.00005
#lr=0.0001
lr=0.0005
for n_p in ${nplist[@]}; do
    echo "python PPO.py --n_p $n_p --n_s $n_s --lr $lr --lr_name $ll --train snuh --vali snuh --n_tr 20 --max_update 100000 --hidden_dim 128 --hidden_dim_actor 64 --hidden_dim_critic 64"
done
