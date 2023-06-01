#/bin/bash
#lrlist=( "1e-5" "2e-5" "3e-5" "4e-5" "5e-5" "1e-6" "2e-6" "3e-6" "4e-6" "5e-6")
#lrlist=("1e-1" "1e-2" "1e-3"  "1e-4" "1e-5" "1e-6")
#lrlist=("1e-2" "1e-3"  "1e-4" "1e-5" "1e-6")
#lrlist=( "1e-4" "5e-4" "1e-5" "5e-5" "1e-6" "5e-6" "1e-7" "5e-7" "1e-8" "5e-8")
#    "5e-7" "1e-7" "5e-8" "1e-8")
#lrlist=( "1e-5" "5e-5" "1e-6" "5e-7" "1e-7" "5e-8" "1e-8")

#n_s = ("5 10 30 44 44")
ll=4
lr1="0.0001"
lr2="5e-05"

n_p=20
n_s=44
python test_eval.py --n_p $n_p --n_s $n_s --lr $lr1 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 77 --log_time 202306010156 --hidden_dim 128 --hidden_dim_actor 64 --hidden_dim_critic 64
# python test_eval.py --n_p $n_p --n_s $n_s --lr $lr1 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 142 --log_time 202301151225 
# n_p=30
# n_s=44
# python test_eval.py --n_p $n_p --n_s $n_s --lr $lr2 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 181 --log_time 202301172110 
# n_p=50
# n_s=44
# python test_eval.py --n_p $n_p --n_s $n_s --lr $lr1 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 131 --log_time 202301151225 
# n_p=100
# n_s=44
# python test_eval.py --n_p $n_p --n_s $n_s --lr $lr1 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 126 --log_time 202301151225 
# n_p=150
# n_s=44
# python test_eval.py --n_p $n_p --n_s $n_s --lr $lr2 --lr_name $ll --train snuh --vali snuh --n_tr 3 --max_update 10000 --vali_model 21 --log_time 202301172110 

