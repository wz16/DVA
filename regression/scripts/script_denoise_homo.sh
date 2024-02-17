log_name="denoise_results_homo.txt"

function="xbyoneplussinx_homo"

x_std_noise=0
uncertainty_model_train_mode='use_mean'

for y_std_noise in {0.707,1.414,2.828}
do
for train_mode in {"separate",}
do
for stochstic_model in "ensemble","MLP" "bayesian","MLP_bayesian"

do IFS=",";set -- $stochstic_model;
static_stochastic_mode=$1
static_model_def=$2
for modes in "homo","fixed","fixed" "homo","homo","fixed"
do IFS=",";set -- $modes;
for seed in {1,2,3,4,5}
do
s2y_est_mode=$1
noise_y_est_mode=$2
noise_x_est_mode=$3


python train_uncertainty_xynoise_bayesianMLP.py --seed $seed --function $function --uncertainty_model_train_mode $uncertainty_model_train_mode --static_stochastic_mode $static_stochastic_mode --static_model_def $static_model_def \
--std_y_noise $y_std_noise --std_x_noise $x_std_noise  --log_name $log_name  \
--train_mode $train_mode --s2y_est_mode $s2y_est_mode --noise_y_est_mode $noise_y_est_mode --noise_x_est_mode $noise_x_est_mode
# done
done
done
done
done
done

