function="dynamics_oneplussinx"
static_model_def='MLP_bayesian'
uncertainty_model_train_mode='use_individual'
y_std_noise=0
log_name="denoise_results_neuralODE.txt"

for seed in {6,7,8}
do
for x_std_noise in {0.707,1.,1.414,2.,2.828}
do
for train_mode in {"separate",}
do
for stochstic_model in  "bayesian","MLP_bayesian"
do IFS=",";set -- $stochstic_model;
static_stochastic_mode=$1
static_model_def=$2
for modes in  "fixed","fixed","homo" "homo","fixed","homo"
do IFS=",";set -- $modes;

s2y_est_mode=$1
noise_y_est_mode=$2
noise_x_est_mode=$3


python train_uncertainty_neuralODE.py --uncertainty_model_train_mode $uncertainty_model_train_mode --seed $seed --function $function --static_stochastic_mode $static_stochastic_mode --static_model_def $static_model_def \
--std_y_noise $y_std_noise --std_x_noise $x_std_noise  --log_name $log_name  \
--train_mode $train_mode --s2y_est_mode $s2y_est_mode --noise_y_est_mode $noise_y_est_mode --noise_x_est_mode $noise_x_est_mode
# done
done
done
done
done
done