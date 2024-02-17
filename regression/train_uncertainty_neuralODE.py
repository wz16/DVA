from ast import parse
from tkinter import TRUE
# from xmlrpc.client import Boolean, boolean
import torch, argparse
from torchdiffeq import odeint

import numpy as np
import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
from utils.stochastic_wrapper import stochastic_wrapper

import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy import stats
from scipy.stats import boxcox

from utils.model_load import save_model, load_saved_model
from utils.uncertainty_estimator import uncertainty_estimator
from utils.train_functions import train_uncertainty_model, train_static_uncertainty_model, train_static_model, train_neuralODE_model,train_neuralODE_uncertainty_model
DEVICE = 'cuda'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--function', default = 'linear_system', type = str)
    parser.add_argument('--t_span', default = 5, type = float)
    parser.add_argument('--step_time', default = 0.1, type = float)

    parser.add_argument('--train_mode', default = 'separate', type = str, 
                        help="separate: train static model first, then uncertainty model, together: train static and uncertainty models together ")
    parser.add_argument('--static_stochastic_mode',default = 'bayesian', type = str, help="ensemble or bayesian")
    parser.add_argument('--static_model_def', default = 'MLP_neuralODE_bayesian', type = str)
    parser.add_argument('--static_model_input_dim', default = 1, type = int)
    parser.add_argument('--static_model_output_dim', default = 1, type = int)
    parser.add_argument('--static_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--save_static_model', default = "test_neuralode_model.tar", type = str)
    parser.add_argument('--load_static_model', default = "test_neuralode_model.tar", type = str)

    parser.add_argument('--uncertainty_model_train_mode',default = 'use_mean', type = str, 
                        help="use_mean: use mean of prediction in the uncertainty training, use_individual use individual f(x)")

    parser.add_argument('--s2y_est_mode', default = "fixed", type=str, 
                        help="fixed: s2y not trained, homo: train single s2y, heter: train s2y(x) depent on x")
    parser.add_argument('--s2y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_y_est_mode', default = "fixed", type=str, 
                        help="fixed: noise estimation = 0, not train, homo: train single sigma2y, heter: train sigma2y(x) depent on x")
    parser.add_argument('--noise_y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_x_est_mode', default = "homo", type=str, 
                        help="fixed: noise estimation = 0, not train, homo: train single sigma2x, heter: train sigma2x(x) depent on x")

    parser.add_argument('--noise_x_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--uncertainty_epochs', default =200, type = int)
    parser.add_argument('--static_epochs', default = 200, type = int)
    parser.add_argument('--total_examples', default = 100, type = int)
    parser.add_argument('--batch_size', default = 10, type = int)
    parser.add_argument('--learning_rate', default = 1e-2, type = float)
    parser.add_argument('--mu_x', default = 5., type = float)
    parser.add_argument('--rho_x', default = 2, type = float)
    parser.add_argument('--std_x_noise', default = 2, type = float)
    parser.add_argument('--std_y_noise', default = 0, type = float)

    parser.add_argument('--seed', default= 20, type=int, help='random seed')
    parser.add_argument('--log_name', default = 'denoise_neuralODE_results.txt', type = str)
    parser.add_argument('--if_opt_log', default = True, type = bool)
    
    return parser.parse_args()


def predict_model(data, data_generator, model, uncertainty_model, args):

    x = data['x']
    x_tilde = data['x_tilde']
    t_eval = data['t_eval']
    # true_y = data_generator.x_to_y(x).squeeze(-1).detach().cpu().numpy()
    samples = 5
    x_ests = []
    loss_corrected = 0
    x0_tilde = x_tilde[:,0,:]

    def func_remove_t_wrapper(t,x):
        return model(x)
    for sample in range(samples):
        model.sample()
        x_est = torch.transpose(odeint(func_remove_t_wrapper, x0_tilde, t_eval),0,1)
        x_ests.append(x_est)
    x_ests = torch.stack(x_ests)
    x_est = torch.mean(x_ests,dim=0)
    mse = torch.mean((x_est-x_tilde)**2)

    log_s2y = uncertainty_model.get_log_s2y(x).detach().cpu().numpy()
    s2y = np.exp(log_s2y).squeeze(-1)
    sy = np.sqrt(s2y)

    # log_sigma2y = uncertainty_model.get_log_sigma2y(x).detach().cpu().numpy()
    # sigma2y = np.exp(log_sigma2y).squeeze(-1) 
    # sigmay = np.sqrt(sigma2y)

    log_sigma2x = uncertainty_model.get_log_sigma2x(x).detach().cpu().numpy()
    sigma2x = np.exp(log_sigma2x).squeeze(-1) 
    sigmax = np.sqrt(sigma2x)

    if args.noise_x_est_mode == "fixed":
        x_noise = torch.zeros(1).to(DEVICE)
        log_inferred_sigma2x = torch.zeros(1).to(DEVICE)
    elif args.noise_x_est_mode == "homo":
        log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x
        x_noise = uncertainty_model.x_noise
    elif args.noise_x_est_mode == "heter":
        log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x(x)
        x_noise = uncertainty_model.x_noise 

    x_noise_hat = torch.exp(0.5*log_inferred_sigma2x)*x_noise

    x_tilde_hat = x_tilde-x_noise_hat

    # predict_y_mean = torch.mean(y_list, dim=0).squeeze(-1).detach().cpu().numpy()
    # predict_y_std = torch.std(y_list, dim=0).squeeze(-1).detach().cpu().numpy()
    # x = x.squeeze(-1).detach().cpu().numpy()

    file = open("./logs/"+args.log_name, "a")  # append mode
    file.write("function:{}, std2_y:{:.2f}, std2_x:{:.2f}, static_stochastic_mode:{}, uncertainty_model_train_mode:{}, train_mode:{}, s2y_est_mode:{}, noise_y_est_mode:{}, noise_x_est_mode:{}, seed:{}, s2y:{:.2f},  sigma2x:{:.2f}, mean_sqrt_err:{:.2f}\n".format(
        args.function, args.std_y_noise**2, args.std_x_noise**2, args.static_stochastic_mode, args.uncertainty_model_train_mode, args.train_mode, args.s2y_est_mode, args.noise_y_est_mode, args.noise_x_est_mode, args.seed, np.mean(sy)**2,  np.mean(sigmax)**2,mse.item()))
    t_eval = data['t_eval'].cpu().detach().numpy()
    ifplot = False
    if ifplot:

        fig, axes = plt.subplots(4,2, figsize=(8,8))

        for i in range(4):
            axes[i,0].scatter(t_eval,x_tilde[i].detach().cpu().numpy(),label='noisy data')
            axes[i,0].scatter(t_eval,x_tilde_hat[i].detach().cpu().numpy(),label='denoised data')
            axes[i,0].scatter(t_eval,x[i].detach().cpu().numpy(),label='true data')

            plt.savefig("figs/test_neuralODE.png")

def draw_noise_hist(noise,true_noise):

    noise = noise.flatten().detach().cpu().numpy()
    true_noise = true_noise.flatten().detach().cpu().numpy()
    print(kstest(noise, stats.norm.cdf))
    fig, ax = plt.subplots(5,1,figsize=(12, 8))
    counts, bins = np.histogram(noise,bins=100)
    ax[0].stairs(counts, bins)
    ax[0].set_xlim([-10,10])
    ax[0].set_ylabel("infered")
    counts, bins = np.histogram(true_noise,bins=100)
    ax[1].stairs(counts, bins)
    ax[1].set_xlim([-10,10])
    ax[1].set_ylabel("true")
    ax[3].scatter(np.linspace(1,len(noise),len(noise)),noise-true_noise)
    counts, bins = np.histogram(noise-true_noise,bins=100)
    ax[2].stairs(counts, bins)
    ax[2].set_xlim([-10,10])
    ax[2].set_ylabel("infer-true")

    stats.probplot(noise, plot=ax[4])
    plt.savefig("inferred_noise.png")



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # static_model = get_model(args, model_type = "static").to(DEVICE)
    static_stoc_model = stochastic_wrapper(args.static_stochastic_mode,args)
    static_stoc_model.to(DEVICE)
    Data_generator = importlib.import_module('regress_funcs.'+args.function).Data_generator
    data_generator = Data_generator(args)

    data = data_generator.get_data(args,args.total_examples,DEVICE)
    x_tildes = data['x_tilde'].clone().detach().requires_grad_(True)
    uncertainty_model = uncertainty_estimator(x_tildes, None, args).to(DEVICE)
    # if args.load_static_model:
    #     load_saved_model(static_stoc_model, args.load_static_model)
    if args.train_mode == "separate":
        print("***********\n start training static model...")
        static_stoc_model = train_neuralODE_model(data, static_stoc_model, args)
        if args.save_static_model:
            save_model(static_stoc_model, args.save_static_model)
        print("***********\n start training uncertainty model...")
        uncertainty_model = train_neuralODE_uncertainty_model(data, static_stoc_model, uncertainty_model, args)
    elif args.train_mode == "together":
        static_stoc_model, uncertainty_model = train_static_uncertainty_model(data, static_stoc_model, uncertainty_model, args)
        args.noise_y_est_mode = "homo"
        uncertainty_model = uncertainty_estimator(x_tildes, None, args).to(DEVICE)
        uncertainty_model = train_uncertainty_model(data, static_stoc_model, uncertainty_model, args)

    print("***********\n start prediction...")

    predict_model(data, data_generator, static_stoc_model, uncertainty_model, args)
   
    return None

if __name__ == "__main__":
    args = get_args()
    main(args)