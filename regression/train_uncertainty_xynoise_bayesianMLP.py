from ast import parse
from tkinter import TRUE
# from xmlrpc.client import Boolean, boolean
import torch, argparse

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
from utils.stochastic_wrapper import stochastic_wrapper

import matplotlib.pyplot as plt
import matplotlib
font = {
        'size'   : 12}

matplotlib.rc('font', **font)

from scipy.stats import kstest
from scipy import stats
from scipy.stats import boxcox

from utils.model_load import save_model, load_saved_model
from utils.uncertainty_estimator import uncertainty_estimator
from utils.train_functions import train_uncertainty_model, train_static_uncertainty_model, train_static_model
DEVICE = 'cuda'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--function', default = 'xbyoneplussinx_homo', type = str)
    parser.add_argument('--train_mode', default = 'separate', type = str, 
                        help="separate: train static model first, then uncertainty model, together: train static and uncertainty models together ")
    parser.add_argument('--static_stochastic_mode',default = 'ensemble', type = str, help="ensemble or bayesian")
    parser.add_argument('--static_model_def', default = 'MLP', type = str)
    parser.add_argument('--static_model_input_dim', default = 1, type = int)
    parser.add_argument('--static_model_output_dim', default = 1, type = int)
    parser.add_argument('--static_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--save_static_model', default = "test_static_model.tar", type = str)
    parser.add_argument('--load_static_model', default = "test_static_model.tar", type = str)

    parser.add_argument('--uncertainty_model_train_mode',default = 'use_mean', type = str, 
                        help="use_mean: use mean of prediction in the uncertainty training, use_individual use individual f(x)")

    # for va methods, set s2y_est_mode=heter or homo, noise_y_est_mode = fixed, s2y is the estimated aleatoric uncertainty of y
    # for dva methods, set s2y_est_mode=heter or homo, noise_y_est_mode = heter or homo, s3y is the estimated aleatoric uncertainty of y 

    parser.add_argument('--s2y_est_mode', default = "homo", type=str, 
                        help="fixed: s2y not trained, homo: train single s2y, heter: train s2y(x) depent on x")
    parser.add_argument('--s2y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_y_est_mode', default = "homo", type=str, 
                        help="fixed: noise estimation = 0, s3y not trained, homo: train single s3y, heter: train s3y(x) depent on x")
    parser.add_argument('--noise_y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_x_est_mode', default = "fixed", type=str, 
                        help="fixed: noise estimation = 0, sigma2x not trained, homo: train single sigma2x, heter: train sigma2x(x) depent on x")

    parser.add_argument('--noise_x_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--uncertainty_epochs', default = 200, type = int)
    parser.add_argument('--static_epochs', default = 200, type = int)
    parser.add_argument('--total_examples', default = 1000, type = int)
    parser.add_argument('--batch_size', default = 200, type = int)
    parser.add_argument('--learning_rate', default = 1e-2, type = float)
    parser.add_argument('--mu_x', default = 5., type = float)
    parser.add_argument('--rho_x', default = 2, type = float)
    parser.add_argument('--std_x_noise', default = 0, type = float)
    parser.add_argument('--std_y_noise', default = 0.707, type = float)

    parser.add_argument('--seed', default= 2, type=int, help='random seed')
    parser.add_argument('--log_name', default = 'denoise_results_test.txt', type = str)
    parser.add_argument('--if_opt_log', default = True, type = bool)
    
    return parser.parse_args()

def draw_model(data, data_generator, model, uncertainty_model_dva, uncertainty_model_va, args):
        legend_font_size = 12
        legend_labelspacing = 0.2
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        x = data['x'].to(DEVICE)
        y_tilde = data['y_tilde'].cpu().detach().numpy()
        y = data['y'].cpu().detach().numpy()
        learned_noise = uncertainty_model_dva.y_noise.cpu().detach().numpy()
        log_s2y = uncertainty_model_va.get_log_s2y(x).detach().cpu().numpy()
        s2y = np.exp(log_s2y).squeeze(-1)
        s2yroot = np.sqrt(s2y)
        
        log_s3y = uncertainty_model_dva.get_log_s3y(x).detach().cpu().numpy()
        s3y = np.exp(log_s3y).squeeze(-1) 
        s3yroot = np.sqrt(s3y)
        x = x.squeeze(-1).detach().cpu().numpy()

        y_std = data_generator.get_y_std(x)

        axes[2].scatter(x, y_tilde, label = 'noisy data', s=1)
        axes[2].scatter(x, y_tilde-np.expand_dims(s3yroot,axis=-1)*learned_noise, label = 'denoised data', s=1, color='C3')
        axes[2].legend(prop={'size': legend_font_size},labelspacing=legend_labelspacing)
        axes[2].set_xlim(-1,11)
        axes[2].set_ylim(-4,30)
        axes[2].set_xlabel("x")
        # axes[2].set_ylabel("y", rotation=0)
        axes[2].title.set_text("noisy vs denoised data points")


        x = torch.linspace(args.mu_x-3*args.rho_x,args.mu_x+3*args.rho_x,args.total_examples).to(DEVICE).unsqueeze(-1)
        true_y = data_generator.x_to_y(x).squeeze(-1).detach().cpu().numpy()
        samples = 10
        y_list = model.forward_multiple(x,samples)

        log_s2y = uncertainty_model_va.get_log_s2y(x).detach().cpu().numpy()
        s2y = np.exp(log_s2y).squeeze(-1)
        s2yroot = np.sqrt(s2y)
        
        log_s3y = uncertainty_model_dva.get_log_s3y(x).detach().cpu().numpy()
        s3y = np.exp(log_s3y).squeeze(-1) 
        s3yroot = np.sqrt(s3y)
        x = x.squeeze(-1).detach().cpu().numpy()

        y_std = data_generator.get_y_std(x)

        predict_y_mean = torch.mean(y_list, dim=0).squeeze(-1).detach().cpu().numpy()
        predict_y_std = torch.std(y_list, dim=0).squeeze(-1).detach().cpu().numpy()

        axes[0].plot(x,true_y, label = "true y", c='C0')
        axes[0].fill_between(x, (true_y-y_std), (true_y+y_std), alpha=0.1, label = "true data uncertainty", color='C0')
        axes[0].plot(x,predict_y_mean, label = r"predict y mean ($\mu_{\theta_1}$)", color='C1')
        axes[0].fill_between(x,(predict_y_mean-predict_y_std), (predict_y_mean+predict_y_std), alpha=0.1, label = r"prediction uncertainty ($s_{\theta_1}$)", color='C1')
        
        axes[0].title.set_text("true data vs prediction")
        # axes[0].set_ylim(y_lo_lim,y_up_lim)
        axes[0].legend(prop={'size': legend_font_size},labelspacing=legend_labelspacing)
        axes[0].set_ylim(-4,30)
        axes[0].set_xlim(-1,11)
        axes[0].set_xlabel("x")
        # axes[0].set_ylabel("y", rotation=0)

        axes[1].plot(x,s2yroot,   color='C1')
        axes[1].plot(x,s3yroot,  color='C3')
        axes[1].plot(x, y_std, color='C0')

        axes[1].fill_between(x,0*x, s2yroot, alpha=0.2, label = r"VA estimation ($\hat{\sigma}_{\theta_2}$)", color='C1')
        axes[1].fill_between(x,0*x, s3yroot, alpha=0.2, label = r"DVA estimation ($\hat{\sigma}_{\epsilon,\theta_3}$)", color='C3')
        axes[1].fill_between(x, 0*x, y_std, alpha=0.2, label = r'true data uncertainty', color='C0')
        axes[1].title.set_text("data uncertainty")
        axes[1].legend(prop={'size': legend_font_size},labelspacing=legend_labelspacing)
        axes[1].set_xlim(-1,11)
        axes[1].set_ylim(0,4)
        axes[1].set_xlabel("x")
        # axes[1].set_ylabel("y", rotation=0)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("drawings/heter_experiment.png")

def predict_model(data, data_generator, model, uncertainty_model, args):

    x = torch.linspace(args.mu_x-2*args.rho_x,args.mu_x+2*args.rho_x,args.total_examples).to(DEVICE).unsqueeze(-1)
    
    true_y = data_generator.x_to_y(x).squeeze(-1).detach().cpu().numpy()
    samples = 10
    y_list = model.forward_multiple(x,samples)

    log_s2y = uncertainty_model.get_log_s2y(x).detach().cpu().numpy()
    s2y = np.exp(log_s2y).squeeze(-1)
    s2yroot = np.sqrt(s2y)
    
    log_s3y = uncertainty_model.get_log_s3y(x).detach().cpu().numpy()
    s3y = np.exp(log_s3y).squeeze(-1) 
    s3yroot = np.sqrt(s3y)

    log_sigma2x = uncertainty_model.get_log_sigma2x(x).detach().cpu().numpy()
    sigma2x = np.exp(log_sigma2x).squeeze(-1) 
    sigmax = np.sqrt(sigma2x)

    predict_y_mean = torch.mean(y_list, dim=0).squeeze(-1).detach().cpu().numpy()
    predict_y_std = torch.std(y_list, dim=0).squeeze(-1).detach().cpu().numpy()
    x = x.squeeze(-1).detach().cpu().numpy()

    y_std = data_generator.get_y_std(x)
    x_std = data_generator.get_x_std(x)

    s3yrootdifftotrue = np.sqrt(np.mean((s3yroot-y_std)**2))
    s2yrootdifftotrue = np.sqrt(np.mean((s2yroot-y_std)**2))
    sigmaxdifftotrue = np.sqrt(np.mean((sigmax-x_std)**2))

    file = open("./logs/"+args.log_name, "a")  # append mode
    message = "function:{}, std2_y:{:.2f}, std2_x:{:.2f}, static_stochastic_mode:{}, uncertainty_model_train_mode:{}, train_mode:{}, s2y_est_mode:{}, noise_y_est_mode:{}, noise_x_est_mode:{}, seed:{}, s2y:{:.2f}, s3y:{:.2f}, sigma2x:{:.2f}, s2yrootdifftotrue:{:.2f}, s3yrootdifftotrue:{:.2f}, sigmaxdifftotrue:{:.2f}\n".format(
        args.function, args.std_y_noise**2, args.std_x_noise**2, args.static_stochastic_mode, args.uncertainty_model_train_mode, args.train_mode, args.s2y_est_mode, args.noise_y_est_mode, args.noise_x_est_mode, args.seed, np.mean(s2yroot)**2, np.mean(s3yroot)**2, np.mean(sigmax)**2,s2yrootdifftotrue,s3yrootdifftotrue,sigmaxdifftotrue)
    file.write(message)
    print(message)
    iffig = False
    if iffig:

        fig, axes = plt.subplots(5,2, figsize=(8,10))
        fig.suptitle('s2y mode:{}, s3y mode:{}'.format(args.s2y_est_mode, args.noise_y_est_mode))

        # y_up_lim = 10
        # y_lo_lim = -7

        y_up_lim = 30
        y_lo_lim = -20

        axes[0,0].plot(x,predict_y_mean)
        axes[0,0].fill_between(x, (predict_y_mean-predict_y_std), (predict_y_mean+predict_y_std), alpha=0.1)
        axes[0,0].fill_between(x, (predict_y_mean-0.5*predict_y_std), (predict_y_mean+0.5*predict_y_std), alpha=0.25)
        axes[0,0].title.set_text("model uncertainty")
        axes[0,0].set_ylim(y_lo_lim,y_up_lim)

        y_std = data_generator.get_y_std(x)
        axes[1,0].plot(x,true_y, label = "true y")
        axes[1,0].plot(x,predict_y_mean, label = "predict y mean")
        axes[1,0].fill_between(x, (true_y-y_std), (true_y+y_std), alpha=0.1)
        axes[1,0].title.set_text("data uncertainty")
        axes[1,0].set_ylim(y_lo_lim,y_up_lim)
        axes[1,0].legend()

        axes[0,1].plot(x,true_y)
        axes[0,1].fill_between(x, (true_y-s2yroot), (true_y+s2yroot), alpha=0.1)
        axes[0,1].title.set_text("s2yroot")
        axes[0,1].set_ylim(y_lo_lim,y_up_lim)

        axes[1,1].plot(x,true_y)
        axes[1,1].fill_between(x, (true_y-s3yroot), (true_y+s3yroot), alpha=0.1)
        axes[1,1].title.set_text("s3yroot")
        axes[1,1].set_ylim(y_lo_lim,y_up_lim)

        axes[2,0].plot(x, predict_y_std+0*x, label = 'pred std')
        axes[2,0].plot(x, y_std+0*x, label = 'y noise std')
        axes[2,0].plot(x, s2yroot+0*x, label = 's2yroot')
        axes[2,0].plot(x, s3yroot+0*x, label = 's3yroot')
        axes[2,0].legend()

        x = data['x'].cpu().detach().numpy()
        y_tilde = data['y_tilde'].cpu().detach().numpy()
        y = data['y'].cpu().detach().numpy()
        learned_noise = uncertainty_model.y_noise.cpu().detach().numpy()
        axes[3,0].scatter(x, y_tilde, label = 'noisy data', s=1)
        axes[3,0].scatter(x, y_tilde-np.expand_dims(s3yroot,axis=-1)*learned_noise, label = 'denoised data', s=1)
        # axes[2,0].plot(x, s2yroot+0*x, label = 's2yroot')
        # axes[2,0].plot(x, s3yroot+0*x, label = 's3yroot')
        axes[3,0].legend()

        axes[4,0].scatter(x, learned_noise+x*0, label = 'learned normalized noise', s=1)
        axes[4,1].scatter(x, (y_tilde[:,0]-true_y)/y_std, label = 'actual normalized noise', s=1)
        axes[4,0].legend()
        axes[4,1].legend()



        # def rolling_window(a, window):
        #     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        #     strides = a.strides + (a.strides[-1],)
        #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        # rolling_data_std = np.std(rolling_window(y_tilde,window=5),1)
        rolling_data_std = np.std(sliding_window_view(y_tilde-y,window_shape=(50,1)),-2)[...,0]

        axes[2,1].plot(x[0:rolling_data_std.shape[0],:], rolling_data_std, label = 'rolling y noise std')
        axes[2,1].legend()
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # y_std = data_generator.get_y_std(x)
        # axes.axvline(x0_data[i], color='b', linestyle='--', label='x0 data')
        # axes.axvline(x0_data_clean[i], color='r', linestyle='--', label='x0 clean')
        plt.savefig("test_bayesianMLP_{}.png".format(args.noise_y_est_mode))

        



    # file = open(args.log_name, "a")  # append mode
    # file.write("function:{}, std_x:{}, std_y:{}, s2y_est_mode:{}, noise_y_est_mode:{}, seed:{}, s2yroot:{}, s3yroot:{}\n".format(
    #     args.function, args.std_x_noise, args.std_y_noise, args.s2y_est_mode, args.noise_y_est_mode, args.seed, np.mean(s2yroot), np.mean(s3yroot)))



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
    y_tildes = data['y_tilde'].clone().detach().requires_grad_(True)
    x_tildes = data['x_tilde'].clone().detach().requires_grad_(True)
    uncertainty_model = uncertainty_estimator(x_tildes, y_tildes, args).to(DEVICE)
    # if args.load_static_model:
    #     load_saved_model(static_stoc_model, args.load_static_model)
    if args.train_mode == "separate":
        print("***********\n start training static model...")
        static_stoc_model = train_static_model(data, static_stoc_model, args)
        if args.save_static_model:
            save_model(static_stoc_model, args.save_static_model)
        print("***********\n start training uncertainty model...")
        # args.s2y_est_mode="heter"
        # args.noise_y_est_mode="fixed"
        # uncertainty_model = train_uncertainty_model(data, static_stoc_model, uncertainty_model, data_generator, args)
        # args.s2y_est_mode="fixed"
        # args.noise_y_est_mode="heter"
        uncertainty_model = train_uncertainty_model(data, static_stoc_model, uncertainty_model, data_generator, args)

        # args.noise_y_est_mode="fixed"
        # uncertainty_model_2 = uncertainty_estimator(x_tildes, y_tildes, args).to(DEVICE)
        # uncertainty_model_2 = train_uncertainty_model(data, static_stoc_model, uncertainty_model_2, data_generator, args)

        # draw_model(data, data_generator, static_stoc_model, uncertainty_model, uncertainty_model_2, args)

    elif args.train_mode == "together":
        static_stoc_model, uncertainty_model = train_static_uncertainty_model(data, static_stoc_model, uncertainty_model, args)
        args.noise_y_est_mode = "homo"
        uncertainty_model = uncertainty_estimator(x_tildes, y_tildes, args).to(DEVICE)
        uncertainty_model = train_uncertainty_model(data, static_stoc_model, uncertainty_model, data_generator, args)

    print("***********\n start prediction...")

    predict_model(data, data_generator, static_stoc_model, uncertainty_model, args)
   
    return None

if __name__ == "__main__":
    args = get_args()
    main(args)