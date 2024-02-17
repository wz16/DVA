import torch
from models.MLP import Net as MLP
from models.linear_function import Net as linear_function
from models.log_linear_function import Net as log_linear_function
import math
class uncertainty_estimator(torch.nn.Module):
    def __init__(self, x_data, y_data, args) -> None:
        super().__init__()
        self.s2y_est_mode = args.s2y_est_mode
        self.noise_y_est_mode = args.noise_y_est_mode
        self.noise_x_est_mode = args.noise_x_est_mode

        # log(s^2_y) in the gaussian log likelyhood
        if self.s2y_est_mode == "fixed":
            self.log_s2y = torch.zeros(1)*0
        elif self.s2y_est_mode == "homo":
            self.log_s2y = torch.nn.Parameter(torch.ones(1)*math.log(2))
        elif self.s2y_est_mode == "heter":
            model_params = {'input_dim': args.static_model_input_dim, 'output_dim': args.static_model_input_dim, 'hidden_dim': args.s2y_model_hidden_dim}
            self.log_s2y = MLP(**model_params)
        else:
            raise Exception("wrong s2y_est_mode")
        
        if self.noise_y_est_mode == "fixed":
            self.y_noise = torch.zeros(1)*0
            self.log_inferred_s3y = torch.zeros(1)+0
        elif self.noise_y_est_mode == "homo":
            self.y_noise = torch.nn.Parameter(torch.randn_like(y_data)*0) # estimator of the normalized noise
            self.log_inferred_s3y = torch.nn.Parameter(torch.log(torch.ones(1)*1)) # estimator of log of the noise variance
        elif self.noise_y_est_mode == "heter":
            self.y_noise = torch.nn.Parameter(torch.randn_like(y_data)) # estimator of the normalized noise
            self.normalize(self.y_noise)
            model_params = {'input_dim': args.static_model_input_dim, 'output_dim': args.static_model_output_dim, 'hidden_dim': args.noise_y_model_hidden_dim}
            self.log_inferred_s3y = MLP(**model_params)
        else:
            raise Exception("wrong s2y_est_mode")

        if self.noise_x_est_mode == "fixed":
            self.x_noise = torch.zeros(1)*0
            self.log_inferred_sigma2x = torch.zeros(1)+0
        elif args.noise_x_est_mode == "homo":
            self.x_noise = torch.nn.Parameter(torch.randn_like(x_data)*0) # estimator of the normalized noise
            # self.normalize(self.x_noise)
            self.log_inferred_sigma2x = torch.nn.Parameter(torch.log(torch.ones(1)*1)) # estimator of log of the noise variance
        elif args.noise_x_est_mode == "heter":
            self.x_noise = torch.nn.Parameter(torch.randn_like(x_data)*0) # estimator of the normalized noise
            model_params = {'input_dim': args.static_model_input_dim, 'output_dim': args.static_model_input_dim, 'hidden_dim': args.noise_x_model_hidden_dim}
            self.log_inferred_sigma2x = MLP(**model_params)
        else:
            raise Exception("wrong s2x_est_mode")
        
    def normalize(self, noise):
        with torch.no_grad():
            std = torch.std(noise)
            mean = torch.mean(noise)
            noise.add_(-mean,alpha=1)
            noise.div_(std)

    def normalize_by_section(self, noise, n_section=10):
        noise_ = noise.detach().clone().reshape(n_section,noise.shape[0]//n_section,1)
        with torch.no_grad():
            std = torch.std(noise_,dim=1,keepdim=True)
            mean = torch.mean(noise_,dim=1,keepdim=True)
            noise_.add_(-mean,alpha=1)
            noise_.div_(std)
            noise__ = torch.reshape(noise_,noise.shape)
            noise.mul_(0)
            noise.add_(noise__,alpha=1)
        return
    
    def normalize_by_section_random(self, noise, n_section=10):
        shift_amount = torch.randint(noise.shape[0], size=(1,)).item()
        indices = torch.arange(noise.shape[0])
        shifted_indices =  torch.roll(indices, shift_amount)
        inverse_indices = torch.argsort(shifted_indices)

        noise_ = noise[shifted_indices].detach().clone().reshape(n_section,noise.shape[0]//n_section,1)
        with torch.no_grad():
            std = torch.std(noise_,dim=1,keepdim=True)
            mean = torch.mean(noise_,dim=1,keepdim=True)
            noise_.add_(-mean,alpha=1)
            noise_.div_(std)
            noise__ = torch.reshape(noise_,noise.shape)
            noise.mul_(0)
            noise.add_(noise__[inverse_indices],alpha=1)
        return
    
    def set_noise(self, noise):
        with torch.no_grad():
            self.y_noise.mul_(0)
            self.y_noise.add_(noise.detach(),alpha=1)

    def set_log_inferred_s3y(self,log_inferred_s3y):
        with torch.no_grad():
            self.log_inferred_s3y.mul_(0)
            self.log_inferred_s3y.add_(log_inferred_s3y.detach(),alpha=1)

    def get_log_s2y(self, x):
        if self.s2y_est_mode == "fixed":
            return self.log_s2y
        elif self.s2y_est_mode == "homo":
            return self.log_s2y
        elif self.s2y_est_mode == "heter":
            return self.log_s2y(x)
        else:
            raise Exception("wrong s2y_est_mode")

    def get_log_s3y(self, x):
        if self.noise_y_est_mode == "fixed":
            return self.log_inferred_s3y
        elif self.noise_y_est_mode == "homo":
            return self.log_inferred_s3y
        elif self.noise_y_est_mode == "heter":
            return self.log_inferred_s3y(x)
        else:
            raise Exception("wrong noise_y_est_mode")
        
    def get_log_sigma2x(self, x):
        if self.noise_x_est_mode == "fixed":
            return self.log_inferred_sigma2x
        elif self.noise_x_est_mode == "homo":
            return self.log_inferred_sigma2x
        elif self.noise_x_est_mode == "heter":
            return self.log_inferred_sigma2x(x)
        else:
            raise Exception("wrong noise_y_est_mode")    