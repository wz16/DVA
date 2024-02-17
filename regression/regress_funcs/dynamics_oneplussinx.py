
import torch, argparse
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


from torchdiffeq import odeint


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--function', default = 'xbyoneplussinx_homo', type = str)
    parser.add_argument('--train_mode', default = 'separate', type = str, 
                        help="separate: train static model first, then uncertainty model, together: train static and uncertainty models together ")
    parser.add_argument('--static_stochastic_mode',default = 'bayesian', type = str, help="ensemble or bayesian")
    parser.add_argument('--static_model_def', default = 'MLP_bayesian', type = str)
    parser.add_argument('--static_model_input_dim', default = 1, type = int)
    parser.add_argument('--static_model_output_dim', default = 1, type = int)
    parser.add_argument('--static_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--save_static_model', default = "test_static_model.tar", type = str)
    parser.add_argument('--load_static_model', default = "test_static_model.tar", type = str)

    parser.add_argument('--uncertainty_model_train_mode',default = 'use_individual', type = str, 
                        help="use_mean: use mean of prediction in the uncertainty training, use_individual use individual f(x)")

    parser.add_argument('--s2y_est_mode', default = "homo", type=str, 
                        help="fixed: s2y not trained, homo: train single s2y, heter: train s2y(x) depent on x")
    parser.add_argument('--s2y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_y_est_mode', default = "fixed", type=str, 
                        help="fixed: noise estimation = 0, not train, homo: train single sigma2y, heter: train sigma2y(x) depent on x")
    parser.add_argument('--noise_y_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--noise_x_est_mode', default = "homo", type=str, 
                        help="fixed: noise estimation = 0, not train, homo: train single sigma2x, heter: train sigma2x(x) depent on x")

    parser.add_argument('--noise_x_model_hidden_dim', default = 100, type = int)

    parser.add_argument('--uncertainty_epochs', default = 400, type = int)
    parser.add_argument('--static_epochs', default = 200, type = int)
    parser.add_argument('--total_examples', default = 1000, type = int)
    parser.add_argument('--batch_size', default = 100, type = int)
    parser.add_argument('--learning_rate', default = 1e-2, type = float)
    parser.add_argument('--mu_x', default = 5., type = float)
    parser.add_argument('--rho_x', default = 2, type = float)
    parser.add_argument('--std_x_noise', default = 2, type = float)
    parser.add_argument('--std_y_noise', default = 0, type = float)

    parser.add_argument('--seed', default= 20, type=int, help='random seed')
    parser.add_argument('--log_name', default = 'denoise_results.txt', type = str)
    parser.add_argument('--if_opt_log', default = True, type = bool)
    
    return parser.parse_args()

class Data_generator():

    def __init__(self,args):
        self.std_x_noise = args.std_x_noise
        self.std_y_noise = args.std_y_noise
        self.rho_x = args.rho_x
        self.mu_x = args.mu_x
        self.step_time = args.step_time
        self.t_eval = torch.linspace(0, args.t_span, int((args.t_span-0)/self.step_time)+1)
        return

    def get_y_std(self , x):
        return self.std_y_noise

    def get_x_std(self , x):
        return self.std_x_noise

    def x_to_dx(self,t,x):
        return (1+torch.sin(x))


        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
    def get_data(self, args, total_examples, device):
        # x = torch.normal(args.mu_x, args.rho_x, size=(args.total_examples,1)).to(DEVICE)
        # x =  (torch.rand(size=(total_examples,1))*2-1)*self.rho_x+self.mu_x
        x = torch.linspace(args.mu_x-2*args.rho_x,args.mu_x+2*args.rho_x,total_examples).to(device).unsqueeze(-1)
        x = x.to(device)
        
        x0 = torch.linspace(args.mu_x-2*args.rho_x,args.mu_x+2*args.rho_x,total_examples).to(device).unsqueeze(-1)
        
        x = torch.transpose(odeint(self.x_to_dx, x0, self.t_eval.to(device)),0,1)

        x_tilde = x+torch.randn(size=x.shape).to(device)*self.get_x_std(x)
        data = {'x':x,'x_tilde':x_tilde,"t_eval":self.t_eval.to(device)}

        return data

# class dynamics(base_dynamics):
    
#     def __init__(self,**kwargs):
#         super().__init__()
#         self.dynamics_dim = 1
#         return None
#     def dynamics_fn(self, t, coords):

#         return 2.

#         # return -2.*coords

#     def conservation_fn(self, coords):
#         q, p = np.split(coords,2,axis=-1)
#         H = p**2 + q**2 # spring hamiltonian (linear oscillator)
#         return H

#     def random_init(self, batch_dim = 1):
#         y0 = np.random.rand(batch_dim,1)*2-1
#         radius = np.random.rand(batch_dim,1)*0.9 + 0.1
#         y0 = y0 / np.sqrt((y0**2).sum()) * radius
#         return y0
    
#     def get_trajectory(self, t_span=[0,5], step_time=0.1, radius=None, y0=None,noise_type='gaussian', noise_level=0., **kwargs):
#         t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/step_time)+1)

#         if y0 is None:
#             y0 = np.squeeze(self.random_init(), axis = 0)

#         spring_ivp = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
#         q = spring_ivp['y'][0]
#         dydt = [self.dynamics_fn(None, y) for y in spring_ivp['y'].T]
#         dydt = np.stack(dydt).T
#         dqdt = dydt
        
#         # add noise
#         if noise_type not in ["gaussian"]:
#             raise ValueError('noise type not in [gaussian].')
#         q_noise = q+np.random.randn(*q.shape)*noise_level


#         dqdt_noise = dqdt+np.random.randn(*dqdt.shape)*noise_level
#         return q, q_noise, dqdt, dqdt_noise, t_eval
    


    
#     def get_data(self, sample=50, test_split=0.5, dim = 2, noise_type = 'gaussian', noise_level = 0., step_time = 0.1, t_span = 5):

#         xs, xs_noise, dxs, dxs_noise, ts = [], [], [], [], []
#         for s in range(sample):
#             x, x_noise, dx, dx_noise, t = self.get_trajectory(noise_type = noise_type, noise_level = noise_level, step_time = step_time, t_span = [0,t_span])
#             xs.append( np.stack( [x]).T )
#             xs_noise.append(np.stack( [x_noise]).T )
#             dxs.append( np.stack( [dx]).T )
#             dxs_noise.append( np.stack( [dx_noise]).T )
#             ts.append(t)
        
#         data = {}
#         if dim == 2:
#             data['t'] = np.concatenate(ts)
#             data['x_clean'] = np.concatenate(xs)
#             data['x'] = np.concatenate(xs_noise)
#             data['fx_clean'] = np.concatenate(dxs).squeeze()
#             data['fx'] = np.concatenate(dxs_noise).squeeze()
            
#         elif dim == 3:
#             data['t'] = np.stack(ts)
#             data['x_clean'] = np.stack(xs)
#             data['x'] = np.stack(xs_noise)
#             data['fx_clean'] = np.stack(dxs)
#             data['fx'] = np.stack(dxs_noise)
#             # data['fx'] = np.stack(dxs).squeeze()
        


#         split_ix = int(len(data['x']) * test_split)
#         split_data = {}
#         for k in ['t', 'x', 'fx',"x_clean", "fx_clean"]:
#             split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
#         data = split_data
#         return data

    
#     def gx(x):
#         return torch.sum(x**2, dim = -1)





def test():
    args = get_args()
    data_generator = Data_generator(args)
    data_generator.get_data(args,1000,'cuda')


if __name__ == "__main__":
    test()
