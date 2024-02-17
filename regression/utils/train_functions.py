import torch
from torch.utils.data import DataLoader,Dataset
DEVICE = 'cuda'
import math
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
def draw_x_y_tilde(x,y,y_est):
    fig, axes = plt.subplots(1,1)
    axes.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    axes.plot(x.detach().cpu().numpy(), y_est.detach().cpu().numpy())
    fig.savefig("x_y.png")

def get_analytical_kktsolution(data, data_generator, uncertainty_model, model, args):
    # x_tildes = torch.tensor( data['x_tilde'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    # y_tildes = torch.tensor( data['y_tilde'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    x_tildes = data['x_tilde'].clone().detach().requires_grad_(True).to(DEVICE)
    y_tildes = data['y_tilde'].clone().detach().requires_grad_(True).to(DEVICE)
    n_samples = 5
    if args.uncertainty_model_train_mode == "use_individual":
        model.sample()
        y_est = model(x_tildes)
    elif args.uncertainty_model_train_mode == "use_mean":
        y_est = []
        loss_corrected = 0
        for sample in range(n_samples):
            model.sample()
            y_est.append(model(x_tildes))
        y_est = torch.stack(y_est)
        y_est = torch.mean(y_est,dim=0)

    count = x_tildes.shape[0]
    avg_y_tilde_minus_fx = torch.sum(y_tildes-y_est,dim=0)/count
    avg_y_tilde_minus_fx_square = torch.sum((y_tildes-y_est)**2,dim=0)/count

    inferred_s3y_kkt = avg_y_tilde_minus_fx_square-avg_y_tilde_minus_fx**2
    s2y_kkt = avg_y_tilde_minus_fx**2
    noise_kkt = (y_tildes-y_est-avg_y_tilde_minus_fx)/math.sqrt(inferred_s3y_kkt)
    # print("kkt:: inferred_s3y:{:.2f}, s2y:{:.2f}".format(inferred_s3y_kkt.item(), s2y_kkt.item()))
    # print("s2y_opt:{:.2f}".format(avg_y_tilde_minus_fx_square.item()))


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

    s3yrootdifftotrue = np.sqrt(np.mean((s3yroot-y_std)**2))
    s2yrootdifftotrue = np.sqrt(np.mean((s2yroot-y_std)**2))

    message = "s2yrootdifftotrue:{:.2f}, s3yrootdifftotrue:{:.2f}\n".format(
       s2yrootdifftotrue,s3yrootdifftotrue)
    
    print(message)

def train_uncertainty_model(data, model, uncertainty_model, data_generator, args):

    x_tildes = data['x_tilde'].clone().detach().requires_grad_()
    xs = data['x'].clone().detach().requires_grad_()
    y_tildes = data['y_tilde'].clone().detach().requires_grad_()
    ys = data['y'].clone().detach().requires_grad_()

    if args.noise_y_est_mode == "homo":
        y_noise_optim = torch.optim.SGD([
                    {'params': uncertainty_model.y_noise}
                ], args.learning_rate, weight_decay=0)
        s3y_optim = torch.optim.SGD([
                {'params': uncertainty_model.log_inferred_s3y}
            ], args.learning_rate, weight_decay=0)  
    elif args.noise_y_est_mode == "heter":
        y_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.y_noise}
                ], args.learning_rate, weight_decay=0)
        s3y_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_s3y.parameters()}
            ], args.learning_rate, weight_decay=0)  

    if args.noise_x_est_mode == "homo":
        x_noise_optim = torch.optim.SGD([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.SGD([
                {'params': uncertainty_model.log_inferred_sigma2x}
            ], args.learning_rate, weight_decay=0)  
    elif args.noise_x_est_mode == "heter":
        x_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_sigma2x.parameters()}
            ], args.learning_rate, weight_decay=0)  

    if args.s2y_est_mode == "homo":
        s2y_optim = torch.optim.SGD([
                    {'params': uncertainty_model.log_s2y}
                ], args.learning_rate, weight_decay=0)    
    elif args.s2y_est_mode == "heter":
        s2y_optim = torch.optim.Adam([
                    {'params': uncertainty_model.log_s2y.parameters()}
                ], args.learning_rate, weight_decay=0)


    train_tuple = []
    for i in range(xs.shape[0]):
      train_tuple.append((x_tildes[i],xs[i],y_tildes[i],ys[i],i))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)


    model = model.to(DEVICE)
    model.sample()
    model.eval()

    for epoch in range(args.uncertainty_epochs): 

        count = 0
        n_samples = 5

        for x_tilde, x, y_tilde, y, index in train_loader:

            count = y_tilde.shape[0]

            if args.noise_y_est_mode != "fixed":
                y_noise_optim.zero_grad()
                s3y_optim.zero_grad()
            if args.noise_x_est_mode != "fixed":
                x_noise_optim.zero_grad()
                sigma2x_optim.zero_grad()
            if args.s2y_est_mode != "fixed":
                s2y_optim.zero_grad()

            if args.noise_y_est_mode == "fixed":
                y_noise = torch.zeros(1).to(DEVICE)
                log_inferred_s3y = torch.zeros(1).to(DEVICE)
            elif args.noise_y_est_mode == "homo":
                log_inferred_s3y = uncertainty_model.log_inferred_s3y
                y_noise = uncertainty_model.y_noise[index]
            elif args.noise_y_est_mode == "heter":
                log_inferred_s3y = uncertainty_model.log_inferred_s3y(x)
                y_noise = uncertainty_model.y_noise[index]

            if args.noise_x_est_mode == "fixed":
                x_noise = torch.zeros(1).to(DEVICE)
                log_inferred_sigma2x = torch.zeros(1).to(DEVICE)
            elif args.noise_x_est_mode == "homo":
                log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x
                x_noise = uncertainty_model.x_noise[index]
            elif args.noise_x_est_mode == "heter":
                log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x(x)
                x_noise = uncertainty_model.x_noise[index]     

            y_noise_hat = torch.exp(0.5*log_inferred_s3y)*y_noise
            y_tilde_hat = y_tilde-y_noise_hat

            x_noise_hat = torch.exp(0.5*log_inferred_sigma2x)*x_noise
            x_tilde_hat = x_tilde-x_noise_hat

            if args.s2y_est_mode == "fixed":
                log_s2y = torch.zeros(1).to(DEVICE)
            elif args.s2y_est_mode == "homo":
                log_s2y = uncertainty_model.log_s2y
            elif args.s2y_est_mode == "heter":
                log_s2y = uncertainty_model.log_s2y(x)

            if args.uncertainty_model_train_mode == "use_individual":
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    y_est = model(x_tilde_hat)
                    # loss_corrected += 0.5*torch.exp(-log_s2y)*((y_est-y_tilde_hat)**2).sum()+0.5*y_tilde_hat.nelement()*log_s2y
                    loss_corrected += 0.5*(torch.exp(-log_s2y)*((y_est-y_tilde_hat)**2)+log_s2y).sum()
                loss_corrected = loss_corrected / n_samples / count    
            elif args.uncertainty_model_train_mode == "use_mean":
                y_est = []
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    y_est.append(model(x_tilde_hat))
                y_est = torch.stack(y_est)
                y_mean = torch.mean(y_est,dim=0)
                loss_corrected = 0.5*(torch.exp(-log_s2y)*((y_mean-y_tilde_hat)**2)+log_s2y).sum()
                loss_corrected /= count    
            else:
                raise Exception("wrong uncertainty_model_train_mode")   
            loss_corrected.backward()

            if args.s2y_est_mode != "fixed":
                s2y_optim.step()
            if args.noise_y_est_mode != "fixed":
                y_noise_optim.step()
                s3y_optim.step()
            if args.noise_x_est_mode != "fixed":
                x_noise_optim.step()
                sigma2x_optim.step()
            # print("")
            # if args.noise_y_est_mode != "fixed":
            #     uncertainty_model.normalize(y_noise)
            # if args.noise_x_est_mode != "fixed":
            #     uncertainty_model.normalize(x_noise)
        if args.noise_y_est_mode == "heter":
            # uncertainty_model.normalize_by_section_random(uncertainty_model.y_noise)
            uncertainty_model.normalize_by_section(uncertainty_model.y_noise)
        elif args.noise_y_est_mode == "homo":
            # uncertainty_model.normalize(uncertainty_model.y_noise)
            uncertainty_model.normalize_by_section(uncertainty_model.y_noise)

        if args.noise_x_est_mode != "fixed":
            uncertainty_model.normalize(uncertainty_model.x_noise)       

        if epoch % 10 == 0:
        
            print("*****************\n epoch:{},loss:{:.2f}".format(epoch,loss_corrected))
            # print("true y noise std:{:.2f}, true x noise std:{:.2f},\n learned: s2y:{:.2f}, s3y:{:.2f}, sigma2x:{:.2f}".\
            #     format(args.std_y_noise**2,args.std_x_noise**2, math.exp(log_s2y.mean().item()),torch.exp(log_inferred_s3y.mean()).item(),torch.exp(log_inferred_sigma2x.mean()).item()))
            get_analytical_kktsolution(data, data_generator, uncertainty_model, model, args)
    return uncertainty_model



def train_static_uncertainty_model(data, model, uncertainty_model, args):

    # x_tildes = torch.tensor( data['x_tilde'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    x_tildes = data['x_tilde'].clone().detach().requires_grad_()
    # xs = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    xs = data['x'].clone().detach().requires_grad_()
    # y_tildes = torch.tensor( data['y_tilde'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    y_tildes = data['y_tilde'].clone().detach().requires_grad_()
    # ys = torch.tensor( data['y'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    ys = data['y'].clone().detach().requires_grad_()

    optim = torch.optim.Adam(model.parameters(), \
                args.learning_rate, weight_decay=0)

    if args.noise_y_est_mode == "homo":
        y_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.y_noise}
                ], args.learning_rate, weight_decay=0)
        s3y_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_s3y}
            ], args.learning_rate, weight_decay=0)  
    elif args.noise_y_est_mode == "heter":
        y_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.y_noise}
                ], args.learning_rate, weight_decay=0)
        s3y_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_s3y.parameters()}
            ], args.learning_rate, weight_decay=0)  

    if args.noise_x_est_mode == "homo":
        x_noise_optim = torch.optim.SGD([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.SGD([
                {'params': uncertainty_model.log_inferred_sigma2x}
            ], args.learning_rate, weight_decay=0)  
    elif args.noise_x_est_mode == "heter":
        x_noise_optim = torch.optim.SGD([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.SGD([
                {'params': uncertainty_model.log_inferred_sigma2x.parameters()}
            ], args.learning_rate, weight_decay=0)  

    if args.s2y_est_mode == "homo":
        s2y_optim = torch.optim.Adam([
                    {'params': uncertainty_model.log_s2y}
                ], args.learning_rate, weight_decay=0)    
    elif args.s2y_est_mode == "heter":
        s2y_optim = torch.optim.Adam([
                    {'params': uncertainty_model.log_s2y.parameters()}
                ], args.learning_rate, weight_decay=0)


    train_tuple = []
    for i in range(xs.shape[0]):
      train_tuple.append((x_tildes[i],xs[i],y_tildes[i],ys[i],i))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)


    model = model.to(DEVICE)
    model.sample()
    model.eval()

    for epoch in range(args.uncertainty_epochs): 

        count = 0
        n_samples = 5

        for x_tilde, x, y_tilde, y, index in train_loader:

            count = y_tilde.shape[0]

            optim.zero_grad()
            if args.noise_y_est_mode != "fixed":
                y_noise_optim.zero_grad()
                s3y_optim.zero_grad()
            if args.noise_x_est_mode != "fixed":
                x_noise_optim.zero_grad()
                sigma2x_optim.zero_grad()
            if args.s2y_est_mode != "fixed":
                s2y_optim.zero_grad()

            if args.noise_y_est_mode == "fixed":
                y_noise = torch.zeros(1).to(DEVICE)
                log_inferred_s3y = torch.zeros(1).to(DEVICE)
            elif args.noise_y_est_mode == "homo":
                log_inferred_s3y = uncertainty_model.log_inferred_s3y
                y_noise = uncertainty_model.y_noise[index]
            elif args.noise_y_est_mode == "heter":
                log_inferred_s3y = uncertainty_model.log_inferred_s3y(x)
                y_noise = uncertainty_model.y_noise[index]

            if args.noise_x_est_mode == "fixed":
                x_noise = torch.zeros(1).to(DEVICE)
                log_inferred_sigma2x = torch.zeros(1).to(DEVICE)
            elif args.noise_x_est_mode == "homo":
                log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x
                x_noise = uncertainty_model.x_noise[index]
            elif args.noise_x_est_mode == "heter":
                log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x(x)
                x_noise = uncertainty_model.x_noise[index]     

            y_noise_hat = torch.exp(0.5*log_inferred_s3y)*y_noise
            y_tilde_hat = y_tilde-y_noise_hat

            x_noise_hat = torch.exp(0.5*log_inferred_sigma2x)*x_noise
            x_tilde_hat = x_tilde-x_noise_hat

            if args.s2y_est_mode == "fixed":
                log_s2y = torch.zeros(1).to(DEVICE)
            elif args.s2y_est_mode == "homo":
                log_s2y = uncertainty_model.log_s2y
            elif args.s2y_est_mode == "heter":
                log_s2y = uncertainty_model.log_s2y(x)

            if args.uncertainty_model_train_mode == "use_individual":
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    y_est = model(x_tilde_hat)
                    # loss_corrected += 0.5*torch.exp(-log_s2y)*((y_est-y_tilde_hat)**2).sum()+0.5*y_tilde_hat.nelement()*log_s2y
                    loss_corrected += 0.5*(torch.exp(-log_s2y)*((y_est-y_tilde_hat)**2)+log_s2y).sum()
                loss_corrected = loss_corrected / n_samples / count    
            elif args.uncertainty_model_train_mode == "use_mean":
                y_est = 0
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    y_est +=  model(x_tilde_hat)
                y_mean = y_est/n_samples
                loss_corrected = 0.5*(torch.exp(-log_s2y)*((y_mean-y_tilde_hat)**2)+log_s2y).sum()
                loss_corrected /= count    
            else:
                raise Exception("wrong uncertainty_model_train_mode")   

            if args.s2y_est_mode != "fixed":
                s2y_optim.step()
            if args.noise_y_est_mode != "fixed":
                y_noise_optim.step()
                s3y_optim.step()
            if args.noise_x_est_mode != "fixed":
                x_noise_optim.step()
                sigma2x_optim.step()
        if args.noise_y_est_mode != "fixed":
            uncertainty_model.normalize(uncertainty_model.y_noise)
        if args.noise_x_est_mode != "fixed":
            uncertainty_model.normalize(uncertainty_model.x_noise)       

        if epoch % 10 == 0:
            print("*****************\n epoch:{},loss:{:.2f}".format(epoch,loss_corrected))
            print("true y noise variance:{:.2f},\n learned: s2y:{:.2f}, s3y:{:.2f}, sigma2x:{:.2f}".\
                format(args.std_y_noise**2,math.exp(log_s2y.mean().item()),torch.exp(log_inferred_s3y.mean()).item(),torch.exp(log_inferred_sigma2x.mean()).item()))
            get_analytical_kktsolution(data, model, args)
    return model, uncertainty_model



def loglikelihood(output, target, sigma):
        sigma = torch.tensor(sigma).to(DEVICE)
        result = -torch.sum((output-target)**2)/2/(sigma**2)+output.nelement()/2*torch.log(1./2./torch.pi/(sigma**2))
        return result

def train_static_model(data, model, args):

    x_tildes = data['x_tilde'].clone().detach().to(DEVICE)
    # xs = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    xs = data['x'].clone().detach().to(DEVICE)
    # y_tildes = torch.tensor( data['y_tilde'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    y_tildes = data['y_tilde'].clone().detach().to(DEVICE)
    # ys = torch.tensor( data['y'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    ys = data['y'].clone().detach().to(DEVICE)

    train_tuple = []
    for i in range(xs.shape[0]):
      train_tuple.append((x_tildes[i],xs[i],y_tildes[i],ys[i],i))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)

    optim = torch.optim.Adam(model.parameters(), \
                args.learning_rate, weight_decay=0)

    model = model.to(DEVICE)
    model.sample()
    model.train()


    for epoch in range(args.static_epochs):

        mean_loss = 0
        count = 0
        n_samples = 5
        if args.static_stochastic_mode == "bayesian":
            for x_tilde, x, y_tilde, y, index in train_loader:

                count = y_tilde.shape[0]

                optim.zero_grad()

                loss_corrected = 0
                loss_y_true = 0
                loss_y_tilde = 0

                loss = 0
                for sample in range(n_samples):
                    model.sample()
                    y_est = model(x_tilde)
                    loss -= loglikelihood(y_est, y_tilde, sigma=1)
                    loss += model.kl_divergence()

                loss = loss / n_samples / count      
                loss.backward()
                optim.step()
            
            if epoch % 10 == 0:
                print("*****************\n epoch:{},loss:{}".format(epoch,loss))

        elif args.static_stochastic_mode == "ensemble":
            for sample in range(n_samples):
                model.sample()
                for x_tilde, x, y_tilde, y, index in train_loader:
                

                    count = y_tilde.shape[0]

                    optim.zero_grad()
                    loss = 0

                    y_est = model(x_tilde)
                    # loss += torch.nn.MSELoss()(y_est, y_tilde)
                    loss -= loglikelihood(y_est, y_tilde, sigma=1)
                    loss += model.kl_divergence()

                    loss = loss / count      
                    loss.backward()
                    optim.step()
            if epoch % 10 == 0:
                print("*****************\n epoch:{},loss:{}".format(epoch,loss))
    return model


def train_neuralODE_uncertainty_model(data, model, uncertainty_model, args):

    x_tildes = data['x_tilde'].clone().detach().requires_grad_().to(DEVICE)
    t_eval = data['t_eval'].clone().detach().to(DEVICE)


    if args.noise_x_est_mode == "homo":
        x_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_sigma2x}
            ], args.learning_rate, weight_decay=0)  
    elif args.noise_x_est_mode == "heter":
        x_noise_optim = torch.optim.Adam([
                    {'params': uncertainty_model.x_noise}
                ], args.learning_rate, weight_decay=0)
        sigma2x_optim = torch.optim.Adam([
                {'params': uncertainty_model.log_inferred_sigma2x.parameters()}
            ], args.learning_rate, weight_decay=0)  

    if args.s2y_est_mode == "homo":
        s2y_optim = torch.optim.SGD([
                    {'params': uncertainty_model.log_s2y}
                ], args.learning_rate, weight_decay=0)    
    elif args.s2y_est_mode == "heter":
        s2y_optim = torch.optim.Adam([
                    {'params': uncertainty_model.log_s2y.parameters()}
                ], args.learning_rate, weight_decay=0)


    train_tuple = []
    for i in range(x_tildes.shape[0]):
      train_tuple.append((x_tildes[i],i))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)


    def func_remove_t_wrapper(t,x):
        return model(x)


    model = model.to(DEVICE)
    model.sample()
    model.eval()

    for epoch in range(args.uncertainty_epochs): 

        count = 0
        n_samples = 5

        for x_tilde,  index in train_loader:

            count = x_tilde.shape[0]

            if args.noise_x_est_mode != "fixed":
                x_noise_optim.zero_grad()
                sigma2x_optim.zero_grad()
            if args.s2y_est_mode != "fixed":
                s2y_optim.zero_grad()

            if args.noise_x_est_mode == "fixed":
                x_noise = torch.zeros(1).to(DEVICE)
                log_inferred_sigma2x = torch.zeros(1).to(DEVICE)
            elif args.noise_x_est_mode == "homo":
                log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x
                x_noise = uncertainty_model.x_noise[index]
            elif args.noise_x_est_mode == "heter":
                raise Exception("heter not supported")

            x_noise_hat = torch.exp(0.5*log_inferred_sigma2x)*x_noise
            x_tilde_hat = x_tilde-x_noise_hat

            if args.s2y_est_mode == "fixed":
                log_s2y = torch.zeros(1).to(DEVICE)
            elif args.s2y_est_mode == "homo":
                log_s2y = uncertainty_model.log_s2y
            elif args.s2y_est_mode == "heter":
                raise Exception("heter not supported")

                

            x0_tilde_hat = x_tilde_hat[:,0,:]
            if args.uncertainty_model_train_mode == "use_individual":
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    x_est = torch.transpose(odeint(func_remove_t_wrapper, x0_tilde_hat, t_eval),0,1)
                    loss_corrected += 0.5*(torch.exp(-log_s2y)*((x_est-x_tilde_hat)**2)+log_s2y).sum()
                loss_corrected = loss_corrected / n_samples / count    
            elif args.uncertainty_model_train_mode == "use_mean":
                x_ests = []
                loss_corrected = 0
                for sample in range(n_samples):
                    model.sample()
                    x_est = torch.transpose(odeint(func_remove_t_wrapper, x0_tilde_hat, t_eval),0,1)
                    x_ests.append(x_est)
                x_ests = torch.stack(x_ests)
                x_est = torch.mean(x_ests,dim=0)
                loss_corrected = 0.5*(torch.exp(-log_s2y)*((x_est-x_tilde_hat)**2)+log_s2y).sum()
                loss_corrected /= count    
            else:
                raise Exception("wrong uncertainty_model_train_mode")   
            loss_corrected.backward()

            if args.s2y_est_mode != "fixed":
                s2y_optim.step()

            if args.noise_x_est_mode != "fixed":
                x_noise_optim.step()
                sigma2x_optim.step()
            # print("")
        if args.noise_y_est_mode != "fixed":
            uncertainty_model.normalize(uncertainty_model.y_noise)
        if args.noise_x_est_mode != "fixed":
            uncertainty_model.normalize(uncertainty_model.x_noise)       

        if epoch % 1 == 0:
        
            print("*****************\n epoch:{},loss:{:.2f}".format(epoch,loss_corrected))
            print("true y noise std:{:.2f}, true x noise std:{:.2f},\n learned: s2y:{:.2f}, sigma2x:{:.2f}".\
                format(args.std_y_noise**2,args.std_x_noise**2, math.exp(log_s2y.mean().item()),torch.exp(log_inferred_sigma2x.mean()).item()))
    return uncertainty_model


def train_neuralODE_model(data, model, args):

    x_tildes = data['x_tilde'].clone().detach().requires_grad_().to(DEVICE)
    t_eval = data['t_eval'].clone().detach().to(DEVICE)

    train_tuple = []
    for i in range(x_tildes.shape[0]):
      train_tuple.append((x_tildes[i],i))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)

    optim = torch.optim.Adam(model.parameters(), \
                args.learning_rate, weight_decay=0)

    model = model.to(DEVICE)
    model.sample()
    model.train()

    def func_remove_t_wrapper(t,x):
        return model(x)


    for epoch in range(args.static_epochs):

        mean_loss = 0
        count = 0
        n_samples = 5
        if args.static_stochastic_mode == "bayesian":
            for x_tilde, index in train_loader:

                count = x_tilde.shape[0]

                optim.zero_grad()

                loss_corrected = 0
                loss_y_true = 0
                loss_y_tilde = 0

                loss = 0
                x0_tilde = x_tilde[:,0,:]
                for sample in range(n_samples):
                    model.sample()
                    x_est = torch.transpose(odeint(func_remove_t_wrapper, x0_tilde, t_eval),0,1)
                    # y_est = model(x_tilde)
                    loss -= loglikelihood(x_est, x_tilde, sigma=1)
                    loss += model.kl_divergence()

                loss = loss / n_samples / count      
                loss.backward()
                optim.step()
            
            if epoch % 10 == 0:
                print("*****************\n epoch:{},loss:{}".format(epoch,loss))

        elif args.static_stochastic_mode == "ensemble":
            for sample in range(n_samples):
                model.sample()
                for x_tilde, x, y_tilde, y, index in train_loader:

                    count = y_tilde.shape[0]

                    optim.zero_grad()
                    loss = 0

                    y_est = model(x_tilde)
                    loss -= loglikelihood(y_est, y_tilde, sigma=1)
                    loss += model.kl_divergence()

                    loss = loss / count      
                    loss.backward()
                    optim.step()
            if epoch % 10 == 0:
                print("*****************\n epoch:{},loss:{}".format(epoch,loss))
    return model