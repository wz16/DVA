import torch

class Data_generator():

    def __init__(self,args):
        self.std_x_noise = args.std_x_noise
        self.std_y_noise = args.std_y_noise
        self.rho_x = args.rho_x
        self.mu_x = args.mu_x
        return

    def get_y_std(self , x):
        return self.std_y_noise

    def get_x_std(self , x):
        return self.std_x_noise

    def x_to_y(self,x):
        return x*(1+torch.sin(x))
    
    def get_data(self, args, total_examples, device):
        # x = torch.normal(args.mu_x, args.rho_x, size=(args.total_examples,1)).to(DEVICE)
        # x =  (torch.rand(size=(total_examples,1))*2-1)*self.rho_x+self.mu_x
        x = torch.linspace(args.mu_x-2*args.rho_x,args.mu_x+2*args.rho_x,total_examples).to(device).unsqueeze(-1)
        x = x.to(device)
        # y = x*a
        y = self.x_to_y(x)

        x_tilde = x+torch.randn(size=x.shape).to(device)*self.get_x_std(x)
        y_tilde = y+torch.randn(size=y.shape).to(device)*self.get_y_std(x)
        # x_tilde = x+math.sqrt(3)*(torch.rand(size=x.shape)*2-1).to(DEVICE)*args.std_x_noise
        # y_tilde = y+math.sqrt(3)*(torch.rand(size=y.shape)*2-1).to(DEVICE)*args.std_y_noise
        data = {'x':x,'y':y,'x_tilde':x_tilde,'y_tilde': y_tilde}

        return data
