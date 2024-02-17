import importlib
import torch
def get_model(args, model_type = 'autoencoder'):
    if model_type == 'dynamics':
        input_dim = args.dynamics_model_input_dim
        output_dim = args.dynamics_model_output_dim
        model_params = {'model_def':args.dynamics_model_def, 'input_dim': args.dynamics_model_input_dim, 'output_dim': args.dynamics_model_output_dim, 'hidden_dim': args.dynamics_model_hidden_dim}
        model_module = importlib.import_module('models.'+args.dynamics_model_def)
    elif model_type == 'traj':
        input_dim = args.traj_model_input_dim
        output_dim = args.traj_model_output_dim
        model_params = {'model_def':args.traj_model_def, 'input_dim': args.traj_model_input_dim, 'output_dim': args.traj_model_output_dim, 'hidden_dim': args.traj_model_hidden_dim}
        model_module = importlib.import_module('models.'+args.traj_model_def)
    elif model_type == 'autoencoder':
        input_dim = args.autoencoder_model_input_dim
        output_dim = args.autoencoder_model_input_dim
        model_params = {'model_def':args.autoencoder_model_def, 'input_dim': args.autoencoder_model_input_dim, 'latent_dim': args.autoencoder_model_latent_dim}
        model_module = importlib.import_module('models.'+args.autoencoder_model_def)
    elif model_type == 'static':     
        input_dim = args.static_model_input_dim
        output_dim = args.static_model_output_dim
        model_params = {'model_def':args.static_model_def, 'input_dim': args.static_model_input_dim, 'output_dim': args.static_model_output_dim, 'hidden_dim': args.static_model_hidden_dim}
        model_module = importlib.import_module('models.'+args.static_model_def)
    model = model_module.Net(**model_params)
    return model
def get_dynamics(args):
    dynamics_module = importlib.import_module('dynamics.'+args.dynamics)
    dynamics = dynamics_module.dynamics()
    return dynamics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def load_saved_model(model, path):
    model.load_state_dict(torch.load(path))
    return

def save_model(model, name):
    torch.save(model.state_dict(), name)
