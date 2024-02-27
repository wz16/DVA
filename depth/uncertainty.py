import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models_depth.model import VPDDepth

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


from configs.test_options import TestOptions
import utils
from models_depth.model import VPDDepth
import torch.backends.cudnn as cudnn

import pickle
import copy
import pandas as pd


class uncertainty_estimator(torch.nn.Module):
    def __init__(self, dataset, noise_type="homo") -> None:
        """
        The dataset here should be train_loader.dataset
        """
        super().__init__()
        # determine noise shape
        y_shape = dataset[0]['depth'].size()
        y_shape = list(y_shape)
        y_shape.insert(0, len(dataset))
        y_shape = tuple(y_shape)
        # this works b/c the gt depth is the same for all images
        self.y_noise = torch.nn.Parameter(torch.randn(y_shape, dtype=torch.float32) * 0.001)

        self.noise_type = noise_type
        if self.noise_type == "homo":
            self.log_sigma2 = torch.nn.Parameter(torch.log(torch.ones(1)*0.1))
            self.log_s2 = torch.nn.Parameter(torch.log(torch.ones(1)*0.1))
        elif self.noise_type == "hetero":
            # create 2-layer NN that takes 1 value as input and output 1 values
            hidden_dim = 100
            self.log_sigma2 = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                )
            self.log_s2 = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )


    def normalize_instance(self, noise, valid_mask=None):
        with torch.no_grad():
            mean, std = self._get_masked_stats(noise, valid_mask)
            noise.add_(-mean,alpha=1)
            noise.div_(std + 1e-8)
            noise.mul_(valid_mask.squeeze(1)) # zero out invalid pixels to prevent exploding


    def _get_masked_stats(self, noise, valid_mask=None):
        """
        Input:
            - noise: tensor of shape (B, H, W)
            - valid_mask: tensor of shape (B, 1, H, W) or (B, H, W)
        Output:
            - mean: 
                - if noise_type=homo, tensor of shape (B, 1, 1)
                - if noise_type=hetero, tensor of shape (B, H, 1)
            - std: a tensor of same dimension as mean
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(noise, dtype=torch.bool)
        else:
            valid_mask = valid_mask.squeeze(1)

        # calculate mean and std of non-masked elements
        if self.noise_type == "homo":
            mean = torch.sum(noise * valid_mask, dim=[-2,-1], keepdim=True) / torch.sum(valid_mask, dim=[-2,-1], keepdim=True)
            std = torch.sqrt(torch.sum((noise - mean) ** 2 * valid_mask, dim=[-2,-1], keepdim=True) \
                            / torch.sum(valid_mask, dim=[-2,-1], keepdim=True))
        elif self.noise_type == "hetero":
            mean = torch.sum(noise * valid_mask, dim=[-1], keepdim=True) / (torch.sum(valid_mask, dim=[-1],  keepdim=True) + 1e-8)
            std = torch.sqrt( torch.sum((noise - mean) ** 2 * valid_mask, dim=[-1], keepdim=True) \
                            / (torch.sum(valid_mask, dim=[-1], keepdim=True) + 1e-8) )
        return mean, std


def vpd_forward(data, class_id, model):
    # transformation on data in accordance to VPD
    bs, _, h, w = data.shape
    assert w > h and bs == 1
    interval_all = w - h
    interval = interval_all // (2-1)
    sliding_images = []
    sliding_masks = torch.zeros((bs, 1, h, w), device=data.device)
    class_ids = []
    for i in range(2): # args.shift_size = 2
        sliding_images.append(data[..., :, i*interval:i*interval+h])
        sliding_masks[..., :, i*interval:i*interval+h] += 1
        class_ids.append(class_id)
    data = torch.cat(sliding_images, dim=0)
    class_ids = torch.cat(class_ids, dim=0)
    data = torch.cat((data, torch.flip(data, [3])), dim=0)
    class_ids = torch.cat((class_ids, class_ids), dim=0)

    # model prediction
    y_pred = model(data, class_ids=class_ids)
    y_pred = y_pred['pred_d']

    # transformation on pred in accordance to VPD
    batch_s = y_pred.shape[0]//2
    y_pred = (y_pred[:batch_s] + torch.flip(y_pred[batch_s:], [3]))/2.0
    pred_s = torch.zeros((bs, 1, h, w), device=y_pred.device)
    for i in range(2):
        pred_s[..., :, i*interval:i*interval+h] += y_pred[i:i+1]
    y_pred = pred_s/sliding_masks

    return y_pred


def train_uncertainty(
        args, model, uncertainty_model, train_loader, 
        epochs=20, mode="DVA", noise_type="homo", noise_amount=1.0,
        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # freeze model
    uncertainty_model.train()
    y_noise_optim = torch.optim.Adam([{'params': uncertainty_model.y_noise}], 0.1, weight_decay=0)
    
    if noise_type == "homo":
        sigma2_optim = torch.optim.Adam([{'params': uncertainty_model.log_sigma2}], 0.1, weight_decay=0)
        s2_optim = torch.optim.Adam([{'params': uncertainty_model.log_s2}], 0.1, weight_decay=0)  
    elif noise_type == "hetero":
        sigma2_optim = torch.optim.Adam(
            [{'params': uncertainty_model.log_sigma2.parameters()}], 0.002, weight_decay=0
            )
        s2_optim = torch.optim.Adam(
            [{'params': uncertainty_model.log_s2.parameters()}], 0.002, weight_decay=0
            )

    for epoch in range(epochs):
        # need to update train_loader such that it also returns the index of the image
        loss_list = []
        true_noise_list = []
        pred_noise_list = []
        error_pred_list = []
        error_denoised_list = []
        valid_mask_list = []
        for batch_idx, sample in enumerate(train_loader):
            # load data
            data = sample['image'].to(device)
            y_tilde = sample['noisy_depth'].to(device)
            index = sample['idx']
            class_id = sample['class_id']

            # zero gradients
            y_noise_optim.zero_grad()
            sigma2_optim.zero_grad()
            s2_optim.zero_grad()

            # model prediction -> predict y
            # y_pred = vpd_forward(data, class_id, model)
            y_pred = sample['depth'].to(device) # use gt as pred for debugging

            # uncertainty model prediction -> predict y_noise
            y_noise = uncertainty_model.y_noise[index].unsqueeze(1)

            # uncertainty model prediction -> predict sigma2
            if noise_type == "homo":
                log_sigma2 = uncertainty_model.log_sigma2
                log_s2 = uncertainty_model.log_s2
            elif noise_type == "hetero":
                height = data.size(2)            
                w = torch.Tensor([[i] for i in range(height)]).to(device)
                log_sigma2 = uncertainty_model.log_sigma2(w)
                log_s2 = uncertainty_model.log_s2(w)

            y_noise_hat = torch.exp(0.5 * log_sigma2) * y_noise

            # masking
            gt = sample['depth'].to(device)
            valid_mask = torch.logical_and(gt > 0.001, gt < 10.0).to(gt.device)
            y_noise_hat = y_noise_hat * valid_mask
            y_pred = y_pred * valid_mask
            y_tilde = y_tilde * valid_mask

            # y_tilde is the denoised y
            y_tilde_hat = y_tilde - y_noise_hat

            # logging stats
            true_noise = sample['depth'] - sample['noisy_depth']
            inferred_noise = true_noise.cpu() - y_noise_hat.cpu()
            true_noise_list.append(true_noise.std().cpu().item())
            pred_noise_list.append(inferred_noise.std().cpu().item())
            error_pred = y_pred - gt
            error_pred_list.append([torch.mean(error_pred).item(), torch.std(error_pred).item(), torch.sqrt(torch.mean(error_pred**2)).item()])
            error_denoised = y_tilde_hat - gt
            error_denoised_list.append([torch.mean(error_denoised).item(), torch.std(error_denoised).item(), torch.sqrt(torch.mean(error_denoised**2)).item()])
            valid_mask_list.append(valid_mask)
            
            # compute loss and update
            y_tilde_hat.squeeze_(1)
            y_pred.squeeze_(1)
            if mode == "DVA":
                loss = (y_tilde_hat - y_pred) ** 2 / (2 * torch.exp(log_s2)) + 0.5 * log_s2
                loss = loss[valid_mask].mean()
            elif mode == "VA":
                loss = (y_tilde - y_pred) ** 2 / (2 * torch.exp(log_s2)) + 0.5 * log_s2
                loss = loss[valid_mask].mean()
                
            loss.backward()
            loss_list.append(loss.item())
            y_noise_optim.step()
            sigma2_optim.step()
            s2_optim.step()

        valid_mask = torch.stack(valid_mask_list, dim=0)
        uncertainty_model.normalize_instance(uncertainty_model.y_noise, valid_mask=valid_mask)

        if epoch % args.log_interval == 0:
            if noise_type == "homo":
                print('Train mode: {}, Epoch: {} \tLoss: {:.3f}, DVA_sigma:{:.3f}, VA_sigma:{:.3f}'.format(
                    mode, epoch, np.mean(loss_list), 
                    torch.exp(0.5 * log_sigma2).detach().cpu().item(),
                    torch.exp(0.5 * log_s2).detach().cpu().item(),
                ))
            elif noise_type == "hetero":
                # compute MSE in variance for heteroskedasticity
                true_std = noise_amount * (1 + 1 * w/height)
                dva_sigma, va_sigma = torch.exp(0.5 * log_sigma2), torch.exp(0.5 * log_s2)
                dva_mse = torch.mean((dva_sigma - true_std) ** 2)
                va_mse = torch.mean((va_sigma - true_std) ** 2)

                print('Train mode: {}, Epoch: {} \tLoss: {:.3f}, DVA MSE:{:.3f}, VA MSE:{:.3f}'.format(
                    mode, epoch, np.mean(loss_list), 
                    dva_mse.detach().cpu().item(),
                    va_mse.detach().cpu().item(),
                ))

    # save sigma values to csv
    if noise_type == "hetero":
        w = w.detach().cpu().numpy()
        true_std = true_std.detach().cpu().numpy()
        dva_sigma = dva_sigma.detach().cpu().numpy()
        va_sigma = va_sigma.detach().cpu().numpy()

        if mode == "DVA":
            df = pd.DataFrame(np.hstack([w, true_std, dva_sigma]), columns=["w", "true_std", "dva_sigma"])
            df.to_csv("hetero/noise_vs_height_dva.csv", index=False)
        elif mode == "VA":
            df = pd.DataFrame(np.hstack([w, true_std, va_sigma]), columns=["w", "true_std", "va_sigma"])
            df.to_csv("hetero/noise_vs_height_va.csv", index=False)

    result = {
        'seed': args.seed,
        'loss': np.mean(loss_list), 
        'true_sigma': args.noise_amount, 
        'pred_sigma': torch.exp(0.5*uncertainty_model.log_sigma2).detach().cpu().item(), 
        'true_noise': np.mean(true_noise_list), 'pred_noise': np.mean(pred_noise_list), 
        'pred_minus_gt': np.mean(error_pred_list, axis=0), 'denoised_minus_gt': np.mean(error_denoised_list, axis=0) # list of [mean, std, RMSE]
        }
    return result
        

def draw_image_with_noise(dataset, model, uncertainty_model, noise_amount, mode):
    log_sigma2 = uncertainty_model.log_sigma2.detach().cpu().numpy()
    y_noise = uncertainty_model.y_noise.detach().cpu().numpy()
    border = 40 # number of pixels as border

    # generate 10 random index from dataset
    indices = np.random.randint(0, len(dataset), size=10)

    for index in indices:
        fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(2, 6, height_ratios=[1, 0.1], width_ratios=[1, 1, 1, 1, 1, 0.10])
        axes = [plt.subplot(gs[i]) for i in range(5)] 

        # input image
        image_rgb = dataset[index]['image'].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(image_rgb[border:-border, border:-border, :])

        # gt depth
        image_clean = dataset[index]['depth']
        valid_mask = torch.logical_and(
            image_clean > 0.001, image_clean < 10.0).to(image_clean.device)
        image_clean = image_clean * valid_mask
        axes[1].imshow(image_clean[border:-border, border:-border].squeeze().cpu().numpy(), cmap='magma_r')

        # gt depth + noise
        image_noise = dataset[index]['noisy_depth']
        image_noise = image_noise * valid_mask
        axes[2].imshow(image_noise[border:-border, border:-border].squeeze().cpu().numpy(), cmap='magma_r')

        # gt depth + noise
        image_denoise = image_noise - np.exp(0.5*log_sigma2) * y_noise[index]
        image_denoise = image_denoise * valid_mask
        axes[3].imshow(image_denoise[border:-border, border:-border].squeeze(), cmap='magma_r')

        # predicted depth
        data = dataset[index]['image'].unsqueeze(0).cuda()
        class_id = torch.tensor([1]).cuda()
        pred = vpd_forward(data, class_id, model)
        pred = pred.squeeze().squeeze().detach().cpu()
        pred[torch.isinf(pred)] = 10.0
        pred[torch.isnan(pred)] = 0.001
        pred = pred * valid_mask
        img = axes[4].imshow(pred[border:-border, border:-border].squeeze().numpy(), cmap='magma_r')

        # colorbar
        cax = plt.subplot(gs[5])
        fig.colorbar(img, cax=cax)
        plt.tight_layout()
        plt.savefig("results/noise_{}/mode_{}/result_{}.png".format(noise_amount, mode, index))


def get_model(args):
    device = torch.device(args.gpu)
        
    model = VPDDepth(args=args)

    # CPU-GPU agnostic settings
    cudnn.benchmark = True
    model.to(device)
    
    from collections import OrderedDict
    model_weight = torch.load("checkpoints/vpd_depth_480x480.pth")['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    return model


def get_dataloader(args):
    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val,
                                             pin_memory=True)
    return val_loader


def add_noise(dataset, noise_amount, noise_type="homo"):
    for i in range(len(dataset)):
        if noise_type == "homo":
            dataset[i]['noisy_depth'] = dataset[i]['depth'] + torch.randn_like(dataset[i]['depth']) * noise_amount
        elif noise_type == "hetero":
            # add heteroskedasticity error according to noise_amount * (1 + x / height of image)
            noisy_depth = copy.deepcopy(dataset[i]['depth'])
            h, w = noisy_depth.shape
            for x in range(h):
                noisy_depth[x] = noisy_depth[x] + torch.randn_like(noisy_depth[x]) * noise_amount * (1 + 1 * x/h)
            dataset[i]['noisy_depth'] = noisy_depth
        else:
            raise NameError("Invalid noise type:", noise_type)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--noise_amount', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
    parser.add_argument('--model_name', type=str, default='AdaBins')
    parser.add_argument('--mode', type=str, default='VA')
    parser.add_argument('--noise_type', type=str, default="hetero")
                        
    # needed for loader
    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='./data/NYUv2/official_splits/test/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='./data/NYUv2/official_splits/test/', type=str,
                        help="path to dataset")
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--input_height', type=int, help='input height', default=352)
    parser.add_argument('--input_width', type=int, help='input width', default=352)


    args = parser.parse_args()
    args.num_threads = args.workers
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    # Define additional args used for VPD
    opt = TestOptions()
    args_vpd = opt.initialize().parse_args()
    args_vpd.save_eval_pngs = True

    args_vpd.ckpt_dir = 'checkpoints/vpd_depth_480x480.pt'
    args_vpd.data_path='./'
    args_vpd.deconv_kernels = [2, 2, 2]
    args_vpd.num_filters=[32, 32, 32]

    print(args_vpd)
    utils.init_distributed_mode_simple(args_vpd)
    print(args_vpd)

    # Define dataset
    print(args.seed)
    train_loader = get_dataloader(args_vpd)
    dataset = train_loader.dataset
    add_noise(dataset, args.noise_amount, args.noise_type)

    # Define model 
    model = get_model(args_vpd)
    model.to(device)

    # Define uncertainty model
    uncertainty_model = uncertainty_estimator(dataset, args.noise_type).to(device)

    # Train
    result = train_uncertainty(
        args, model, uncertainty_model, train_loader, 
        args.epochs, args.mode, args.noise_type, args.noise_amount
        )
    print(result)
    # save result dictionary to pickle file
    with open('./results/noise_{}/mode_{}/result_seed_{}.pkl'.format(args.noise_amount, args.mode, args.seed), 'wb') as f:
        pickle.dump(result, f)
    torch.save(uncertainty_model.state_dict(), "./checkpoints/uncertainty_{}.pt".format(args.model_name))

    # Visualize
    uncertainty_model.load_state_dict(torch.load("./checkpoints/uncertainty_{}.pt".format(args.model_name)))
    draw_image_with_noise(dataset, model, uncertainty_model, args.noise_amount, args.mode)


if __name__ == '__main__':
    main()