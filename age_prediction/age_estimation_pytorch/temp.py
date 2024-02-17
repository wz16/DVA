from infer import *
import torch
import argparse
from dataloader import DepthDataLoader
import copy


class uncertainty_estimator(torch.nn.Module):
    def __init__(self, dataset) -> None:
        """
        The dataset here should be train_loader.dataset
        """
        super().__init__()
        # determine noise shape
        y_shape = dataset[0]['depth'].size()
        y_shape = list(y_shape)
        y_shape[0] = len(dataset)
        y_shape = tuple(y_shape)
        # this works b/c the gt depth is the same for all images
        self.y_noise = torch.nn.Parameter(torch.randn(y_shape, dtype=torch.float32) * 0.001)
        self.log_inferred_sigma2x = torch.nn.Parameter(torch.log(torch.ones(1)*0.1)) # estimator of log of the noise variance

    def normalize(self, noise):
        with torch.no_grad():
            std = torch.std(noise)
            mean = torch.mean(noise)
            noise.add_(-mean,alpha=1)
            noise.div_(std)

    def normalize_instance(self, noise):
        with torch.no_grad():
            std = torch.std(noise,dim=[-2,-1],keepdim=True)
            mean = torch.mean(noise,dim=[-2,-1],keepdim=True)
            noise.add_(-mean,alpha=1)
            noise.div_(std)


def train_uncertainty(args, model, uncertainty_model, train_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # freeze model
    uncertainty_model.train()
    y_noise_optim = torch.optim.Adam([{'params': uncertainty_model.y_noise}], 1, weight_decay=0)
    sigma2x_optim = torch.optim.Adam([{'params': uncertainty_model.log_inferred_sigma2x}], 1, weight_decay=0)  

    for epoch in range(epochs):
        # need to update train_loader such that it also returns the index of the image
        loss_list = []
        for batch_idx, sample in enumerate(train_loader):
            # load data
            data = sample['image']
            target = sample['noisy_depth']
            index = sample['idx']
            data, target = data.to(device), target.to(device)

            # zero gradients
            y_noise_optim.zero_grad()
            sigma2x_optim.zero_grad()

            # model prediction -> predict y
            y_pred = model(data)

            # uncertainty model prediction -> predict y_noise 
            y_noise = uncertainty_model.y_noise[index].unsqueeze(1)    
            y_noise_hat = torch.exp(0.5 * uncertainty_model.log_inferred_sigma2x) * y_noise

            # y_tilde is the denoised y
            # y_pred = sample['depth'].to(device) # use gt depth as y_pred temporarily
            y_tilde = y_pred - y_noise_hat
            print(y_noise.mean().item(), y_noise.std().item(), y_noise.abs().mean().item())
            print(uncertainty_model.log_inferred_sigma2x.item())

            true_noise = sample['depth'] - sample['noisy_depth']
            print(true_noise.std().cpu(), (true_noise.cpu() - y_noise_hat.cpu()).std().cpu())

            # compute loss and update
            loss = torch.nn.MSELoss()(y_tilde, target)
            loss.backward()
            loss_list.append(loss.item())
            y_noise_optim.step()
            sigma2x_optim.step()
            # if epoch > 0:
            #     print("grad:", uncertainty_model.y_noise.grad.sum().item())
            if epoch == epochs-1:
                # Create a subplot with 1 row and 3 columns
                fig, axs = plt.subplots(1, 3)

                # Assign each individual figure to a subplot
                axs[0].imshow(data[2].cpu().permute(1, 2, 0).numpy(),cmap='magma_r')
                axs[0].set_title('Data')

                axs[1].imshow(target[2].squeeze(0).cpu().numpy(),cmap='magma_r')
                axs[1].set_title('Target')

                axs[2].imshow(y_tilde[2].squeeze(0).detach().cpu().numpy(),cmap='magma_r')
                axs[2].set_title('Y Tilde')

                # Adjust the spacing between subplots
                plt.tight_layout()

                # Save the subplot to a file
                plt.savefig('subplot.png')

        uncertainty_model.normalize_instance(uncertainty_model.y_noise)

        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}, sigmax:{:.4f}'.format(
                epoch, np.mean(loss_list), torch.exp(0.5*uncertainty_model.log_inferred_sigma2x).detach().cpu().item())
                )
    

class InferenceModel(nn.Module):
    def __init__(self, model_name='AdaBins'):
        super(InferenceModel, self).__init__()
        self.min_depth = 1e-3
        self.max_depth = 10
        self.model_name = model_name
        self.model = self._load_model()    

    def _load_model(self):
        if self.model_name == 'AdaBins':
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_nyu.pt"
        elif self.model_name == "Gaussian":
            model = UnetGaussian.build(min_val=self.min_depth, max_val=self.max_depth, gaussian=False)
            pretrained_path = "./pretrained/Gaussian_nyu.pt"
        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        return model


    def forward(self, image):
        _, pred = self.model(image)

        pred = torch.clamp(pred, self.min_depth, self.max_depth)
        final = pred

        # Interpolate
        final = nn.functional.interpolate(final, image.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)

        final = torch.where(final < self.min_depth, self.min_depth, final)
        final = torch.where(final > self.max_depth, self.max_depth, final)
        final = torch.where(torch.isinf(final), self.max_depth, final)
        final = torch.where(torch.isnan(final), self.min_depth, final)
        return final


def draw_image_with_noise(dataset, model, uncertainty_model):
    fig, axes = plt.subplots(6,4, figsize=(20,80))
    log_inferred_sigma2x = uncertainty_model.log_inferred_sigma2x.detach().cpu().numpy()
    y_noise = uncertainty_model.y_noise.detach().cpu().numpy()

    for index in range(min(6, len(dataset))):
        image_clean = dataset[index]['depth']
        axes[index][0].imshow(image_clean.squeeze().cpu().numpy(), cmap='magma_r')

        image_noise = dataset[index]['noisy_depth']
        axes[index][1].imshow(image_noise.squeeze().cpu().numpy(), cmap='magma_r')
        
        # input_image = dataset_noise[index]['image'].unsqueeze(0)
        # model.to("cpu")
        # pred = model(input_image).squeeze(0).detach().cpu().numpy()

        image_denoise = image_noise - np.exp(0.5*log_inferred_sigma2x) * y_noise[index]
        # image_denoise = pred - y_noise[index]
        axes[index][2].imshow(image_denoise.squeeze(), cmap='magma_r')

        image_denoise = image_noise + np.exp(0.5*log_inferred_sigma2x) * y_noise[index]
        # image_denoise = pred + y_noise[index]
        axes[index][3].imshow(image_denoise.squeeze(), cmap='magma_r')
    plt.savefig("noise.png")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
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
    parser.add_argument('--noise_amount', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='AdaBins')
                        
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
    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)


    args = parser.parse_args()
    args.num_threads = args.workers
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    # Define dataset
    train_loader = DepthDataLoader(args, 'train').data
    dataset = train_loader.dataset

    # add noise
    noise_amount = args.noise_amount
    for i in range(len(dataset)):
        dataset[i]['noisy_depth'] = dataset[i]['depth'] + torch.randn_like(dataset[i]['depth']) * noise_amount

    # Define model 
    model = InferenceModel(args.model_name)
    model.to(device)

    # Define uncertainty model
    uncertainty_model = uncertainty_estimator(dataset).to(device)

    # Train
    train_uncertainty(args, model, uncertainty_model, train_loader, args.epochs)
    torch.save(uncertainty_model.state_dict(), "./pretrained/uncertainty_{}.pt".format(args.model_name))

    # Visualize
    uncertainty_model.load_state_dict(torch.load("./pretrained/uncertainty_{}.pt".format(args.model_name)))
    draw_image_with_noise(dataset, model, uncertainty_model)


if __name__ == '__main__':
    main()

***********************************************************************************

import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss, GaussianLogLikelihoodLoss, MSELoss, L1Loss
from utils import RunningAverage, colorize

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "MDE-AdaBins"
logging = True


def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    if args.loss == 'GaussianNLL':
        model = models.UnetGaussian.build(min_val=args.min_depth, max_val=args.max_depth)
    elif args.loss == 'MSE' or args.loss == 'L1':
        model = models.UnetGaussian.build(min_val=args.min_depth, max_val=args.max_depth, gaussian=False)
    else:
        model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                              norm=args.norm)

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    loss_str = args.loss
    if args.beta > 0:
        loss_str += f'_{args.beta}'
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{loss_str}-bs{args.bs}-ep{epochs}-lr{lr}-wd{args.wd}-{str(uuid.uuid4())[:8]}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    if args.loss == 'GaussianNLL':
        criterion_ueff = GaussianLogLikelihoodLoss(beta=args.beta)
    elif args.loss == 'MSE':
        criterion_ueff = MSELoss()
    elif args.loss == 'L1':
        criterion_ueff = L1Loss()
    else:
        criterion_ueff = SILogLoss()
        criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            bin_edges, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True, variance=bin_edges)
            loss = l_dense

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
                loss = l_dense + args.w_chamfer * l_chamfer

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)

            step += 1
            scheduler.step()

            ########################################################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                             root=os.path.join(root, "checkpoints"))

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
                                             root=os.path.join(root, "checkpoints"))
                    best_loss = metrics['abs_rel']
                model.train()
                #################################################################################################

    # wandb.last_step()

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True, variance=bins)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--dry', action='store_true', help='Dry run')
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--loss', default='GaussianNLL', help='loss function')
    parser.add_argument('--beta', default=0.0, type=float, help='beta param')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.0, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

    if sys.argv.__len__() >= 2 and os.path.isfile(sys.argv[1]):
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)