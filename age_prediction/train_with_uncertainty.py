import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from age_estimation_pytorch.model import get_model, get_model_regress, get_model_regress_uncertainty, get_model_regress_withlastlayer
from age_estimation_pytorch.dataset import FaceDataset, FaceDataset_noise, FaceDataset_noise_full_vote
from age_estimation_pytorch.defaults import _C as cfg
import matplotlib.pyplot as plt
import math
import pickle

class fixed_output_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return 0*torch.mean(x,dim=1)
def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="./age_estimation_pytorch/appa-real-release", help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--uncertainty_load_prediction_resume", type=str, default="./age_estimation_pytorch/checkpoint/epoch025_0.38765_4.8373.pth", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--tensorboard", type=str, default="tf_log", help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument("--s3y_mode", type=str, default="heter")
    parser.add_argument("--s2y_mode", type=str, default="heter")
    args = parser.parse_args()
    return args


class bias_module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return self.bias+0*torch.mean(x,dim=1)
class divide_module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.bias = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return x/100

class normalize_module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.bias = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        h = (x-mean)/(std+1e-8)
        return h

class uncertainty_estimator(torch.nn.Module):
    def __init__(self, dataset, dim_feats) -> None:
        """
        The dataset here should be train_loader.dataset
        """
        super().__init__()
        # determine noise shape
        # y_shape = dataset.y.size()
        # y_shape = list(y_shape)
        # y_shape[0] = len(dataset)
        # y_shape = tuple(y_shape)
        # y_tensor = torch.clone(dataset.y)
        self.y_noise = torch.nn.Parameter(torch.randn(dataset.y.shape, dtype=torch.float32) * 0)
        # self.log_s3y = nn.Linear(dim_feats, 1)
        self.log_s3y = nn.Sequential(divide_module(),
                                    nn.Linear(1, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 1)
                                )
        # self.log_s3y = fixed_output_module()

        # self.log_s3y = bias_module()
        # self.log_s2y = nn.Linear(dim_feats, 1)
        self.log_s2y = nn.Sequential(
                                    divide_module(),
                                    nn.Linear(1, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 1)
                                )
        # self.log_s3y = torch.nn.Parameter(torch.log(torch.ones(1)*1)) # estimator of log of the noise variance

    def normalize(self, noise):
        with torch.no_grad():
            std = torch.std(noise)
            mean = torch.mean(noise)
            noise.add_(-mean,alpha=1)
            noise.div_(std)
    def normalize_by_section(self, noise, ranking_index, n_section = 10):
        num_cols = math.floor(noise.shape[0]/n_section)
        # padding_length = n_section * num_cols - noise.shape[0]
        after_clip_length = num_cols*n_section

        # indices = torch.arange(noise.shape[0])
        shift_amount = torch.randint(noise.shape[0], size=(1,)).item()
        shifted_indices =  torch.roll(ranking_index, shift_amount)
        inverse_indices = torch.argsort(shifted_indices)

        after_clip_shifted_indices = shifted_indices[0:after_clip_length]
        after_clip_inverse_indices = inverse_indices[0:after_clip_length]

        noise_ = noise.detach().clone()
        noise_clip = noise_[after_clip_shifted_indices]
        noise_clip_reshape = noise_clip.reshape(n_section,num_cols,1)

        # noise_ranked = noise_[ranking_index]
        # padded_vector = torch.cat([noise_ranked, torch.zeros(padding_length).to(noise.device)])
        # noise_ranked_reshaped = padded_vector.reshape(n_section,num_cols,1)

        with torch.no_grad():
            std = torch.std(noise_clip_reshape,dim=1,keepdim=True)
            mean = torch.mean(noise_clip_reshape,dim=1,keepdim=True)
            noise_clip_reshape.add_(-mean,alpha=1)
            noise_clip_reshape.div_((std+1e-8))
            noise__ = torch.reshape(noise_clip_reshape,noise_clip.shape)
            noise__ = torch.cat([noise__, noise_[shifted_indices][-(noise.shape[0]-after_clip_length):]])
            noise.mul_(0)
            noise.add_(noise__[inverse_indices],alpha=1)
        return
    def normalize_instance(self, noise):
        with torch.no_grad():
            std = torch.std(noise,dim=[-2,-1],keepdim=True)
            mean = torch.mean(noise,dim=[-2,-1],keepdim=True)
            noise.add_(-mean,alpha=1)
            noise.div_(std)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train_regress(train_loader, model, optimizer, epoch, device):
    criterion = nn.MSELoss().to(device)
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y, idx in _tqdm:
            x = x.to(device)
            idx = idx.to(device)
            y = y.to(device).float().unsqueeze(-1)

            # compute output
            outputs,_ = model(x)

            # calc loss
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate_regress(validate_loader, model, epoch, device):
    model.eval()
    criterion = nn.MSELoss().to(device)
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, idx) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device).float().unsqueeze(-1)

                # compute output
                outputs,_ = model(x)
                preds.append(outputs.cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    # ages = np.arange(0, 101)
    # ave_preds = (preds * ages).sum(axis=-1)
    diff = preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae




def train(train_loader, model, optimizer, epoch, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y, idx in _tqdm:
            x = x.to(device)
            idx = idx.to(device)
            y = y.to(device)

            # compute output
            outputs = model(x)

            # calc loss
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, epoch, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, idx) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

def main_regress(args):


    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model_regress_withlastlayer(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset_noise(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset_noise(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train_regress(train_loader, model,  optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate_regress(val_loader, model,  epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

    return model

def train_regress_uncertainty(train_loader, model, uncertainty_model, optimizer,  y_noise_optim, s3y_optim,s2y_optim, epoch, device):
    criterion = nn.MSELoss().to(device)
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    # uncertainty_model.normalize_by_section(uncertainty_model.y_noise, train_loader.dataset.index_rank_by_y)

    s3yroot_list = []
    denoise_list = []
    with tqdm(train_loader) as _tqdm:
        for x, y, idx in _tqdm:
            x = x.to(device)
            idx = idx.to(device)
            y = y.to(device).unsqueeze(-1)
            y_tilde = train_loader.dataset.y_tilde.to(device)[idx].unsqueeze(-1)
            y_noise = uncertainty_model.y_noise.to(device)[idx].unsqueeze(-1)
            predict_y, last_layer = model(x)
            s3yroot_s2yroot_input = last_layer
            s3yroot_s2yroot_input = predict_y
            s3yroot = torch.exp(0.5 * uncertainty_model.log_s3y(s3yroot_s2yroot_input))
            s3yroot_list.append(s3yroot)
            log_s2y = uncertainty_model.log_s2y(s3yroot_s2yroot_input)
            y_noise_hat =  s3yroot * y_noise
            y_tilde_hat = y_tilde - y_noise_hat

            denoise_list.append(y_tilde_hat-y)

            # compute output
            

            # calc loss
            if args.s3y_mode != "fixed":
                loss = 0.5*(torch.exp(-log_s2y)*((predict_y-y_tilde_hat)**2)+log_s2y).mean()
            else:
                loss = 0.5*(torch.exp(-log_s2y)*((predict_y-y_tilde)**2)+log_s2y).mean()
            cur_loss = loss.item()

            # calc accuracy
            # _, predicted = outputs.max(1)
            correct_num = 0
            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            # optimizer.zero_grad()
            y_noise_optim.zero_grad()
            s3y_optim.zero_grad()
            s2y_optim.zero_grad()
            loss.backward()
            # optimizer.step()
            y_noise_optim.step()
            s3y_optim.step()
            s2y_optim.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)
        # uncertainty_model.normalize(uncertainty_model.y_noise)
        uncertainty_model.normalize_by_section(uncertainty_model.y_noise, train_loader.dataset.index_rank_by_y, n_section = 100)
        print("s3y:{}".format(torch.cat(s3yroot_list).mean().item()))
        print("mean_y_noise:{}".format(torch.mean(uncertainty_model.y_noise).item()))
        print("std_y_noise:{}".format(torch.std(uncertainty_model.y_noise).item()))
        print("actual noise std:{}".format(torch.std(train_loader.dataset.y_tilde-train_loader.dataset.y).item()))
        print("denoised noise std:{}".format(torch.std(torch.cat(denoise_list)).item()))



    return loss_monitor.avg, accuracy_monitor.avg


def validate_regress_uncertainty(validate_loader, model, epoch, device):
    model.eval()
    criterion = nn.MSELoss().to(device)
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, idx) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device).float().unsqueeze(-1)

                # compute output
                outputs,_ = model(x)
                preds.append(outputs.cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)

    diff = preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def plot_prediction(model, uncertainty_model, dataloader, epoch, device, args):

    predict_y_list = []
    y_list = []
    ys_list = []
    s3yroot_list = []
    s2yroot_list = []
    real_age_list = []
    std_list = []
    y_noise_list = []
    i = 0

    for x, y, idx in dataloader:
        # i+=1
        # print(i)
        x = x.to(device)
        idx = idx.to(device)
        y = y.to(device).unsqueeze(-1)
        std = np.array(dataloader.dataset.std)[idx.cpu().numpy()]
        real_age = np.array(dataloader.dataset.real_age)[idx.cpu().numpy()]
        ys = [dataloader.dataset.ys[i] for i in list(idx.cpu().numpy())]
        y_tilde = dataloader.dataset.y_tilde.to(device)[idx].unsqueeze(-1)
        y_noise = uncertainty_model.y_noise.to(device)[idx].unsqueeze(-1)
        predict_y, last_layer = model(x)
        predict_y_list.append(predict_y.detach().cpu().numpy())

        s3yroot_s2yroot_input = predict_y
        s3yroot = torch.exp(0.5 * uncertainty_model.log_s3y(s3yroot_s2yroot_input))
        s2yroot = torch.exp(0.5 * uncertainty_model.log_s2y(s3yroot_s2yroot_input))
        s3yroot_list.append(s3yroot.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())
        std_list.append(std) 
        s2yroot_list.append(s2yroot.detach().cpu().numpy())

        y_noise_list.append(y_noise.detach().cpu().numpy())
        ys_list += ys
        real_age_list.append(real_age)

    y = np.concatenate(y_list)
    std = np.concatenate(std_list)
    s3yroot = np.concatenate(s3yroot_list)
    s2yroot = np.concatenate(s2yroot_list)
    y_noise = np.concatenate(y_noise_list)
    real_age = np.concatenate(real_age_list)

    dict_ = {"s2yroot":s2yroot, "s3yroot":s3yroot, "y":y, "std":std, "y_noise":y_noise,"real_age":real_age,"ys_list":ys_list}
    with open('save_dict/{}_{}_epoch{}.pickle'.format(args.s2y_mode,args.s3y_mode,epoch), 'wb') as file:
        pickle.dump(dict_, file, protocol=pickle.HIGHEST_PROTOCOL)


def main_uncertainty(args, model=None):

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> loading prediction model '{}'".format(cfg.MODEL.ARCH))
    

    # if cfg.TRAIN.OPT == "sgd":
    #     optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
    #                                 momentum=cfg.TRAIN.MOMENTUM,
    #                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    resume_path = args.uncertainty_load_prediction_resume

    if resume_path or model == None:
        model = get_model_regress_withlastlayer(model_name=cfg.MODEL.ARCH)
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    model = model.to(device)

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset_noise_full_vote(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=False,
                                age_stddev=cfg.TRAIN.AGE_STDDEV, age_fixed_stddev = 0.0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    uncertainty_model = uncertainty_estimator(train_dataset, model.dim_feats)
    uncertainty_model.to(device)

    

    if cfg.TRAIN.OPT == "sgd":
        y_noise_optim = torch.optim.SGD([{'params': uncertainty_model.y_noise}], lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        s3y_optim = torch.optim.SGD([{'params': uncertainty_model.log_s3y.parameters()}], lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)  
        s2y_optim = torch.optim.SGD([{'params': uncertainty_model.log_s2y.parameters()}], lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)  

    else:
        y_noise_optim = torch.optim.Adam([{'params': uncertainty_model.y_noise}], lr=0.001)
        s3y_optim = torch.optim.Adam([{'params': uncertainty_model.log_s3y.parameters()}], lr=0.001)  
        s2y_optim = torch.optim.Adam([{'params': uncertainty_model.log_s2y.parameters()}], lr=0.001)  

    val_dataset = FaceDataset_noise_full_vote(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    # scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
    #                    last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):


        

        # train
        train_loss, train_acc = train_regress_uncertainty(train_loader, model, uncertainty_model, None, y_noise_optim, s3y_optim, s2y_optim, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate_regress_uncertainty(val_loader, model,  epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else uncertainty_model.state_dict()
            # torch.save(
            #     {
            #         'epoch': epoch + 1,
            #         'arch': cfg.MODEL.ARCH,
            #         'state_dict': model_state_dict,
            #         # 'optimizer_state_dict': optimizer.state_dict()
            #     },
            #     str(checkpoint_dir.joinpath("uncertainty_epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            # )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # plot_prediction(model, uncertainty_model, train_loader, epoch, device)
        plot_prediction(model, uncertainty_model, train_loader, epoch, device, args)
        # adjust learning rate
        # scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # main_regress(args)
    main_uncertainty(args)
