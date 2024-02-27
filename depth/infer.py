import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import model_io
import utils
from models import UnetAdaptiveBins, UnetGaussian

import matplotlib.pyplot as plt
from time import time


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class InferenceHelper:
    def __init__(self, dataset='nyu', device='cuda:0', model_name='AdaBins'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            self.model_name = model_name
            if model_name == 'AdaBins':
                model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
                pretrained_path = "./pretrained/AdaBins_nyu.pt"
                # pretrained_path = "./pretrained/Gaussian_nyu.pt"
            elif model_name == "Gaussian":
                model = UnetGaussian.build(min_val=self.min_depth, max_val=self.max_depth, gaussian=False)
                pretrained_path = "./pretrained/Gaussian_nyu.pt"
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # # Flip
        # image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        # pred_lr = self.model(image)[-1]
        # pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # # Take average of original and mirror
        # final = 0.5 * (pred + pred_lr)
        final = pred
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        if self.model_name == "AdaBins":
            centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
            centers = centers.cpu().squeeze().numpy()
            centers = centers[centers > self.min_depth]
            centers = centers[centers < self.max_depth]
        elif self.model_name == "Gaussian":
            centers = bins
        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final).save(save_path)


if __name__ == '__main__':
    # # set seed
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # np.random.seed(0)

    # specify model to use
    model_name = "AdaBins"

    img = Image.open("test_imgs/classroom__rgb_00283.jpg")
    # img = Image.open("test_imgs/rgb_00307.jpg")
    start = time()
    inferHelper = InferenceHelper(model_name=model_name)
    centers, pred = inferHelper.predict_pil(img)

    print(f"took :{time() - start}s")
    print(pred.shape, pred.dtype, pred.min(), pred.max())
    plt.figure()
    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.colorbar()
    plt.savefig("test_imgs/processed_{}.png".format(model_name))

    # load gt
    gt = Image.open("test_imgs/sync_depth_00283.png")
    gt = np.asarray(gt, dtype='float32') / 1000.
    print(gt.shape, gt.dtype, gt.min(), gt.max())
    plt.figure()
    plt.imshow(gt, cmap='magma_r')
    plt.colorbar()
    plt.savefig("test_imgs/gt.png")