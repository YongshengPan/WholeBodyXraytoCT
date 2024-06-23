import os
import random
from typing import List, Dict

import torch
import torchvision.transforms.functional
from torch import nn
from torch.nn import functional as F
from itertools import product
from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import time
import matplotlib.pyplot as plt

def dice_coeff(pred, target, classes=None):
    smooth = 1.
    coeff = []
    all_classes = range(pred.size(1)) if classes is None else classes
    for idx in range(len(all_classes)):
        m1, m2 = pred[:, idx] > 0.5, (target == all_classes[idx]) * 1.0
        intersection = (m1 * m2).sum()
        union = m1.sum() + m2.sum()
        coeff.append((2. * intersection + smooth) / (union + smooth))
    return torch.tensor(coeff)


def dice_loss(pred, target):
    smooth = 1.
    coeff = 0
    for idx in range(0, pred.size(1)):
        m1, m2 = pred[:, idx], (target[:, 0] == idx)
        intersection = (m1 * m2).sum()
        coeff += (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return 1-coeff/pred.size(1)


def thd_dice_loss(y_pred, y_true, thres=None):
        smooth = 1.
        coeff = 0
        if thres is None:
            thres = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scale = 10 * len(thres)
        for idx in range(len(thres) - 1):
            region = torch.sigmoid(scale*(y_true - thres[idx])) - torch.sigmoid(scale*(y_true - thres[idx + 1]))
            predict = torch.sigmoid(scale*(y_pred - thres[idx])) - torch.sigmoid(scale*(y_pred - thres[idx + 1]))
            intersection = (region * predict).sum()
            union = region.sum() + predict.sum()
            coeff += (2. * intersection + smooth) / (union + smooth)
        return 1-coeff/(len(thres)-1)


def thd_dice(y_pred, y_true, thres=None):
    smooth = 1.
    coeff = []
    if thres is None:
        thres = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scale = 10 * len(thres)
    for idx in range(len(thres) - 1):
        region = np.where(np.logical_and(y_true > thres[idx], y_true <= thres[idx+1]), 1, 0)
        predict = np.where(np.logical_and(y_pred > thres[idx], y_pred <= thres[idx + 1]), 1, 0)
        # region = torch.sigmoid(scale * (y_true - thres[idx])) - torch.sigmoid(scale * (y_true - thres[idx + 1]))
        # predict = torch.sigmoid(scale * (y_pred - thres[idx])) - torch.sigmoid(scale * (y_pred - thres[idx + 1]))
        intersection = (region * predict).sum()
        union = region.sum() + predict.sum()
        coeff.append((2. * intersection + smooth) / (union + smooth))
    return torch.tensor(coeff)


class Trusteeship:
    def __init__(self, module: nn.Module, loss_fn, volin=('NAC',), volout=('CT',), metrics=('mae', ), advmodule: nn.Module=None, device="cpu", ckpt_prefix='', volume_names=None):
        self.volin, self.volout = volin, volout
        self.module = module
        self.advmodule = advmodule
        self.device = device
        self.ckpt_prefix = ckpt_prefix
        self.to_device(device)
        self.loss_fun = loss_fn
        self.metrics = metrics
        self.volume_names = volume_names
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)

    def to_device(self, device):
        self.module.to(device)
        if self.advmodule is not None:
            self.advmodule.to(device)

    def loss_function(self, pred, target, add_con=None, classes=None):
        total_loss = torch.mean(pred[1])*0
        if total_loss.isnan():
            print(total_loss)
        if 'dice' in self.loss_fun:
            total_loss += self._dice_loss_(pred[1], target, classes)
        if 'crep' in self.loss_fun:
            # total_loss += nn.functional.binary_cross_entropy_with_logits(pred[0], target)
            total_loss += self._cross_entropy_(pred[0], target, classes)
        if 'mse' in self.loss_fun:
            total_loss += nn.functional.mse_loss(pred[1], target)
        if 'mae' in self.loss_fun:
            total_loss += nn.functional.l1_loss(pred[1], target)
        if 'thd' in self.loss_fun:
            total_loss += thd_dice_loss(pred[1], target, thres=(-1000, 0.1, 0.85, 0.99, 1.15, 1000))
        if 'pdc' in self.loss_fun:
            pr, tr = torch.relu(pred[1]), torch.relu(target)
            coeff = (torch.minimum(pr, tr)) / (torch.maximum(pr, tr) + 1.0e-6)
            total_loss += 1-coeff.mean()
        return total_loss


    def metrics_function(self, pred, target, add_con=None, classes=None):
        metrics = {}
        if 'dice' in self.metrics:
            metrics['dice'] = dice_coeff(pred[1], target, classes)
        if 'crep' in self.metrics:
            # total_loss += nn.functional.binary_cross_entropy_with_logits(pred[0], target)
            metrics['crep'] += self._cross_entropy_(pred[0], target, classes)
        if 'rmse' in self.metrics:
            metrics['mse'] = torch.sqrt(nn.functional.mse_loss(pred[1], target))
        if 'mae' in self.metrics:
            metrics['mae'] = nn.functional.l1_loss(pred[1], target)
        if 'thd' in self.metrics:
            metrics['thd'] = thd_dice(pred[1], target,  thres=(-1000, 0.1, 0.85, 0.99, 1.15, 1000))
        # if 'psnr' in self.metrics:
            # metrics['psnr'] = torch.psnr(pred[1], target)
        # if 'ssim' in self.metrics:
        #     metrics['ssim'] = self._thres_dice_(pred[1], target, thres=None)
        return metrics


    def train_step(self, datadict):
        modalities = {modal: datadict[modal].to(self.device) for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)

        pred = self.module(inputs,)  # , coord.unique()
        loss = self.loss_function(pred, outputs, )

        if 'adv' in self.loss_fun or 'msl' in self.loss_fun:
            adv_grth, adv_pred = self.advmodule(outputs), self.advmodule(pred[1].detach())
            adv_loss = torch.mean(torch.abs(adv_pred[0] - 1) + torch.abs(adv_grth[0]))

            self.advmodule.optimizer.zero_grad()
            adv_loss.backward()
            self.advmodule.optimizer.step()

            adv_pred = self.advmodule(pred[1])
            fmloss = sum([torch.mean(torch.abs(adv_pred[1][ly] - adv_grth[1][ly].detach())) for ly in range(len(adv_pred[1]) - 1)])

        else:
            fmloss = 0
        loss = loss + fmloss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def train_step_diffusion(self, datadict, sqrt_alpha_cp):
        t = random.randint(0, len(sqrt_alpha_cp[0])-1)
        modalities = {modal: datadict[modal].to(self.device) for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)

        outputs_t = outputs*sqrt_alpha_cp[0][t]+(torch.randn(outputs.size(), device=outputs.device)+1) * sqrt_alpha_cp[1][t]
        pred = self.module(torch.concat((inputs, outputs_t), dim=1),)  # , coord.unique()
        loss = nn.functional.mse_loss(outputs*sqrt_alpha_cp[0][t]+(pred[0]+1) * sqrt_alpha_cp[1][t], outputs_t) / (sqrt_alpha_cp[1][t] * sqrt_alpha_cp[1][t])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __save_memory_infer(self, inputs, split_size=None, stride=(96, 32, 32), outdevice=None):
        predicts = None
        if outdevice is None:
            outdevice = inputs.device
        size = inputs.size()
        weight = 1
        for dm in range(0, len(split_size)):
            weight = weight * torch.exp(-(torch.arange(0, split_size[dm])/split_size[dm]-0.5)**2).view([1]*dm+[-1]+[1]*(len(split_size)-dm-1))
        weight = weight.to(outdevice)
        iterXYZ = list(product(range(0, size[2]-split_size[0]+stride[0], stride[0]), range(0, size[3]-split_size[1]+stride[1], stride[1]), range(0, size[4]-split_size[2]+stride[2], stride[2])))
        for iXYZ in iterXYZ:
            iXYZ = np.minimum(iXYZ, [size[2] - split_size[0], size[3] - split_size[1], size[4] - split_size[2]])
            pred = self.module(inputs[:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]].to(self.device),)
            if predicts is None:
                predicts = torch.zeros(pred[0].size()[0:2] + inputs.size()[2::], device=outdevice), torch.zeros(pred[1].size()[0:2] + inputs.size()[2::], device=outdevice)
                weights = torch.zeros((1, 1) + inputs.size()[2::], device=outdevice)+1.0e-6
            predicts[0][:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += pred[0].to(outdevice)
            predicts[1][:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += pred[1].to(outdevice)
            weights[:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += 1
        predicts = predicts[0] / weights, predicts[1] / weights
        return predicts

    def eval_step(self, datadict, split_size=None, stride=(96, 32, 32), outdevice='cpu'):
        modalities = {modal: datadict[modal] for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)
        outdevice = outputs.device
        if split_size is not None:
            predicts = self.__save_memory_infer(inputs, split_size=split_size, stride=stride, outdevice=outdevice)
        else:
            predicts = self.module(inputs.to(self.device), )
        loss = self.loss_function(predicts, outputs)
        self.metrics = self.metrics_function(predicts, outputs)
        return loss, predicts

    def infer_step(self, datadict, split_size=None, stride=(96, 32, 32),):
        inputs = torch.concat([datadict[modal] for modal in datadict if modal in self.volin], axis=1)
        outdevice = inputs.device
        if split_size is not None:
            predicts = self.__save_memory_infer(inputs, split_size=split_size, stride=stride,)
        else:
            predicts = self.module(inputs.to(self.device),)
            predicts = [item.to(outdevice) for item in predicts]
        return predicts

    def _dice_loss_(self, pred, target, classes=None):
        smooth = 1.
        coeff = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target[:, 0] == all_classes[idx])*1.0
            intersection = (m1 * m2).sum()
            union = m1.sum() + m2.sum()
            coeff += (2. * intersection + smooth) / (union + smooth)
        return 1-coeff/len(all_classes)

    def _cross_entropy_(self, pred, target, classes=None):
        smooth = 1.
        crossentropy = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target[:, 0] == all_classes[idx])*1.0
            crossentropy += F.binary_cross_entropy_with_logits(m1, m2)
        coeff = crossentropy / len(all_classes)
        return coeff

    def load_dict(self, dict_name, strict=True):
        state_dict = torch.load(os.path.join('models', '_'.join(self.loss_fun), self.ckpt_prefix, self.ckpt_prefix+'_'+dict_name))
        # pop_item = [xx for xx in state_dict.keys() if 'begin_conv' in xx or 'class_conv' in xx]
        # pop_item = [xx for xx in state_dict.keys() if 'begin_conv' in xx and 'weight' in xx and 'norm' not in xx and state_dict[xx].size(1) == 2]
        # for xx in pop_item:
        #     state_dict.pop(xx)
        self.module.load_state_dict(state_dict, strict=strict)

    def save_dict(self, dict_name):
        if not os.path.exists(os.path.join('models', '_'.join(self.loss_fun), self.ckpt_prefix)): os.makedirs(os.path.join('models', '_'.join(self.loss_fun), self.ckpt_prefix))
        torch.save(self.module.state_dict(), os.path.join('models', '_'.join(self.loss_fun), self.ckpt_prefix, self.ckpt_prefix+'_'+dict_name))

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()


class MultipleModalityTrusteeship:
    def __init__(self, modules: nn.ModuleDict, loss_fn, advmodules:nn.ModuleDict={}, device="cpu", prefix_filename=None, volume_names=None):
        if volume_names is None:
            volume_names = {'input': 'orig', 'output': 'mask'}
        self.modules = nn.ModuleDict(modules)
        self.device = device
        self.advmodules = advmodules
        self.toDevice(device)
        self.loss_fun = loss_fn
        self.volume_names = volume_names
        self.prefix_filename = prefix_filename
        # for modal in self.modules:
        #     self.optimizer[modal] = torch.optim.Adam(self.modules[modal].parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.modules.parameters(), lr=1e-3)

    def toDevice(self, device):
        for module in self.modules:
            self.modules[module].to(device)
        for module in self.advmodules:
            self.advmodules[module].to(device)

    def loss_function(self, pred, target, loss_fun=('mse',), region_mask=None, classes=None):

        total_loss = torch.mean(pred[1])*0
        region_mask = 1 if region_mask == None else region_mask + 0.1
        if total_loss.isnan():
            print(total_loss)
        if 'dice' in loss_fun:
            total_loss += self._dice_loss_(pred[1], target, classes=classes)
        if 'crep' in loss_fun:
            # total_loss += nn.functional.binary_cross_entropy_with_logits(pred[0], target)
            total_loss += self._cross_entropy_(pred[0], target, classes)
        if 'mse' in loss_fun:
            total_loss += torch.sqrt(torch.mean(torch.square(pred[1] - target)*region_mask))
        if 'mae' in loss_fun:
            total_loss += torch.mean(torch.abs(pred[1] - target)*region_mask)
        if 'thd' in loss_fun:
            total_loss += self._thres_dice_(pred[1], target, thres=None)
        if 'pdc' in loss_fun:
            pr, tr = torch.relu(pred[1]), torch.relu(target)
            coeff = (torch.minimum(pr, tr)+1.0e-6) / (torch.maximum(pr, tr) + 1.0e-6)
            total_loss += 1-coeff.mean()
        return total_loss

    def loss_function_extra(self, pred, target, loss_fun=('mse',)):
        for ly in range(len(pred)):
            return torch.mean(torch.abs(pred[ly]-target[ly]))

    def grid_sample(self, data, theta, align_corners=False):
        grid = F.affine_grid(theta, data.size(), align_corners=align_corners)
        return F.grid_sample(data, grid, align_corners=align_corners)

    def train_step(self, inputs, regionmask=None):
        modalities = {modal: inputs[modal][0].to(self.device) for modal in inputs}
        print(sum([inputs[it][1] for it in inputs]))
        # affine_theta = {modal: self.modules[modal].affine_theta(modalities[modal]) for modal in inputs if modal in self.modules}
        theta = torch.eye(3, 4, dtype=torch.float32, device='cuda').reshape([1, 3, 4])
        affine_theta = {modal: [theta, theta] for modal in inputs if modal in self.modules}
        # avg_latent = torch.mean(torch.stack(tuple(modal_latents.values()), dim=0), dim=0)
        if sum([inputs[it][1] for it in inputs]) == 0:
            modal_latents = {modal: self.modules[modal](self.grid_sample(modalities[modal], affine_theta[modal][0]), 'backward') for modal in inputs if modal in self.modules}
        else:
            modal_latents = {modal: self.modules[modal](self.grid_sample(modalities[modal], affine_theta[modal][0]), 'backward') for modal in inputs if inputs[modal][1]==1 and modal in self.modules}

        if not isinstance(list(modal_latents.values())[0], list):
            avg_latent = torch.mean(torch.stack(tuple(modal_latents.values()), dim=0), dim=0)
        else:
            avg_latent = [torch.mean(torch.stack(tup, dim=0), dim=0) for tup in list(zip(*modal_latents.values()))]
        # for iter in range(random.randrange(0, 1)):
        #     trans_latent = avg_latent if 'Trans' not in self.modules else self.modules['Trans'](avg_latent, 'translation')
        #     for modal in inputs:
        #         if modal in self.modules and inputs[modal][2] == 1:
        #             modal_latents[modal] = self.modules[modal](self.modules[modal](trans_latent, 'forward')[1].detach(), 'backward')
        #     avg_latent = [torch.mean(torch.stack(tup, dim=0), dim=0) for tup in list(zip(*modal_latents.values()))]

        # plt.imshow(np.transpose(torch.cat([ml[0, :, 32, :, :] for ml in modal_latents + [avg_latent]], dim=2).detach().cpu().numpy(), axes=(1, 2, 0)))

        if 'Trans' in self.modules:
            trans_latent = self.modules['Trans'](avg_latent, 'translation')
        else:
            trans_latent = avg_latent
        loss_essamble = {}
        loss_all = 0
        for modal in inputs:
            if not modal in self.modules: continue
            if inputs[modal][2] == 0: continue
            pred = self.modules[modal](trans_latent, 'forward')
            grid = F.affine_grid(affine_theta[modal][1], modalities[modal].size(), align_corners=False)
            pred[0] = F.grid_sample(pred[0], grid, align_corners=False)
            pred[1] = F.grid_sample(pred[1], grid, align_corners=False)

            loss = self.loss_function(pred, modalities[modal], self.loss_fun[modal], region_mask=regionmask)
            if 'adv' in self.loss_fun[modal]:
                adv_grth = self.advmodules[modal](modalities[modal])
                adv_pred = self.advmodules[modal](pred[1].detach())
                adv_loss = torch.mean(torch.abs(adv_pred[0] - 1) + torch.abs(adv_grth[0]))
                self.advmodules[modal].optimizer.zero_grad()
                adv_loss.backward()
                self.advmodules[modal].optimizer.step()

                adv_pred = self.advmodules[modal](pred[1])
                fmloss = sum([torch.mean(torch.abs(adv_pred[1][ly] - adv_grth[1][ly].detach())) for ly in range(len(adv_pred[1])-1)]) #+[torch.mean(torch.abs(adv_pred[1][-1]))]
            else:
                fmloss = 0
            loss_essamble[modal] = loss + fmloss
            loss_all = loss_all + fmloss + loss
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        return loss_essamble


    def train_step1(self, inputs, regionmask=None):
        modalities = {modal: inputs[modal][0].to(self.device) for modal in inputs}
        print(sum([inputs[it][1] for it in inputs]))
        # affine_theta = {modal: self.modules[modal].affine_theta(modalities[modal]) for modal in inputs if modal in self.modules}
        theta = torch.eye(3, 4, dtype=torch.float32, device='cuda').reshape([1, 3, 4])
        affine_theta = {modal: [theta, theta] for modal in inputs if modal in self.modules}
        # avg_latent = torch.mean(torch.stack(tuple(modal_latents.values()), dim=0), dim=0)
        avg_latent, num_of_modin = None, 0
        for modal in inputs:
            if (inputs[modal][1] == 1 or sum([inputs[it][1] for it in inputs]) == 0) and modal in self.modules:
                modal_latents = self.modules[modal](modalities[modal], 'backward')
                if avg_latent is None:
                    avg_latent = modal_latents
                elif not isinstance(modal_latents, list):
                    avg_latent = avg_latent + modal_latents
                else:
                    avg_latent = [avg_latent[idx] + modal_latents[idx] for idx in range(len(modal_latents))]
                num_of_modin += 1
        if not isinstance(avg_latent, list):
            avg_latent = avg_latent / num_of_modin
        else:
            avg_latent = [avg_latent[idx] / num_of_modin for idx in range(len(avg_latent))]

        if 'Trans' in self.modules:
            avg_latent = self.modules['Trans'](avg_latent, 'translation')

        loss_essamble = {}
        loss_all = 0
        for modal in inputs:
            if not modal in self.modules: continue
            if inputs[modal][2] == 0: continue
            pred = self.modules[modal](avg_latent, 'forward')
            grid = F.affine_grid(affine_theta[modal][1], modalities[modal].size(), align_corners=False)
            pred[0] = F.grid_sample(pred[0], grid, align_corners=False)
            pred[1] = F.grid_sample(pred[1], grid, align_corners=False)

            loss = self.loss_function(pred, modalities[modal], self.loss_fun[modal], region_mask=regionmask)
            if 'adv' in self.loss_fun[modal]:
                adv_grth = self.advmodules[modal](modalities[modal])
                adv_pred = self.advmodules[modal](pred[1].detach())
                adv_loss = torch.mean(torch.abs(adv_pred[0] - 1) + torch.abs(adv_grth[0]))
                self.advmodules[modal].optimizer.zero_grad()
                adv_loss.backward()
                self.advmodules[modal].optimizer.step()

                adv_pred = self.advmodules[modal](pred[1])
                fmloss = sum([torch.mean(torch.abs(adv_pred[1][ly] - adv_grth[1][ly].detach())) for ly in range(len(adv_pred[1])-1)]) #+[torch.mean(torch.abs(adv_pred[1][-1]))]
            else:
                fmloss = 0
            loss_essamble[modal] = loss + fmloss
            loss_all = loss_all + fmloss + loss
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        return loss_essamble


    def eval_step(self, inputs):
        modalities = {modal: inputs[modal][0].to(self.device) for modal in inputs}
        # print(sum([inputs[it][1] for it in inputs]))
        theta = torch.eye(3, 4, dtype=torch.float32, device='cuda').reshape([1, 3, 4])
        affine_theta = {modal: [theta, theta] for modal in inputs if modal in self.modules}
        # affine_theta = {modal: self.modules[modal].affine_theta(modalities[modal]) for modal in inputs if modal in self.modules}

        if sum([inputs[it][1] for it in inputs]) == 0:
            modal_latents = {modal: self.modules[modal](self.grid_sample(modalities[modal], affine_theta[modal][0]), 'backward') for modal in inputs if modal in self.modules}
        else:
            modal_latents = {modal: self.modules[modal](self.grid_sample(modalities[modal], affine_theta[modal][0]), 'backward') for modal in inputs if inputs[modal][1] == 1 and modal in self.modules}

        # avg_latent = torch.mean(torch.stack(tuple(modal_latents.values()), dim=0), dim=0)
        if not isinstance(list(modal_latents.values())[0], list):
            avg_latent = torch.mean(torch.stack(tuple(modal_latents.values()), dim=0), dim=0)
        else:
            avg_latent = [torch.mean(torch.stack(tup, dim=0), dim=0) for tup in list(zip(*modal_latents.values()))]
        for iter in range(0):
            trans_latent = avg_latent if 'Trans' not in self.modules else self.modules['Trans'](avg_latent, 'translation')
            for modal in inputs:
                if modal in self.modules and inputs[modal][2] == 1:
                    modal_latents[modal] = self.modules[modal](self.modules[modal](trans_latent, 'forward')[1], 'backward')
            avg_latent = [torch.mean(torch.stack(tup, dim=0), dim=0) for tup in list(zip(*modal_latents.values()))]
        # latent_image = np.transpose(torch.cat([ml[0, :, :, 48, :] for ml in modal_latents[::-1] + [avg_latent]], dim=2).detach().cpu().numpy(), axes=(1, 2, 0))
        # plt.imsave('latent_image.png', latent_image)
        if 'Trans' in self.modules:
            trans_latent = self.modules['Trans'](avg_latent, 'translation')
        else:
            trans_latent = avg_latent
        pred = {modal: self.modules[modal](trans_latent, 'forward') for modal in inputs if modal in self.modules}
        for modal in pred:
            grid = F.affine_grid(affine_theta[modal][1], modalities[modal].size(), align_corners=False)
            pred[modal][0] = F.grid_sample(pred[modal][0], grid, align_corners=False)
            pred[modal][1] = F.grid_sample(pred[modal][1], grid, align_corners=False)

        loss = {modal: self.loss_function(pred[modal], modalities[modal], self.loss_fun[modal]) for modal in inputs if modal in self.loss_fun}
        return loss, pred

    def infer_step(self, inputs):
        modalities = {modal: inputs[modal][0].to(self.device) for modal in inputs}
        # affine_theta = {modal: self.modules[modal].affine_theta(modalities[modal]) for modal in inputs if modal in self.modules}
        theta = torch.eye(3, 4, dtype=torch.float32, device='cuda').reshape([1, 3, 4])
        affine_theta = {modal: [theta, theta] for modal in inputs if modal in self.modules}

        # modal_latents = {modal: self.modules[modal](self.grid_sample(modalities[modal], affine_theta[modal][0]), 'backward') for modal in inputs if inputs[modal][1] == 1 and modal in self.modules}
        avg_latent, num_of_modin = None, 0
        for modal in inputs:
            if inputs[modal][1] == 1 and modal in self.modules:
                modal_latents = self.modules[modal](modalities[modal], 'backward')
                if avg_latent is None:
                    avg_latent = modal_latents
                elif not isinstance(modal_latents, list):
                    avg_latent = avg_latent + modal_latents
                else:
                    avg_latent = [avg_latent[idx]+modal_latents[idx] for idx in range(len(modal_latents))]
                num_of_modin += 1
        if not isinstance(avg_latent, list):
            avg_latent = avg_latent/num_of_modin
        else:
            avg_latent = [avg_latent[idx]/num_of_modin for idx in range(len(avg_latent))]
        if 'Trans' in self.modules:
            avg_latent = self.modules['Trans'](avg_latent, 'translation')

        pred = {modal: self.modules[modal](avg_latent, 'forward') for modal in self.modules}
        # ref = [modal for modal in inputs if inputs[modal][1] == 1][0]
        # grid = F.affine_grid(affine_theta[ref][1], modalities[ref].size(), align_corners=False)
        # for modal in pred:
        #     pred[modal][0] = F.grid_sample(pred[modal][0], grid, align_corners=False)
        #     pred[modal][1] = F.grid_sample(pred[modal][1], grid, align_corners=False)
        return pred

    def _dice_loss_(self, pred, target, classes=None):
        smooth = 1.
        coeff = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target == all_classes[idx])*1.0
            intersection = (m1 * m2).sum()
            union = m1.sum() + m2.sum()
            coeff += (2. * intersection + smooth) / (union + smooth)
        return 1-coeff/len(all_classes)

    def _cross_entropy_(self, pred, target, classes=None):
        smooth = 1.
        crossentropy = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target[:, 0] == all_classes[idx])*1.0
            crossentropy += F.binary_cross_entropy_with_logits(m1, m2)
        coeff = crossentropy / len(all_classes)
        return coeff

    def _thres_dice_(self, y_pred, y_true, thres=None):
        smooth = 1.
        coeff = 0
        if thres is None:
            thres=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scale = 10*len(thres)
        for idx in range(len(thres) - 1):
            region = torch.sigmoid(scale*(y_true - thres[idx])) - torch.sigmoid(scale*(y_true - thres[idx + 1]))
            predict = torch.sigmoid(scale*(y_pred - thres[idx])) - torch.sigmoid(scale*(y_pred - thres[idx + 1]))
            intersection = (region * predict).sum()
            union = region.sum() + predict.sum()
            coeff += (2. * intersection + smooth) / (union + smooth)
        return 1-coeff/len(thres)

    def load_dict(self, modal, epoch, prefix='chkpt', strict=True):
        dict_name = prefix + '_' + '{0}_{1}.h5'.format(epoch, modal)
        state_dict = torch.load(dict_name)

        # for item in state_dict:
        #     if state_dict[item].size()[2:5] in [(1, 3, 3), (1, 5, 5)]:
        #         state_dict[item] = torch.concat([state_dict[item]] * 3, dim=2)

        # pop_item = [xx for xx in state_dict.keys() if 'affine_theta' in xx or 'inten_shift' in xx]
        # for xx in pop_item:
        #     state_dict.pop(xx)
        # state_dict.pop('affine_theta.localization.0.weight')
        # state_dict.pop('begin_conv.0.conv1.weight')
        # state_dict.pop('begin_conv.0.conv2.weight')
        self.modules[modal].load_state_dict(state_dict, strict=strict)
        if modal in self.advmodules:
            dict_name = prefix + '_' + '{0}_{1}_adv.h5'.format(epoch, modal)
            if os.path.exists(dict_name):
                state_dict = torch.load(dict_name)
                self.advmodules[modal].load_state_dict(state_dict)

    def save_dict(self, modal, epoch, prefix='chkpt'):
        dict_name = prefix + '_' + '{0}_{1}.h5'.format(epoch, modal)
        torch.save(self.modules[modal].state_dict(), dict_name)
        if modal in self.advmodules:
            dict_name = prefix + '_' + '{0}_{1}_adv.h5'.format(epoch, modal)
            torch.save(self.advmodules[modal].state_dict(), dict_name)

    def train(self):
        for modal in self.modules:
            self.modules[modal].train()

    def eval(self):
        for modal in self.modules:
            self.modules[modal].eval()



