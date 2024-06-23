import random
from itertools import combinations, product

import torch
# import torch_directml
import torchvision.transforms.functional
from torch import nn
import os
import glob
from typing import List
from dataset_xraytoct_openaccess_torch import DataBase
from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from core.models import Generic_UNetwork, ResEncoderDecoder, AdverserialNetwork
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import time
from core.simulator import Trusteeship
import SimpleITK as sitk
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(0)


# input_size = (256, 288, 16)
input_size=(128, 192, 192)
subsitelist = ['CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'HNSCC', 'NSCLC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA']

training_data = DataBase(input_path="E:/dataset/NACdata1",
                    side_len=(128, 224, 224),
                    center_shift=(0, 0, 0),
                    data_shape=input_size,
                    subset=['HNSCC', ],
                    cycload=True,
                    use_augment=True,
                    aug_model='random',
                    random_rotation=False,
                    random_views=((0, 90,), (0, 60, 120), (0, 45, 90, 135), (0, 36, 72, 108, 144,), (0, 30, 60, 90, 120, 150)),
                    aug_side=(256, 128, 128),
                    aug_stride = (1, 1, 1),
                    angle_of_views=(0, 90,),
                    randomcrop=(0, 1),
                    return_original=False,
                    randomflip=('sk', 'flr', 'fud', 'r90'))

test_data = DataBase(input_path="E:/dataset/NACdata1",
                    side_len=(128, 224, 224),
                    center_shift=(0, 0, 0),
                    data_shape=input_size,
                    subset=['NSCLC', 'TCGA-HNSC', 'TCGA-LUAD',],
                    cycload=False,
                    use_augment=False,
                    random_rotation=False,
                    aug_side=(0, 0, 0),
                    aug_stride = (1, 1, 1),
                    angle_of_views=(0, 90, ),
                    randomcrop=(0, 1),
                    return_original=False,
                    randomflip=('sk', 'flr', 'fud', 'r90'))

extra_data = DataBase(input_path="E:/dataset/NACdata1",
                    side_len=(128, 224, 224),
                    center_shift=(0, 0, 0),
                    data_shape=input_size,
                    subset=['CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA'],
                    cycload=True,
                    use_augment=False,
                    random_rotation=False,
                    aug_side=(0, 0, 0),
                    aug_stride = (1, 1, 1),
                    angle_of_views=(0, 90,),
                    randomcrop=(0, 1),
                    randomflip=('sk', 'flr', 'fud', 'r90'))



batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size,)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
extra_dataloader = DataLoader(extra_data, batch_size=batch_size)

# for essamble, subjname in test_dataloader:
#     for ida in essamble:
#         print(f"Shape of {ida} [N, C, H, W]: {essamble[ida].shape}")
#     break
# torch_directml.device_count()
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs are {torch.cuda.get_device_name(0), torch.cuda.device_count()}")
print(f"Using {device} device")
basedim = 16

SYN_XraytoCT = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl'), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoCT_mv',)

SYN_XraytoCT_thd = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoCT_thd_mv',)

SYN_XraytoSKT = Trusteeship(Generic_UNetwork(1,5, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='softmax'),
    loss_fn=('crep', 'dice'), volin=('Xrays', ), volout=('SKT',), metrics=('dice', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoSKT_mv',)

SYN_XraySKTtoCT = Trusteeship(Generic_UNetwork(6,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl'), volin=('Xrays', 'SKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT_mv',)

SYN_XraySKTtoCT_thd = Trusteeship(Generic_UNetwork(6,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'SKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT_thd_mv',)

SYN_XraysSKTtoCT = Trusteeship(Generic_UNetwork(6,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraysSKTtoCT_mv',)

SYN_XraysSKTtoCT_thd = Trusteeship(Generic_UNetwork(6,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraysSKTtoCT_thd_mv',)

SYN_XraysSKTtoCTe = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT_mv',)

SYN_XraysSKTtoCT_thde = Trusteeship(Generic_UNetwork(6,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT_thd_mv',)

Trans_XraysSKTtoCT_thd = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraysSKTtoCT_thd',)

Trans_XraytoSKT = Trusteeship(Generic_UNetwork(1,5, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='softmax'),
    loss_fn=('crep', 'dice'), volin=('Xrays', ), volout=('SKT',), metrics=('dice', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraytoSKT_mv',)

Trans_XraytoCT = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=False, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', ), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraytoCT_mv',)


def train(dataloader, modules, seg_module=None):
    size = len(dataloader.dataset)
    for trustship in modules: trustship.train()
    total_loss = []
    for batch, (essamble, subj) in enumerate(dataloader):
        datadict = {it: essamble[it].to(device) for it in essamble}
        if 'SKT' in datadict:
            datadict['SKT'] = torch.eye(5, device=device)[datadict['SKT'].to(torch.int)].transpose(dim0=1, dim1=-1).squeeze(-1)
        if torch.sum(datadict['CT']) == 0:
            print(subj, "zero value")
        if len(datadict) == 0: continue
        if seg_module is not None:
            # SKT_SEG = torch.argmax(seg_module.infer_step(datadict)[0], dim=1, keepdim=True)
            SKT_SEG = seg_module.infer_step(datadict)[0].detach()
            datadict['sSKT'] = SKT_SEG
        loss_ensamble = []
        for trustship in modules:
            loss = trustship.train_step(datadict)
            loss_ensamble.append(loss.item())
        if any([np.isnan(ls) for ls in loss_ensamble]) or any([lt > 0.8 for lt in loss_ensamble]):
            print(subj)
        total_loss.append(loss_ensamble)
        if batch % 2 == 0:
            current = batch * batch_size
            print(f"current loss: {','.join(['%7f' % lt for lt in loss_ensamble])}  [{current:>5d}/{size:>5d}]")
    print(f"total loss: {','.join(['%7f' % lt for lt in np.mean(total_loss, 0)])} ")
    pass


def test(dataloader, modules, seg_module=None, ):
    for trustship in modules:
        trustship.eval()
    essamble_metrics = {trustship.ckpt_prefix: [] for trustship in modules}
    essamble_dice = {trustship.ckpt_prefix: [] for trustship in modules}
    with (torch.no_grad()):
        for essamble, subj in dataloader:
            datadict = {it: essamble[it].to(device) for it in essamble}
            images_all = [datadict[lt][0, :, 8, :, :].detach().cpu().numpy() for lt in datadict]
            if seg_module is not None:
                # SKT_SEG = torch.argmax(seg_module.infer_step(datadict)[0], dim=1, keepdim=True)
                SKT_SEG = seg_module.infer_step(datadict)[0]
                datadict['sSKT'] = SKT_SEG

            print(subj['flnm'], )
            for trustship in modules:
                loss_ensamble, predictions = trustship.eval_step(datadict)
                test_loss = loss_ensamble.item()
                y_pred = np.maximum(predictions[1].cpu().numpy()[0,], 0)
                if trustship is seg_module:
                    y_pred = np.argmax(y_pred, axis=0, keepdims=True)
                images_all.append(y_pred[:, 8, :, :])
                y_true = np.maximum(torch.concat([datadict[modal] for modal in datadict if modal in trustship.volout], axis=1).cpu().numpy()[0,], 0)
                correct_mae = np.mean(np.abs(y_pred-y_true)/2.0)
                correct_psnr = psnr(y_pred-1, y_true-1, data_range=4.0)
                correct_rmse = np.sqrt(mse(y_pred, y_true)/2.0)
                correct_ssim = ssim(y_pred[0]-1, y_true[0]-1, data_range=4.0)
                correct_dice = trustship.metrics['thd'] if 'thd' in trustship.metrics else trustship.metrics['dice']
                print(trustship.ckpt_prefix, f"loss: {'%2.6f' % test_loss}", f"psnr: {'%2.6f' % correct_psnr}",
                    f"rmae: {'%2.6f' % correct_mae}", f"rmse: {'%2.6f' % correct_rmse}", f"ssim: {'%2.6f' % correct_ssim}",
                    f"dice: {','.join(['%2.6f' % lt for lt in correct_dice])}")
                essamble_metrics[trustship.ckpt_prefix].append([test_loss, correct_mae, correct_psnr, correct_rmse, correct_ssim])
                essamble_dice[trustship.ckpt_prefix].append(correct_dice.numpy())
            all_img = np.concatenate(images_all, axis=1)
            plt.imshow(np.transpose(all_img, axes=(1, 2, 0)), vmin=0, vmax=1.0,)
            pass

            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'FLAIR'+'.png'), all_img[0])
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'missone' + '.png'), all_img[0])

        for trustship in modules:
            average_loss, average_mae, average_psnr, average_rmse, average_ssim = np.mean(essamble_metrics[trustship.ckpt_prefix], 0)
            average_dice = np.mean(essamble_dice[trustship.ckpt_prefix], 0)
            print(trustship.ckpt_prefix, f"loss: {'%2.6f' % average_loss}", f"psnr: {'%2.6f' % average_psnr}",
                  f"rmae: {'%2.6f' % average_mae}", f"rmse: {'%2.6f' % average_rmse}", f"ssim: {'%2.6f' % average_ssim}",
                  f"dice: {','.join(['%2.6f' % lt for lt in average_dice])}")


def multiple_instensity_metrics(prediction, groundtruth, data_range=1.0):
    prediction, groundtruth = prediction / data_range, groundtruth / data_range
    diff_map = prediction - groundtruth
    MAE = np.mean(np.abs(diff_map))
    RMSE = np.sqrt(np.mean(np.square(diff_map)))
    SSIM = np.mean(ssim(groundtruth, prediction, data_range=data_range, full=False, channel_axis=-1))
    PSNR = 10 * np.log10((data_range ** 2) / RMSE) / 100
    NCC = np.mean(np.multiply(prediction - np.mean(prediction), groundtruth - np.mean(groundtruth)) / (np.std(prediction) * np.std(groundtruth))+1e-6)
    return MAE, RMSE, PSNR, NCC, SSIM

def test_whole_image(dataloader, modules, seg_module=None):
    dataloader.dataset.return_original = True
    for trustship in modules:
        trustship.eval()
    essamble_metrics = {trustship.ckpt_prefix: [] for trustship in modules}
    essamble_dice = {trustship.ckpt_prefix: [] for trustship in modules}
    with (torch.no_grad()):
        for essamble, subj in dataloader:
            datadict = {it: essamble[it] for it in essamble}
            print(subj['flnm'], torch.max(essamble['CT']), [torch.mean(datadict['CT'][datadict['SKT']==chl]) for chl in range(5)])

            if 'SKT' in datadict:
                datadict['SKT'] = torch.eye(5, device=datadict['SKT'].device)[datadict['SKT'].to(torch.int)].transpose(dim0=1, dim1=-1).squeeze(-1)
            if seg_module is not None:
                    # SKT_SEG = torch.argmax(seg_module.infer_step(datadict, split_size=input_size)[1], dim=1, keepdim=True)
                    SKT_SEG = seg_module.infer_step(datadict, split_size=input_size)[0]
                    datadict['sSKT'] = SKT_SEG
                # images_all = [datadict[lt][0, :, 8, :, :].detach().cpu().numpy() for lt in datadict]

            for trustship in modules:
                    loss_ensamble, predictions = trustship.eval_step(datadict, split_size=input_size)
                    test_loss = loss_ensamble.item()
                    y_pred = np.maximum(predictions[1].cpu().numpy()[0,], 0)
                    if trustship is SYN_XraytoSKT:
                        y_pred = np.argmax(y_pred, axis=0, keepdims=True)

                    y_true = torch.concat([datadict[modal] for modal in datadict if modal in trustship.volout], axis=1).cpu().numpy()[0,]
                    correct_mae = np.mean(np.abs(y_pred-y_true)/2.0)
                    correct_psnr = psnr(y_pred, y_true, data_range=4.0)
                    # correct_psnr = 10 * np.log10((data_range ** 2) / RMSE) / 100
                    correct_rmse = np.sqrt(mse(y_pred/2.0, y_true/2.0))
                    correct_ssim = ssim(y_pred[0], y_true[0], data_range=4.0)
                    correct_dice = trustship.metrics['thd'] if 'thd' in trustship.metrics else trustship.metrics['dice']
                    print(trustship.ckpt_prefix, f"loss: {'%2.6f' % test_loss}", f"psnr: {'%2.6f' % correct_psnr}",
                            f"rmae: {'%2.6f' % correct_mae}", f"rmse: {'%2.6f' % correct_rmse}", f"ssim: {'%2.6f' % correct_ssim}",
                            f"dice: {','.join(['%2.6f' % lt for lt in correct_dice])}")
                    essamble_metrics[trustship.ckpt_prefix].append([test_loss, correct_mae, correct_psnr, correct_rmse, correct_ssim])
                    essamble_dice[trustship.ckpt_prefix].append(correct_dice.numpy())
            pass

        for trustship in modules:
            average_loss, average_mae, average_psnr, average_rmse, average_ssim = np.mean(essamble_metrics[trustship.ckpt_prefix], 0)
            average_dice = np.mean(essamble_dice[trustship.ckpt_prefix], 0)
            print(trustship.ckpt_prefix, f"loss: {'%2.6f' % average_loss}", f"psnr: {'%2.6f' % average_psnr}",
                  f"rmae: {'%2.6f' % average_mae}", f"rmse: {'%2.6f' % average_rmse}", f"ssim: {'%2.6f' % average_ssim}",
                  f"dice: {','.join(['%2.6f' % lt for lt in average_dice])}")


def apply(dataloader, modules, seg_module=None):
    dataloader.dataset.return_original = True
    for trustship in modules:
        trustship.eval()

    with torch.no_grad():
        for essamble, subjinfo in dataloader:
            datadict = {it: essamble[it].to(device) for it in essamble}
            print(subjinfo['flnm'], )
            tarfile = os.path.join('outputs', subjinfo['flnm'][0][0])
            if not os.path.exists(tarfile): os.makedirs(tarfile)
            if seg_module is not None:
                SKT_SEG = torch.argmax(seg_module.infer_step(datadict, split_size=input_size)[1], dim=1, keepdim=True)
                datadict['sSKT'] = SKT_SEG
            images_all = [datadict[lt][0, :, 64, :, :].detach().cpu().numpy() for lt in datadict]
            ref_file = os.path.join(dataloader.dataset.input_path, subjinfo['flnm'][0][0])
            if os.path.exists(os.path.join(ref_file, 'RS1_NAC_2mm.nii.gz')):
                ref_img = sitk.ReadImage(os.path.join(ref_file, 'RS1_NAC_2mm.nii.gz'))
            else:
                ref_img = sitk.ReadImage(os.path.join(ref_file, 'RS2_NAC_2mm.nii.gz'))
            for trustship in modules:
                predictions = trustship.infer_step(datadict, split_size=input_size)
                if trustship is SYN_XraytoSKT:
                    y_pred = np.uint8(np.argmax(predictions[1].cpu().numpy()[0,], 0, keepdims=True))
                    modal_itk = sitk.GetImageFromArray(y_pred[0, ])
                    images_all.append(y_pred[:, 64, :, :]/2)
                else:
                    y_pred = np.maximum(predictions[1].cpu().numpy()[0,], 0)
                    modal_itk = sitk.GetImageFromArray(np.int16(1000 * y_pred[0, ]))
                    images_all.append(y_pred[:, 64, :, :])
                modal_itk.SetOrigin(ref_img.GetOrigin())
                modal_itk.SetSpacing(ref_img.GetSpacing())
                modal_itk.SetDirection(ref_img.GetDirection())

                sitk.WriteImage(modal_itk, os.path.join(tarfile, trustship.ckpt_prefix+'.nii.gz'), useCompression=True)
            all_img = np.concatenate(images_all, axis=1)
            plt.imshow(np.transpose(all_img, axes=(1, 2, 0)), vmin=0, vmax=2.0, )
            pass


epochs = 301
start_epochs = 0
# all_trustships = [SYN_XraytoCT, SYN_XraytoCT_thd, SYN_XraytoSKT, SYN_XraySKTtoCT, SYN_XraySKTtoCT_thd, SYN_XraysSKTtoCT, SYN_XraysSKTtoCT_thd] #
all_trustships = [SYN_XraySKTtoCT_thd, SYN_XraysSKTtoCT, SYN_XraysSKTtoCT_thd] #
if start_epochs > 0:
    for trustship in all_trustships:
        trustship.load_dict(dict_name='chkpt_%d.h5' % start_epochs, strict=False)
# SYN_XraytoSKT.load_dict(dict_name='chkpt_200.h5', strict=False)

for t in range(start_epochs, epochs):
    print(f"Epoch {t}\n-------------------------------")
    start_time = time.time()
    # train(train_dataloader, all_trustships, seg_module=SYN_XraytoSKT)
    # train(train_dataloader, {SYN_XraySKTtoCT_thd, SYN_XraysSKTtoCT, SYN_XraysSKTtoCT_thd,}, seg_module=SYN_XraytoSKT)
    # test(test_dataloader, {SYN_XraytoSKT,}, seg_module=None)
    # test(test_dataloader, [SYN_XraytoSKT, SYN_XraysSKTtoCT, SYN_XraysSKTtoCT_thd], seg_module=SYN_XraytoSKT)
    # test_whole_image(test_dataloader, all_trustships, seg_module=SYN_XraytoSKT, )
    test_whole_image(test_dataloader, {SYN_XraysSKTtoCT_thd}, seg_module=SYN_XraytoSKT, )

    # apply(test_dataloader, all_trustships, seg_module=SYN_XraytoSKT, )

    end_time = time.time()
    print(f"Epoch {t} takes {end_time-start_time} seconds")
    # break
    if t % 10 == 9:
        for trustship in all_trustships:
            trustship.save_dict(dict_name='chkpt_%d.h5' % (t+1))

print("Done!")


