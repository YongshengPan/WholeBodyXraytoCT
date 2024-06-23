import random
from itertools import combinations

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
from core.models import Generic_UNetwork, AdverserialNetwork, AdverserialResidualNetwork
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from core.simulator import thd_dice_loss, dice_coeff
import matplotlib.pyplot as plt
import time
from core.simulator import Trusteeship
import SimpleITK as sitk
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(0)


# input_size = (256, 288, 16)
input_size=(128, 224, 224)
subsitelist = ['CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'HNSCC', 'NSCLC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA']

training_data = DataBase(input_path="E:/dataset/NACdata1",
                    side_len=(128, 224, 224),
                    center_shift=(0, 0, 0),
                    data_shape=input_size,
                    subset=['HNSCC', ],
                    cycload=False,
                    use_augment=True,
                    aug_model='random',
                    random_rotation=False,
                    random_views=[[0, c*15] for c in range(1, 12)],
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
                    subset=['NSCLC', 'TCGA-HNSC', 'TCGA-LUAD', ],
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


# extra_data = [DataBase(
#     root="E:/dataset/ASL_to_CBV/extra_validation",
#     datafolder=datafolder,
#     data_shape=input_size,
#     run_stage='extra',
#     use_augment=False,
#     return_original=True,
# ) for datafolder in ['extra_validation2', "Grade&PrognosisE", "Correlation_test"]]



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

SYN_XraytoCT = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', ), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoCT',)

SYN_XraytoCT_thd = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoCT_thd',)

SYN_XraytoSKT = Trusteeship(Generic_UNetwork(1,5, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='sigmoid'),
    loss_fn=('crep', 'dice'), volin=('Xrays', ), volout=('SKT',), metrics=('dice', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraytoSKT',)

SYN_XraySKTtoCT = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', ), volin=('Xrays', 'SKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT',)

SYN_XraySKTtoCT_thd = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=5, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'rSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraySKTtoCT_thd',)

SYN_XraysSKTtoCT = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', ), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraysSKTtoCT',)

SYN_XraysSKTtoCT_thd = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    # advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    advmodule=AdverserialResidualNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraysSKTtoCT_thd',)

Trans_XraySKTtoCT_thd = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'rSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraySKTtoCT_thd',)

Trans_XraytoCT = Trusteeship(Generic_UNetwork(1,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', ), volin=('Xrays',), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraytoCT',)

Trans_XraytoSKT = Trusteeship(Generic_UNetwork(1,5, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=True, use_triD=False, activation_function='softmax'),
    loss_fn=('crep', 'dice'), volin=('Xrays', ), volout=('SKT',), metrics=('dice', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraytoSKT',)

Trans_XraysSKTtoCT_thd = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, istransunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='Trans_XraysSKTtoCT_thd',)

SYN_XraysSKTtoCT_thd_ar = Trusteeship(Generic_UNetwork(2,1, basedim=basedim, downdepth=4, model_type='3D', isresunet=True, use_triD=False, activation_function='lrelu'),
    loss_fn=('mae', 'msl', 'thd'), volin=('Xrays', 'sSKT'), volout=('CT',), metrics=('thd', ),
    # advmodule=AdverserialNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    advmodule=AdverserialResidualNetwork(1, basedim=16, downdepth=3, model_type='3D', activation_function=None),
    device=device, ckpt_prefix='XraysSKTtoCT_thd_ar',)


def train(dataloader, modules, seg_module=None):
    size = len(dataloader.dataset)
    for trustship in modules: trustship.train()
    total_loss = []
    channel_weights = torch.Tensor([0.0120, 0.3267, 0.923, 1.0434, 1.3548]).to(modules[0].device)
    for batch, (essamble, subj) in enumerate(dataloader):
        datadict = {it: essamble[it].to(device) for it in essamble}
        if len(datadict) == 0: continue
        if 'SKT' in datadict:
            datadict['rSKT'] = channel_weights[datadict['SKT'].to(torch.int)]
        if seg_module is not None:
            SKT_SEG = torch.sum(channel_weights.view(1, -1, 1, 1, 1) * seg_module.infer_step(datadict, split_size=input_size)[1], dim=1, keepdim=True).detach()
            # SKT_SEG = channel_weights[torch.argmax(seg_module.infer_step(datadict)[1], dim=1, keepdim=True)]
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


def test(dataloader, modules, seg_module=None):
    for trustship in modules:
        trustship.eval()
    essamble_metrics = {trustship.ckpt_prefix: [] for trustship in modules}
    essamble_dice = {trustship.ckpt_prefix: [] for trustship in modules}
    with (torch.no_grad()):
        for essamble, subj in dataloader:
            datadict = {it: essamble[it].to(device) for it in essamble}
            if seg_module is not None:
                SKT_SEG = torch.argmax(seg_module.infer_step(datadict)[0], dim=1, keepdim=True)
                datadict['sSKT'] = SKT_SEG

            images_all = [datadict[lt][0, :, 8, :, :].detach().cpu().numpy() for lt in datadict]
            print(subj['flnm'], )
            for trustship in modules:
                loss_ensamble, predictions = trustship.eval_step(datadict)
                test_loss = loss_ensamble.item()
                y_pred = np.maximum(predictions[1].cpu().numpy()[0,], 0)
                if trustship in [SYN_XraytoSKT, Trans_XraytoSKT]:
                    y_pred = np.argmax(y_pred, axis=0, keepdims=True)
                images_all.append(y_pred[:, 8, :, :])
                y_true = torch.concat([datadict[modal] for modal in datadict if modal in trustship.volout], axis=1).cpu().numpy()[0,]
                correct_mae = np.mean(np.abs(y_pred-y_true))
                correct_psnr = psnr(y_pred, y_true, data_range=4.0)
                correct_rmse = np.sqrt(mse(y_pred, y_true))
                correct_ssim = ssim(y_pred[0], y_true[0], data_range=4.0)
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


# XraysSKTtoCT_thd loss: 0.392173 psnr: 29.536929 rmae: 0.025514 rmse: 0.066709 ssim: 0.830876 dice: 0.952159,0.603495,0.313838,0.740374,0.685189
# Trans_XraysSKTtoCT_thd loss: 0.332821 psnr: 30.337939 rmae: 0.022345 rmse: 0.060832 ssim: 0.870434 dice: 0.948924,0.622478,0.543409,0.771224,0.673428
# XraysSKTtoCT_thd loss: 0.336763 psnr: 30.241644 rmae: 0.022693 rmse: 0.061510 ssim: 0.866993 dice: 0.946583,0.611342,0.535548,0.772610,0.677074
def test_whole_image(dataloader, modules, seg_module=None):
    dataloader.dataset.return_original = True
    for trustship in modules:
        trustship.eval()
    essamble_metrics = {trustship.ckpt_prefix: [] for trustship in modules}
    essamble_dice = {trustship.ckpt_prefix: [] for trustship in modules}

    with (torch.no_grad()):
        channel_weights = torch.Tensor([0.0120, 0.3267, 0.923, 1.0434, 1.3548]).to('cpu')
        for essamble, subj in dataloader:
            datadict = {it: essamble[it] for it in essamble}
            print(subj['flnm'], torch.max(essamble['CT']), [torch.mean(datadict['CT'][datadict['SKT']==chl]) for chl in range(5)])
            if 'SKT' in datadict:
                datadict['rSKT'] = channel_weights[datadict['SKT'].to(torch.int)]
            if seg_module is not None:
                # SKT_SEG = torch.sum(channel_weights.view(1, -1, 1, 1, 1) * seg_module.infer_step(datadict, split_size=input_size, stride=(32, 32, 32))[1], dim=1, keepdim=True)
                SKT_SEG = torch.sum(channel_weights.view(1, -1, 1, 1, 1) * seg_module.infer_step(datadict, split_size=input_size)[1], dim=1, keepdim=True).detach()
                # SKT_SEG = channel_weights[torch.argmax(seg_module.infer_step(datadict, split_size=input_size)[1], dim=1, keepdim=True)]
                datadict['sSKT'] = SKT_SEG
                # images_all = [datadict[lt][0, :, 8, :, :].detach().cpu().numpy() for lt in datadict]
            # 0.946507, 0.611081, 0.534982, 0.772403, 0.677456
            for trustship in modules:
                    loss_ensamble, predictions = trustship.eval_step(datadict, split_size=input_size, stride=(96, 32, 32))
                    test_loss = loss_ensamble.item()
                    y_pred = np.maximum(predictions[1].cpu().numpy()[0,], 0)
                    if trustship in [SYN_XraytoSKT, Trans_XraytoSKT]:
                        y_pred = np.argmax(y_pred, axis=0, keepdims=True)

                    y_true = torch.concat([datadict[modal] for modal in datadict if modal in trustship.volout], axis=1).cpu().numpy()[0,]
                    correct_mae = np.mean(np.abs(y_pred-y_true)/2.0)
                    correct_psnr = psnr(y_pred, y_true, data_range=4.0)
                    correct_ncc = np.mean(np.multiply(y_pred - np.mean(y_pred), y_true - np.mean(y_true)) / (np.std(y_pred) * np.std(y_true)) + 1e-6)
                    # correct_psnr = 10 * np.log10((data_range ** 2) / RMSE) / 100
                    correct_rmse = np.sqrt(mse(y_pred/2.0, y_true/2.0))
                    correct_ssim = ssim(y_pred[0], y_true[0], data_range=4.0)
                    correct_dice = trustship.metrics['thd'] if 'thd' in trustship.metrics else trustship.metrics['dice']
                    print(trustship.ckpt_prefix, f"loss: {'%2.6f' % test_loss}", f"psnr: {'%2.6f' % correct_psnr}", f"ncc: {'%2.6f' % correct_ncc}",
                            f"rmae: {'%2.6f' % correct_mae}", f"rmse: {'%2.6f' % correct_rmse}", f"ssim: {'%2.6f' % correct_ssim}",
                            f"dice: {','.join(['%2.6f' % lt for lt in correct_dice])}")
                    essamble_metrics[trustship.ckpt_prefix].append([test_loss, correct_mae, correct_psnr, correct_ncc, correct_rmse, correct_ssim])
                    essamble_dice[trustship.ckpt_prefix].append(correct_dice.numpy())

                    # print(multiple_instensity_metrics(y_pred[0], y_true[0], data_range=2.0))
            pass

            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'FLAIR'+'.png'), all_img[0])
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'missone' + '.png'), all_img[0])

        for trustship in modules:
            average_loss, average_mae, average_psnr, average_ncc, average_rmse, average_ssim = np.mean(essamble_metrics[trustship.ckpt_prefix], 0)
            std_loss, std_mae, std_psnr, std_ncc, std_rmse, std_ssim = np.std(essamble_metrics[trustship.ckpt_prefix], 0)
            average_dice = np.mean(essamble_dice[trustship.ckpt_prefix], 0)
            std_dice = np.std(essamble_dice[trustship.ckpt_prefix], 0)
            print(trustship.ckpt_prefix, f"loss: {'%2.6f' % average_loss}", f"psnr: {'%2.6f' % average_psnr}", f"ncc: {'%2.6f' % average_ncc}",
                  f"rmae: {'%2.6f' % average_mae}", f"rmse: {'%2.6f' % average_rmse}", f"ssim: {'%2.6f' % average_ssim}",
                  f"dice: {','.join(['%2.6f' % lt for lt in average_dice])}", f"psnr: {'%2.6f' % std_psnr}", f"ncc: {'%2.6f' % std_ncc}",
                  f"rmae: {'%2.6f' % std_mae}", f"rmse: {'%2.6f' % std_rmse}", f"ssim: {'%2.6f' % std_ssim}",
                  f"dice: {','.join(['%2.6f' % lt for lt in std_dice])}")


def hu_statistics(dataloader,):
    dataloader.dataset.return_original = True
    stats = []
    with (torch.no_grad()):
        for essamble, subj in dataloader:
            datadict = {it: essamble[it] for it in essamble}
            stats.append([torch.mean(datadict['CT'][datadict['SKT']==chl]) for chl in range(5)])
        print(torch.mean(torch.Tensor(stats), dim=0, keepdims=True))



epochs = 201
start_epochs = 200
# all_trustships = [SYN_XraytoCT, SYN_XraytoCT_thd, SYN_XraytoSKT, SYN_XraySKTtoCT, SYN_XraySKTtoCT_thd, SYN_XraysSKTtoCT, SYN_XraysSKTtoCT_thd]
all_trustships = [Trans_XraysSKTtoCT_thd, SYN_XraysSKTtoCT_thd, SYN_XraysSKTtoCT_thd_ar, ]
if start_epochs > 0:
    for trustship in all_trustships:
        trustship.load_dict(dict_name='chkpt_%d.h5' % start_epochs, strict=False)
SYN_XraytoSKT.load_dict(dict_name='chkpt_200.h5', strict=False)

for t in range(start_epochs, epochs):
    print(f"Epoch {t}\n-------------------------------")
    start_time = time.time()
    # train(train_dataloader, all_trustships)
    # train(train_dataloader, [SYN_XraysSKTtoCT_thd, ], seg_module=SYN_XraytoSKT)
    # test(test_dataloader, all_trustships, seg_module=SYN_XraytoSKT)
    test_whole_image(test_dataloader, all_trustships, seg_module=SYN_XraytoSKT, )
    # test_whole_image(test_dataloader, [SYN_XraysSKTtoCT_thd, Trans_XraysSKTtoCT_thd, SYN_XraytoSKT], seg_module=SYN_XraytoSKT, )
    # hu_statistics(train_dataloader, )
    end_time = time.time()
    print(f"Epoch {t} takes {end_time-start_time} seconds")
    # break
    if t % 10 == 9:
        for trustship in all_trustships:
            trustship.save_dict(dict_name='chkpt_%d.h5' % (t+1))

print("Done!")


# Trans_XraysSKTtoCT_thd loss: 0.371768 psnr: 29.813304 rmae: 0.025725 rmse: 0.065546 ssim: 0.847594 dice: 0.952560,0.629231,0.650674,0.721353,0.620564 psnr: 1.418507 rmae: 0.004982 rmse: 0.011857 ssim: 0.029620 dice: 0.012499,0.069702,0.098047,0.052469,0.069969
# XraysSKTtoCT_thd loss: 0.384994 psnr: 29.557792 rmae: 0.028009 rmse: 0.067410 ssim: 0.843859 dice: 0.951709,0.638187,0.624988,0.713214,0.624718 psnr: 1.348731 rmae: 0.004994 rmse: 0.011577 ssim: 0.029916 dice: 0.012455,0.067309,0.101843,0.054538,0.071619
# XraysSKTtoCT_thd_ar loss: 0.369876 psnr: 29.751250 rmae: 0.025450 rmse: 0.066039 ssim: 0.846416 dice: 0.954633,0.631289,0.657825,0.722374,0.619625 psnr: 1.433500 rmae: 0.005138 rmse: 0.012134 ssim: 0.031539 dice: 0.012610,0.068787,0.097436,0.050954,0.069338
