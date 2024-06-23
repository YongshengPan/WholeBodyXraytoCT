import os
import numpy as np
import time
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def cls_loss_with_logits(y_pred, y_true, model='categorical', from_logits=False):
    if model == 'categorical':
        return torch.mean(torch.binary_cross_entropy_with_logits(y_pred=y_pred, y_true=y_true, from_logits=from_logits))
    elif model == 'binary':
        return torch.mean(torch.binary_cross_entropy_with_logits(y_pred=y_pred, y_true=y_true, from_logits=from_logits))
    else:
        return torch.mean(torch.binary_cross_entropy_with_logits(y_pred=y_pred, y_true=y_true, from_logits=from_logits))


def seg_loss(y_pred, y_true, model='dice'):
    ex_axis = [0, 2, 3, 4]
    ex_axis = [0] + [ex for ex in range(2, len(y_true.shape))]
    if model == 'dice':
        value = 1 - (2 * torch.sum(y_true * y_pred, dim=ex_axis) + 1) / (torch.sum(y_true, dim=ex_axis) + torch.sum(y_pred, dim=ex_axis) + 1)
    elif model == 'jaccard':
        value = 1 - (torch.sum(torch.minimum(y_true, y_pred), dim=ex_axis) + 1) / (torch.sum(torch.maximum(y_true, y_pred), dim=ex_axis) + 1)
    else:
        value = 1 - (2 * torch.sum(y_true * y_pred, dim=ex_axis) + 1) / (torch.sum(y_true, dim=ex_axis) + torch.sum(y_pred, dim=ex_axis) + 1)
    return torch.mean(value)


def mae_loss(con_feat, fake_feat, weight=1.0):
    # return tf.reduce_mean(tf.abs(con_feat - fake_feat)) * weight
    # weight = tf.tanh(tf.abs(con_feat - fake_feat)*1000)+0.001
    # weight = weight/tf.reduce_mean(weight)
    return torch.mean(torch.abs(con_feat - fake_feat) * weight)


def mse_loss(con_feat, fake_feat, weight=1):
    return torch.reduce_mean(torch.square(con_feat - fake_feat)) * weight


def mae_loss_with_weight(con_feat, fake_feat, weight):
    return torch.reduce_mean(torch.abs(con_feat - fake_feat) * weight)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ct_segmentation_loss(y_pred, y_true, model='dice'):
    thres = -1100, -900, -400, -150, -100, -10, -100, 150, 250, 400, 20000
    scale, offset = 1000, -1000
    y_true = y_true*scale+offset
    y_pred = y_pred*scale+offset
    dice_value = 0
    for idx in range(len(thres)-1):
        # region = tf.minimum(tf.nn.relu(y_true - thres[idx]), 1) - tf.minimum(tf.nn.relu(y_true - thres[idx+1]), 1)
        # predict = tf.minimum(tf.nn.relu(y_pred - thres[idx]), 1) - tf.minimum(tf.nn.relu(y_pred - thres[idx+1]), 1)
        region = torch.sigmoid(y_true - thres[idx]) - torch.sigmoid(y_true - thres[idx + 1])
        predict = torch.sigmoid(y_pred - thres[idx]) - torch.sigmoid(y_pred - thres[idx + 1])
        dice_value = dice_value + seg_loss(region, predict, model=model)

    return dice_value/(len(thres)-1)

# region, predict = np.logical_and(tissue_thres < groundtruth, groundtruth < bone_thres), np.logical_and(
#     tissue_thres < prediction, prediction < bone_thres)
# Dice1 = (2 * np.sum(predict * region, axis=ex_axis) + 1e-3) / (
#             np.sum(predict, ex_axis) + np.sum(region, axis=ex_axis) + 1e-3)
#
# region, predict = groundtruth > bone_thres, prediction > bone_thres
# Dice2 = (2 * np.sum(predict * region, axis=ex_axis) + 1e-3) / (
#             np.sum(predict, ex_axis) + np.sum(region, axis=ex_axis) + 1e-3)


def wavelet_loss(y_pred, y_true):
    x_d = torch.reshape(torch.Tensor([1.0, 1.0], torch.float32), (2, 1, 1, 1, 1)), torch.reshape(torch.Tensor([-1.0, 1.0], torch.float32), (2, 1, 1, 1, 1))
    y_d = torch.reshape(torch.Tensor([1.0, 1.0], torch.float32), (1, 2, 1, 1, 1)), torch.reshape(torch.Tensor([-1.0, 1.0], torch.float32), (1, 2, 1, 1, 1))
    z_d = torch.reshape(torch.Tensor([1.0, 1.0], torch.float32), (1, 1, 2, 1, 1)), torch.reshape(torch.Tensor([-1.0, 1.0], torch.float32), (1, 1, 2, 1, 1))

    kernels = torch.concat([x_d[0]*y_d[0]*z_d[0], x_d[0]*y_d[0]*z_d[1], x_d[0]*y_d[1]*z_d[0], x_d[0]*y_d[1]*z_d[1],
                            x_d[1]*y_d[0]*z_d[0], x_d[1]*y_d[0]*z_d[1], x_d[1]*y_d[1]*z_d[0], x_d[1]*y_d[1]*z_d[1]], dim=1)

    wl_pred = F.conv3d(y_pred, kernels, stride=[1, 1, 1], padding='VALID')
    wl_true = F.conv3d(y_true, kernels, stride=[1, 1, 1], padding='VALID')
    return F.l1_loss(wl_pred, wl_true)



# angles_count = 30
# radon_transform_matrix = {}

# def sinogram_loss(y_pred, y_true):
#     diff_trans = tf.transpose(y_pred-y_true, (4, 0, 1, 2, 3))
#     window_size = diff_trans.shape[-2]
#
#     # time_start = time.time()
#     radon_transform_matrix_index = 'space_indexs_%d_%d.npz' % (angles_count, window_size)
#     if radon_transform_matrix_index in radon_transform_matrix:
#         space_indexs = radon_transform_matrix[radon_transform_matrix_index]
#     elif os.path.exists(radon_transform_matrix_index):
#         space_indexs = np.load(radon_transform_matrix_index, allow_pickle=True)
#         space_indexs = tf.sparse.SparseTensor(indices=space_indexs['indices'], values=space_indexs['values'],
#                                               dense_shape=space_indexs['dense_shape'])
#         radon_transform_matrix[radon_transform_matrix_index] = space_indexs
#     else:
#         space_indexs = rtt.create_radon_kernel(window_size, np.linspace(0, 180, angles_count, endpoint=False))
#         np.savez(radon_transform_matrix_index, dense_shape=space_indexs.dense_shape, indices=space_indexs.indices,
#                      values=space_indexs.values)
#         radon_transform_matrix[radon_transform_matrix_index] = space_indexs
#
#     diff_trans = tf.unstack(tf.reshape(diff_trans, shape=(-1, diff_trans.shape[-3], diff_trans.shape[-2], diff_trans.shape[-1])), axis=0)
#     sinogram_diff = tf.reduce_mean([tf.reduce_mean(tf.abs(rtt.radon_transform(diff_trans[idx], space_indexs, axis=0))) for idx in range(len(diff_trans))])
#
#     # time_end = time.time()
#     # print(time_end - time_start)
#     return sinogram_diff


# def maxproj_loss(y_pred, y_true):
#
#     x_d = mae_loss(tf.reduce_max(y_pred, 1), tf.reduce_max(y_true, 1))
#     y_d = mae_loss(tf.reduce_max(y_pred, 2), tf.reduce_max(y_true, 2))
#     z_d = mae_loss(tf.reduce_max(y_pred, 3), tf.reduce_max(y_true, 3))
#     return x_d + y_d + z_d
#
# np.linspace(0, 180, angles_count, endpoint=False, dtype=np.float32)
# rotation_matrix = tf.convert_to_tensor([[[np.cos(theta), -np.sin(theta)], [np.sin(theta), -np.cos(theta)]] for theta in np.linspace(0, 180, angles_count, endpoint=False, dtype=np.float32)])

# def maxproj_loss(y_pred, y_true):
#     # time_start = time.time()
#     y_pred = tf.transpose(y_pred, (2, 3, 0, 1, 4))
#     y_true = tf.transpose(y_true, (2, 3, 0, 1, 4))
#     image_shape = tf.shape(y_true, out_type=tf.int32)
#     label_prop_x = tf.range(0, image_shape[0], dtype=tf.float32) - tf.cast(image_shape[0], tf.float32) * 0.5 - 0.5
#     label_prop_y = tf.range(0, image_shape[1], dtype=tf.float32) - tf.cast(image_shape[1], tf.float32) * 0.5 - 0.5
#     label_prop = tf.meshgrid(label_prop_x, label_prop_y, indexing='ij')
#     label_prop = [tf.expand_dims(lp, axis=-1) for lp in label_prop]
#     label_meshgrid = tf.concat(label_prop, axis=-1)
#     all_loss = 0
#     for theta in np.linspace(0, 180, angles_count, endpoint=False):
#         rotation_matrix = [[tf.cos(theta), -tf.sin(theta)], [tf.sin(theta), -tf.cos(theta)]]
#         label_prop_trans = tf.matmul(label_meshgrid, rotation_matrix) + tf.cast(image_shape[0:2], tf.float32) * 0.5 - 0.5
#         label_prop_trans = tf.cast(label_prop_trans, tf.int32)
#         y_true_trans = tf.gather_nd(y_true, label_prop_trans)
#         y_pred_trans = tf.gather_nd(y_pred, label_prop_trans)
#         y_true_trans_s = tf.split(y_true_trans, [image_shape[0]//2, image_shape[0]//2], axis=0)
#         y_true_trans_s = tf.split(y_true_trans_s[0], [image_shape[1]//2, image_shape[1]//2], axis=1) + tf.split(y_true_trans_s[1], [image_shape[1]//2, image_shape[1]//2], axis=1)
#         y_pred_trans_s = tf.split(y_pred_trans, [image_shape[0] // 2, image_shape[0] // 2], axis=0)
#         y_pred_trans_s = tf.split(y_pred_trans_s[0], [image_shape[1]//2, image_shape[1]//2], axis=1) + tf.split(y_pred_trans_s[1], [image_shape[1]//2, image_shape[1]//2], axis=1)
#         x_d = [mae_loss(tf.reduce_max(y_true_trans_s[idx], 0), tf.reduce_max(y_pred_trans_s[idx], 0)) for idx in [0, 1, 2, 3]]
#         y_d = [mae_loss(tf.reduce_max(y_true_trans_s[idx], 1), tf.reduce_max(y_pred_trans_s[idx], 1)) for idx in [0, 1, 2, 3]]
#         # x_d = mae_loss(tf.reduce_max(y_true_trans, 0), tf.reduce_max(y_pred_trans[image_shape[0]//2::, :], 0))
#         # y_d = mae_loss(tf.reduce_max(y_true_trans, 1), tf.reduce_max(y_pred_trans[:, 0:image_shape[1]//2], 1))
#         all_loss = all_loss + tf.reduce_mean(x_d) + tf.reduce_mean(y_d)
#     # time_end = time.time()
#     # print(time_end - time_start)
#     return all_loss/angles_count


def multi_feat_loss(con_feats, fake_feats, weight=None):
    if weight is None:
        multi_weight = [1 for slt in range(len(con_feats))]
    else:
        strips = [np.int32(np.ceil(np.divide(weight.shape, fake_feats[slt].shape))) for slt in range(len(con_feats))]
        multi_weight = [weight[strips[slt][0]//2::strips[slt][0], strips[slt][1]//2::strips[slt][1], strips[slt][2]//2::strips[slt][2], strips[slt][3]//2::strips[slt][3]] for slt in range(len(con_feats))]
    return torch.sum(torch.Tensor([F.mse_loss(con_feats[slt], fake_feats[slt])*multi_weight[slt] for slt in range(len(con_feats))]))


def basic_loss_essamble(y_pred, y_true, lossses):
    total_loss = 0
    for loss in lossses:
        if loss == 'maxp':
            total_loss = total_loss + maxproj_loss(y_true, y_pred)
        elif loss == 'sin':
            total_loss = total_loss + sinogram_loss(y_true, y_pred)
        elif loss == 'wll':
            total_loss = total_loss + wavelet_loss(y_true, y_pred)
        elif loss == 'p2p':
            total_loss = total_loss + F.l1_loss(y_true, y_pred)
        elif loss == 'cre':
            total_loss = total_loss + F.cross_entropy(y_pred, y_true)
        elif loss == 'dice':
            total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='dice')
        elif loss == 'jac':
            total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='jaccard')
        elif loss == 'ct_dice':
            total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='dice')
        elif loss == 'ct_jac':
            total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='jac')
        else:
            pass
    return total_loss


def extra_loss_essamble(loss_maps, ext_fun, ext_const, ext_alter, ext_prob, ext_tar=None):
    syn_total_loss = 0

    def kconcat(inputs, axis=1):
        inputs = [torch.Tensor(var).to('cuda') for var in inputs if var is not None]
        return None if len(inputs) == 0 else torch.concat(inputs, dim=axis)

    if len(loss_maps) > 0:
        con_logit, con_prob, con_feat = ext_fun(kconcat([ext_alter, ext_const], axis=1))
        fake_con_logit, fake_con_prob, fake_con_feat = ext_fun(kconcat([ext_prob, ext_const], axis=1))
        if 'dis' in loss_maps:
            syn_total_loss = syn_total_loss + mae_loss(fake_con_logit, 1) / 10
        if 'cyc' in loss_maps:
            syn_total_loss = syn_total_loss + mae_loss(fake_con_prob, ext_tar)
        if 'cls' in loss_maps:
            syn_total_loss = syn_total_loss + cls_loss_with_logits(y_pred=fake_con_prob, y_true=ext_tar)
        if 'msl' in loss_maps or 'scl' in loss_maps or 'fcl' in loss_maps:
            syn_total_loss = syn_total_loss + multi_feat_loss(con_feat, fake_con_feat)
    return syn_total_loss


def matrics_ct_segmentation(y_pred, y_true, model='dice'):

    def dice_value(predict, region, model=model):
        ex_axis = [0, 1, 2, 3, 4]
        # ex_axis = tuple(ex_axis[0: np.ndim(region) - 1])
        dv = (2 * np.sum(predict * region) + 1e-3) / (np.sum(predict) + np.sum(region) + 1e-3)
        return dv

    thres = -1100, -900, -150, -10, 150, 20000
    scale, offset = 1000, -1000
    y_true = y_true*scale+offset
    y_pred = y_pred*scale+offset

    dice_values = []

    for idx in range(len(thres)-1):
        region = np.where(np.logical_and(y_true > thres[idx], y_true < thres[idx+1]), 1, 0)
        predict = np.where(np.logical_and(y_pred > thres[idx], y_pred < thres[idx + 1]), 1, 0)
        # region = np.minimum(np.maximum(y_true - thres[idx], 0), 1) - np.minimum(np.maximum(y_true - thres[idx+1], 0), 1)
        # predict = np.minimum(np.maximum(y_pred - thres[idx], 0), 1) - np.minimum(np.maximum(y_pred - thres[idx+1], 0), 1)
        dice_values.append(dice_value(region, predict, model=model))

    return dice_values


def multiple_instensity_metrics(prediction, groundtruth, data_range=1.0):
    prediction, groundtruth = prediction / data_range, groundtruth / data_range
    diff_map = prediction - groundtruth
    MAE = np.mean(np.abs(diff_map))
    RMSE = np.sqrt(np.mean(np.square(diff_map)))
    SSIM = np.mean(ssim(groundtruth, prediction, data_range=data_range, full=False, channel_axis=-1))
    PSNR = 10 * np.log10((data_range ** 2) / RMSE) / 100
    NCC = np.mean(np.multiply(prediction - np.mean(prediction), groundtruth - np.mean(groundtruth)) / (np.std(prediction) * np.std(groundtruth))+1e-6)
    return MAE, RMSE, PSNR, NCC, SSIM


def multiple_projection_metrics(prediction, groundtruth, data_range=1):
    ex_axis = [0, 1, 2, 3, 4]

    MAE = np.mean(np.abs(prediction / data_range - groundtruth / data_range))
    MSE = np.mean(np.square(prediction / data_range - groundtruth / data_range))
    SSIM = np.mean(ssim(groundtruth / data_range, prediction / data_range, full=False, multichannel=True))
    PSNR = 10 * np.log10((data_range ** 2) / MSE) / 100
    NCC = np.mean(np.multiply(prediction - np.mean(prediction), groundtruth - np.mean(groundtruth)) / (
                np.std(prediction) * np.std(groundtruth)))
    return [MAE], [MSE], [PSNR], [NCC], [SSIM]


def matrics_synthesis(prediction, groundtruth, data_range=1.0, isinstance=False):
    prediction = prediction
    groundtruth = groundtruth
    ex_axis = [0, 1, 2, 3, 4]

    if isinstance:
        INMT = [multiple_instensity_metrics(prediction, groundtruth, data_range=data_range)]
        DICE = [matrics_ct_segmentation(prediction, groundtruth, model='dice')]

    else:
        INMT = [multiple_instensity_metrics(prediction[idx], groundtruth[idx], data_range=data_range) for idx in range(np.shape(groundtruth)[0])]
        DICE = [matrics_ct_segmentation(prediction[idx], groundtruth[idx], model='dice') for idx in range(np.shape(groundtruth)[0])]

    # return 100*np.mean(INMT, axis=0), 100*np.mean(DICE, axis=0)
    return np.concatenate((100*np.mean(INMT, axis=0), 100*np.mean(DICE, axis=0)), axis=-1)
    # return np.mean(MAE), np.mean(MSE), np.mean(SSIM), np.mean(PSNR), np.std(MAE), np.std(MSE), np.std(SSIM), np.std(PSNR)


def matrics_classification(testvals, labels, thres=None):
    # print(np.mean(testvals, axis=0))

    def softmax(logits):
        return np.exp(logits) / np.sum(np.exp(logits), -1, keepdims=True)

    if thres is not None:
        testvals = testvals - np.array(thres)
    else:
        testvals = testvals - np.mean(testvals, axis=0)
    testvals = softmax(testvals)
    print(np.shape(testvals))
        # losslist = -np.sum(np.multiply(labels, np.log(testvals)), 1)
        # total_loss = np.average(losslist)
    AUC = metrics.roc_auc_score(y_score=testvals, y_true=labels, average='macro')
    ACC = metrics.accuracy_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    BAC = metrics.balanced_accuracy_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    APS = metrics.average_precision_score(y_score=testvals, y_true=labels, average='macro')
    SEN = metrics.recall_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), pos_label=1, average='macro')
    SPE = metrics.recall_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), pos_label=0, average='macro')
    COM = metrics.confusion_matrix(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    F1S = metrics.f1_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), average='macro')
    MCC = metrics.matthews_corrcoef(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    # return [AUC*100, ACC*100, BAC*100, REC*100, F1S*100, MCC*100]
    return [AUC*100, ACC*100, SEN*100, SPE*100, F1S*100, MCC*100], metrics.classification_report(y_true=np.argmax(labels, axis=-1), y_pred=np.argmax(testvals, axis=-1))


def matrics_segmentation(prediction, groundtruth, labeltype='category', threshold=0.5):

    if labeltype == 'category' and np.shape(prediction)[-1] > 1:
        prediction_hard = np.argmax(prediction, axis=-1)
        groundtruth_hard = np.argmax(groundtruth, axis=-1)
        prediction_hard = np.concatenate([np.expand_dims(prediction_hard == idx, axis=-1) for idx in range(np.shape(prediction)[-1])], axis=-1)
        groundtruth_hard = np.concatenate([np.expand_dims(groundtruth_hard == idx, axis=-1) for idx in range(np.shape(prediction)[-1])], axis=-1)
    else:
        prediction_hard = np.array(prediction > threshold)
        groundtruth_hard = np.array(groundtruth > threshold)
    # Intersection = [np.array(prediction_hard == idx) & np.array(groundtruth_hard == idx) for idx in range(np.shape(prediction)[-1])]
    # Union = [np.array(prediction_hard == idx) | np.array(groundtruth_hard == idx) for idx in range(np.shape(prediction)[-1])]
        # Intersection = np.array(prediction > threshold) & np.array(groundtruth > threshold)
        # Union = np.array(prediction > threshold) | np.array(groundtruth > threshold)
    # ex_axis = [dd for dd in range(0, np.ndim(Intersection)-1)]
    ex_axis = [0, 1, 2, 3, 4]
    ex_axis = tuple(ex_axis[0: np.ndim(prediction) - 1])
    IoU = (np.sum(prediction_hard & groundtruth_hard, axis=ex_axis)+1e-3)/(np.sum(prediction_hard | groundtruth_hard, axis=ex_axis)+1e-3)
    Jaccard = (np.sum(np.minimum(prediction, groundtruth), axis=ex_axis)+1e-3) / (np.sum(np.maximum(prediction, groundtruth), axis=ex_axis)+1e-3)
    DICE1 = (2*np.sum(prediction_hard*groundtruth_hard, axis=ex_axis)+1e-3)/(np.sum(prediction_hard, axis=ex_axis) + np.sum(groundtruth_hard, axis=ex_axis)+1e-3)
    DICE2 = (2 * np.sum(prediction * groundtruth, axis=ex_axis)+1e-3) / (np.sum(prediction, ex_axis) + np.sum(groundtruth, axis=ex_axis)+1e-3)
    return IoU*100, Jaccard*100, DICE1*100, DICE2*100


def getbondingbox(image, fctr=0, thres=0.5):
        org_shp = np.shape(image)
        locsx, locsy = np.nonzero(np.sum(image > 0.5, axis=0)), np.nonzero(np.sum(image > 0.5, axis=1))
        if len(locsx[0]) == 0 or len(locsy[0]) == 0: return None
        region = np.array([[min(locsy[0]), (max(locsy[0]) + 1 + fctr * org_shp[0])],
                           [min(locsx[0]), (max(locsx[0]) + 1 + fctr * org_shp[1])]]) // (fctr + 1)
        region = region.astype(np.int)
        region[0] = np.minimum(np.maximum(region[0], 0), org_shp[0])
        region[1] = np.minimum(np.maximum(region[1], 0), org_shp[1])
        return region




