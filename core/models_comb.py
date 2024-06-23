import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import basic_loss_essamble, extra_loss_essamble


ConvBlock = {'2d': nn.Conv2d, '2.5d': nn.Conv3d, '3d': nn.Conv3d, }
InstanceNorm = {'2d': nn.InstanceNorm2d, '2.5d': nn.InstanceNorm3d, '3d': nn.InstanceNorm3d, }


def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        if alpha is None: alpha = 0.00
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input)
    else:
        output = input
    return output


def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [krnlsz] * 2 + [(np.array(krnlsz) * 0 + 1) * half_dim]
    else:
        outsz = [krnlsz]
    return outsz


class SimpleClassifier(nn.Module):
    # def extdim(krnlsz, halfdim=1):
    #     return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
    defaultparams = {
        'input_shape': [1, 128, 128, 128],
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'activation': 'softmax',
    }

    def __init__(self, params, model_type='3D'):
        super(SimpleClassifier, self).__init__()
        self.params = dict(SimpleClassifier.defaultparams, **params)
        self.use_spatial_kernel = self.params['use_spatial_kernel']
        self.use_local_l2 = self.params['use_local_l2']
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d

        self.bulid_network(self.params['input_shape'], self.params['basedim'])

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]
        self.conv1 = self.ConvBlock(inchl, basedim, self.extdim(3), padding=self.extdim(1, 0), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(basedim, affine=True)
        self.conv2 = self.ConvBlock(basedim*1, basedim*2, self.extdim(3), padding=self.extdim(1, 0), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(basedim*2, affine=True)
        self.conv3 = self.ConvBlock(basedim*2, basedim*4, self.extdim(3), padding=self.extdim(1, 0), padding_mode='reflect')
        self.norm3 = self.InstanceNorm(basedim * 4, affine=True)
        self.conv4 = self.ConvBlock(basedim*4, basedim*4, self.extdim(3), padding=self.extdim(1, 0), padding_mode='reflect')
        self.norm4 = self.InstanceNorm(basedim * 4, affine=True)
        self.conv5 = self.ConvBlock(basedim*4, basedim*4, self.extdim(3), padding=self.extdim(1, 0), padding_mode='reflect')
        self.norm5 = self.InstanceNorm(basedim * 4, affine=True)
        if self.use_spatial_kernel:
            if self.model_type == '2.5d':
                featsize = np.prod([insize[0]//2**5, insize[1]//2**5, insize[2]])
            else:
                featsize = np.prod([isz // (2 ** 5) for isz in insize])
            if self.params['use_second_order']:
                featsize = featsize * 2
        else:
            featsize = basedim * 4
        self.fc = nn.Linear(featsize * basedim * 4, self.output_channel)

    def forward(self, x):
        o_c1 = self.max_pool(F.relu(self.norm1(self.conv1(x))), self.extdim(3), self.extdim(2), padding=self.extdim(1, 0))
        o_c2 = self.max_pool(F.relu(self.norm2(self.conv2(o_c1))), self.extdim(3), self.extdim(2), padding=self.extdim(1, 0))
        o_c3 = self.max_pool(F.relu(self.norm3(self.conv3(o_c2))), self.extdim(3), self.extdim(2), padding=self.extdim(1, 0))
        o_c4 = self.max_pool(F.relu(self.norm4(self.conv4(o_c3))), self.extdim(3), self.extdim(2), padding=self.extdim(1, 0))
        o_c5 = self.avg_pool(F.relu(self.norm5(self.conv5(o_c4))), self.extdim(3), self.extdim(2), padding=self.extdim(1, 0))

        if self.use_spatial_kernel:
            if self.params['use_second_order']:
                x = torch.cat((o_c5, torch.square(o_c5)-torch.tensor(1.0)), 2)
            if self.use_local_l2:
                x = F.normalize(x, p=2, dim=1)
        else:
            x = self.avg_pool(x, x.size()[2:])
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_flat_features(x))
        logit = self.fc(x)
        prob = build_end_activation(logit, self.activation_function)
        return [logit, prob, [o_c1, o_c2, o_c3, o_c4, o_c5, ]]


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class GeneralDiscriminator(nn.Module):
    # def extdim(krnlsz, halfdim=1):
    #     return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
    defaultparams = {
        'input_shape': [1, 128, 128, 128],
        'basedim': 16,
        'filter_size': 4,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'activation': 'softmax',
    }

    def __init__(self, params, model_type='3D'):
        super(GeneralDiscriminator, self).__init__()
        self.params = dict(SimpleClassifier.defaultparams, **params)
        self.use_spatial_kernel = self.params['use_spatial_kernel']
        self.use_local_l2 = self.params['use_local_l2']
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d

        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
        self.bulid_network(self.params['input_shape'], self.params['basedim'])

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]
        self.conv1 = self.ConvBlock(inchl, basedim, self.extdim(4), stride=self.extdim(2), padding=self.extdim(1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(basedim, affine=True)
        self.conv2 = self.ConvBlock(basedim*1, basedim*2,  self.extdim(4), stride=self.extdim(2), padding=self.extdim(1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(basedim*2, affine=True)
        self.conv3 = self.ConvBlock(basedim*2, basedim*4,  self.extdim(4), stride=self.extdim(2), padding=self.extdim(1), padding_mode='reflect')
        self.norm3 = self.InstanceNorm(basedim * 4, affine=True)
        self.conv4 = self.ConvBlock(basedim*4, basedim*8,  self.extdim(4), stride=1, padding=1, padding_mode='reflect')
        self.norm4 = self.InstanceNorm(basedim * 8, affine=True)
        self.conv5 = self.ConvBlock(basedim*8, 1,  self.extdim(4), stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        o_c1 = F.relu(self.conv1(x))
        o_c2 = F.relu(self.norm2(self.conv2(o_c1)))
        o_c3 = F.relu(self.norm3(self.conv3(o_c2)))
        o_c4 = F.relu(self.norm4(self.conv4(o_c3)))
        o_c5 = self.conv5(o_c4)
        prob = build_end_activation(o_c5, self.activation_function)
        return [o_c5, prob, [o_c1, o_c2, o_c3, o_c4, ]]


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            model_type='3d',
            change_dimension=False,
            device=None,
            dtype=None):
        super(ResidualBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.change_dimension = change_dimension
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
        self.short_cut_conv = self.ConvBlock(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = self.InstanceNorm(out_channels, affine=True)
        self.conv1 = self.ConvBlock(in_channels, out_channels, extdim(kernel_size), extdim(stride), padding=extdim(padding, 0), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(out_channels, affine=True)
        self.conv2 = self.ConvBlock(out_channels, out_channels, extdim(kernel_size), extdim(1), padding=extdim(padding, 0), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True)

    def forward(self, x):
        if self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = F.relu(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        out_res = F.relu(o_c2+short_cut_conv)
        return out_res


class TransformerBlock(nn.Module):
    def __init__(self, dimofmodel, numofheads=2, style='3d', name="residual"):
        super(TransformerBlock, self).__init__()
        self.dimofhead = dimofmodel // numofheads
        self.numofheads = numofheads

        self.layernorm = nn.LayerNorm(dimofmodel)
        self.linearK = nn.Linear(dimofmodel, dimofmodel)
        self.linearV = nn.Linear(dimofmodel, dimofmodel)
        self.linearQ = nn.Linear(dimofmodel, dimofmodel)
        self.attenfactor = self.dimofhead ** -0.5
        self.multiheadattention = nn.MultiheadAttention(dimofmodel, numofheads)
        self.attenaction = nn.Softmax()
        self.applyatten = nn.layers.Dot(axes=(2, 1))
        self.mlp = nn.Sequential([nn.layers.LayerNormalization(axis=(-1)),
                                  nn.layers.Dense(dimofmodel*4),
                                  nn.layers.ReLU(),
                                  nn.layers.Dense(dimofmodel)])

    def forward(self, input_tensor, training=None, mask=None):
        o_rb = self.layernorm(input_tensor)
        linear_k = self.linearK(o_rb)
        linear_v = self.linearV(o_rb)
        linear_q = self.linearQ(o_rb)
        QmKmV = self.multiheadattention(linear_q, linear_k, linear_v)
        inputres = input_tensor + torch.concat(QmKmV, dim=1)
        out_res = inputres + self.mlp(inputres)
        return out_res


class SimpleEncoderDecoder(nn.Module):

    defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }

    def __init__(self, params, model_type='3D'):
        super(SimpleEncoderDecoder, self).__init__()
        self.params = dict(SimpleEncoderDecoder.defaultparams, **params)
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d
        self.bulid_network(self.params['input_shape'], self.params['basedim'], self.model_type)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]

        self.begin_conv = nn.Sequential(self.ConvBlock(inchl, basedim, self.extdim(7), stride=self.extdim(1), padding=self.extdim(3, 0), padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        nn.ReLU())

        self.encoding_block = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.encoding_block.append(
                nn.Sequential(
                    self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), self.extdim(3), stride=self.extdim(2), padding=self.extdim(1, 0), padding_mode='reflect'),
                    self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                    nn.ReLU()))

        trans_dim = basedim * 2 ** self.params['downdeepth']
        self.trans_block = nn.Sequential(
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,)
        )

        self.decoding_block = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.decoding_block.append(
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx), affine=True),
                nn.ReLU()))

        self.end_conv = nn.Sequential(self.ConvBlock(basedim*2, basedim, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                                      self.InstanceNorm(basedim, affine=True),
                                      self.ConvBlock(basedim, self.params['output_channel'], self.extdim(7), stride=1, padding=self.extdim(3, 0), padding_mode='reflect'),
                                      self.InstanceNorm(self.params['output_channel'], affine=True)
                                      )

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)
        o_c2 = self.trans_block(o_c1)

        for convidx in range(self.params['downdeepth'], 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")

            # o_c2 = self.decoding_block[convidx - 1](o_c2)
            # o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")
            # o_c2 = torch.concat((o_c2, feats[convidx-1]), dim=1)

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]

class ResEncoderDecoder(nn.Module):

    defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }

    def __init__(self, params, model_type='3D'):
        super(ResEncoderDecoder, self).__init__()
        self.params = dict(ResEncoderDecoder.defaultparams, **params)
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d
        self.bulid_network(self.params['input_shape'], self.params['basedim'], self.model_type)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]

        self.begin_conv = nn.Sequential(self.ConvBlock(inchl, basedim, self.extdim(7), stride=self.extdim(1), padding=self.extdim(3, 0), padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        )

        self.encoding_block = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.encoding_block.append(
                nn.Sequential(
                    ResidualBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                    self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), self.extdim(3), stride=self.extdim(2), padding=self.extdim(1, 0), padding_mode='reflect'),
                    self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                    ))

        trans_dim = basedim * 2 ** self.params['downdeepth']
        self.trans_block = nn.Sequential(
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
        )

        self.decoding_block = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.decoding_block.append(
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx), affine=True),
                nn.ReLU()))

        self.end_conv = nn.Sequential(self.ConvBlock(basedim*2, basedim, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                                      self.InstanceNorm(basedim, affine=True),
                                      self.ConvBlock(basedim, self.params['output_channel'], self.extdim(7), stride=1, padding=self.extdim(3, 0), padding_mode='reflect'),
                                      self.InstanceNorm(self.params['output_channel'], affine=True)
                                      )

    def tridis_mix(self, x:torch.Tensor, p=0.5, alpha=0.1, eps=1.0e-6):
        shp = x.shape
        smpshp = [shp[0], shp[1]] + [1]*(len(shp)-2)
        if torch.rand(1) > p:
            return x
        mu = torch.mean(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        var = torch.var(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        sig = torch.sqrt(var+eps)
        x_normed = (x-mu.detach()) / sig.detach()
        mu_r = torch.rand(smpshp, device=mu.device)
        sig_r = torch.rand(smpshp, device=mu.device)
        lmda = torch.distributions.Beta(alpha, alpha).sample(smpshp)
        bern = torch.bernoulli(lmda).to(mu.device)
        mu_mix = mu_r*bern + mu * (1.0-bern)
        sig_mix = sig_r * bern + sig * (1.0 - bern)
        return x_normed * sig_mix + mu_mix

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            if self.training:
                o_c1 = self.tridis_mix(o_c1)
            o_c1 = self.encoding_block[convidx](F.relu(o_c1))
            feats.append(o_c1)
        o_c2 = self.trans_block(F.relu(o_c1))

        for convidx in range(self.params['downdeepth'], 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)

            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")

            # o_c2 = self.decoding_block[convidx - 1](o_c2)
            # o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")
            # o_c2 = torch.concat((o_c2, feats[convidx-1]), dim=1)

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]



class Simple2DEncoder3DDecoder(nn.Module):

    defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }

    def __init__(self, params, model_type='3D'):
        super(Simple2DEncoder3DDecoder, self).__init__()
        self.params = dict(Simple2DEncoder3DDecoder.defaultparams, **params)
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d
        self.bulid_network(self.params['input_shape'], self.params['basedim'], self.model_type)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]//2
        insize = input_shape[1::]

        self.begin_conv1 = nn.Sequential(self.ConvBlock(inchl, basedim, [7, 1, 7], stride=1, padding=(3, 0, 3), padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        nn.ReLU())
        self.begin_conv2 = nn.Sequential(self.ConvBlock(inchl, basedim, [7, 7, 1], stride=1, padding=(3, 3, 0), padding_mode='reflect'),
                                        nn.ReLU())

        self.encoding_block1 = nn.Sequential()
        self.encoding_block2 = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.encoding_block1.append(
                nn.Sequential(
                    ResidualBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                    self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), (3, 1, 3), stride=(2, 1, 2), padding=(1, 0, 1), padding_mode='reflect'),
                    self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                    nn.ReLU()))
            self.encoding_block1.append(
                nn.Sequential(
                    ResidualBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                    self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), (3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0), padding_mode='reflect'),
                    self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                    nn.ReLU()))

        trans_dim = basedim * 2 ** self.params['downdeepth']
        self.trans_block = nn.Sequential(
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            ResidualBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
        )

        self.decoding_block = nn.Sequential()
        for convidx in range(0, self.params['downdeepth']):
            self.decoding_block.append(
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx), affine=True),
                nn.ReLU()))

        self.end_conv = nn.Sequential(self.ConvBlock(basedim*2, basedim, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                                      self.InstanceNorm(basedim, affine=True),
                                      self.ConvBlock(basedim, self.params['output_channel'], self.extdim(7), stride=1, padding=self.extdim(3, 0), padding_mode='reflect')
                                      )

    def forward(self, x):
        x = torch.split(x, 1, dim=1)
        x_slc1 = x[:, 0:1, :, :, 0:1]
        x_slc2 = x[:, 1:2, :, 0:1, :]
        o_c1 = self.begin_conv1(x_slc1), self.begin_conv2(x_slc2)
        feats = [o_c1[0]+o_c1[1], ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block1[convidx](o_c1[0]), self.encoding_block1[convidx](o_c1[1])
            feats.append(o_c1[0]+o_c1[1])
        o_c2 = self.trans_block(o_c1[0]+o_c1[1])

        for convidx in range(self.params['downdeepth'], 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")

            # o_c2 = self.decoding_block[convidx - 1](o_c2)
            # o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")
            # o_c2 = torch.concat((o_c2, feats[convidx-1]), dim=1)

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]


class StandardUNet(nn.Module):
    # def extdim(krnlsz, halfdim=1):
    #     return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
    defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }

    def __init__(self, params, model_type='3D'):
        super(StandardUNet, self).__init__()
        self.params = dict(StandardUNet.defaultparams, **params)
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()
        self.bulid_network(self.params['input_shape'], self.params['basedim'])
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]
        self.begin_conv = nn.Sequential(self.ConvBlock(inchl, basedim, 7, stride=1, padding=3, padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        nn.ReLU())
        self.encoding_block = [
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=2, padding=1, padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU(),
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=2, padding=1, padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU()
            )
            for convidx in range(0, self.params['downdeepth'])]

        trans_dim = basedim * 2 ** self.params['downdeepth']
        self.trans_block = nn.Sequential(
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1)
        )

        self.decoding_block = [
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=1, padding=1, padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU(),
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=1, padding=1, padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU()
            )
            for convidx in range(0, self.params['downdeepth'])]

        self.end_conv = nn.Sequential(self.ConvBlock(basedim, self.params['output_channel'], 7, stride=1, padding=3, padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True))

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)
            o_c1 = self.max_pool(o_c1, 3, 2, padding=1)

        o_c2 = self.trans_block(o_c1)

        for convidx in range(self.params['downdeepth'], 0, -1):
            o_c2 = self.decoding_block[convidx-1](torch.concat((o_c2, feats[convidx]), dim=1))
            F.interpolate(o_c2, scale_factor=2, mode='nearest')

        o_c3 = self.end_conv(o_c2)
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]


class TransfermerEncoderDecoder(nn.Module):
    # def extdim(krnlsz, halfdim=1):
    #     return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
    defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
                     'downdeepth': 2,
                     'use_skip': True,
                     'input_shape': None,
                     'activation': 'tanh',
                     'padding_mode': "REFLECT",
                     'data_format': None
                     }

    def __init__(self, params, model_type='3D'):
        super(TransfermerEncoderDecoder, self).__init__()
        self.params = dict(TransfermerEncoderDecoder.defaultparams, **params)
        self.use_spatial_kernel = self.params['use_spatial_kernel']
        self.use_local_l2 = self.params['use_local_l2']
        self.output_channel = self.params['output_channel']
        self.activation_function = self.params['activation']
        self.model_type = model_type.lower()
        self.bulid_network(self.params['input_shape'], self.params['basedim'])
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d

    def bulid_network(self, input_shape, basedim, model_type='3d'):
        inchl = input_shape[0]
        insize = input_shape[1::]
        self.begin_conv = nn.Sequential(self.ConvBlock(inchl, basedim, 7, stride=1, padding=3, padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        nn.ReLU())

        self.encoding_block = [nn.Sequential(
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=2, padding=1,
                               padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU())
            for convidx in range(0, self.params['downdeepth'])]

        trans_dim = basedim * 2 ** self.params['downdeepth']
        self.trans_block = nn.Sequential(
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            ResidualBlock(trans_dim, trans_dim, 3, 1),
            # ResidualBlock(trans_dim, trans_dim, 3, 1),
            # ResidualBlock(trans_dim, trans_dim, 3, 1),
            # ResidualBlock(trans_dim, trans_dim, 3, 1),
            # ResidualBlock(trans_dim, trans_dim, 3, 1)
        )

        self.decoding_block = [
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, stride=1, padding=1,
                               padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                nn.ReLU())
            for convidx in range(0, self.params['downdeepth'])]

        self.end_conv = nn.Sequential(
            self.ConvBlock(basedim, self.params['output_channel'], 7, stride=1, padding=3, padding_mode='reflect'),
            self.InstanceNorm(basedim, affine=True))

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)

        o_c2 = self.trans_block(o_c1)

        for convidx in range(self.params['downdeepth'], 0, -1):
            o_c2 = self.decoding_block[convidx - 1](torch.concat((o_c2, feats[convidx]), dim=1))
            F.interpolate(o_c2, scale_factor=2, mode='nearest')

        o_c3 = self.end_conv(o_c2)
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]


class Trusteeship:
    modelmap = {'simpleunet': SimpleEncoderDecoder,
                'resunet': ResEncoderDecoder,
                'simple23dunet': Simple2DEncoder3DDecoder,
                'transunet': TransfermerEncoderDecoder,
                'standardunet': StandardUNet,
                 'simpleclassifier': SimpleClassifier,
                 'generaldiscriminator': GeneralDiscriminator,
                }

    def __init__(self, network_params, model_type, device="cpu"):
        self.module = self.modelmap[network_params['network']](network_params, model_type=model_type)
        self.device = device
        self.module.to(device)
        self.loss_fun = network_params['losses']
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)

    def toDevice(self, device):
        self.module.to(device)
        self.device = device

    def loss_function(self, pred, target, add_con=None):
        total_loss = torch.zeros(1, device=self.device)
        if 'dice' in self.loss_fun:
            total_loss += self._dice_loss_(pred[1], target)
        if 'crep' in self.loss_fun:
            total_loss += nn.functional.cross_entropy(pred[0], target)
        if 'mse' in self.loss_fun:
            total_loss += nn.functional.mse_loss(pred[1], target)
        if 'mae' in self.loss_fun:
            total_loss += nn.functional.l1_loss(pred[1], target)

        return total_loss

    def train_step(self, inputs, outputs):
        T1_raw_LR = inputs.to(self.device)
        coord = outputs.to(self.device)
        pred = self.module(T1_raw_LR)
        loss = self.loss_function(pred, coord)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred

    def extra_train_step(self, inputs, outputs, ext_funs=None):
        source = inputs.to(self.device)
        target = outputs.to(self.device)
        pred = self.module(source)
        loss = basic_loss_essamble(pred[1], target, self.loss_fun)
        for fid in ext_funs:
            if fid == 'adv':
                loss_fun = set(self.loss_fun) & {'msl', 'dis'}
            elif fid == 'cls':
                loss_fun = set(self.loss_fun) & {'fsl', 'cls'}
            elif fid == 'syn':
                loss_fun = set(self.loss_fun) & {'scl', 'cyc'}
            else:
                loss_fun = {}
            if len(loss_fun) > 0:
                ext_fun = ext_funs[fid]
                loss += extra_loss_essamble(loss_fun, ext_fun['fun'], ext_fun['const'], ext_fun['alter'], pred[1], ext_fun['tar'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred

    def eval_step(self, inputs, outputs):
        T1_raw_LR = inputs.to(self.device)
        coord = outputs.to(self.device)
        pred = self.module(T1_raw_LR)
        loss = self.loss_function(pred, coord)
        return loss, pred

    def infer_step(self, inputs):
        T1_raw_LR = inputs.to(self.device)
        pred = self.module(T1_raw_LR)
        return pred

    def _dice_loss_(self, pred, target):
        smooth = 1.
        coeff = 0
        for idx in range(0, pred.size(1)):
            m1, m2 = pred[:, idx], (target == idx)
            intersection = (m1 * m2).sum()
            coeff += (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
        return 1-coeff/pred.size(1)

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

    def state_dict(self):
        return self.module.state_dict()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

