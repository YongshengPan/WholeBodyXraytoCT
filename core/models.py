import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [(np.array(krnlsz) * 0 + 1) * half_dim] + [krnlsz] * 2
    else:
        outsz = [krnlsz]
    return tuple(outsz)


def getKernelbyType(model_type='3D'):
    if model_type == '3d':
        ConvBlock, InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        max_pool, avg_pool = nn.MaxPool3d, nn.AvgPool3d
    elif model_type == '2.5d':
        ConvBlock, InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        max_pool, avg_pool = nn.MaxPool3d, nn.AvgPool3d
    else:
        ConvBlock, InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
        max_pool, avg_pool = nn.MaxPool2d, nn.AvgPool2d
    return ConvBlock, InstanceNorm, max_pool, avg_pool

def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'sigmoid':
        output = torch.sigmoid(input)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input) #* (3-2.0*torch.relu(1-torch.relu(input*100)))
    else:
        output = input
    return output


class TriDis_Mix(nn.Module):
    def __init__(self, prob=0.5, alpha=0.1, eps=1.0e-6,):
        super(TriDis_Mix, self).__init__()

        self.prob = prob
        self.alpha = alpha
        self.eps = eps
        self.mu = []
        self.var = []

    def forward(self, x: torch.Tensor, ):
        shp = x.shape
        smpshp = [shp[0], shp[1]] + [1]*(len(shp)-2)
        if torch.rand(1) > self.prob:
            return x
        mu = torch.mean(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        var = torch.var(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        sig = torch.sqrt(var+self.eps)
        if (sig == 0).any():
            print(sig)
        x_normed = (x-mu.detach()) / sig.detach()
        mu_r = torch.rand(smpshp, device=mu.device)
        sig_r = torch.rand(smpshp, device=mu.device)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(smpshp)
        bern = torch.bernoulli(lmda).to(mu.device)
        mu_mix = mu_r*bern + mu * (1.0-bern)
        sig_mix = sig_r * bern + sig * (1.0 - bern)
        return x_normed * sig_mix + mu_mix


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
            model_type='3d', residualskip=False,  device=None, dtype=None):
        super(StackConvBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size//2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

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
        self.conv1 = self.ConvBlock(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride), padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(mid_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = self.ConvBlock(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1), padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.relu2(o_c2+short_cut_conv)
        else:
            out_res = self.relu2(o_c2)
        return out_res


class GlobalResUNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalResUNetGenerator, self).__init__()
        self.isresunet = True
        self.n_downsampling = n_downsampling

        self.begin_conv = StackConvBlock(input_nc, ngf, 3, 1, model_type='2d', residualskip=self.isresunet)
        self.encoding_block = nn.ModuleList([nn.Sequential(
            StackConvBlock(ngf * 2 ** convidx, ngf * 2 ** (convidx + 1), 3, 2, model_type='2d',
                           residualskip=self.isresunet)) for
            convidx in range(0, n_downsampling)])
        trans_dim = ngf * 2 ** n_downsampling
        self.trans_block = nn.Sequential(
            StackConvBlock(trans_dim, trans_dim, 1, 1, model_type='2d', residualskip=self.isresunet),
            StackConvBlock(trans_dim, trans_dim, 1, 1, model_type='2d', residualskip=self.isresunet),
        )
        self.decoding_block = nn.ModuleList([
            StackConvBlock(ngf * 2 ** (convidx + 2), ngf * 2 ** convidx, 3, 1, model_type='2d',
                           mid_channels=ngf * 2 ** (convidx + 1), residualskip=self.isresunet) for convidx in
            range(0, n_downsampling)
        ])
        self.end_conv = StackConvBlock(ngf * 2, ngf, 3, 1, model_type='2d', residualskip=self.isresunet)
        self.class_conv = nn.Conv2d(ngf, output_nc, 3, stride=1, dilation=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)

        o_c2 = self.trans_block(o_c1)
        for convidx in range(self.n_downsampling, 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=2)

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        o_cls = self.class_conv(o_c3)
        prob = nn.Tanh()(o_cls, )
        return prob


class LocalizationNetwork(nn.Module):
    def __init__(self, output_channel=2, model_type='3D',):
        super().__init__()
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d

        def extdim(self, krnlsz, halfdim=1):
            return self.extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

        self.flatten = nn.Flatten()
        self.feats_extraction = nn.Sequential(
            nn.Conv2d(1, 16, 7, 1, 3),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64*2, 1),
            # nn.InstanceNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(512, 10)
    )

    def forward(self, x):
        x = self.feats_extraction(x)
        x = torch.cat((x, torch.square(x)-torch.tensor(1.0)), 1)
        x = F.normalize(x, p=2, dim=1)
        logits = self.linear_relu_stack(x)
        return logits


class SimplePredication(nn.Module):
    def __init__(self, input_shape, in_channels, output_channel, basedim=8, model_type='3D', activation_function='sigmoid',
                 use_spatial_kernel=True, use_local_l2=True, use_second_order=True):
        super(SimplePredication, self).__init__()
        self.output_channel = output_channel
        self.activation_function = activation_function
        self.use_spatial_kernel = use_spatial_kernel
        self.use_second_order = use_second_order
        self.use_local_l2 = use_local_l2
        self.model_type = model_type.lower()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.basedim = basedim

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d

        self.bulid_network(in_channels, input_shape, basedim)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, in_channels, input_shape, basedim):
        self.conv1 = self.ConvBlock(in_channels, basedim, self.extdim(3), padding='same', padding_mode='reflect')
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
                featsize = np.prod([input_shape[0]//2**5, input_shape[1]//2**5, input_shape[2]])
            else:
                featsize = np.prod([isz // (2 ** 5) for isz in input_shape])
            if self.use_second_order:
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
            if self.use_second_order:
                x = torch.cat((o_c5, torch.square(o_c5)-torch.tensor(1.0)), 1)
            if self.use_local_l2:
                x = F.normalize(x, p=2, dim=1)
        else:
            x = self.avg_pool(x, x.size()[2:])
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_flat_features(x))
        logit = self.fc(x)
        prob = build_end_activation(logit, self.activation_function)
        return [logit, prob, [o_c1, o_c2, o_c3, o_c4, o_c5, ]]


class SegmentationNetwork(nn.Module):

    def __init__(self, in_channels, output_channel=2, basedim=8, downdeepth=2, model_type='3D', activation_function='sigmoid', use_max_pool=False, use_triD=True, isresunet=False, use_skip=True):
        super(SegmentationNetwork, self).__init__()
        self.output_channel = output_channel
        self.model_type = model_type.lower()
        self.downdeepth = downdeepth
        self.activation_function = activation_function
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        self.bulid_network(in_channels, basedim, downdeepth, output_channel, self.model_type)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

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

    def bulid_network(self, in_channels, basedim, downdeepth=2, output_channel=2, model_type='3d'):
        self.begin_conv = nn.Sequential(self.ConvBlock(in_channels, basedim, self.extdim(3), stride=self.extdim(1), padding=self.extdim(3, 0), padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        nn.ReLU())
        self.encoding_block = nn.Sequential()
        for convidx in range(0, downdeepth):

            if self.use_max_pool:
                self.encoding_block.append(
                    nn.Sequential(StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                                  self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), self.extdim(3),  stride=self.extdim(1), padding=self.extdim(1, 0), padding_mode='reflect'),
                                  self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                                  self.MaxPool(self.extdim(3), self.extdim(2), padding=self.extdim(1, 0)),
                                  nn.ReLU()))
            else:
                self.encoding_block.append(
                    nn.Sequential(StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                        self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), self.extdim(3), stride=self.extdim(2), padding=self.extdim(1, 0), padding_mode='reflect'),
                        self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                        nn.ReLU()))
        trans_dim = basedim * 2 ** downdeepth
        self.trans_block = nn.Sequential(
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
        )
        self.decoding_block = nn.Sequential()
        for convidx in range(0, downdeepth):
            self.decoding_block.append(
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx), affine=True),
                nn.ReLU()))
        self.end_conv = nn.Sequential(self.ConvBlock(basedim*2, basedim, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                                      self.InstanceNorm(basedim, affine=True),
                                      self.ConvBlock(basedim, output_channel, self.extdim(3), stride=1, padding=self.extdim(3, 0), padding_mode='reflect'),
                                      self.InstanceNorm(output_channel, affine=True)
                                      )

    def forward(self, x, classes=None):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            if self.training and self.use_triD:
                o_c1 = self.tridis_mix(o_c1)
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)
        o_c2 = self.trans_block(o_c1)

        for convidx in range(self.downdeepth, 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="nearest")

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        prob = build_end_activation(o_c3, self.activation_function)
        # prob = F.softmax(o_c3, dim=1)
        return [o_c3, prob, feats]


class ResEncoderDecoder(nn.Module):
    def __init__(self, in_channels, output_channel=2, basedim=8, downdepth=2, model_type='3D', isresunet=True,
                 istransunet=False, activation_function='sigmoid', use_max_pool=False, use_attention=False,
                 use_triD=True, use_skip=True):
    # defaultparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
    #     'downdeepth': 2,
    #     'use_skip': True,
    #     'input_shape': None,
    #     'activation': 'tanh',
    #     'padding_mode': "REFLECT",
    #     'data_format': None
    # }
        super(ResEncoderDecoder, self).__init__()

        self.output_channel = output_channel
        self.model_type = model_type.lower()
        self.downdepth = downdepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        self.use_attention = use_attention

        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.max_pool, self.avg_pool = F.max_pool2d, F.avg_pool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.max_pool, self.avg_pool = F.max_pool3d, F.avg_pool3d

        self.bulid_network(in_channels, basedim)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, in_channels, basedim):
        inchl = in_channels

        self.begin_conv = nn.Sequential(self.ConvBlock(inchl, basedim, self.extdim(7), stride=self.extdim(1), padding=self.extdim(3, 0), padding_mode='reflect'),
                                        self.InstanceNorm(basedim, affine=True),
                                        )

        self.encoding_block = nn.Sequential()
        for convidx in range(0, self.downdepth):
            self.encoding_block.append(
                nn.Sequential(
                    StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** convidx, 3, 1, model_type=self.model_type, ),
                    self.ConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), self.extdim(3), stride=self.extdim(2), padding=self.extdim(1, 0), padding_mode='reflect'),
                    self.InstanceNorm(basedim * 2 ** (convidx + 1), affine=True),
                    ))

        trans_dim = basedim * 2 ** self.downdepth
        self.trans_block = nn.Sequential(
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type,),
        )

        self.decoding_block = nn.Sequential()
        for convidx in range(0, self.downdepth):
            self.decoding_block.append(
            nn.Sequential(
                self.ConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (convidx), affine=True),
                nn.ReLU()))

        self.end_conv = nn.Sequential(self.ConvBlock(basedim*2, basedim, self.extdim(3), stride=1, padding=self.extdim(1, 0), padding_mode='reflect'),
                                      self.InstanceNorm(basedim, affine=True),
                                      self.ConvBlock(basedim, self.output_channel, self.extdim(7), stride=1, padding=self.extdim(3, 0), padding_mode='reflect'),
                                      self.InstanceNorm(self.output_channel, affine=True)
                                      )

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](F.relu(o_c1))
            feats.append(o_c1)
        o_c2 = self.trans_block(F.relu(o_c1))

        for convidx in range(self.downdepth, 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)

            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.upsample(o_c2, scale_factor=self.extdim(2), mode="nearest")

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        prob = build_end_activation(o_c3, self.activation_function)
        return [o_c3, prob, feats]



class Generic_UNetwork(nn.Module):
    def __init__(self, in_channels, output_channel=2, basedim=8, downdepth=2, model_type='3D', isresunet=True,
                 istransunet=False, activation_function='sigmoid', use_max_pool=False, use_attention=False, use_triD=False, use_skip=True):
        super(Generic_UNetwork, self).__init__()
        self.output_channel = output_channel
        self.model_type = model_type.lower()
        self.downdepth = downdepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        self.use_attention = use_attention
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d

        self.tridis_mix = TriDis_Mix(prob=0.5, alpha=0.1, eps=1.0e-6)
        self.bulid_network(in_channels, basedim, downdepth, output_channel)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, in_channels, basedim, downdepth=2, output_channel=2):
        self.begin_conv = StackConvBlock(in_channels, basedim, 7, 1, model_type=self.model_type, residualskip=self.isresunet)
        if self.use_max_pool:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                self.MaxPool(self.extdim(3), self.extdim(2), padding=self.extdim(1, 0)),
                StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 1,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdepth)])
        else:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdepth)])

        trans_dim = basedim * 2 ** downdepth
        if self.istransunet:
            self.trans_block = nn.Sequential(nn.TransformerEncoder(nn.TransformerEncoderLayer(trans_dim, 8), 12))
        else:
            self.trans_block = nn.Sequential(
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
            )

        self.decoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, 3, 1, model_type=self.model_type,
                           mid_channels=basedim * 2 ** (convidx + 1), residualskip=self.isresunet) for convidx in range(0, downdepth)
        ])
        self.end_conv = StackConvBlock(basedim * 2, basedim, 3, 1, model_type=self.model_type, residualskip=self.isresunet)
        self.class_conv = self.ConvBlock(basedim, output_channel, self.extdim(7), stride=1, dilation=1, padding=self.extdim(3, 0), padding_mode='reflect')

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            if self.training and self.use_triD:
                o_c1 = self.tridis_mix(o_c1)
            o_c1 = self.encoding_block[convidx](o_c1)
            # x = F.interpolate(x, scale_factor=self.extdim(1/2), mode="trilinear")
            feats.append(o_c1)
        if self.istransunet:
            o_c2 = torch.transpose(o_c1.view([*o_c1.size()[0:2], -1]), 1, 2)
            o_c2 = self.trans_block(o_c2)
            o_c2 = torch.transpose(o_c2, 1, 2).view(o_c1.size())
        else:
            o_c2 = self.trans_block(o_c1)

        for convidx in range(self.downdepth, 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="trilinear")

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        o_cls = self.class_conv(o_c3)
        prob = build_end_activation(o_cls, self.activation_function)

        return [o_cls, prob, ]


class MultimodalityNetwork(nn.Module):
    def __init__(self, modality_channel, latent_channel=2, basedim=8, downdepth=2, unetdepth=2, model_type='3D', isresunet=True,
                 istransunet=False, activation_function='sigmoid', use_max_pool=False, use_attention=False, use_triD=False):
        super(MultimodalityNetwork, self).__init__()
        self.latent_channel = latent_channel
        self.model_type = model_type.lower()
        self.downdepth = downdepth
        self.unetdepth = unetdepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        self.use_attention = use_attention
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d

        self.tridis_mix = TriDis_Mix(prob=0.5, alpha=0.1, eps=1.0e-6)
        self.bulid_network(modality_channel, latent_channel, basedim, downdepth, unetdepth)
        self.affine_theta = AffineRegister(model_type=self.model_type, inchl=latent_channel * 2, basedim=8,)
        self.inten_shift = IntensityShift(model_type=self.model_type, inchl=modality_channel, basedim=8, )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_unet_kernel(self, basedim=8, depth=2, skipopt='concat'):
        encoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** (idcv + 1), 3, 2,
                           model_type=self.model_type, residualskip=self.isresunet) for idcv in range(0, depth)])
        trans_dim = basedim * 2 ** depth
        trans_block = nn.Sequential(
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type, residualskip=self.isresunet),
            StackConvBlock(trans_dim, trans_dim, 3, 1, model_type=self.model_type, residualskip=self.isresunet),
        )
        dimplus = 1 if skipopt == 'concat' else 0
        decoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** (idcv + dimplus), basedim * 2 ** (idcv - 1), 3, 1, model_type=self.model_type,
                           mid_channels=basedim * 2 ** idcv, residualskip=self.isresunet) for idcv in range(depth, 0, -1)])
        return nn.ModuleList([encoding_block, decoding_block, trans_block])

    def bulid_network(self, modal_channel, latent_channel=1,  basedim=8, downdepth=2, UBdepth=2):
        self.begin_conv = nn.Sequential(StackConvBlock(modal_channel, basedim, 3, 1, model_type=self.model_type, residualskip=self.isresunet),
                                        *[StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** (idcv + 1), 3, 2,
                                                         model_type=self.model_type, residualskip=self.isresunet) for
                                          idcv in range(0, downdepth)])

        self.forward_process = self.bulid_unet_kernel(basedim * 2 ** (downdepth+1), UBdepth)
        self.backward_process = self.bulid_unet_kernel(basedim * 2 ** downdepth, UBdepth)
        self.translate_process = self.bulid_unet_kernel(basedim * 2 ** (downdepth+1), UBdepth, skipopt='add')

        self.end_conv = nn.Sequential(*[nn.Sequential(StackConvBlock(basedim * 2 ** (idcv+2), basedim * 2 ** (idcv+1), 3, 1,
                                                                     model_type=self.model_type, residualskip=self.isresunet),
                                                      nn.Upsample(scale_factor=self.extdim(2), mode='trilinear', )) for idcv in range(downdepth, 0, -1)],
                                      nn.Sequential(self.ConvBlock(basedim * 4, modal_channel, self.extdim(5), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect'),
                                                    self.InstanceNorm(modal_channel, affine=True),
                                                    ))

        self.latent2process = StackConvBlock(latent_channel, basedim * 2 ** (downdepth+1), 5, 1, model_type=self.model_type, residualskip=self.isresunet)
        self.process2latent = StackConvBlock(basedim * 2 ** (downdepth+1), latent_channel, 5, 1, model_type=self.model_type, residualskip=self.isresunet)
        # self.latent2process = self.ConvBlock(latent_channel, basedim * 2 ** downdepth, self.extdim(5), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect')
        # self.process2latent = nn.Sequential(self.ConvBlock(basedim * 2 ** (downdepth + 1), latent_channel, self.extdim(5), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect'),
        #                                     self.InstanceNorm(latent_channel, affine=True),
        #                                     nn.Softmax())
        # self.discriminator = AdverserialNetwork(modal_channel, basedim, 3)

    def forward(self, x, stage='any', **kwargs):
        if stage.lower() == 'forward':
            x = self.latent2process(x)
            x = self.inferUkernel(x, *self.forward_process)
            x = self.end_conv(x)
        elif stage.lower() == 'backward':
            x = self.begin_conv(x)
            x = self.inferUkernel(x, *self.backward_process)
            x = self.process2latent(x)
            return x
        elif stage.lower() == 'translation':
            x = self.latent2process(x)
            x = self.inferUkernel(x, *self.translate_process, skipopt='sum')
            x = self.process2latent(x)
            return x
        elif stage.lower() == 'end':
            x = self.end_conv(x)
        else:
            x = self.begin_conv(x)
            x = self.inferUkernel(x, *self.backward_process)
            x = self.process2latent(x)
            x = build_end_activation(x, self.activation_function)
            x = self.latent2process(x)
            x = self.inferUkernel(x, *self.forward_process)
            x = self.end_conv(x)
        prob = build_end_activation(x, self.activation_function)
        return [x, prob, ]

    def inferUkernel(self, x, en_block, de_block, tr_block, skipopt='concat'):
        feats = [x, ]
        for idcv in range(0, len(en_block)):
            if self.training and self.use_triD:
                x = self.tridis_mix(x)
            x = en_block[idcv](x)
            feats.append(x)
        x = tr_block(x)
        for idcv in range(0, len(de_block)):
            x = torch.concat((x, feats[-idcv-1]), dim=1) if skipopt == 'concat' else x + feats[-idcv-1]
            x = de_block[idcv](x)
            x = F.interpolate(x, scale_factor=self.extdim(2), mode="trilinear")
        return torch.concat((x, feats[0]), dim=1) if skipopt == 'concat' else x + feats[0]


class AdverserialResidualNetwork(nn.Module):
    def __init__(self, in_channel, basedim=8, downdepth=2, model_type='3D', activation_function=None):
        super(AdverserialResidualNetwork, self).__init__()
        self.model_type = model_type.lower()
        self.activation_function = activation_function
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d

        self.adverserial_network = self.bulid_network(in_channel, basedim, downdepth)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, channel, basedim=8, depth=3):
        adverserial_network = nn.Sequential(
            StackConvBlock(channel, basedim, 3, 2, model_type=self.model_type, residualskip=True),
            *[StackConvBlock(basedim * 2 ** dpt, basedim * 2 ** (dpt + 1), 3, 2, model_type=self.model_type, residualskip=True)
              for dpt in range(0, depth)],
            self.ConvBlock(basedim * 2 ** depth, 1, self.extdim(4), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect'))
        return adverserial_network

    def forward(self, x, **kwargs):
        features = []
        for layer in self.adverserial_network:
            x = layer(x)
            features.append(x)
        prob = build_end_activation(x, self.activation_function)
        return prob, features


class AdverserialNetwork(nn.Module):
    def __init__(self, in_channel, basedim=8, downdepth=2, model_type='3D', activation_function=None):
        super(AdverserialNetwork, self).__init__()
        self.model_type = model_type.lower()
        self.activation_function = activation_function
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d

        self.adverserial_network = self.bulid_network(in_channel, basedim, downdepth)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, channel, basedim=8, depth=3):
        adverserial_network = nn.Sequential(
            nn.Sequential(self.ConvBlock(channel, basedim, self.extdim(4), stride=2, dilation=1,
                                         padding=self.extdim(2, 0), padding_mode='reflect'),
                          nn.LeakyReLU()),
            *[nn.Sequential(
                self.ConvBlock(basedim * 2 ** dpt, basedim * 2 ** (dpt + 1), self.extdim(4), stride=2, dilation=1,
                               padding=self.extdim(2, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (dpt + 1)),
                nn.LeakyReLU()
            ) for dpt in range(0, depth)],
            self.ConvBlock(basedim * 2 ** depth, 1, self.extdim(4), stride=1, dilation=1,
                               padding=self.extdim(2, 0), padding_mode='reflect'))
        return adverserial_network

    def forward(self, x, **kwargs):
        features = []
        for layer in self.adverserial_network:
            x = layer(x)
            features.append(x)
        prob = build_end_activation(x, self.activation_function)
        return prob, features


class AffineRegister(nn.Module):
    def __init__(self, model_type='3d', inchl=1, basedim=8, channel_wise=False, name='affine'):
        super(AffineRegister, self).__init__()
        self.channel_wise = channel_wise
        self.model_type = model_type.lower()
        self.basedim=basedim
        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
        self.ConvBlock, self.InstanceNorm, self.max_pool, self.avg_pool = getKernelbyType(model_type=model_type)
        self.localization = nn.Sequential(
            self.ConvBlock(inchl, basedim, extdim(3, 3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 1, basedim * 2, extdim(3, 3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 2, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 2, basedim * 4, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 4, basedim * 4, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 4, basedim * 4, extdim(3, 3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        if self.model_type in ('3d', '2.5d'):
            self.fc = nn.Linear(basedim * 4, 12)
            self.fc.weight.data.zero_()
            self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc = nn.Linear(basedim * 4, 6)
            self.fc.weight.data.zero_()
            self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.localization(x)
        xs = x.view(-1, self.basedim * 4)
        theta = self.fc(xs)
        if self.model_type in ('3d', '2.5d'):
            theta = theta.view(-1, 3, 4)
            thetaA= torch.zeros([1, 4, 4], dtype=theta.dtype, device=theta.device)
        else:
            theta = theta.view(-1, 2, 3)
            thetaA = torch.zeros([1, 3, 3], dtype=theta.dtype, device=theta.device)
        thetaA[:, 0:-1, :] = theta
        thetaA[:, -1, -1] = 1
        theta_inv = thetaA.inverse()[:, 0:-1, :]
        return theta, theta_inv
        # return F.log_softmax(x, dim=1)

class IntensityShift(nn.Module):
    def __init__(self, model_type='3d', inchl=1, basedim=8, channel_wise=False, name='affine'):
        super(IntensityShift, self).__init__()
        self.channel_wise = channel_wise
        self.model_type = model_type.lower()
        self.basedim=basedim
        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
        self.ConvBlock, self.InstanceNorm, self.max_pool, self.avg_pool = getKernelbyType(model_type=model_type)
        self.localization = nn.Sequential(
            self.ConvBlock(inchl, basedim, extdim(3, 3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 1, basedim * 2, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 2, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 2, basedim * 4, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 4, basedim * 4, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            self.max_pool(kernel_size=extdim(3), stride=extdim(2), padding=extdim(1, 0)),
            self.ConvBlock(basedim * 4, basedim * 4, extdim(3,3), padding=extdim(1, 1), padding_mode='reflect'),
            self.InstanceNorm(basedim * 4, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(basedim * 4, 2)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0], dtype=torch.float))


    def forward(self, x):
        x = self.localization(x)
        xs = x.view(-1, self.basedim * 4)
        theta = self.fc(xs)
        return theta
        # return F.log_softmax(x, dim=1)

class MultimodalityHalfNetwork(nn.Module):
    def __init__(self, modality_channel, latent_channel=2, basedim=8, downdepth=2, unetdepth=2, model_type='3D', isresunet=True,
                 istransunet=False, activation_function='sigmoid', use_max_pool=False, use_attention=False, use_triD=False):
        super(MultimodalityHalfNetwork, self).__init__()
        self.latent_channel = latent_channel
        self.model_type = model_type.lower()
        self.downdepth = downdepth
        self.unetdepth = unetdepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        self.use_attention = use_attention
        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d

        self.tridis_mix = TriDis_Mix(prob=0.5, alpha=0.1, eps=1.0e-6)
        self.bulid_network(modality_channel, latent_channel, basedim, downdepth, unetdepth)
        self.affine_theta = AffineRegister(model_type=self.model_type, inchl=modality_channel, basedim=8,)
        self.inten_shift = IntensityShift(model_type=self.model_type, inchl=modality_channel, basedim=8, )
        # self.adverserial = AdverserialNetwork(modality_channel, basedim, downdepth=3, model_type=model_type, activation_function=None),

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_unet_kernel(self, basedim=8, depth=2, skipopt='concat'):
        encoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** (idcv + 1), 3, 2,
                           model_type=self.model_type, residualskip=self.isresunet) for idcv in range(0, depth)])
        trans_dim = basedim * 2
        trans_block = nn.ModuleList([nn.Sequential(
            StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** idcv, 3, 1, model_type=self.model_type, residualskip=self.isresunet),
            StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** idcv, 3, 1, model_type=self.model_type, residualskip=self.isresunet),
        ) for idcv in range(0, depth+1)])
        dimplus = 1 if skipopt == 'concat' else 0
        decoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** (idcv + dimplus), basedim * 2 ** (idcv - 1), 3, 1, model_type=self.model_type,
                           mid_channels=basedim * 2 ** idcv, residualskip=self.isresunet) for idcv in range(depth, 0, -1)])
        return nn.ModuleList([encoding_block, decoding_block, trans_block])

    def bulid_network(self, modal_channel, latent_channel=1,  basedim=8, downdepth=2, UBdepth=2):
        self.begin_conv = nn.Sequential(StackConvBlock(modal_channel, basedim, 5, 1, model_type=self.model_type, residualskip=self.isresunet),
                                        *[StackConvBlock(basedim * 2 ** idcv, basedim * 2 ** (idcv + 1), 3, 2,
                                                         model_type=self.model_type, residualskip=self.isresunet) for
                                          idcv in range(0, downdepth)])

        process = self.bulid_unet_kernel(basedim * 2 ** downdepth, UBdepth)
        self.forward_process = nn.ModuleList([None, process[1], process[2]])
        self.backward_process = nn.ModuleList([process[0], None, None])
        self.translate_process = nn.ModuleList([None, None, process[2]])

        self.end_conv = nn.Sequential(*[nn.Sequential(StackConvBlock(basedim * 2 ** (idcv+2), basedim * 2 ** (idcv+1), 3, 1,
                                                                     model_type=self.model_type, residualskip=self.isresunet),
                                                      nn.Upsample(scale_factor=self.extdim(2), mode='trilinear', )) for idcv in range(downdepth, 0, -1)],
                                      nn.Sequential(self.ConvBlock(basedim * 2, modal_channel, self.extdim(5, 3), stride=1, dilation=1, padding=self.extdim(2, 1), padding_mode='reflect'),
                                                    self.InstanceNorm(modal_channel, affine=True),
                                                    ))

        self.latent2process = StackConvBlock(latent_channel, basedim * 2 ** (downdepth+1), 5, 1, model_type=self.model_type, residualskip=self.isresunet)
        self.process2latent = StackConvBlock(basedim * 2 ** (downdepth+1), latent_channel, 5, 1, model_type=self.model_type, residualskip=self.isresunet)
        # self.latent2process = self.ConvBlock(latent_channel, basedim * 2 ** downdepth, self.extdim(5), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect')
        # self.process2latent = nn.Sequential(self.ConvBlock(basedim * 2 ** (downdepth + 1), latent_channel, self.extdim(5), stride=1, dilation=1, padding=self.extdim(2, 0), padding_mode='reflect'),
        #                                     self.InstanceNorm(latent_channel, affine=True),
        #                                     nn.Softmax())
        # self.discriminator = AdverserialNetwork(modal_channel, basedim, 3)

    def forward(self, x, stage='any', **kwargs):
        if stage.lower() == 'forward':
            x = self.inferUkernel(x, *self.forward_process, )
            x = self.end_conv(x)
        elif stage.lower() == 'backward':
            x = self.begin_conv(x)
            x = self.inferUkernel(x, *self.backward_process, )
            return x
        elif stage.lower() == 'translation':
            x = self.inferUkernel(x, *self.translate_process, skipopt='sum')
            return x
        elif stage.lower() == 'end':
            x = self.end_conv(x)
        else:
            x = self.begin_conv(x)
            x = self.inferUkernel(x, *self.backward_process)
            x = self.inferUkernel(x, *self.forward_process)
            x = self.end_conv(x)
        prob = build_end_activation(x, self.activation_function)
        return [x, prob, ]

    def inferUkernel(self, x, en_block=None, de_block=None, tr_block=None, skipopt='concat'):
        if en_block is None:
            feats = x
        else:
            feats = [x, ]
            for idcv in range(0, len(en_block)):
                if self.training and self.use_triD:
                    x = self.tridis_mix(x)
                x = en_block[idcv](x)
                feats.append(x)
        x = feats[-1]
        if tr_block is not None:
            feats = [tr_block[idx](feats[idx]) for idx in range(len(feats))]
            # for idx in range(len(feats)):
            #     feats[idx] = tr_block[idx](feats[idx])
        if de_block is None:
            return feats
        else:
            for idcv in range(0, len(de_block)):
                x = torch.concat((x, feats[-idcv-1]), dim=1) if skipopt == 'concat' else x + feats[-idcv-1]
                x = de_block[idcv](x)
                x = F.interpolate(x, scale_factor=self.extdim(2), mode="trilinear")
            return torch.concat((x, feats[0]), dim=1) if skipopt == 'concat' else x + feats[0]



