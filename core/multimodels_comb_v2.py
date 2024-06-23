import csv
import time
import random

import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

from .models_comb import *
from .losses import *


class MultiModels(object):
    def __init__(self, database,
                 output_path,
                 losses=('dis', 'fcl'),
                 subdir=None,
                 model_type='2D',
                 model_task='synthesis',
                 basedim=16,
                 batchsize=10,
                 numcls=17,
                 numchs=None,
                 training_modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'),
                 network_params=None,
                 fuse_from_logit=True,
                 max_num_images=100000,
                 cls_num_epoch=100,
                 syn_num_epoch=2000,
                 learning_rate=0.001):
        self.training_modules = training_modules
        self.clsA_params = {
            'network': 'simpleclassifier',
            'input_shape': (224, 224, 1),
            'activation': 'softmax',
            'output_channel': numcls,
            'basedim': basedim * 2,
            'input_alter': ['low_PET', ],
            'input_const': ['std_PET', ],
            'output': ['inputC', ],
            'use_spatial_kernel': True,
            'use_syn': False,
            'optimizer': 'adam'
        }
        self.clsB_params = {
            'network': 'simpleclassifier',
            'input_shape': (224, 224, 3),
            'activation': 'softmax',
            'output_channel': numcls,
            'basedim': basedim * 2,
            'input_alter': ['low_PET', ],
            'input_const': ['std_PET', ],
            'output': ['inputC', ],
            'use_spatial_kernel': True,
            'use_syn': False,
            'optimizer': 'adam'
        }
        self.synA_params = {
            'network': 'simpleunet',
            'input_shape': (224, 224, 3),
            'activation': 'tanh',
            'task_type': 'synthesis',
            'output_channel': numcls,
            'input_alter': ['low_PET', ],
            'input_const': [],
            'output': ['CT', ],
            'basedim': basedim * 2,
            'losses': losses,
            'use_fake': 0,
            'optimizer': 'adam'
        }
        self.synB_params = {
            'network': 'simpleunet',
            'input_shape': (224, 224, 1),
            'activation': 'tanh',
            'task_type': 'synthesis',
            'output_channel': numcls,
            'input_alter': [],
            'input_const': ['low_PET', ],
            'output': ('std_PET',),
            'basedim': basedim,
            'losses': losses,
            'use_fake': 0,
            'optimizer': 'adam',
        }
        self.advA_params = {
            'network': 'generaldiscriminator',
            'input_shape': (224, 224, 4),
            'activation': None,
            'output_channel': 1,
            'basedim': basedim * 2,
            'input_alter': ['CT', ],
            'input_const': ['low_PET', ],
            'output': None,
            'model_type': 'condition',
            'optimizer': 'adam'
        }
        self.advB_params = {
            'network': 'generaldiscriminator',
            'input_shape': (224, 224, 4),
            'activation': None,
            'output_channel': 1,
            'input_alter': ['std_PET', ],
            'input_const': ['low_PET', ],
            'output': None,
            'basedim': basedim * 2,
            'model_type': 'condition',
            'optimizer': 'adam'
        }
        if network_params is not None:
            if 'clsA' in network_params:
                self.clsA_params.update(network_params['clsA'])
                num_of_inchl = np.sum([numchs[md] for md in self.clsA_params['input_alter']+self.clsA_params['input_const']])
                self.clsA_params['input_shape'] = [num_of_inchl]+[num for num in self.clsA_params['input_shape']]
                num_of_outchl = np.sum([numchs[md] for md in self.clsA_params['output']])
                self.clsA_params['output_channel'] = num_of_outchl
            if 'clsB' in network_params:
                self.clsB_params.update(network_params['clsB'])
                num_of_inchl = np.sum([numchs[md] for md in self.clsB_params['input_alter'] + self.clsB_params['input_const']])
                self.clsB_params['input_shape'] = [num_of_inchl]+[num for num in self.clsB_params['input_shape']]
                num_of_outchl = np.sum([numchs[md] for md in self.clsB_params['output']])
                self.clsB_params['output_channel'] = num_of_outchl
            if 'synA' in network_params:
                # print(network_params['synA'])
                self.synA_params.update(network_params['synA'])
                num_of_inchl = np.sum([numchs[md] for md in self.synA_params['input_alter'] + self.synA_params['input_const']])
                self.synA_params['input_shape'] = [num_of_inchl]+[num for num in self.synA_params['input_shape']]
                num_of_outchl = np.sum([numchs[md] for md in self.synA_params['output']])
                self.synA_params['output_channel'] = num_of_outchl
            if 'synB' in network_params:
                self.synB_params.update(network_params['synB'])
                num_of_inchl = np.sum([numchs[md] for md in self.synB_params['input_alter'] + self.synB_params['input_const']])
                self.synB_params['input_shape'] = [num_of_inchl]+[num for num in self.synB_params['input_shape']]
                num_of_outchl = np.sum([numchs[md] for md in self.synB_params['output']])
                self.synB_params['output_channel'] = num_of_outchl
            if 'advA' in network_params:
                self.advA_params.update(network_params['advA'])
                num_of_inchl = np.sum([numchs[md] for md in self.advA_params['input_alter'] + self.advA_params['input_const']])
                self.advA_params['input_shape'] = [num_of_inchl]+[num for num in self.advA_params['input_shape']]
                # num_of_outchl = np.sum([numchs[md] for md in self.advA_params['output']])
                # self.advA_params['output_channel'] = num_of_outchl
            if 'advB' in network_params:
                self.advB_params.update(network_params['advB'])
                num_of_inchl = np.sum([numchs[md] for md in self.advB_params['input_alter'] + self.advB_params['input_const']])
                self.advB_params['input_shape'] = [num_of_inchl]+[num for num in self.advB_params['input_shape']]
                # num_of_outchl = np.sum([numchs[md] for md in self.advB_params['output']])
                # self.advB_params['output_channel'] = num_of_outchl
        self.basedim = basedim
        self.batchsize = batchsize
        self.numcls = numcls
        self.numchs = numchs
        self.model_type = model_type.lower()
        self.model_task = model_task
        self.learning_rate = learning_rate
        self.max_num_images = max_num_images
        self.cls_num_epoch = cls_num_epoch
        self.syn_num_epoch = syn_num_epoch
        self.output_path = output_path
        self.losses = tuple(sorted(losses))
        if subdir is None:
            self.subdir = self.losses
        else:
            self.subdir = subdir
        check_dir = self.output_path + "/multimodels_v1/"
        self.result_path = os.path.join(output_path, "{0}/mask/".format(''.join(self.subdir)))
        self.sample_path = os.path.join(output_path, "{0}/samples/".format(''.join(self.subdir)))
        chkpt_format = check_dir + '{loss}'
        self.chkpt_syn_fname = chkpt_format.format(loss=self.subdir)
        self.chkpt_cls_fname = chkpt_format.format(loss=self.subdir)

        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        # if not os.path.exists(self.result_path_mask):
        #     os.makedirs(self.result_path_mask)
        if not os.path.exists(self.chkpt_syn_fname):
            os.makedirs(self.chkpt_syn_fname)
        if not os.path.exists(self.chkpt_cls_fname):
            os.makedirs(self.chkpt_cls_fname)
        self.database = database
        self.databaseV = copy.deepcopy(database)
        self.databaseV.change_run_stage('valid')
        self.databaseT = copy.deepcopy(database)
        self.databaseT.change_run_stage('test')
        self.dataloader = DataLoader(self.database, batch_size=batchsize)
        self.dataloaderV = DataLoader(self.databaseV, batch_size=batchsize)
        self.dataloaderT = DataLoader(self.databaseT, batch_size=batchsize)
        self.iteration_modules = self.iteration_modules_v2

    def model_setup(self):
        self.clsAs = [Trusteeship(self.clsA_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]
        self.clsBs = [Trusteeship(self.clsB_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]
        self.synAs = [Trusteeship(self.synA_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]
        self.synBs = [Trusteeship(self.synB_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]
        self.advAs = [Trusteeship(self.advA_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]
        self.advBs = [Trusteeship(self.advB_params, model_type=self.model_type) for idx in
                      range(len(self.database.train_combinations))]

    def model_save(self, globel_epoch=0, modules=('cls', 'syn')):
        network_modules = {'clsA': self.clsAs, 'clsB': self.clsBs, 'synA': self.synAs, 'synB': self.synBs, 'advA': self.advAs, 'advB': self.advBs}
        if 'cls' in modules:
            modules = set(modules) | {'clsA', 'clsB'}
        if 'syn' in modules:
            modules = set(modules) | {'synA', 'synB', 'advA', 'advB'}
        for idx in range(len(self.database.train_combinations)):
            for modname in network_modules:
                if modname in modules:
                    torch.save(network_modules[modname][idx].state_dict(), self.chkpt_cls_fname + '/{0}_model{1}_{2}.h5'.format(modname, idx, globel_epoch))

    def model_load(self, globel_epoch=0, modules=('cls', 'syn'), by_name=False, skip_mismatch=False):
        if 'cls' in modules:
            modules = set(modules) | {'clsA', 'clsB'}
        if 'syn' in modules:
            modules = set(modules) | {'synA', 'synB', 'advA', 'advB'}

        network_modules = {'clsA': self.clsAs, 'clsB': self.clsBs, 'synA': self.synAs, 'synB': self.synBs,
                           'advA': self.advAs, 'advB': self.advBs}
        for idx in range(len(self.database.train_combinations)):
            for modname in ['clsA', 'clsB', 'synA', 'synB', 'advA', 'advB']:
                if modname in modules:
                    state_dict = torch.load(self.chkpt_cls_fname + '/{0}_model{1}_{2}.h5'.format(modname, idx, globel_epoch))
                    network_modules[modname][idx].load_state_dict(state_dict)

    def iteration_modules_v2(self, inputs, cv_index,
                             modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'), outkeys=None, flags=None):

        def kconcat(inputs, axis=1):
            inputs = [torch.Tensor(var).to('cuda') for var in inputs if var is not None]
            return None if len(inputs) == 0 else torch.concat(inputs, dim=axis)

        def npconcat(inputs, axis=-1):
            inputs = [var for var in inputs if var is not None]
            return None if len(inputs) == 0 else np.concatenate(inputs, axis=axis)

        attributes = {'labelA': npconcat([inputs[key] for key in self.clsA_params['output']], axis=-1), 'refA': npconcat([inputs[key] for key in self.synA_params['output']], axis=-1),
                      'labelB': npconcat([inputs[key] for key in self.clsB_params['output']], axis=-1), 'refB': npconcat([inputs[key] for key in self.synB_params['output']], axis=-1)}

        clsA_alter = kconcat([inputs[ind] for ind in self.clsA_params['input_alter']], axis=1)
        clsA_const = kconcat([inputs[ind] for ind in self.clsA_params['input_const']], axis=1)

        clsB_alter = kconcat([inputs[ind] for ind in self.clsB_params['input_alter']], axis=1)
        clsB_const = kconcat([inputs[ind] for ind in self.clsB_params['input_const']], axis=1)

        advA_alter = kconcat([inputs[ind] for ind in self.advA_params['input_alter']], axis=1)
        advA_const = kconcat([inputs[ind] for ind in self.advA_params['input_const']], axis=1)
        advB_alter = kconcat([inputs[ind] for ind in self.advB_params['input_alter']], axis=1)
        advB_const = kconcat([inputs[ind] for ind in self.advB_params['input_const']], axis=1)

        synB_alter = kconcat([inputs[ind] for ind in self.synB_params['input_alter']], axis=1)
        synB_const = kconcat([inputs[ind] for ind in self.synB_params['input_const']], axis=1)
        synB_tar = kconcat([inputs[ind] for ind in self.synB_params['output']], axis=1)
        synA_alter = kconcat([inputs[ind] for ind in self.synA_params['input_alter']], axis=1)
        synA_const = kconcat([inputs[ind] for ind in self.synA_params['input_const']], axis=1)
        synA_tar = kconcat([inputs[ind] for ind in self.synA_params['output']], axis=1)



        if 'clsA' in modules:
            self.clsAs[cv_index].train()
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index].train_step(kconcat([clsA_alter, clsA_const], axis=1), attributes['labelA'])
            attributes.update({'logitA': clsA_logit.detach().to('cpu').numpy(), 'probA': clsA_prob.detach().to('cpu').numpy()})
        elif 'probA' in outkeys or 'logitA' in outkeys:
            self.clsAs[cv_index].eval()
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index].infer_step(kconcat([clsA_alter, clsA_const], axis=1))
            attributes.update({'logitA': clsA_logit.detach().to('cpu').numpy(), 'probA': clsA_prob.detach().to('cpu').numpy()})

        if 'clsB' in modules:
            self.clsBs[cv_index].train()
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index].train_step(kconcat([clsB_alter, clsB_const], axis=1), attributes['labelB'])
            attributes.update({'logitB': clsB_logit.numpy(), 'probB': clsB_prob.numpy()})
        elif 'probB' in outkeys or 'logitB' in outkeys:
            self.clsBs[cv_index].eval()
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index].infer_step(kconcat([clsB_alter, clsB_const], axis=1))
            attributes.update({'logitB': clsB_logit.detach().to('cpu').numpy(), 'probB': clsB_prob.detach().to('cpu').numpy()})

        if 'synA' in modules:
            self.synAs[cv_index].train()
            # synA_logit, synA_prob, _ = self.synAs[cv_index].train_step(kconcat([synA_alter, synA_const], axis=1), synA_tar)
            synA_logit, synA_prob, _ = self.synAs[cv_index].extra_train_step(kconcat([synA_alter, synA_const], axis=1), synA_tar,
                                                ext_funs={'adv': {'fun': self.advAs[cv_index].module, 'const': advA_const, 'alter': advA_alter, 'tar': torch.Tensor(1)},
                                                          'cls': {'fun': self.clsAs[cv_index].module, 'const': clsA_const, 'alter': clsA_alter, 'tar': inputs['label']},
                                                          'cyc': {'fun': self.synBs[cv_index].module, 'const': synB_const, 'alter': synB_alter, 'tar': synB_tar},
                                                          })
        else:
            self.synAs[cv_index].eval()
            synA_logit, synA_prob, _ = self.synAs[cv_index].infer_step(kconcat([synA_alter, synA_const], axis=1))
        attributes.update({'synA': synA_prob.detach().to('cpu').numpy()})

        if 'probAs' in outkeys:
            self.clsAs[cv_index].eval()
            cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index].infer_step(kconcat([synA_prob, clsA_const], axis=1))
            attributes.update({'probAs': cls_sA_prob.detach().to('cpu').numpy()})

        if 'advA' in modules:
            self.advAs[cv_index].train()
            conA_logit, _, conA_feat = self.advAs[cv_index].train_step(kconcat([advA_alter, advA_const], axis=1), torch.ones(1))
            fake_conA_logit, _, fake_conA_feat = self.advAs[cv_index].train_step(kconcat([synA_prob.detach(), advA_const], axis=1), torch.zeros(1))

        if 'synB' in modules:
            self.synBs[cv_index].train()
            if self.synB_params['use_fake'] < random.uniform(0, 1) and synA_prob is not None and synB_alter is not None:
                synB_alter = synA_prob.detach()

            synB_logit, synB_prob, _ = self.synBs[cv_index].extra_train_step(kconcat([synB_alter, synB_const], axis=1), synB_tar,
                                       ext_funs={'adv': {'fun': self.advBs[cv_index].module, 'const': advB_const, 'alter': advB_alter, 'tar': torch.ones(1)},
                                                 'cls': {'fun': self.clsBs[cv_index].module, 'const': clsB_const, 'alter': clsB_alter, 'tar': inputs['label']},
                                                 'syn': {'fun': self.synAs[cv_index].module, 'const': synA_const, 'alter': synA_alter, 'tar': synA_tar}, })
        else:
            self.synBs[cv_index].eval()
            synB_logit, synB_prob, _ = self.synBs[cv_index].infer_step(kconcat([synB_alter, synB_const], axis=1))
        attributes.update({'synB': synB_prob.detach().to('cpu').numpy()})

        if 'probBs' in outkeys:
            cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index].infer_step(kconcat([synB_prob, clsB_const], axis=1))
            attributes.update({'probBs': cls_sB_prob.detach().to('cpu').numpy()})

        # attributes.update({'synB': synB_const})
        if 'advB' in modules:
            self.advBs[cv_index].train()
            conB_logit, _, conB_feat = self.advBs[cv_index].train_step(kconcat([advB_alter, advB_const], axis=1), torch.ones(1))
            fake_conB_logit, _, fake_conB_feat = self.advBs[cv_index].train_step(kconcat([synB_prob.detach(), advB_const], axis=1), torch.zeros(1))

        if self.synB_params['use_fake'] > 0 and synA_prob is not None and synB_alter is not None:
            # synA_logit, synA_prob, _ = self.synAs[cv_index](kconcat([synB_prob, synA_const], axis=-1))
            synB_logit, synB_prob, _ = self.synBs[cv_index].infer_step(kconcat([synA_prob, synB_const], axis=1))
            attributes.update({'synB': synB_prob.detach().to('cpu').numpy()})

        if 'clsA' in modules and self.clsA_params['use_syn'] is True:
            clsA_alter = synA_prob
            self.clsAs[cv_index].train()
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index].train_step(kconcat([clsA_alter, clsA_const], axis=1), attributes['labelA'])

        if 'clsB' in modules and self.clsB_params['use_syn'] is True:
            clsB_alter = synB_prob
            self.clsBs[cv_index].train()
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index].train_step(kconcat([clsB_alter, clsB_const], axis=1), attributes['labelB'])
        return attributes

    def iteration_modules_v3(self, inputs, cv_index,
                             modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'), outkeys=None, flags=None):

        def kconcat(inputs, axis=1):
            inputs = [torch.Tensor(var).to('cuda') for var in inputs if var is not None]
            return None if len(inputs) == 0 else torch.concat(inputs, dim=axis)

        def npconcat(inputs, axis=-1):
            inputs = [var for var in inputs if var is not None]
            return None if len(inputs) == 0 else np.concatenate(inputs, axis=axis)

        attributes = {'labelA': npconcat([inputs[key] for key in self.clsA_params['output']], axis=-1), 'refA': npconcat([inputs[key] for key in self.synA_params['output']], axis=-1),
                      'labelB': npconcat([inputs[key] for key in self.clsB_params['output']], axis=-1), 'refB': npconcat([inputs[key] for key in self.synB_params['output']], axis=-1)}

        clsA_alter = kconcat([inputs[ind] for ind in self.clsA_params['input_alter']], axis=1)
        clsA_const = kconcat([inputs[ind] for ind in self.clsA_params['input_const']], axis=1)

        clsB_alter = kconcat([inputs[ind] for ind in self.clsB_params['input_alter']], axis=1)
        clsB_const = kconcat([inputs[ind] for ind in self.clsB_params['input_const']], axis=1)

        synB_alter = kconcat([inputs[ind] for ind in self.synB_params['input_alter']], axis=1)
        synB_const = kconcat([inputs[ind] for ind in self.synB_params['input_const']], axis=1)
        synA_alter = kconcat([inputs[ind] for ind in self.synA_params['input_alter']], axis=1)
        synA_const = kconcat([inputs[ind] for ind in self.synA_params['input_const']], axis=1)

        self.clsAs[cv_index].toDevice('cuda')
        self.clsBs[cv_index].toDevice('cuda')
        self.synAs[cv_index].toDevice('cuda')
        self.synBs[cv_index].toDevice('cuda')
        with torch.no_grad():
            if 'probA' in outkeys or 'logitA' in outkeys:
                self.clsAs[cv_index].eval()
                clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index].infer_step(kconcat([clsA_alter, clsA_const], axis=1))
                attributes.update({'logitA': clsA_logit.detach().to('cpu').numpy(), 'probA': clsA_prob.detach().to('cpu').numpy()})

            if 'probB' in outkeys or 'logitB' in outkeys:
                self.clsBs[cv_index].eval()
                clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index].infer_step(kconcat([clsB_alter, clsB_const], axis=1))
                attributes.update({'logitB': clsB_logit.detach().to('cpu').numpy(), 'probB': clsB_prob.detach().to('cpu').numpy()})

            if 'synA' in outkeys or 'probAs' in outkeys:
                self.synAs[cv_index].eval()
                self.clsAs[cv_index].eval()
                synA_logit, synA_prob, _ = self.synAs[cv_index].infer_step(kconcat([synA_alter, synA_const], axis=1))
                cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index].infer_step(kconcat([synA_prob, clsA_const], axis=1))
                attributes.update({'synA': synA_prob.detach().to('cpu').numpy(), 'probAs': cls_sA_prob.detach().to('cpu').numpy()})
            else:
                attributes.update({'synA': None, 'probAs': None})

            if 'synB' in outkeys or 'probBs' in outkeys:
                self.synBs[cv_index].eval()
                self.clsBs[cv_index].eval()
                if self.synB_params['use_fake'] > 0 and synA_prob is not None and synB_alter is not None:
                    synB_alter = synA_prob.detach()
                synB_logit, synB_prob, _ = self.synBs[cv_index].infer_step(kconcat([synB_alter, synB_const], axis=1))
                cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index].infer_step(kconcat([synB_prob, clsB_const], axis=1))
                attributes.update({'synB': synB_prob.detach().to('cpu').numpy(), 'probBs': cls_sB_prob.detach().to('cpu').numpy()})
            else:
                attributes.update({'synB': None, 'probBs': None})
            return attributes

    def iteration_loop(self, dataloader, cv_index, train_modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'),
                       sample_output_rate=None, outkeys=None):
        attr_summary = {'flnm': []}

        self.clsAs[cv_index].toDevice('cuda')
        self.clsBs[cv_index].toDevice('cuda')
        self.synAs[cv_index].toDevice('cuda')
        self.synBs[cv_index].toDevice('cuda')
        self.advAs[cv_index].toDevice('cuda')
        self.advBs[cv_index].toDevice('cuda')
        for ptr, (data, info) in enumerate(dataloader):
            if ptr > self.max_num_images: break
            attrs = self.iteration_modules(data, cv_index, train_modules, outkeys=outkeys)
            attrs.update({'flnm': info['flnm']})
            if sample_output_rate is None or ptr % 10 < sample_output_rate * 10:
                if outkeys is None:
                    outkeys = attrs.keys()
                for key in outkeys:
                    if key not in attr_summary:
                        attr_summary[key] = []
                    if outkeys is None or key in outkeys or key is outkeys:
                        attr_summary[key].append(attrs[key])

        for key in attr_summary:
            attr_summary[key] = np.concatenate(attr_summary[key], axis=0)
        return attr_summary

    def train(self, start_epoch=0, inter_epoch=10, sample_output_rate=0.1):
        for epoch in range(start_epoch, 1 + np.max(self.cls_num_epoch + self.syn_num_epoch)):
            outkeys = ['flnm']
            modules = []
            if epoch in self.cls_num_epoch:
                modules = [mid for mid in ('clsA', 'clsB') if mid in self.training_modules]
                if 'clsA' in self.training_modules or 'clsB' in self.training_modules:
                    outkeys = outkeys + ['probA', 'probB', 'labelA', 'labelB']
            if epoch in self.syn_num_epoch:
                modules = modules + [mid for mid in ('synA', 'synB', 'advA', 'advB') if mid in self.training_modules]
                outkeys = outkeys + ['synA', 'refA', 'synB', 'refB']

            for sp in range(len(self.database.train_combinations)):
                start_time = time.perf_counter()
                print("Epoch {0}, Split {1}".format(epoch, sp))
                attr_summary = self.iteration_loop(self.dataloader, sp, train_modules=modules,
                                                   sample_output_rate=sample_output_rate, outkeys=outkeys)
                if epoch in self.cls_num_epoch:
                    clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
                    clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
                    print('clsA train: ', clsA_metrics[0])
                    print(clsA_metrics[1])
                    print('clsB train: ', clsB_metrics[0])
                    print(clsB_metrics[1])
                    clsAB_metrics = matrics_classification(
                        np.average((attr_summary['probA'], attr_summary['probB']), axis=0), attr_summary['labelA'])
                    print('clsAB train: ', clsAB_metrics[0])
                    print(clsAB_metrics[1])

                    if epoch % inter_epoch == inter_epoch - 1:
                        self.model_save(epoch + 1, 'cls')
                if epoch in self.syn_num_epoch:
                    if 'synA' in self.training_modules:
                        prediction = np.transpose(attr_summary['synA'],
                                                  axes=[0] + [num for num in range(2, np.ndim(attr_summary['synA']))] + [1])
                        groundtruth = np.transpose(attr_summary['refA'],
                                                   axes=[0] + [num for num in range(2, np.ndim(attr_summary['refA']))] + [1])
                        if self.synA_params['task_type'] == 'synthesis':
                            syn_metrics = matrics_synthesis(prediction, groundtruth)
                            print('synA train:', syn_metrics)
                        else:
                            syn_metrics = matrics_segmentation(prediction, groundtruth)
                            print('segA train:', syn_metrics)
                    if 'synB' in self.training_modules:
                        prediction = np.transpose(attr_summary['synB'],
                                                  axes=[0] + [num for num in range(2, np.ndim(attr_summary['synB']))] + [1])
                        groundtruth = np.transpose(attr_summary['refB'],
                                                   axes=[0] + [num for num in range(2, np.ndim(attr_summary['refB']))] + [1])
                        if self.synB_params['task_type'] == 'synthesis':
                            syn_metrics = matrics_synthesis(prediction, groundtruth)
                            print('synB train:', syn_metrics)
                        else:
                            syn_metrics = matrics_segmentation(prediction, groundtruth)
                            print('segB train:', syn_metrics)
                    if epoch % inter_epoch == inter_epoch - 1:
                        self.model_save(epoch + 1, 'syn')
                end_time = time.perf_counter()
                print(end_time - start_time)

                if epoch % 10 == 9:
                    attr_summary = self.iteration_loop(self.dataloaderV, sp, train_modules=(),
                                                       sample_output_rate=1.1, outkeys=outkeys)
                    if epoch in self.cls_num_epoch:
                        clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
                        clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
                        print('clsA test: ', clsA_metrics[0])
                        print(clsA_metrics[1])
                        print('clsB test: ', clsB_metrics[0])
                        print(clsB_metrics[1])
                        clsAB_metrics = matrics_classification(
                            np.average((attr_summary['probA'], attr_summary['probB']), axis=0), attr_summary['labelA'])
                        print('clsAB test: ', clsAB_metrics[0])
                        print(clsAB_metrics[1])

                    if epoch in self.syn_num_epoch:
                        if 'synA' in self.training_modules:
                            prediction = np.transpose(attr_summary['synA'],
                                                      axes=[0] + [num for num in range(2, np.ndim(attr_summary['synA']))] + [1])
                            groundtruth = np.transpose(attr_summary['refA'],
                                                       axes=[0] + [num for num in range(2, np.ndim(attr_summary['refA']))] + [1])
                            if self.synA_params['task_type'] == 'synthesis':
                                syn_metrics = matrics_synthesis(prediction, groundtruth)
                                print('synA valid:', syn_metrics)
                            else:
                                syn_metrics = matrics_segmentation(prediction, groundtruth)
                                print('segA valid:', syn_metrics)
                        if 'synB' in self.training_modules:
                            prediction = np.transpose(attr_summary['synB'],
                                                      axes=[0] + [num for num in range(2, np.ndim(attr_summary['synB']))] + [1])
                            groundtruth = np.transpose(attr_summary['refB'],
                                                       axes=[0] + [num for num in range(2, np.ndim(attr_summary['refB']))] + [1])
                            if self.synB_params['task_type'] == 'synthesis':
                                syn_metrics = matrics_synthesis(prediction, groundtruth)
                                print('synB valid:', syn_metrics)
                            else:
                                syn_metrics = matrics_segmentation(prediction, groundtruth)
                                print('segB valid:', syn_metrics)

    def cross_validation(self, check_epochs=(3, 200)):
        outkeys = ['flnm', 'probA', 'probB', 'labelA', 'labelB', 'probAs', 'probBs', 'synA', 'refA', 'synB', 'refB']
        for sp in range(len(self.database.train_combinations)):
            # start_time = time.perf_counter()
            print("Epoch {0}, Split {1}".format(check_epochs, sp))
            attr_summary = self.iteration_loop(self.database.imdb_valid_split[sp], sp, train_modules=(), outkeys=outkeys)
            result2csv = ['{0},{1},{2},{3},{4}'.format(attr_summary['flnm'][idx], attr_summary['labelA'][idx], attr_summary['probA'][idx], attr_summary['labelB'][idx], attr_summary['probB'][idx]) for idx in range(len(attr_summary['flnm']))]
            with open('cross_validation_{0}.csv'.format(sp), 'w', ) as csvfile:
                result2csv = csv.writer(csvfile, delimiter=',')
                result2csv.writerow(['flnm', 'labelA', 'probA', 'labelB', 'probB'])
                for idx in range(len(attr_summary['flnm'])):
                    result2csv.writerow([attr_summary['flnm'][idx], attr_summary['labelA'][idx][1], attr_summary['probA'][idx][1], attr_summary['labelB'][idx][1], attr_summary['probB'][idx][1]])

            # np.savetxt('cross_validation_{0}.txt'.format(sp), result2csv, delimiter=',')
            clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
            clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
            clsAs_metrics = matrics_classification(attr_summary['probAs'], attr_summary['labelA'])
            clsBs_metrics = matrics_classification(attr_summary['probBs'], attr_summary['labelB'])
            print('cls valid modality A: ', clsA_metrics[0])
            print('cls valid modality B: ', clsB_metrics[0])
            print('cls valid synthesis A: ', clsAs_metrics[0])
            print('cls valid synthesis B: ', clsBs_metrics[0])
            clsAB_metrics = matrics_classification(np.average((attr_summary['probA'], attr_summary['probB']), axis=0),
                                                   attr_summary['labelA'])
            clsABs_metrics = matrics_classification(np.average((attr_summary['probA'], attr_summary['probBs']), axis=0),
                                                    attr_summary['labelB'])
            print('cls valid modality A + modality B: ', clsAB_metrics[0])
            print('cls valid modality A + synthesis B: ', clsABs_metrics[0])
            synA_metrics = matrics_synthesis(attr_summary['synA'], attr_summary['refA'])
            synB_metrics = matrics_synthesis(attr_summary['synB'], attr_summary['refB'])
            print('synA valid:', synA_metrics)
            print('synB valid:', synB_metrics)

            # end_time = time.perf_counter()
            # print(end_time - start_time)

    def test(self, check_epochs=(100, 110)):
        outkeys = ['flnm', 'probA', 'probB', 'labelA', 'labelB', 'probBs']
        attr_summary = {}
        # start_time = time.perf_counter()
        for sp in range(len(self.database.train_combinations)):
            print("Epoch {0}, Split {1}".format(check_epochs, sp))
            attrs = self.iteration_loop(self.database.imdb_test, sp, train_modules=(), outkeys=outkeys)
            for key in attrs:
                if key not in attr_summary:
                    attr_summary[key] = []
                if outkeys is None or key in outkeys:
                    attr_summary[key].append(attrs[key])
        print('probA', np.average(attr_summary['probA'], axis=0))
        print('probB', np.average(attr_summary['probB'], axis=0))
        clsA_metrics = matrics_classification(np.average(attr_summary['probA'], axis=0),
                                              np.average(attr_summary['labelA'], axis=0))
        clsB_metrics = matrics_classification(np.average(attr_summary['probB'], axis=0),
                                              np.average(attr_summary['labelB'], axis=0))
        clsBs_metrics = matrics_classification(np.average(attr_summary['probBs'], axis=0),
                                               np.average(attr_summary['labelB'], axis=0))
        print('cls valid single: clsA_metrics', clsA_metrics[0], clsA_metrics[1])
        print('cls valid single: clsB_metrics', clsB_metrics[0], clsB_metrics[1])
        print('cls valid single: clsBs_metrics', clsBs_metrics[0], clsBs_metrics[1])
        clsABs_metrics = matrics_classification(
            (np.average(attr_summary['probA'], axis=0) + np.average(attr_summary['probBs'], axis=0)) / 2,
            np.average(attr_summary['labelA'], axis=0))
        clsAB_metrics = matrics_classification(
            (np.average(attr_summary['probA'], axis=0) + np.average(attr_summary['probB'], axis=0)) / 2,
            np.average(attr_summary['labelA'], axis=0))

        print('cls valid dual: clsABs_metrics', clsABs_metrics)
        print('cls valid dual: clsAB_metrics', clsAB_metrics)
        # syn_metrics = matrics_synthesis(np.average(attr_summary['probsyn'], axis=0),
        #                                 np.average(attr_summary['inputB'], axis=0))
        # print('syn valid:', syn_metrics)
        # end_time = time.perf_counter()
        # print(end_time - start_time)

    @staticmethod
    def expand_apply_synthesis(inputA, synmodels):
        if not isinstance(synmodels, list): synmodels = [synmodels]
        syn_probs = [model(inputA)[1] for model in synmodels]
        syn_prob = np.mean(syn_probs, axis=0)
        return syn_prob

    @staticmethod
    def expand_apply_classification(inputB, clsmodels):
        if not isinstance(clsmodels, list): clsmodels = [clsmodels]
        cls_probs = [model(inputB)[0] for model in clsmodels]
        cls_prob = np.mean(cls_probs, axis=0)
        return cls_prob

    def extra_test(self):
        nums_of_samps = len(self.database.imdb_test) if 'nums_of_samps' not in self.database.imdb_test else \
            self.database.imdb_test['nums_of_samps']
        print(nums_of_samps)
        for ptr in range(0, min(self.max_num_images, nums_of_samps)):
            flnm, inputA, inputB, label = self.database.inputAB(self.database.imdb_test, index=ptr)
            syn_B = self.expand_apply_synthesis(inputA, self.synAs)
            cls_B = self.expand_apply_classification(syn_B, self.clsBs)
            cls_A = self.expand_apply_classification(inputA, self.clsAs)
            if isinstance(flnm, bytes): flnm = flnm.decode()
            if not flnm.endswith('.png'): flnm = flnm + '.png'
            cv2.imwrite(self.result_path + "/f_{0}".format(flnm),
                        (np.concatenate((inputB[0], syn_B[0]), axis=0) + 1) * 126)
            print(np.argmax(cls_A, axis=-1), np.argmax(cls_B, axis=-1), np.argmax((cls_A + cls_B) / 2, axis=-1))


    def test_classification(self):
        nums_of_samps = len(self.database.imdb_test) if 'nums_of_samps' not in self.database.imdb_test else \
            self.database.imdb_test['nums_of_samps']
        print(nums_of_samps)
        for ptr in range(0, min(self.max_num_images, nums_of_samps)):
            flnm, inputA, inputB, label = self.database.inputAB(self.database.imdb_test, index=ptr)
            cls_A = self.expand_apply_classification(inputA, self.clsAs)
            cls_B = self.expand_apply_classification(inputB, self.clsBs)

            if isinstance(flnm, bytes): flnm = flnm.decode()
            if not flnm.endswith('.png'): flnm = flnm + '.png'
            print(np.argmax(cls_A, axis=-1), np.argmax(cls_B, axis=-1), np.argmax((cls_A + cls_B) / 2, axis=-1))


    def test_synthesis(self):
        nums_of_samps = len(self.database)
        print(nums_of_samps)
        synA_metrics, synB_metrics = [], []

        for ptr in range(len(self.databaseT)):
            load_pregenerated = False
            time_start = time.time()
            if load_pregenerated:
                eval_out = self.database.read_output(self.result_path, self.database.imdb_test, index=ptr)
                # eval_out['affine'] = inputCs['affine'][0]
                flnm, refA, refB, synA, synB = eval_out['flnm'], eval_out['refA'], eval_out['refB'], eval_out['synA'], eval_out['synB']
            else:
                outkeys = ['flnm'] + ['probA', 'probB', 'labelA', 'labelB'] + ['synA', 'refA', 'synB', 'refB']
                if False:
                    mm_images, info = self.databaseT.inputAB(aug_model='sequency', index=ptr, aug_count=1, aug_index=(0,))
                    full_size_A = [self.synA_params['output_channel']] + info['orig_size'].tolist()
                    full_size_B = [self.synB_params['output_channel']] + info['orig_size'].tolist()
                    syn_A, syn_B = np.zeros(full_size_A, np.float32), np.zeros(full_size_B, np.float32)
                    ref_A, ref_B = np.zeros(full_size_A, np.float32), np.zeros(full_size_B, np.float32)
                    cusum_A, cusum_B = np.zeros(full_size_A, np.float32) + 1e-6, np.zeros(full_size_B, np.float32) + 1e-6
                    flnm = info['flnm']
                    for fctr in range(50):
                        mm_images, info = self.databaseT.inputAB(aug_model='sequency', index=ptr, aug_count=1, aug_index=(fctr,))
                        attrs = self.iteration_modules_v3(mm_images, 0, (), outkeys=outkeys)
                        sX1, sY1, sZ1, sX2, sY2, sZ2 = info['aug_crops'][0]

                        ref_A[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['refA'][0]
                        ref_B[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['refB'][0]
                        syn_A[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['synA'][0]
                        syn_B[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['synB'][0]
                        cusum_A[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += 1
                        cusum_B[:, sX1:sX2, sY1:sY2, sZ1:sZ2] += 1
                        if fctr >= info['count_of_augs']: break
                    synA, synB = syn_A/cusum_A, syn_B/cusum_B
                    refA, refB = ref_A/cusum_A, ref_B/cusum_B
                    synA = np.where(cusum_A > 1.0, synA, np.min(synA[cusum_A > 1.0]))
                    synB = np.where(cusum_B > 1.0, synB, np.min(synB[cusum_B > 1.0]))
                    refA = np.where(cusum_A > 1.0, refA, np.min(refA[cusum_A > 1.0]))
                    refB = np.where(cusum_B > 1.0, refB, np.min(refB[cusum_B > 1.0]))

                else:
                    mm_images, info = self.databaseT.inputAB(aug_model='center', index=ptr, aug_count=1, aug_index=(0,))
                    flnm = info['flnm']
                    attrs = self.iteration_modules(mm_images, 0, (), outkeys=outkeys)
                    refA = attrs['refA'][0]
                    refB = attrs['refB'][0]
                    synA = attrs['synA'][0]
                    synB = attrs['synB'][0]
                eval_out = {'flnm': flnm, 'refA': refA, 'refB': refB, 'synA': synA, 'synB': synB, 'affine': info['affine']}
                self.database.save_output(self.result_path, flnm, eval_out)

            print(flnm)
            if refA is not None:
                synA = np.transpose(synA, axes=[num for num in range(1, np.ndim(synA))] + [0])
                refA = np.transpose(refA, axes=[num for num in range(1, np.ndim(refA))] + [0])
                syn_metrics = matrics_synthesis(synA, refA, isinstance=True) \
                    if self.synA_params['task_type'] == 'synthesis' else matrics_segmentation(synA, refA)
                print('synA: ', syn_metrics)
                synA_metrics.append(syn_metrics)
            if refB is not None:
                synB = np.transpose(synB, axes=[num for num in range(1, np.ndim(synA))] + [0])
                refB = np.transpose(refB, axes=[num for num in range(1, np.ndim(refA))] + [0])
                syn_metrics = matrics_synthesis(synB, refB, data_range=2, isinstance=True) \
                    if self.synB_params['task_type'] == 'synthesis' else matrics_segmentation(synB, refB)
                print('synB: ', syn_metrics)
                synB_metrics.append(syn_metrics)
            # time_end = time.time()
            # print(time_end - time_start)

        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print('synA', np.mean(synA_metrics, axis=0), np.std(synA_metrics, axis=0))
        print('synB', np.mean(synB_metrics, axis=0), np.std(synB_metrics, axis=0))

