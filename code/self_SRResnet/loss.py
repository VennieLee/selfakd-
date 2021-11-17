import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.feature_loss_module = nn.ModuleList()
        self.feature_loss_used = args.feature_loss_used
        
        # SR loss
        LS_weight = 1 - args.alpha #Label loss (hr, shallow sr)
        OS_weight = args.alpha #Output loss (deep sr, shallow sr)
        FS_weight = 1e-6
        
        self.loss.append({'type': "LS", 'weight': LS_weight, 'function': nn.L1Loss()})
        self.loss.append({'type': "OS", 'weight': OS_weight, 'function': nn.L1Loss()})
          


        FS_weight = {'type': 'FS', 'weight': FS_weight, 'function': FeatureLoss(loss=nn.L1Loss())}
        self.loss.append(FS_weight)
        self.feature_loss_module.append(FS_weight['function'])
       
      
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
        
        
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.log = torch.Tensor()      
        self.feature_loss_module.to(device)
        
        if not args.cpu and args.n_GPUs > 1:
            self.feature_loss_module = nn.DataParallel(
                self.feature_loss_module, range(args.n_GPUs)
            )

        if args.resume == 1: 
            self.load(ckp.dir, cpu=args.cpu)



    def forward(self, hr, output, middle_output1, middle_feature1, middle_output2, middle_feature2, middle_output3, middle_feature3, final_feature):
        # LS Loss
        LS_loss = self.loss[0]['function'](middle_output1, hr)* self.loss[0]['weight'] + self.loss[0]['function'](middle_output2, hr)* self.loss[0]['weight'] + \
                  self.loss[0]['function'](middle_output3, hr)* self.loss[0]['weight'] + self.loss[0]['function'](output, hr)
        self.log[-1, 0] += LS_loss.item()
        
        # OS Loss
        OS_loss = self.loss[1]['function'](middle_output1, output) * self.loss[1]['weight'] +self.loss[1]['function'](middle_output2, output) * self.loss[1]['weight'] + \
            self.loss[1]['function'](middle_output3, output) * self.loss[1]['weight']
        self.log[-1, 1] += OS_loss.item()
        
        loss_sum = LS_loss + OS_loss
        
        # FS loss
        for i in range(len(self.feature_loss_module)):   
            feature_loss = self.feature_loss_module[i](middle_feature1[i], final_feature[i])*1e-6 + self.feature_loss_module[i](middle_feature2[i], final_feature[i])*1e-6 + \
                self.feature_loss_module[i](middle_feature3[i], final_feature[i])*1e-6
            self.log[-1, 2 + i] += feature_loss.item()
            loss_sum += feature_loss
        
        self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.feature_loss_module
        else:
            return self.feature_loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.feature_loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()





class FeatureLoss(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss



