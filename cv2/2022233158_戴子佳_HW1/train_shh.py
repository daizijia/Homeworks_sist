import numpy as np
import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from resnet50 import CrowdCounter
from conf import cfg
from utils import *
import pdb


class Counter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net = CrowdCounter(cfg.GPU_ID).cuda()
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0

        self.train_loader, self.val_loader, self.restore_transform = dataloader()
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp')


    def forward(self):

        # self.validate_V3()
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
                
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if epoch%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()

                self.validate()

                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )


    def train(self): # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss = self.net.loss
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg_data.LOG_PARA, pred_map[0].sum().data/self.cfg_data.LOG_PARA) )           


    def validate(self):

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA
                    
                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)

seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET

from loadingdata import loading_data_SHHA, loading_data_SHHB 
from conf import cfg_SHHA, cfg_SHHB

#------------Prepare Trainer------------
net = cfg.NET

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
if data_mode == 'SHHA':
    cc_trainer = Counter(loading_data_SHHA,cfg_SHHA,pwd)
    cc_trainer.forward()
elif data_mode == 'SHHB':
    cc_trainer = Counter(loading_data_SHHB,cfg_SHHB,pwd)
    cc_trainer.forward()
