# Copyright (c) 2022, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
import os
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from builder.utils.metrics import Evaluator


class Logger:
    def __init__(self, args):
        self.args = args
        self.args_save = copy.deepcopy(args)
        
        # Evaluator
        self.evaluator = Evaluator(self.args)
        
        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(self.args.dir_result, self.args.project_name)
        # self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_log = os.path.join(self.dir_root, 'logs_{}'.format(str(args.log_fold)))
        self.dir_save = os.path.join(self.dir_root, 'ckpts')
        self.log_iter = args.log_iter
        self.scores_list = []

        if args.reset and os.path.exists(self.dir_root):
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif os.path.exists(os.path.join(self.dir_save, 'last.pth')) and os.path.exists(self.dir_log):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=30)
        
        # Log variables
        self.loss = 0
        self.val_loss = 0
        self.best_auc = 0
        self.best_iter = 0
        self.best_result_so_far = np.array([])
        self.best_results = []

        # test results
        self.test_results = {}

        # test results
        self.val_results = {}

    def log_tqdm(self, epoch, step, pbar):
        tqdm_log = "Epochs: {}, Iteration: {}, Loss: {}".format(str(epoch), str(step), str(self.loss / step))
        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        self.writer.add_scalar('train/loss', self.loss / step, global_step=step)
    
    def log_lr(self, lr, step):
        self.writer.add_scalar('train/lr', lr, global_step=step)

    def log_val_loss(self, val_step, step):
        self.writer.add_scalar('val/loss', self.val_loss / val_step, global_step=step)

    def add_validation_logs(self, step):
        result = self.evaluator.performance_metric()

        if (self.args.model_types == "detection") or (self.args.model_types == "bce_rmse"):
            auc, apr, f1 = result
            anchor = auc + apr
            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(auc), str(apr), str(f1)))
            
            self.writer.add_scalar('val/auc', np.mean(auc), global_step=step)
            self.writer.add_scalar('val/apr', np.mean(apr), global_step=step)
            self.writer.add_scalar('val/f1', np.mean(f1), global_step=step)
            
            if self.best_auc < anchor:
                self.best_iter = step
                self.best_auc = anchor
                self.best_result_so_far = list(result)
            
        elif (self.args.model_types == "classification"):
            nresult, wresult = result
            aucs, aprs, f1s = nresult
            wauc, wapr, wf1 = wresult
            anchor = wf1
            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'Weighted auc: {}, apr: {}, f1: {}\'".format(str(wauc), str(wapr), str(wf1)))
            os.system("echo  \'Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))

            self.writer.add_scalar('val/avg_auc', np.mean(aucs), global_step=step)
            self.writer.add_scalar('val/avg_apr', np.mean(aprs), global_step=step)
            self.writer.add_scalar('val/avg_f1', np.mean(f1s), global_step=step)
            self.writer.add_scalar('val/w_auc', wauc, global_step=step)
            self.writer.add_scalar('val/w_apr', wapr, global_step=step)
            self.writer.add_scalar('val/w_f1', wf1, global_step=step)

            if self.best_auc < anchor:
                self.best_iter = step
                self.best_auc = anchor
                self.best_result_so_far = list(result)

            nresult, wresult = self.best_result_so_far 
            aucs, aprs, f1s = nresult
            wauc, wapr, wf1 = wresult
            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system("echo  \'Weighted auc: {}, apr: {}, f1: {}\'".format(str(wauc), str(wapr), str(wf1)))
            os.system("echo  \'Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))

            self.writer.flush()

    def save(self, model, optimizer, step, epoch, k_indx, last=None):
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_step': step, 'last_step' : last, 'score' : self.best_auc, 'epoch' : epoch}
        
        if step == self.best_iter:
            self.save_ckpt(ckpt, 'best_fold{}_seed{}.pth'.format(str(k_indx), str(self.args.seed)))
            
        if last:
            self.evaluator.get_attributions()
            self.save_ckpt(ckpt, 'last_fold{}_seed{}.pth'.format(str(k_indx), str(self.args.seed)))
    
    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))

    def test_result_only(self):
        result = self.evaluator.performance_metric()
        
        if (self.args.model_types == "detection"):
            auc, apr, f1 = result
            anchor = auc + apr
            os.system("echo  \'##### Test results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(auc), str(apr), str(f1)))
            
        elif (self.args.model_types == "classification"):
            nresult, wresult = result
            aucs, aprs, f1s = nresult
            wauc, wapr, wf1 = wresult
            anchor = wf1
            os.system("echo  \'##### Test results #####\'")
            os.system("echo  \'Weighted auc: {}, apr: {}, f1: {}\'".format(str(wauc), str(wapr), str(wf1)))
            os.system("echo  \'Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))        

        self.test_results = list([self.args.seed, result])

    def val_result_only(self):
        if (self.args.model_types == "detection"):
            auc, apr, f1 = self.best_result_so_far 
            anchor = auc + apr
            os.system("echo  \'##### Validation results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(auc), str(apr), str(f1)))
            
        elif (self.args.model_types == "classification"):
            nresult, wresult = self.best_result_so_far 
            aucs, aprs, f1s = nresult
            wauc, wapr, wf1 = wresult
            anchor = wf1
            os.system("echo  \'##### Validation results #####\'")
            os.system("echo  \'Weighted auc: {}, apr: {}, f1: {}\'".format(str(wauc), str(wapr), str(wf1)))
            os.system("echo  \'Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))       

        self.val_results = list([self.args.seed, self.best_result_so_far])
            
          