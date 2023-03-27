# Copyright (c) 2021, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import numpy as np
import pickle as pkl
# from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score, auc
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from control.config import args
from torchmetrics import AUROC, AveragePrecision, ROC, F1Score
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassF1Score
from torchmetrics.functional import f1_score

# from torchmetrics.classification import BinaryF1Score

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.n_labels = args.output_dim
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.batch_size = args.batch_size
        self.best_auc = 0
        self.labels_list = [i for i in range(self.n_labels)]

        self.y_true_multi = []
        self.y_pred_multi = []
        self.nauroc = MulticlassAUROC(num_classes=12, average="none")
        self.nauprc = MulticlassAveragePrecision(num_classes=12, average="none")
        self.nf1 = MulticlassF1Score(num_classes=12, average="none").to(args.device)
        
        self.wauroc = MulticlassAUROC(num_classes=12, average="weighted")
        self.wauprc = MulticlassAveragePrecision(num_classes=12, average="weighted")
        self.wf1 = MulticlassF1Score(num_classes=12, average="weighted").to(args.device)
        
        self.auroc = AUROC(task="binary")
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary")
        # self.roc = ROC(task="binary")
    
    def binary_normalize(self, i):
        proba_list = [i[0], max(i[1:])]
        return np.array(proba_list)/sum(proba_list)

    def add_batch(self, y_true, y_pred_multi):
        self.y_pred_multi.append(y_pred_multi.detach())
        self.y_true_multi.append(y_true.detach())

    def performance_metric(self):
        self.y_true_multi = torch.stack(self.y_true_multi).type(torch.ByteTensor).cuda()
        self.y_pred_multi = torch.stack(self.y_pred_multi).cuda()
        self.y_pred_multi = torch.nan_to_num(self.y_pred_multi)
        
        if self.args.model_types == "classification":
            trues = self.y_true_multi.reshape(-1).cuda()
            preds = self.y_pred_multi.reshape(-1, 12).cuda()

            auc = self.nauroc(preds, trues)
            apr = self.nauprc(preds, trues)
            f1 = self.nf1(preds, trues)
            
            wauc = self.wauroc(preds, trues)
            wapr = self.wauprc(preds, trues)
            wf1 = self.wf1(preds, trues)
            
            scores_list = list([np.round(np.array([auc.clone().detach().cpu().numpy(), 
                                                apr.clone().detach().cpu().numpy(), 
                                                f1.clone().detach().cpu().numpy()]), 4),
                                np.round(np.array([wauc.clone().detach().cpu().numpy(), 
                                                wapr.clone().detach().cpu().numpy(), 
                                                wf1.clone().detach().cpu().numpy()]), 4)])        
            del wauc
            del wapr
            del wf1
            
        elif self.args.model_types == "detection":     
            trues = self.y_true_multi.cuda()
            preds = self.y_pred_multi.cuda()
            auc = self.auroc(preds, trues)
            apr = self.auprc(preds, trues)
            f1 = 0
            for i in range(1, 100):
                threshold = i / 100.0    
                temp_output = preds.detach().clone()
                temp_output[temp_output>=threshold] = 1
                temp_output[temp_output<threshold] = 0        
                temp_score = f1_score(temp_output, trues, task="binary", threshold = threshold)
                if temp_score > f1:
                    f1 = temp_score
            scores_list = list(np.round(np.array([auc.clone().detach().cpu().numpy(), 
                                                apr.clone().detach().cpu().numpy(), 
                                                f1.clone().detach().cpu().numpy()]), 4))    
        del trues
        del preds
        del auc
        del apr
        del f1
        torch.cuda.empty_cache()
        return scores_list

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.y_true_multi = []
        self.y_pred_multi = []
