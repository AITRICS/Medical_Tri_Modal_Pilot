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
        if args.model_types == "classification" and args.loss_types == "rmse":
            self.best_auc = float('inf')
        else:
            self.best_auc = 0
        self.labels_list = [i for i in range(self.n_labels)]

        self.y_true_multi = []
        self.y_pred_multi = []
        self.rmse = []
        # self.nauroc = MulticlassAUROC(num_classes=12, average="none")
        # self.nauprc = MulticlassAveragePrecision(num_classes=12, average="none")
        # self.nf1 = MulticlassF1Score(num_classes=12, average="none").to(args.device)
        
        # self.wauroc = MulticlassAUROC(num_classes=12, average="weighted")
        # self.wauprc = MulticlassAveragePrecision(num_classes=12, average="weighted")
        # self.wf1 = MulticlassF1Score(num_classes=12, average="weighted").to(args.device)
        
        self.auroc = AUROC(task="binary")
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary")
        # self.roc = ROC(task="binary")
    
    def binary_normalize(self, i):
        proba_list = [i[0], max(i[1:])]
        return np.array(proba_list)/sum(proba_list)

    def add_batch(self, y_true, y_pred_multi, rmse = None):
        self.y_pred_multi.append(y_pred_multi.detach())
        self.y_true_multi.append(y_true.detach())
        if rmse is not None:
            self.rmse.append(rmse.detach())

    def performance_metric(self):
        self.y_true_multi = torch.stack(self.y_true_multi).type(torch.ByteTensor).cuda()
        self.y_pred_multi = torch.stack(self.y_pred_multi).cuda()
        self.y_pred_multi = torch.nan_to_num(self.y_pred_multi)
        
<<<<<<< HEAD
        if self.args.model_types == "classification":
            if "softmax" == args.loss_types or "bceandsoftmax" == args.loss_types:
                trues = self.y_true_multi.reshape(-1).cuda()
                preds = self.y_pred_multi.reshape(-1, 12).cuda()
                
                auc = self.nauroc(preds, trues).detach().cpu().numpy()
                apr = self.nauprc(preds, trues).detach().cpu().numpy()
                f1 = self.nf1(preds, trues).detach().cpu().numpy()
                
                wauc = self.wauroc(preds, trues).detach().cpu().numpy()
                wapr = self.wauprc(preds, trues).detach().cpu().numpy()
                wf1 = self.wf1(preds, trues).detach().cpu().numpy()
                
            elif "bces" == args.loss_types:
                trues = self.y_true_multi.reshape(-1, 12).cuda()
                preds = self.y_pred_multi.reshape(-1, 12).cuda()
                
                auc = []
                apr = []
                f1 = []
                for i in range(12):
                    trues_i = trues[:,i]
                    preds_i = preds[:,i]
                    auc.append(self.auroc(preds_i, trues_i).detach().cpu().numpy())
                    apr.append(self.auprc(preds_i, trues_i).detach().cpu().numpy())
                    f1i = 0
                    for i in range(1, 100):
                        threshold = i / 100.0    
                        temp_output = preds.detach()
                        temp_output[temp_output>=threshold] = 1
                        temp_output[temp_output<threshold] = 0        
                        temp_score = f1_score(temp_output, trues, task="binary", threshold = threshold)
                        if temp_score > f1i:
                            f1i = temp_score
                    f1.append(f1i.detach().cpu().numpy())
                    
                auc = np.array(auc)
                apr = np.array(apr)
                f1 = np.array(f1)
                wauc = np.mean(auc)
                wapr = np.mean(apr)
                wf1 = np.mean(f1)
                
            elif "rmse" == args.loss_types:
                preds = self.y_pred_multi.detach().cpu().numpy()
                rmse_avg = np.mean(preds)
                scores_list = list([np.round(np.array([rmse_avg, rmse_avg, rmse_avg]), 4),
                                np.round(np.array([rmse_avg, rmse_avg, rmse_avg]), 4)])
                
                del rmse_avg
                
            if "rmse" != args.loss_types:
                scores_list = list([np.round(np.array([auc, apr, f1]), 4),
                                    np.round(np.array([wauc, wapr, wf1]), 4)])        
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
                temp_output = preds.detach()
                temp_output[temp_output>=threshold] = 1
                temp_output[temp_output<threshold] = 0        
                temp_score = f1_score(temp_output, trues, task="binary", threshold = threshold)
                if temp_score > f1:
                    f1 = temp_score
            if "rmse" in args.auxiliary_loss_type:
                rmse =  torch.mean(torch.tensor(self.rmse).cuda())
                scores_list = list(np.round(np.array([auc.detach().cpu().numpy(), 
                                                    apr.detach().cpu().numpy(), 
                                                    f1.detach().cpu().numpy(),
                                                    rmse.detach().cpu().numpy()]), 4))    
            else:
                scores_list = list(np.round(np.array([auc.detach().cpu().numpy(), 
                                                    apr.detach().cpu().numpy(), 
                                                    f1.detach().cpu().numpy()]), 4))    
=======
        trues = self.y_true_multi.cuda()
        preds = self.y_pred_multi.cuda()
        auc = self.auroc(preds, trues)
        apr = self.auprc(preds, trues)
        f1 = 0
        
        for i in range(1, 100):
            threshold = i / 100.0    
            temp_output = preds.detach()
            temp_output[temp_output>=threshold] = 1
            temp_output[temp_output<threshold] = 0        
            temp_score = f1_score(temp_output, trues, task="binary", threshold = threshold)
            if temp_score > f1:
                f1 = temp_score
        if "rmse" in args.auxiliary_loss_type:
            rmse =  torch.mean(torch.tensor(self.rmse).cuda())
            scores_list = list(np.round(np.array([auc.detach().cpu().numpy(), 
                                                apr.detach().cpu().numpy(), 
                                                f1.cpu(),
                                                rmse.detach().cpu().numpy()]), 4))    
        else:
            scores_list = list(np.round(np.array([auc.detach().cpu().numpy(), 
                                                apr.detach().cpu().numpy(), 
                                                f1.cpu()]), 4))    
>>>>>>> 4c80bba8089573b532360f10b220efd43bb28567
        
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
        self.rmse = []
