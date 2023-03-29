
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
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
from control.config import args

def missing_trainer(args, iteration, train_x, static_x, input_lengths, train_y, 
                                            model, logger, device, scheduler=None, optimizer=None, criterion=None, 
                                            scaler=None, flow_type=None, output_lengths=None, 
                                            seq_lengths=None, x_img=None, x_txt=None, txt_lengths=None, imgtxt_time=None, missing=None, reports_tokens=None, reports_lengths=None):
    
    # (tensor([[0., 0., 0.],
    #     [0., 0., 1.],
    #     [0., 1., 0.],
    #     [0., 1., 1.]])
    # print("train_x: ", train_x.shape)  # TIE: ([batch_size, 10000, 3]) Carry-forward:  torch.Size([batch_size, 3, 24, 16])
    img_time, txt_time = imgtxt_time
    img_time = img_time.type(torch.HalfTensor).to(device, non_blocking=True)
    txt_time = txt_time.type(torch.HalfTensor).to(device, non_blocking=True)
    
    if args.vslt_type == "carryforward":
        train_x = train_x.permute(1, 0, 2, 3)
        data = train_x[0]
        # most likely for GRUD
        h0 = torch.zeros(data.size(0), args.hidden_size).to(device, non_blocking=True)
        mask = train_x[1]
        delta = train_x[2]

    else: # TIE
        max_input_lengths = torch.max(input_lengths).detach().clone()
        data = train_x[:, :max_input_lengths, :]
        h0 = None
        mask = None
        delta = None
    mean = args.feature_means.to(device, non_blocking=True)
        
    if args.model_types == "classification":
        if "softmax" == args.loss_types:
            final_target = train_y.type(torch.LongTensor).to(device, non_blocking=True)
        elif "bces" == args.loss_types: # 1
            final_target = train_y.type(torch.FloatTensor).to(device, non_blocking=True)
        elif "rmse" == args.loss_types:
            final_target = train_y.type(torch.FloatTensor).to(device, non_blocking=True)
            final_target = final_target.unsqueeze(1)
        elif "bceandsoftmax" == args.loss_types: # 1
            final_target = (train_y[0].type(torch.LongTensor).to(device, non_blocking=True), train_y[1].type(torch.FloatTensor).to(device, non_blocking=True))
    elif args.model_types == "bce_rmse":
        final_target = (train_y[0].type(torch.FloatTensor).to(device, non_blocking=True), train_y[1].type(torch.FloatTensor).to(device, non_blocking=True).unsqueeze(1))
    else:
        final_target = train_y.type(torch.FloatTensor).to(device, non_blocking=True)
    
    if args.fullmodal_definition == "txt1":
        missing = torch.stack([missing[:,0], missing[:,2]]).permute(1,0).detach().clone() # in case of vslt_txt
        sample_missing = torch.tensor([[0., 0.],
        [0., 1.]])
        if flow_type == "train" and args.multitoken:
            final_target = final_target.repeat(2,1,1).permute(1,0,2).reshape(-1,12) # [2, 64, 12] -> [64, 2, 12] -> [128, 12]
        
    elif args.fullmodal_definition == "img1":
        sample_missing = torch.tensor([[0., 0.],
        [0., 1.]])
        missing = missing[:,:2].detach().clone() # in case of vslt_img
        if flow_type == "train" and args.multitoken:
            final_target = final_target.repeat(2,1,1).permute(1,0,2).reshape(-1,12) # [2, 64, 12] -> [64, 2, 12] -> [128, 12]
    
    else:
        sample_missing = torch.tensor([[0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 1.]])
        missing = missing.detach().clone()
        # if flow_type == "train" and args.multitoken:
        #     final_target = final_target.repeat(4,1,1).permute(1,0,2).reshape(-1,12) # [2, 64, 12] -> [64, 2, 12] -> [128, 12]
    
    
    sample_missing = torch.cat([sample_missing, missing], dim = 0)
    _, missing_num = torch.unique(sample_missing, dim=0, sorted=True, return_inverse=True)
    
    missing_num = missing_num[4:].type(torch.LongTensor).cuda()
    if args.fullmodal_definition == "txt1_img1":
        missing_multitoken = torch.tensor([[0., 0., 0., 0.],
        [1., 0., 1., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.]]).cuda()
        missing = missing_multitoken[missing_num]
    missing = missing.cuda()
    
    static_x = static_x.permute(1,0)
    feasible_indices = None
    if output_lengths is not None:
        feasible_indices = output_lengths.type(torch.IntTensor).to(device, non_blocking=True)

    # Static Data
    age = static_x[1]
    gender = static_x[0]
    age = age.type(torch.FloatTensor).to(device, non_blocking=True)
    gender = gender.type(torch.FloatTensor).to(device, non_blocking=True)
    x_txt = x_txt.to(device, non_blocking=True)
    
    if flow_type == "train":
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if "tdecoder" not in args.auxiliary_loss_type:
                output, aux_loss = model(data, h0, mask, delta, mean, age, gender, input_lengths, x_txt, txt_lengths, x_img, missing_num, feasible_indices, img_time, txt_time)
            else:
                output, aux_loss = model(data, h0, mask, delta, mean, age, gender, input_lengths, x_txt, txt_lengths, x_img, missing_num, feasible_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths)
            output = output.squeeze()
            
            if "bceandsoftmax" == args.loss_types:
                loss1 = criterion[0](output, final_target[0])
                loss2 = criterion[1](output, final_target[1])
                loss = loss1 + loss2
            elif "rmse" == args.loss_types:
                loss = criterion(output, final_target)
            elif  args.model_types == "bce_rmse":
                loss1 = criterion[0](output[:,0], final_target[0])
                loss2 = criterion[1](output[:,1], final_target[1])[:,0]
                loss2 = torch.sqrt(torch.mean(loss2[final_target[0] == 1]))
                loss2 = torch.nan_to_num(loss2, nan=0.0)
                loss = loss1 + loss2
            else:
                loss = criterion(output, final_target)
                
            if aux_loss is not None:
                loss = loss + (args.auxiliary_loss_weight * aux_loss)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        # optimizer.step()
        scheduler.step(iteration)
        logger.log_lr(scheduler.get_lr()[0], iteration)
    else:
        test_loss = []
        with torch.cuda.amp.autocast():
            if "tdecoder" not in args.auxiliary_loss_type:
                output, aux_loss = model(data, h0, mask, delta, mean, age, gender, input_lengths, x_txt, txt_lengths, x_img, missing_num, feasible_indices, img_time, txt_time)
            else:
                output, aux_loss = model(data, h0, mask, delta, mean, age, gender, input_lengths, x_txt, txt_lengths, x_img, missing_num, feasible_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths)
            output = output.squeeze()
            loss2 = None
            if "bceandsoftmax" == args.loss_types:
                loss1 = criterion[0](output, final_target[0])
                loss2 = criterion[1](output, final_target[1])
                final_target = final_target[0]
                loss = loss1 + loss2
            elif  args.model_types == "bce_rmse":
                loss1 = criterion[0](output[:,0], final_target[0])
                loss2 = criterion[1](output[:,1], final_target[1])[:,0]
                loss2 = torch.sqrt(torch.mean(loss2[final_target[0] == 1]))
                loss2 = torch.nan_to_num(loss2, nan=0.0)
                output = output[:,0]
                final_target = final_target[0]
                
                loss = loss1 + loss2
            else:
                loss = criterion(output, final_target)
            
            if args.model_types == "classification":
                if "rmse" == args.loss_types:
                    output = loss
                else:
                    output = torch.softmax(output, dim=1)
            else: # detection
                output = torch.sigmoid(output)
            
        test_loss.append(loss)
        logger.evaluator.add_batch(final_target, output)

    return model, loss.item()

