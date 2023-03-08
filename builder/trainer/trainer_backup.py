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
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import datetime
from control.config import args


def binary_classification_vslt(args, iteration, train_x, static_x, input_lengths, train_y, model, logger, device, scheduler=None, optimizer=None, criterion=None, seq_lengths=None, flow_type=None):
    # final_target = train_y.type(torch.LongTensor).squeeze()
    train_x = train_x.permute(1,0,2,3)
    data = train_x[0]
    delta = train_x[1]
    mask = train_x[2]

    data = data.type(torch.FloatTensor)
    final_target = train_y.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    delta = delta.type(torch.FloatTensor)
    mean = args.feature_means.type(torch.FloatTensor)
    length = (input_lengths - 1).type(torch.LongTensor)

    if flow_type == "train":
        optimizer.zero_grad()

        h0 = torch.zeros(data.size(0), args.hidden_size).type(torch.FloatTensor)
        output = model(data, h0, mask, delta, mean, length)
        if args.output_type == "all":
            output = output.squeeze().permute(1,0)
        else:
            output = output.squeeze()
        loss = criterion(output, final_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step(iteration)    
        logger.log_lr(scheduler.get_lr()[0], iteration)

    else:
        test_loss = []

        data = data.type(torch.FloatTensor)

        shift_start = 0
        shift_num = data.shape[1]-args.window_size

        min_interval = args.window_size
        max_interval = args.window_size + args.mortality_after

        for i in range(shift_start, shift_num):
            sliced_data = data[:, i:i+args.window_size, :]
            sliced_mask = mask[:, i:i+args.window_size, :]
            sliced_delta = delta[:, i:i+args.window_size, :]
            sliced_delta = sliced_delta / torch.amax(sliced_delta)
            h0 = torch.zeros(sliced_data.size(0), 64).type(torch.FloatTensor)

            output = model(sliced_data, h0, sliced_mask, sliced_delta, mean, age, gender, length)
            output = output.squeeze()

            seq_not_over = (length - min_interval >= i)

            final_target = final_target.unsqueeze(0)
            negatives = (final_target == 0) | (i + max_interval < final_target)
            positives = (final_target >= i + min_interval) & (final_target <= i + max_interval)
            current_target = positives.float()
            counts_loss = negatives | positives & seq_not_over

            current_target = torch.masked_select(current_target, counts_loss)
            output = torch.masked_select(output, counts_loss)

            loss = criterion(output, current_target)
            loss = torch.mean(loss)
            test_loss.append(loss)

            logger.evaluator.add_batch(np.array(current_target.cpu()), np.array(output.cpu()))

        loss = np.mean(test_loss)

    return model, loss.item()
    # return model, loss

def multiTaskLearningVslt(args, iteration, train_x, static_x, input_lengths, train_y, model, logger, device, scheduler=None, optimizer=None, criterion=None, seq_lengths=None, flow_type=None):
    # final_target = train_y.type(torch.LongTensor).squeeze()
    train_x = train_x.permute(1,0,2,3)
    static_x = static_x.permute(1,0)
    data = train_x[0]
    delta = train_x[1]
    mask = train_x[2]
    age = static_x[1]
    gender = static_x[0]

    data = data.type(torch.FloatTensor)
    final_target = train_y.type(torch.FloatTensor)
    age = age.type(torch.FloatTensor)
    gender = gender.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    delta = delta.type(torch.FloatTensor)
    mean = args.feature_means.type(torch.FloatTensor)
    length = (input_lengths - 1).type(torch.LongTensor)
    print("finally")
    if flow_type == "train":
        optimizer.zero_grad()

        h0 = torch.zeros(data.size(0), args.hidden_size).type(torch.FloatTensor)
        output = model(data, h0, mask, delta, mean, age, gender, length)
        if args.output_type == "all":
            output = output.squeeze().permute(1,0)
        else:
            output = output.squeeze()
        loss = criterion(output, final_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step(iteration)    
        logger.log_lr(scheduler.get_lr()[0], iteration)

    else:
        test_loss = []

        data = data.type(torch.FloatTensor)

        shift_start = 0
        shift_num = data.shape[1]-args.window_size

        min_interval = args.window_size
        max_interval = args.window_size + args.mortality_after

        for i in range(shift_start, shift_num):
            sliced_data = data[:, i:i+args.window_size, :]
            sliced_mask = mask[:, i:i+args.window_size, :]
            sliced_delta = delta[:, i:i+args.window_size, :]
            sliced_delta = sliced_delta / torch.amax(sliced_delta)
            h0 = torch.zeros(sliced_data.size(0), 64).type(torch.FloatTensor)

            output = model(sliced_data, h0, sliced_mask, sliced_delta, mean, age, gender, length)
            output = output.squeeze()

            seq_not_over = (length - min_interval >= i)

            final_target = final_target.unsqueeze(0)
            negatives = (final_target == 0) | (i + max_interval < final_target)
            positives = (final_target >= i + min_interval) & (final_target <= i + max_interval)
            current_target = positives.float()
            counts_loss = negatives | positives & seq_not_over

            current_target = torch.masked_select(current_target, counts_loss)
            output = torch.masked_select(output, counts_loss)

            loss = criterion(output, current_target)
            loss = torch.mean(loss)
            test_loss.append(loss)

            logger.evaluator.add_batch(np.array(current_target.cpu()), np.array(output.cpu()))

        loss = np.mean(test_loss)

    return model, loss.item()

def binary_classification_vs_lt_image(args, iteration, train_x, static_x, image_x, train_y, model, logger, device, scheduler=None, optimizer=None, criterion=None, seq_lengths=None, flow_type="train"):
    # final_target = train_y.type(torch.LongTensor).squeeze()

    if flow_type == "train":
        optimizer.zero_grad()

        train_x = train_x.permute(1,0,2,3)
        static_x = static_x.permute(1,0)
        data = train_x[0]
        delta = train_x[1]
        mask = train_x[2]
        age = static_x[1]

        h0 = torch.zeros(data.size(0), 64).type(torch.FloatTensor)
        data = data.type(torch.FloatTensor)
        final_target = train_y.type(torch.FloatTensor)
        age = age.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor)
        delta = delta.type(torch.FloatTensor)
        mean = args.feature_means.type(torch.FloatTensor)
        length = torch.Tensor([args.window_size-1] * args.batch_size).type(torch.LongTensor)

        output = model(data, h0, mask, delta, mean, age, length, image_x)
        output = output.squeeze()

        loss = criterion(output, final_target)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step(iteration)    
        logger.log_lr(scheduler.get_lr()[0], iteration)
        
    else:
        test_loss = []
        
        train_x = train_x.permute(1,0,2,3)
        static_x = static_x.permute(1,0)
        data = train_x[0]
        delta = train_x[1]
        mask = train_x[2]
        age = static_x[1]

        data = data.type(torch.FloatTensor)
        final_target = train_y.type(torch.FloatTensor)
        age = age.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor)
        delta = delta.type(torch.FloatTensor)
        mean = args.feature_means.type(torch.FloatTensor)
        length = torch.Tensor([args.window_size-1] * args.batch_size).type(torch.LongTensor)
        seq_lengths = torch.Tensor(seq_lengths)

        shift_start = 0
        shift_num = data.shape[1]-args.window_size

        min_interval = args.window_size
        max_interval = args.window_size + args.mortality_after

        for i in range(shift_start, shift_num):
            sliced_data = data[:, i:i+args.window_size, :]
            sliced_mask = mask[:, i:i+args.window_size, :]
            sliced_delta = delta[:, i:i+args.window_size, :]
            sliced_delta = sliced_delta / torch.amax(sliced_delta)
            h0 = torch.zeros(sliced_data.size(0), 64).type(torch.FloatTensor)

            output = model(sliced_data, h0, sliced_mask, sliced_delta, mean, age, length, image_x)
            output = output.squeeze()

            seq_not_over = (seq_lengths - min_interval >= i)

            final_target = final_target.unsqueeze(0)
            negatives = (final_target == 0) | (i + max_interval < final_target)
            positives = (final_target >= i + min_interval) & (final_target <= i + max_interval)
            current_target = positives.float()
            counts_loss = negatives | positives & seq_not_over

            current_target = torch.masked_select(current_target, counts_loss)
            output = torch.masked_select(output, counts_loss)

            loss = criterion(output, current_target)
            loss = torch.mean(loss)
            test_loss.append(loss)

            logger.evaluator.add_batch(np.array(current_target.cpu()), np.array(output.cpu()))

        loss = np.mean(test_loss)

    return model, loss.item()

def binary_classification_txt(args, iteration, train_x, train_y, input_lengths, model, logger, device, scheduler=None, optimizer=None, criterion=None, flow_type=None):
    train_y = train_y.type(torch.FloatTensor)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(train_x, input_lengths.type(torch.LongTensor))
        output = output.squeeze()
        #print("Output ", output)
        #print("Train Y ", train_y)

        loss = criterion(output, train_y.to(torch.long).to(device)).mean()
        loss.backward()
        optimizer.step()
        scheduler.step(iteration)
        logger.log_lr(scheduler.get_last_lr()[0], iteration)

    else:
        output = model(train_x, input_lengths.type(torch.LongTensor))
        output = output.squeeze()
        loss = criterion(output, train_y.to(torch.long).to(device)).mean()
        logger.evaluator.add_batch(np.array(train_y.cpu()), np.array(output[:,1].cpu()))

    return model, loss.item()