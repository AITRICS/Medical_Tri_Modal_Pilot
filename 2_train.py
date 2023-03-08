# Copyright (c) 2022, AITRICS. All rights reserved.
#
# "But seek first his kingdom and his righteousness, and all these things will be given to you as well." (Matthew 6:33)
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
import math

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from control.config import args
from builder.data.data_preprocess import get_data_loader
from builder.models import get_model
from builder.trainer import get_trainer
from builder.utils.utils import *
from builder.utils.result_utils import *
from builder.utils.lars import LARC
from builder.utils.logger import Logger
from builder.utils.cosine_annealing_with_warmup_v2 import CosineAnnealingWarmupRestarts
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle


# set gpu device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# set trainer, setting file, and seed number 
args.seed = 0
make_setting_file(args)
if args.cross_fold_val == 1:
    set_seeds(args)
name_trainer(args)

# define result class
save_valid_results = experiment_results_validation(args)
save_test_results = experiment_results(args)

# get patient_dict: {pat_id: pkl list}
patient_dict, keys_list = patient_wise_ordering(args)
print("Selected Dataset: ", args.train_data_path.split("/")[-2])
if args.cross_fold_val == 1:
    print("K-number of seeds (K-fold-cross-validation): ", len(args.seed_list))
else:
    print("K-number of seeds (K-seeds average): ", len(args.seed_list))

for k_indx, seed_num in enumerate(args.seed_list):
    args.log_fold = k_indx
    if args.cross_fold_val != 1:
        args.seed = seed_num
        set_seeds(args)
    scaler = torch.cuda.amp.GradScaler()
    # set device
    seed_num = 0
    device = set_devices(args)
    args.device = device
    
    # set logger
    logger = Logger(args)
    logger.evaluator.best_auc = 0
    
    print("########## Experiment Begins ##########")
    print(args.input_types)
    print(args.modality_inclusion)
    
    train_loader, val_loader, test_loader = get_data_loader(args, patient_dict, keys_list, k_indx)
    # exit(1)
    # for train_batch in train_loader:
    #     train_x, static_x, train_y, input_lengths, train_img, train_txt, txt_lengths, missing, f_indices = train_batch

    # print("done without any error - Train loader")
    # for train_batch in val_loader:
    #     train_x, static_x, train_y, input_lengths, train_img, train_txt, txt_lengths, missing, f_indices = train_batch
    
    # print("done without any error - Validation loader")
    # for train_batch in test_loader:
    #     train_x, static_x, train_y, input_lengths, train_img, train_txt, txt_lengths, missing, f_indices = train_batch

    # print("done without any error - Test loader")
    # exit(1)

    # get model
    model = get_model(args) 
    model = model(args).to(device)
    
    # set loss function
    # criterion = nn.BCELoss(reduction='mean')
    criterion = nn.BCEWithLogitsLoss(size_average=True, reduction='mean')

    # check whether to use model checkpoint
    if args.checkpoint:
        # check type of model checkpoint
        if args.last:
            ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/last_fold{}.pth'.format(str(k_indx))
        elif args.best:
            ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_fold{}.pth'.format(str(k_indx))
        else:
            raise ValueError('invalid type of model checkpoint: last, best')

        # load model checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        # load train states - best score, epoch
        logger.best_auc = checkpoint['score']
        start_epoch     = checkpoint['epoch']
        # delete checkpoint
        del checkpoint
    
    # no model checkpoint: train model from scratch
    else:
        logger.best_auc = 0
        start_epoch = 1

    # set optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
    
    elif args.optim == 'adam_lars':
        optimizer = optim.Adam(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    
    elif args.optim == 'sgd_lars':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    
    elif args.optim == 'adamw_lars':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    
    else:
        raise ValueError('invalid optimizer: adam, sgd, adamw, adam_lars, sgd_lars, adamw_lars')

    # get number of iteration
    iter_num_per_epoch  = len(train_loader)
    iter_num_total      = args.epochs * iter_num_per_epoch
    print("# of Iterations (per epoch): ",  iter_num_per_epoch)
    print("# of Iterations (total): ",      iter_num_total)

    # set learning scheduler
    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                  first_cycle_steps=args.t_0*iter_num_per_epoch, 
                                                  cycle_mult=args.t_mult, 
                                                  max_lr=args.lr_init * math.sqrt(args.batch_size), 
                                                  min_lr=1e-6,
                                                  warmup_steps=args.t_up*iter_num_per_epoch, gamma=args.gamma)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), 
                                                epochs=args.epochs, steps_per_epoch=iter_num_per_epoch, 
                                                div_factor=math.sqrt(args.batch_size))
    else:
        raise ValueError('invalid lr-scheduler: Single, CosineAnnealing')

    # intialize train step
    model.train()
    iteration               = 0
    total_epoch_iteration   = 0

    # start model training
    pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    for epoch in range(start_epoch, args.epochs+1):
        # initialize vars: (per epoch)
        epoch_losses    = []
        loss            = 0
        logger.loss     = 0
        iter_in_epoch   = 0

        for train_batch in train_loader:
            # get X, y, input_lengths, ...
            train_x, static_x, train_y, input_lengths, train_img, train_txt, txt_lengths, missing, f_indices = train_batch
            if "vslt" in args.input_types:
                input_lengths = input_lengths.to(device)
                static_x      = static_x.to(device)
        
            if "txt" in args.input_types:
                train_txt     = train_txt.to(device)
                txt_lengths   = txt_lengths.to(device)

            if args.auxiliary_loss_input is None:
                f_indices     = None    
            else:
                f_indices   = f_indices.to(device)
                
            if "img" in args.input_types:
                train_img     = train_img.to(device)
        
            # set vars to selected device
            train_x         = train_x.to(device)
            train_y         = train_y.to(device)
            
            # update iter counts
            iteration               += 1
            iter_in_epoch           += 1
            total_epoch_iteration   += 1

            # get trainer: model, iter_loss                    
            model, iter_loss = get_trainer(args              = args,
                                            iteration        = iteration,
                                            x                = train_x,
                                            static           = static_x,
                                            input_lengths    = input_lengths,
                                            y                = train_y,
                                            output_lengths   = f_indices,
                                            model            = model,
                                            logger           = logger,
                                            device           = device,
                                            scheduler        = scheduler,
                                            optimizer        = optimizer,
                                            criterion        = criterion,
                                            x_txt            = train_txt,
                                            x_img            = train_img,
                                            txt_lengths      = txt_lengths,
                                            scaler           = scaler,
                                            missing          = missing,
                                            flow_type        = "train"
                                            )

            # update loss (in logger)
            logger.loss += iter_loss
            # print(logger.loss)
            ### LOGGING
            if iter_in_epoch % args.log_iter == 0:
                logger.log_tqdm(epoch, iter_in_epoch, pbar)
                logger.log_scalars(total_epoch_iteration)

            ### VALIDATION
            # if iteration % (iter_num_per_epoch) == 0:
            if iteration % (iter_num_per_epoch) == 0 and epoch > (args.epochs//2):
                # initialize valid step
                model.eval()
                logger.evaluator.reset()
                val_iteration   = 0
                logger.val_loss = 0

                with torch.no_grad():
                    for idx, val_batch in enumerate(tqdm(val_loader)):
                        # get X, y, input_lengths, ...
                        val_x, val_static_x, val_y, input_lengths, val_img, val_txt, txt_lengths, missing, f_indices = val_batch
            
                        if "vslt" in args.input_types:
                            input_lengths = input_lengths.to(device)
                            val_static_x  = val_static_x.to(device)
                    
                        if "txt" in args.input_types:
                            val_txt       = val_txt.to(device)
                            txt_lengths   = txt_lengths.to(device)

                        if args.auxiliary_loss_input is None:
                            f_indices     = None    
                        else:
                            f_indices     = f_indices.to(device)
                            
                        if "img" in args.input_types:
                            val_img       = val_img.to(device)
                    
                        # set vars to selected device
                        val_x             = val_x.to(device)
                        val_y             = val_y.to(device)
                        
                        # input_lengths   = input_lengths.to(device)

                        # get trainer: model, val_loss
                        model, val_loss = get_trainer(args   = args,
                                            iteration        = iteration,
                                            x                = val_x,
                                            static           = val_static_x,
                                            input_lengths    = input_lengths,
                                            y                = val_y,
                                            output_lengths   = f_indices,
                                            model            = model,
                                            logger           = logger,
                                            device           = device,
                                            scheduler        = scheduler,
                                            optimizer        = optimizer,
                                            criterion        = criterion,
                                            x_txt            = val_txt,
                                            x_img            = val_img,
                                            txt_lengths      = txt_lengths,
                                            scaler           = scaler,
                                            missing          = missing,
                                            flow_type        = "test"
                                            )
                        
                        
                        # update loss, iter count
                        logger.val_loss += val_loss
                        val_iteration   += 1

                    # update logger - end of valid step
                    logger.log_val_loss(val_iteration, iteration)
                    logger.add_validation_logs(iteration)
                    logger.save(model, optimizer, iteration, epoch, str(k_indx))

                # reset to train mode
                model.train()

        # update progress bar - end of epoch
        pbar.update(1)
    
    logger.val_result_only()
    save_valid_results.results_all_seeds(logger.val_results)
    
    # get model checkpoint - end of train step
    # initalize model (again)
    del model
    model = get_model(args) 
    model = model(args).to(device)
    # load model checkpoint  
    if args.last:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/last_fold{}_seed{}.pth'.format(str(k_indx), str(args.seed))
    elif args.best:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_fold{}_seed{}.pth'.format(str(k_indx), str(args.seed))
    if not os.path.exists(ckpt_path):
        print("Final model for test experiment doesn't exist...")
        exit(1)
    # load model & state
    ckpt    = torch.load(ckpt_path, map_location=device)
    state   = {k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(state)

    # initialize test step
    model.eval()
    logger.evaluator.reset()

    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), 
                               bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            # get X, y, input_lengths, ...
            test_x, test_static_x, test_y, input_lengths, test_img, test_txt, txt_lengths, missing, f_indices = test_batch
            if "vslt" in args.input_types:
                input_lengths = input_lengths.to(device)
                test_static_x = test_static_x.to(device)
        
            if "txt" in args.input_types:
                test_txt      = test_txt.to(device)
                txt_lengths   = txt_lengths.to(device)

            if args.auxiliary_loss_input is None:
                f_indices     = None    
            else:
                f_indices     = f_indices.to(device)
                
            if "img" in args.input_types:
                test_img      = test_img.to(device)
        
            # set vars to selected device
            test_x            = test_x.to(device)
            test_y            = test_y.to(device)

            # get trainer: model
            model, _ = get_trainer(args   = args,
                                    iteration        = iteration,
                                    x                = test_x,
                                    static           = test_static_x,
                                    input_lengths    = input_lengths,
                                    y                = test_y,
                                    output_lengths   = f_indices,
                                    model            = model,
                                    logger           = logger,
                                    device           = device,
                                    scheduler        = scheduler,
                                    optimizer        = optimizer,
                                    criterion        = criterion,
                                    x_txt            = test_txt,
                                    x_img            = test_img,
                                    txt_lengths      = txt_lengths,
                                    scaler           = scaler,
                                    missing          = missing,
                                    flow_type        = "test"
                                    )

    # update logger - end of test step
    logger.test_result_only()
    logger.writer.close()
    del model
    
    # save test results
    save_test_results.results_all_seeds(logger.test_results)

# check: whether to save cross-validation results or seed average
save_test_results.results_per_cross_fold()
save_valid_results.results_per_cross_fold()