import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn

from control.config import args
from builder.data.data_preprocess import get_test_data_loader
from builder.models import get_model
from builder.utils.utils import search_walk
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
from builder.trainer.trainer import *
from builder.utils.utils import set_seeds, set_devices
from builder.trainer import get_trainer
import warnings
from builder.utils.result_utils import *
from builder.utils.utils import *

# log_directory = os.path.join(args.dir_result, args.project_name)
# settings_file_name = os.path.join(log_directory, "settings.txt")
# settings_file = open(settings_file_name, 'r')

# # Settings which are not taken from file
# blacklist_settings = ["seed", "seed_list", "device", "gpus", "reset", "project_name", "checkpoint", \
#                     "train_data_path", "test_data_path", "dir_result", "image_data_path", "dir_root", "train_mode"]

# # Settings which are of string type
# string_settings = ["lr_scheduler", "input_types", "output_type", "predict_type", "modality_inclusion", "model", \
#                     "activation", "optim", "txt_tokenization"]

# # Settings which are of boolean type
# boolean_settings = ["cross_fold_val", "carry_back", "patient_time", "quantization", "best", "last", "show_roc"]

# settings_dict = {}
# while True:
#     line = settings_file.readline().strip()
#     if not line:
#         break
#     setting = line.split(" # ")[0]
#     value = line.split(" # ")[1]

#     # print(setting)

#     if setting in blacklist_settings:
#         continue

#     elif setting in string_settings:
#         args.__dict__[setting] = value
    
#     elif setting in boolean_settings:
#         if value == "False":
#             args.__dict__[setting] = False
#         else:
#             args.__dict__[setting] = True
    
#     elif "vitalsign_labtest" in setting:
#         args.__dict__[setting] = value.split("'")[1::2]
    
#     elif "." in value or "-" in value:
#         args.__dict__[setting] = float(value)
#     else:
#         args.__dict__[setting] = int(value)

# warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# set trainer, setting file, and seed number 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
args.seed = 0

if args.cross_fold_val == 1:
    set_seeds(args)
    
# define result class
save_valid_results = experiment_results_validation(args)
save_test_results = experiment_results_test(args)

label_method_max = True
scheduler = None
optimizer = None

# get patient_dict: {pat_id: pkl list}
patient_dict, keys_list = patient_wise_ordering(args)
print("Selected Dataset: ", args.train_data_path.split("/")[-2])
if args.cross_fold_val == 1:
    print("K-number of seeds (K-fold-cross-validation): ", len(args.seed_list))
else:
    print("K-number of seeds (K-seeds average): ", len(args.seed_list))



iteration = 1
name = args.project_name

result_dir = search_walk({"path": args.dir_result + "/" + name + "/ckpts", "extension": ".pth"})
print("result",result_dir)



for model_numm, ckpt in enumerate(result_dir):
    args.log_fold = model_numm
    if args.cross_fold_val != 1:
        args.seed = args.seed_list[model_numm]
        set_seeds(args)
    scaler = torch.cuda.amp.GradScaler()
    # set device
    seed_num = 0
    device = set_devices(args)
    args.device = device
    
    test_loader = get_test_data_loader(args, patient_dict, keys_list)
    model = get_model(args)
    model = model(args).to(device)
    
    # set logger
    logger = Logger(args)
    if args.model_types == "classification" and args.loss_types == "rmse":
        logger.evaluator.best_auc = float('inf')
    else:
        logger.evaluator.best_auc = 0
    logger.loss = 0
    
    print("########## Experiment Begins ##########")
    print(args.input_types)
    print(args.modality_inclusion)
    
    model_ckpt = torch.load(ckpt, map_location = device)
    state = {k:v for k,v in model_ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()

    print('Model for Seed ' + ckpt.split("seed")[-1].split(".pth")[0] + " Loaded...")

    logger.evaluator.reset()
    result_list = []
    # set loss function
    if args.model_types == "classification":
        if "softmax" == args.loss_types:
            criterion = nn.CrossEntropyLoss(reduction='mean')
            args.output_dim = 12
        elif "bces" == args.loss_types: 
            criterion = nn.BCEWithLogitsLoss(size_average=True, reduction='mean')    
            args.output_dim = 12
        elif "bceandsoftmax" == args.loss_types:
            criterion = (nn.CrossEntropyLoss(reduction='mean'), nn.BCEWithLogitsLoss(size_average=True, reduction='mean'))    
            args.output_dim = 12
        elif "rmse" == args.loss_types: 
            criterion = nn.MSELoss(reduction='none')
            args.output_dim = 1
        
    elif args.model_types == "detection":   
        criterion = nn.BCEWithLogitsLoss(size_average=True, reduction='mean')
        args.output_dim = 1
        
    pad_id = 0
    criterion_img_aux = nn.CrossEntropyLoss(ignore_index = pad_id).to(device, non_blocking=True)
    criterion_vslt_aux = nn.MSELoss(reduction='none')

    
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            
            # get X, y, input_lengths, ...
            test_x, test_static_x, test_y, input_lengths, test_img, img_time, test_txt, txt_lengths, txt_time, missing, f_indices, test_y2, test_reports_tokens, test_reports_lengths = test_batch
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
                if "tdecoder" in args.auxiliary_loss_type:
                    test_reports_tokens = test_reports_tokens.to(device)
                    test_reports_lengths = test_reports_lengths.to(device)
        
            # set vars to selected device
            test_x            = test_x.type(torch.HalfTensor).to(device)
            if "rmse" in args.auxiliary_loss_type:
                test_y         = (test_y.to(device, non_blocking=True), test_y2.to(device, non_blocking=True))
            else:
                test_y         = test_y.to(device, non_blocking=True)

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
                                    imgtxt_time      = (img_time, txt_time),
                                    scaler           = scaler,
                                    missing          = missing,
                                    flow_type        = "test",
                                    reports_tokens   = test_reports_tokens,
                                    reports_lengths  = test_reports_lengths,
                                    criterion_aux    = (criterion_img_aux, criterion_vslt_aux)
                                    )

    # update logger - end of test step
    logger.test_result_only()
    logger.writer.close()
    del model
    # save test results
    save_test_results.results_all_seeds(logger.test_results)
    
save_test_results.results_per_cross_fold()