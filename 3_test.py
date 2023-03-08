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

log_directory = os.path.join(args.dir_result, args.project_name)
settings_file_name = os.path.join(log_directory, "settings.txt")
settings_file = open(settings_file_name, 'r')

# Settings which are not taken from file
blacklist_settings = ["seed", "seed_list", "device", "gpus", "reset", "project_name", "checkpoint", \
                    "train_data_path", "test_data_path", "dir_result", "image_data_path", "dir_root", "train_mode"]

# Settings which are of string type
string_settings = ["lr_scheduler", "input_types", "output_type", "predict_type", "modality_inclusion", "model", \
                    "activation", "optim", "txt_tokenization"]

# Settings which are of boolean type
boolean_settings = ["cross_fold_val", "carry_back", "patient_time", "quantization", "best", "last", "show_roc"]

settings_dict = {}
while True:
    line = settings_file.readline().strip()
    if not line:
        break
    setting = line.split(" # ")[0]
    value = line.split(" # ")[1]

    # print(setting)

    if setting in blacklist_settings:
        continue

    elif setting in string_settings:
        args.__dict__[setting] = value
    
    elif setting in boolean_settings:
        if value == "False":
            args.__dict__[setting] = False
        else:
            args.__dict__[setting] = True
    
    elif "vitalsign_labtest" in setting:
        args.__dict__[setting] = value.split("'")[1::2]
    
    elif "." in value or "-" in value:
        args.__dict__[setting] = float(value)
    else:
        args.__dict__[setting] = int(value)

warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

label_method_max = True
scheduler = None
optimizer = None
# criterion = nn.CrossEntropyLoss(reduction='none')
criterion = nn.BCELoss(reduction='mean')
iteration = 1
set_seeds(args)
device = set_devices(args)
logger = Logger(args)
logger.loss = 0
# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

# select trainer
if args.input_types == 'vslt':
    if args.predict_type == "binary":
        args.trainer = "binary_classification_vslt"
    
    elif args.predict_type == "multi_task_within":
        args.trainer = "multi_task_within"
        
    elif args.predict_type == "multi_task_range":
        args.trainer = "multi_task_range"

elif args.input_types == "txt":
    if args.model == "lstm":
        args.trainer = "binary_classification_txt_lstm"
    else:
        args.trainer = "binary_classification_txt"
    
elif args.input_types == "img":
    args.trainer = "binary_classification_img"

elif args.input_types == "vslt_txt":
    if args.predict_type == "multi_task_within":
        args.trainer = "multi_task_within_txt"

    elif args.predict_type == "multi_task_range":
        args.trainer = "multi_task_range_txt"
        
else:
    raise NotImplementedError("unimodal: vslt, img, txt / bimodal: vslt_txt")

test_loader = get_test_data_loader(args)
model = get_model(args)
model = model(args).to(device)
name = args.project_name

result_dir = search_walk({"path": args.dir_result + "/" + name + "/ckpts", "extension": ".pth"})


for model_numm, ckpt in enumerate(result_dir):
    model_ckpt = torch.load(ckpt, map_location = device)
    state = {k:v for k,v in model_ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()

    print('Model for Seed ' + ckpt.split("seed")[-1].split(".pth")[0] + " Loaded...")

    logger.evaluator.reset()
    result_list = []

    
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            
            # get X, y, input_lengths, ...
            if "binary" in args.trainer:
                if args.input_types == 'img':
                    test_x, test_y = test_batch
                    input_lengths  = None
                    test_static_x  = None
                else:
                    test_x, test_y, input_lengths = test_batch
                    test_static_x = None

            elif 'multi_task' in args.trainer:
                if args.input_types == 'vslt':
                    test_x, test_static_x, test_y, input_lengths = test_batch
                    test_static_x = test_static_x.to(device)

                elif args.input_types == "vslt_txt":
                    test_x, test_static_x, test_y, input_lengths, test_txt, txt_lengths = test_batch

            # set vars to selected device
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            # get trainer: model
            if 'binary' in args.predict_type:
                if args.input_types == 'vslt':  # binary_classification_vslt
                    model, _ = get_trainer(args             = args, 
                                           iteration        = iteration, 
                                           x                = test_x,  
                                           static           = test_static_x, 
                                           input_lengths    = input_lengths, 
                                           y                = test_y, 
                                           output_lengths   = None,
                                           model            = model, 
                                           logger           = logger, 
                                           device           = device, 
                                           scheduler        = scheduler, 
                                           optimizer        = optimizer, 
                                           criterion        = criterion,
                                           flow_type        = "test"
                                          )
                elif args.input_types == 'txt': # binary_classification_txt, binary_classification_txt_lstm 
                    model, _ = get_trainer(args             = args,
                                           iteration        = iteration,
                                           x                = test_x,
                                           static           = None,
                                           input_lengths    = input_lengths,
                                           y                = test_y,
                                           output_lengths   = None,
                                           model            = model,
                                           logger           = logger,
                                           device           = device,
                                           scheduler        = scheduler,
                                           optimizer        = optimizer,
                                           criterion        = criterion,
                                           flow_type        = "test"
                                          )
                else:
                    model, _ = get_trainer(args             = args,
                                           iteration        = iteration,
                                           x                = test_x,
                                           static           = None,
                                           input_lengths    = None,
                                           y                = test_y,
                                           output_lengths   = None,
                                           model            = model,
                                           logger           = logger,
                                           device           = device,
                                           scheduler        = scheduler,
                                           optimizer        = optimizer,
                                           criterion        = criterion,
                                           flow_type        = "test"
                                          )
            
            elif 'multi_task' in args.predict_type:
                if args.input_types == 'vslt': # multi_task_within, multi_task_range 
                    model, _ = get_trainer(args             = args, 
                                           iteration        = iteration, 
                                           x                = test_x,  
                                           static           = test_static_x, 
                                           input_lengths    = input_lengths, 
                                           y                = test_y, 
                                           output_lengths   = None,
                                           model            = model, 
                                           logger           = logger, 
                                           device           = device, 
                                           scheduler        = scheduler, 
                                           optimizer        = optimizer, 
                                           criterion        = criterion,
                                           flow_type        = "test"
                                          )

                elif args.input_types == "vslt_txt": # multi_task_within_txt, multi_task_range_txt
                    model, _ = get_trainer(args             = args,
                                           iteration        = iteration,
                                           x                = test_x,
                                           static           = test_static_x,
                                           input_lengths    = input_lengths,
                                           y                = test_y,
                                           x_txt            = test_txt,
                                           txt_lengths      = txt_lengths,
                                           output_lengths   = None,
                                           model            = model,
                                           logger           = logger,
                                           device           = device,
                                           scheduler        = scheduler,
                                           optimizer        = optimizer,
                                           criterion        = criterion,
                                           flow_type        = "test"
                                          )

    logger.test_result_only()