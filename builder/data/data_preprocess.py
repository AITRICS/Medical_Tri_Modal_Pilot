import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from builder.utils.utils import *
from builder.data.data_utils import *
from builder.data.dataset_new import *

# 실시간 데이터 전처리기 구상도
# [vslt + txt + img VS. vslt + txt VS. vslt + img VS. vslt]ㅌ
# 	a. training dataset: missing data or full data
# 	b. test dataset: missing data or full data
 
# 'trian-full_test-full', 'trian-missing_test-missing', 'trian-full_test-missing'
# 1-1. train-full의 기준: vslt + txt + img
# 1-2. train-missing의 기준: vslt
# 2-1. test-full의 기준: vslt + txt + img
# 2-2. test-missing의 기준: vslt


def get_data_loader(args, patient_dict, keys_list, k_indx):
    # if args.cross_fold_val == 1:
    #     folds       = list(range(len(args.seed_list)))
    #     folds_val   = folds.pop(k_indx)
    #     train_keys  = [keys_list[fold] for fold in folds]
    #     train_keys  = [item for sublist in train_keys for item in sublist]
    #     val_keys    = keys_list[folds_val]
    # else:
    train_keys  = keys_list[0]
    val_keys    = keys_list[1]
    # flatten to data list & shuffle train set (again)
    val_data_list   = [patient_dict[key] for key in val_keys]
    val_data_list   = [item for sublist in val_data_list for item in sublist]
    train_data_list = [patient_dict[key] for key in train_keys]
    train_data_list = [item for sublist in train_data_list for item in sublist]

    random.shuffle(train_data_list)
    
    # get test data
    test_dir = search_walk({'path': args.test_data_path, 'extension': ".pkl"})
    args.vslt_mask = [True if i not in args.vitalsign_labtest else False for i in VITALSIGN_LABTEST] # True to remove

    print("train_data_list: ", len(train_data_list))
    print("val_data_list: ", len(val_data_list))
    print("test_dir: ", len(test_dir))
    # train_data_list = train_data_list[:2048]
    # val_data_list = val_data_list[:512]
    # test_dir = test_dir[:512]

    # For severance VC dataset, only onset time exist for intubation...!
    # For MIMIC-ER dataset, future 24 hrs of labtest is measured...
    # For dataset, we load every input data and differentiate target types according to args.predict_type
    if args.output_type == 'mortality':
        train_data        = Onetime_Outbreak_Training_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data          = Onetime_Outbreak_Test_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data         = Onetime_Outbreak_Test_Dataset(args, data=test_dir, data_type="test dataset")

    elif args.output_type == 'cpr' or args.output_type == 'intubation' or\
            args.output_type == 'vasso' or args.output_type == 'transfer':
        train_data        = Multiple_Outbreaks_Training_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data          = Multiple_Outbreaks_Test_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data         = Multiple_Outbreaks_Test_Dataset(args, data=test_dir, data_type="test dataset")
    
    # set sampler - target type (0: negative, 1: positive, 2: currently negative)
    if args.output_type != "seq":
        class_sample_count = np.unique(train_data._type_list, return_counts=True)[1]
        print("class_sample_count: ", class_sample_count)
        weight = 1. / class_sample_count
        print("weight: ", weight)
        samples_weight = weight[train_data._type_list]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler        = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    # args.feature_mins     = torch.Tensor(train_data.train_min)
    # args.feature_maxs     = torch.Tensor(train_data.train_max)
    # args.feature_max_mins = args.feature_maxs - args.feature_mins
    args.feature_mins = torch.Tensor([0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 9.0, 0.0, 5.0, 0.0, 0.0, 0.94, 2.0, 0.0, 0.0, 0.8, 67.0, 0.2])
    args.feature_maxs = torch.Tensor([295.0, 120.0, 43.05555555556, 299.0, 298.0, 100.0, 15.0, 68.6, 1000.0, 100.0, 75.0, 9.38, 50.0, 20.0, 20.0, 14.7, 185.0, 531.3])
    args.feature_max_mins = args.feature_maxs - args.feature_mins
    args.feature_means    = torch.Tensor(train_data.feature_means)    
    args.feature_means = np.delete(args.feature_means, args.vslt_mask, axis = 0) 
    
    # For validation dataset, we prioritize full window_size subsequences over smaller ones
    # The indexes for sampling are random for every iteration
    # For testing dataset, we prioritize full window_size subsequences over smaller ones
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True, sampler=sampler)

    val_loader   = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True)                
    test_loader  = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True)
            
    return train_loader, val_loader, test_loader


def get_test_data_loader(args, patient_dict, keys_list):
    train_keys  = keys_list[0]
    # flatten to data list & shuffle train set (again)
    train_data_list = [patient_dict[key] for key in train_keys]
    train_data_list = [item for sublist in train_data_list for item in sublist]

    random.shuffle(train_data_list)
    # get test data
    test_dir = search_walk({'path': args.test_data_path, 'extension': ".pkl"})
    args.vslt_mask = [True if i not in args.vitalsign_labtest else False for i in VITALSIGN_LABTEST] # True to remove
    
    if args.output_type == 'mortality':
        train_data        = Onetime_Outbreak_Training_Dataset(args, data=train_data_list, data_type="training dataset")
        test_data         = Onetime_Outbreak_Test_Dataset(args, data=test_dir, data_type="test dataset")
    elif args.output_type == 'cpr' or args.output_type == 'intubation' or\
            args.output_type == 'vasso' or args.output_type == 'transfer':
        train_data        = Multiple_Outbreaks_Training_Dataset(args, data=train_data_list, data_type="training dataset")
        test_data         = Multiple_Outbreaks_Test_Dataset(args, data=test_dir, data_type="test dataset")
    
    args.feature_mins = torch.Tensor([0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 9.0, 0.0, 5.0, 0.0, 0.0, 0.94, 2.0, 0.0, 0.0, 0.8, 67.0, 0.2])
    args.feature_maxs = torch.Tensor([295.0, 120.0, 43.05555555556, 299.0, 298.0, 100.0, 15.0, 68.6, 1000.0, 100.0, 75.0, 9.38, 50.0, 20.0, 20.0, 14.7, 185.0, 531.3])
    args.feature_max_mins = args.feature_maxs - args.feature_mins
    args.feature_means    = torch.Tensor(train_data.feature_means)    
    args.feature_means = np.delete(args.feature_means, args.vslt_mask, axis = 0)    
    test_loader  = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True)
    return test_loader