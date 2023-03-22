import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from builder.data.data_utils import *
import pickle5 as pickle
import math 

FEATURE_LIST = [
    'PULSE', 'RESP', 'TEMP', 'SBP', 'DBP', 'SpO2', 'GCS',
    'HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
    'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP',
]

FEATURE_MEAN = { 
    'PULSE'     : 85.93695802, 
    'RESP'      : 20.10544135, 
    'TEMP'      : 36.97378611, 
    'SBP'       : 120.00165406, 
    'DBP'       : 62.85878326, 
    'SpO2'      : 96.7560417, 
    'GCS'       : 14.58784295, 
    'HEMATOCRIT': 29.44163972, 
    'PLATELET'  : 200.15499694, 
    'WBC'       : 12.11825286, 
    'BILIRUBIN' : 3.79762327, 
    'pH'        : 7.37816261, 
    'HCO3'      : 24.38824869, 
    'CREATININE': 1.5577265, 
    'LACTATE'   : 2.51239096, 
    'POTASSIUM' : 4.12411448, 
    'SODIUM'    : 138.91951009, 
    'CRP'       : 88.96706267,
} # feature mean values gained from only training dataset...

def optimizer(args):

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
    
    return optimizer

def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False

def make_setting_file(args) -> None:
    print(f"### Project name is: {args.project_name} ###")
    log_directory = os.path.join(args.dir_result, args.project_name)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    settings_file_name = os.path.join(log_directory, "settings.txt")
    settings_file = open(settings_file_name, 'w')

    for key in args.__dict__:
        settings_file.write(key + " # " + str(args.__dict__[key]) + "\n")

    settings_file.close()
    

def carry_forward(np_arr):
    # for (seq_len x feature_num)
    df = pd.DataFrame(np_arr, columns=FEATURE_LIST)
    df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    # df = df.fillna(value=FEATURE_MEAN)
    df = df.fillna(0.0)
    data = df.to_numpy()
    
    return data

def set_seeds(args) -> None:
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

def set_devices(args):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False ### ==torch.use_deterministic_algorithms(True)
    
	if args.cpu or not torch.cuda.is_available():
		return torch.device('cpu')
	else:
		return torch.device('cuda')

def search_walk(info):### 파일 돌아가면서 가지고 옮
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    if searched_list:
        return searched_list
    else:
        return False

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def patient_wise_ordering(args):
    train_dir = search_walk({"path": args.train_data_path, "extension": ".pkl"})
    patient_dict = {}
    for pkl in train_dir:
        pat_id = pkl.split("/")[-1].split("_")[0]
        patient_dict[pat_id] = patient_dict.get(pat_id, list())
        patient_dict[pat_id].append(pkl)

    keys = sorted(list(patient_dict.keys()))
    
    if args.cross_fold_val == 1:
        keys_list = sorted(partition(keys, len(args.seed_list)))
    else:
        # random.seed(args.seed)
        # random.shuffle(keys)
        keys = sorted(keys)
        val, train = np.split(keys, [int(len(keys)*(args.val_data_ratio/0.9))])
        keys_list = [train, val]
    
    return patient_dict, keys_list

def text_patient_wise_ordering(args):
    datasetName = args.train_data_path.split("/")[-2]
    train_dir = "builder/data/text/textDataset/" + datasetName + "_train_" + args.txt_tokenization + "_textDataset.txt"
    
    patient_dict = {}
    with open(train_dir, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            patientId = int(line.split("/")[0].split()[0])
            patient_dict[patientId] = patient_dict.get(patientId, list())
            patient_dict[patientId].append(line.strip())
    
    keys = list(patient_dict.keys())
    keys_list = sorted(partition(keys, len(args.seed_list)))
    
    return patient_dict, keys_list

def img_patient_wise_ordering(args):
    train_dir = search_walk({"path": args.train_data_path, 
                             "extension": ".pkl"})
    patient_dict = dict()
    for pkl in train_dir:
        filename = pkl.split('/')[-1]
        pat_id   = filename.split('_')[0]
        # check: select samples only with cxr images
        if 'img1' not in filename:
            continue
        patient_dict[pat_id] = patient_dict.get(pat_id, list())
        patient_dict[pat_id].append(pkl)    # {pat_id: list(pkl_data)}
    
    keys = list(patient_dict.keys())
    keys_list = sorted(partition(keys, len(args.seed_list)))
    print('[img] train patients: ', len(patient_dict.keys()))
    
    return patient_dict, keys_list

def onetime_outbreak_valdataset_maker(args, patdictPath, winsizePath):
    train_dir = search_walk({"path": args.train_data_path, "extension": ".pkl"})
    _data_list = []
    patDict = {}
    winDict = {}
    # txtDict = txtDictLoad("train")
    txtDict = txtDictLoad("train")
    txtDict.update(txtDictLoad("test"))

    # time_len = 0
    for idx, pkl_path in enumerate(tqdm(train_dir, desc="Creating fixed indices of validation data...")):
        file_name = pkl_path.split("/")[-1]
        
        with open(pkl_path, 'rb') as _f:
            data_info = pkl.load(_f)
            
        # data_in_time = data_info['data_in_time']
        # if time_len < data_in_time.shape[0]:
        #     time_len = data_in_time.shape[0]
        #     print(time_len)
        
        if "cxr_input" in data_info:
            if data_info["cxr_input"] == None:
                del data_info["cxr_input"]
                    
        if "cxr_input" in data_info:
            new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
            if len(new_cxr_inputs) > 0:
                data_info["cxr_input"] = new_cxr_inputs
            else:
                del data_info['cxr_input']
                file_name = file_name.replace("_img1", "_img0")
    
        if 'test-full' in args.modality_inclusion:
            if args.fullmodal_definition not in file_name:
                continue
            if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                continue
            if "txt1" in args.fullmodal_definition:
                if (int(data_info['pat_id']), int(data_info['chid'])) not in txtDict:
                    continue
                if (len(txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                    continue
        else: # missing modality
            if "txt1" in file_name:
                if (int(data_info['pat_id']), int(data_info['chid'])) not in txtDict:
                    continue
                if (len(txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                    file_name = file_name.replace("_txt1_", "_txt0_")
            if ("cxr_input" not in data_info and "img1" in file_name):
                file_name = file_name.replace("_img1", "_img0")

        pat_id = int(data_info['pat_id'])
        chid = int(data_info['chid'])
        sequenceLength = data_info['data'].shape[0]
        # We need args.min_inputlen data points (hours) in the features,
        if sequenceLength < args.min_inputlen:
            continue
        
        # if features from args.vitalsign_labtest are not prepared, exclude
        if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
            continue
        
        ##### get possibles indices and max lengths here #####
        # If patient hasn't died, sample random sequence of 24 hours
        # target_type = 0: non patient
        # target_type = 1: patient with outbreak soon
        # target_type = 2: patient with outbreak on far future or far past but currently in normal group
        #
        # target = 0: normal (negative)
        # target = 1: abnormal (positive)
        #
        # modality_target_type = 1~12 (1: vslt_pp, 2: vslt_pn, 3: vslt_nn, 4: vslt+txt_pp, 5: vslt+txt_pn, 6: vslt+txt_nn, 
        #                               7: vslt+img_pp, 8: vslt+img_pn, 9: vslt+img_nn, 10: vslt+txt+img_pp, 11: vslt+txt+img_pn, 12: vslt+txt+img_nn)
        possible_indices_dict = {}
        # possible_indices_keys: 0 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pp, nn)
        # possible_indices_keys_with_img: 1 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pp, nn)     Case3: (wimgwtxt_pp: 0, wimgwtxt-nn: 2, wimgw/otxt_pp: 3, wimgw/otxt-nn: 5)
        # possible_indices_keys_without_img: 2 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt_pp: 6, w/oimgwtxt-nn: 8, w/oimgw/otxt_pp: 9, w/oimgw/otxt-nn: 11)
        # pat_neg_indices_keys = 3 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pn)
        # pat_neg_indices_keys_with_img = 4 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pn)         Case3: (wimgwtxt-pn: 1, wimgw/otxt-pn: 4)
        # pat_neg_indices_keys_without_img = 5 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt-pn: 7, w/oimgw/otxt-pn: 10)
        possible_indices_keys_alltypes = [[] for _ in range(6)]
        if(data_info['death_yn'] == 0):
            target = 0
            target_type = 0
            
            possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
            
        else:
            death_time = data_info['death_time']
            # If death time is beyond the prediction range of the given data or happened too early, change to 0 target
            if (death_time > sequenceLength + args.prediction_range - 1) or (death_time < args.min_inputlen):
                target = 0
                target_type = 2
                
                possible_indices_keys_alltypes[3] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
                
            else:              
                target = 1
                target_type = 1
                # For within-n-hours task, the data must be within prediction_range of death time
                # range 2~3 means 2 < x =< 3
                death_time = math.ceil(death_time)
                possible_indices = [(death_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (death_time >= args.min_inputlen+i-1) and (death_time - i < sequenceLength)] or None
                
                if possible_indices is None:
                    print("SeqLength : " + str(sequenceLength))
                    #print(data_info)
                    raise Exception('Classification Error')
            
                for p_index in possible_indices:
                    if p_index[0] not in possible_indices_dict:
                        possible_indices_dict[p_index[0]] = []
                    if p_index[1] not in possible_indices_dict[p_index[0]]:
                        possible_indices_dict[p_index[0]].append(p_index[1])
                    if p_index[0] not in possible_indices_keys_alltypes[0]:
                        possible_indices_keys_alltypes[0].append(p_index[0])
                        
                possible_indices_keys_alltypes[0].sort()
                    
        if target_type in [0, 1]:
            if (("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
                possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
            elif ('test-missing' in args.modality_inclusion):
                possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
        
        if ("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion and target_type != 2):
            if not possible_indices_keys_alltypes[1]: 
                continue
        
        if target == 1 or target_type == 2:
            if target == 1:
                all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
                possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]

            if (("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
            elif ('test-missing' in args.modality_inclusion):
                possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])    
        
        possibleWinSizes = data_info['possibleWinSizes']
        new_possibleWinSizes = {}
        for win_index in possibleWinSizes:
            new_list = [i for i in possibleWinSizes[win_index] if i >= args.min_inputlen]
            if len(new_list) > 0:
                new_possibleWinSizes[win_index] = new_list
        possibleWinSizes = dict(new_possibleWinSizes)
        possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
        
        if isListEmpty(possible_indices_keys_alltypes):
            continue       
        
        for pidx, possible_indices_keys_type in enumerate(possible_indices_keys_alltypes):
            if len(possible_indices_keys_type) == 0:
                continue
            if pidx < 3:
                if len(possible_indices_keys_type) >= args.PatPosSampleN:
                    possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatPosSampleN)
            else:        
                if len(possible_indices_keys_type) >= args.PatNegSampleN:
                    possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatNegSampleN)
                
        _data_list.append([pkl_path, possible_indices_keys_alltypes, possibleWinSizes, target])
        patDict[(pat_id, chid)] = possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type

    for idx, sample in enumerate(_data_list):
        pkl_pth, p_keys, possibleWinSizes, t = sample
        # t_type = _type_list[idx]
        for keylist_idx, keys_list in enumerate(p_keys):
            for key in keys_list:
                win_key_name = "_".join(pkl_pth.split("/")[-1].split("_")[:2]) + f"_{str(keylist_idx)}" + "_" + f"_{str(key)}"
                if win_key_name not in winDict:     
                    # win_size = random.choice(possibleWinSizes[key])
                    win_size = max(possibleWinSizes[key])
                    winDict[win_key_name] = win_size
                
    with open(patdictPath, 'wb') as f:
        pickle.dump(patDict, f, pickle.HIGHEST_PROTOCOL)
    with open(winsizePath, 'wb') as f:
        pickle.dump(winDict, f, pickle.HIGHEST_PROTOCOL)
        
def multiple_outbreaks_valdataset_maker(args, patdictPath, winsizePath):
    train_dir = search_walk({"path": args.train_data_path, "extension": ".pkl"})
    _data_list = []
    _type_list = []
    patDict = {}
    winDict = {}
    tmpTasks = ['vasso', 'intubation', 'cpr']
    tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
    tmptimes = ['vasso_time', 'intubation_time', 'cpr_time']
    taskIndex = tmpTasks.index(args.output_type)
    
    # if "txt" in args.input_types:
    # txtDict = txtDictLoad("train")
    txtDict = txtDictLoad("train")
    txtDict.update(txtDictLoad("test"))

    for idx, pkl_path in enumerate(tqdm(train_dir, desc="Creating fixed indices of validation data...")):
        file_name = pkl_path.split("/")[-1]
        
        with open(pkl_path, 'rb') as _f:
            data_info = pkl.load(_f)
        
        if "cxr_input" in data_info:
            if data_info["cxr_input"] == None:
                del data_info["cxr_input"]
                
        if "cxr_input" in data_info:
            new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
            if len(new_cxr_inputs) > 0:
                data_info["cxr_input"] = new_cxr_inputs
            else:
                del data_info['cxr_input']
                file_name = file_name.replace("_img1", "_img0")
            
        if 'test-full' in args.modality_inclusion:
            if args.fullmodal_definition not in file_name:
                continue
            if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                continue
            if "txt1" in args.fullmodal_definition:
                if (int(data_info['pat_id']), int(data_info['chid'])) not in txtDict:
                    continue
                if (len(txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                    continue
        else: # missing modality
            if "txt1" in file_name:
                if (int(data_info['pat_id']), int(data_info['chid'])) not in txtDict:
                    continue
                if (len(txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                    file_name = file_name.replace("_txt1_", "_txt0_")
            if ("cxr_input" not in data_info and "img1" in file_name):
                file_name = file_name.replace("_img1", "_img0")

        pat_id = int(data_info['pat_id'])
        chid = int(data_info['chid'])
        sequenceLength = data_info['data'].shape[0]
        # We need args.min_inputlen data points (hours) in the features,
        if sequenceLength < args.min_inputlen:
            continue
        
        # if features from args.vitalsign_labtest are not prepared, exclude
        if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
            continue
        ##### get possibles indices and max lengths here #####
        # If patient hasn't died, sample random sequence of 24 hours
        # target_type = 0: non patient
        # target_type = 1: patient with outbreak soon
        # target_type = 2: patient with outbreak on far future or far past but currently in normal group
        #
        # target = 0: normal (negative)
        # target = 1: abnormal (positive)
        #
        # modality_target_type = 1~12 (1: vslt_pp, 2: vslt_pn, 3: vslt_nn, 4: vslt+txt_pp, 5: vslt+txt_pn, 6: vslt+txt_nn, 
        #                               7: vslt+img_pp, 8: vslt+img_pn, 9: vslt+img_nn, 10: vslt+txt+img_pp, 11: vslt+txt+img_pn, 12: vslt+txt+img_nn)
        possible_indices_dict = {}
        # possible_indices_keys: 0 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pp, nn)
        # possible_indices_keys_with_img: 1 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pp, nn)     Case3: (wimgwtxt_pp: 0, wimgwtxt-nn: 2, wimgw/otxt_pp: 3, wimgw/otxt-nn: 5)
        # possible_indices_keys_without_img: 2 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt_pp: 6, w/oimgwtxt-nn: 8, w/oimgw/otxt_pp: 9, w/oimgw/otxt-nn: 11)
        # pat_neg_indices_keys = 3 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pn)
        # pat_neg_indices_keys_with_img = 4 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pn)         Case3: (wimgwtxt-pn: 1, wimgw/otxt-pn: 4)
        # pat_neg_indices_keys_without_img = 5 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt-pn: 7, w/oimgw/otxt-pn: 10)
        possible_indices_keys_alltypes = [[] for _ in range(6)]
        
        # possibleWinSizes = data_info['possibleWinSizes']
        
        ##### get possibles indices and max lengths here #####
        # If there are no positive cases in indices below args.min_inputlen, we change the case to negative.
        # If outbreak_time is beyond the prediction range of the given data or happened too early, change to 0 target
        outbreak_times = data_info[tmptimes[taskIndex]]
        if outbreak_times is not None and len(outbreak_times) != 0:
            outbreak_times = sorted(outbreak_times)
            if isinstance(outbreak_times[0], tuple):
                outbreak_times = list([i for i in outbreak_times if (i[0] >= args.min_inputlen) and (i[0] <= sequenceLength + args.prediction_range - 1)])
            else:
                outbreak_times = list([i for i in outbreak_times if (i >= args.min_inputlen) and (i <= sequenceLength + args.prediction_range - 1)])
            
            if len(outbreak_times) == 0:
                target = 0
            else:
                target = 1
        else:
            target = 0

        # If target variable negative, get random subsequence of length 3 ~ args.window_size
        if target == 0:
            target_type = 0
            possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
            
        else:
            # For within-n-hours task, the data must be within prediction_range of outbreak times
            # range 2~3 means 2 < x =< 3
            dup_counts = []
            possible_indices_list = []
            target_type = 1
            for idx_outbreak, outbreak_time in enumerate(outbreak_times):
                if isinstance(outbreak_times[0], tuple):
                    outbreak_time = math.ceil(outbreak_time[0])
                else:
                    outbreak_time = math.ceil(outbreak_time)
                if outbreak_time in dup_counts:
                    continue
                else:
                    dup_counts.append(outbreak_time)
                possible_indices = [(outbreak_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_time >= args.min_inputlen+i-1) and (outbreak_time - i < sequenceLength)] or None
                
                if possible_indices is None:
                    print("SeqLength : " + str(sequenceLength))
                    print(data_info)
                    raise Exception('Classification Error')
                possible_indices_list.append(possible_indices)
                # print(f"{str(idx_outbreak)}: ", possible_indices)
                
            # p_indices: [(32, [0, 1]), (31, [1, 2]), (30, [2, 3]), (29, [3, 4]), (28, [4, 5]), (27, [5, 6]), (26, [6, 7]), (25, [7, 8]), (24, [8, 9]), (23, [9, 10]), (22, [10, 11]), (21, [11, 12])]
            for p_indices in possible_indices_list: 
                # p_index: (32, [0, 1])
                for p_index in p_indices:
                    if p_index[0] not in possible_indices_dict:
                        possible_indices_dict[p_index[0]] = []
                    if p_index[1] not in possible_indices_dict[p_index[0]]:
                        possible_indices_dict[p_index[0]].append(p_index[1])
                    if p_index[0] not in possible_indices_keys_alltypes[0]:
                        possible_indices_keys_alltypes[0].append(p_index[0])
            possible_indices_keys_alltypes[0].sort()
        
        if (("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
            earliest_img_time = min([j[0] for j in data_info['cxr_input']])
            possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
            possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
        elif ('test-missing' in args.modality_inclusion):
            possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
            
        if ("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion):
            if not possible_indices_keys_alltypes[1]: 
                continue
            
        if target == 1:
            all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
            possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]

            if len(possible_indices_keys_alltypes[3]) > 0:
                if (("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                    possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                    possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
                elif ('test-missing' in args.modality_inclusion):
                    possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])  
        ######################################################
        possibleWinSizes = data_info['possibleWinSizes']
        new_possibleWinSizes = {}
        for win_index in possibleWinSizes:
            new_list = [i for i in possibleWinSizes[win_index] if i >= args.min_inputlen]
            if len(new_list) > 0:
                new_possibleWinSizes[win_index] = new_list
        possibleWinSizes = dict(new_possibleWinSizes)
        # possible_indices_keys = [key for key in possible_indices_keys if key in possibleWinSizes]
        possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
        
        if isListEmpty(possible_indices_keys_alltypes):
            continue
        
        for pidx, possible_indices_keys_type in enumerate(possible_indices_keys_alltypes):
            if len(possible_indices_keys_type) == 0:
                continue
            if pidx < 3:
                if len(possible_indices_keys_type) >= args.PatPosSampleN:
                    possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatPosSampleN)
            else:        
                if len(possible_indices_keys_type) >= args.PatNegSampleN:
                    possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatNegSampleN)
                
        _data_list.append([pkl_path, possible_indices_keys_alltypes, possibleWinSizes, target])
        patDict[(pat_id, chid)] = possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type
                
    for idx, sample in enumerate(_data_list):
        pkl_pth, p_keys, possibleWinSizes, t = sample
        # t_type = _type_list[idx]
        for keylist_idx, keys_list in enumerate(p_keys):
            for key in keys_list:
                win_key_name = "_".join(pkl_pth.split("/")[-1].split("_")[:2]) + f"_{str(keylist_idx)}" + "_" + f"_{str(key)}"
                if win_key_name not in winDict:     
                    # win_size = random.choice(possibleWinSizes[key])
                    win_size = max(possibleWinSizes[key])
                    winDict[win_key_name] = win_size
                
    with open(patdictPath, 'wb') as f:
        pickle.dump(patDict, f, pickle.HIGHEST_PROTOCOL)
    with open(winsizePath, 'wb') as f:
        pickle.dump(winDict, f, pickle.HIGHEST_PROTOCOL)