import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from control.config import args
from itertools import groupby
import torch
from builder.utils.utils import *
import torch.nn.functional as F
from torchvision import transforms
from os.path import exists
from math import floor


# import numpy as np
# a = np.asarray([[1,1,1,1],[2,2,2,2],[3,3,3,3],[1,3,1,1],[1,1,4,1]])
# print(a)
# print(a.shape)
# b = [False, False, True, False, True]
# a = np.delete(a, b, axis = 0)
# print(a)

def collate_emr_image_train(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    image_batch = []
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation((-5,5)),
        transforms.ToTensor(),
    ])
    for pkl_path in train_data:
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
        seq_inputs = data_pkl['data']
        delta_inputs = data_pkl['delta']
        mask_inputs = data_pkl['mask']

        if data_pkl['Gender'] == "M":
            gender = 1
        else:
            gender = 0
        
        static_inputs = torch.Tensor([gender, data_pkl['Age']])

        IMG_DIR = '/nfs/thena/MedicalAI/ImageData/public/MIMIC_CXR/data' 
        img_path = os.path.join(IMG_DIR, data_pkl['CXR'].replace('files','files_jpg'))
        image = Image.open(img_path)
        image = np.array(image) / 255. # rescale 0~1
        image = Image.fromarray(image)
        image = preprocess(image)
        image_batch.append(image)
        
        max_start_index = seq_inputs.shape[0] - args.window_size
        
        if data_pkl['Death_YN'] == 0:
            target = 0
            min_start_index = 0 
        else:
            # min_start_index = max_start_index - args.mortality_after
            max_start_d_index = round(data_pkl['Death_time'] - args.window_size)
            min_start_index = round(max_start_d_index - args.mortality_after)

            if min_start_index < 0:
                min_start_index = 0 
            
            # print("data_pkl['Death_time']: ", data_pkl['Death_time'])
            # print("max_start_index: ", max_start_index)
            # print("min_start_index: ", min_start_index)

            if max_start_index < min_start_index:
                target = 0
                min_start_index = 0 
            else:
                target = 1

        # print("Death_YN: {}, Target: {}".format(str(data_pkl['Death_YN']), str(target)))

        start_index = random.randint(min_start_index, max_start_index)	
        end_index = start_index+args.window_size
        seq_input = torch.Tensor(seq_inputs[start_index:end_index, :])

        seq_input = torch.div((seq_input - args.feature_mins), args.feature_max_mins)
        mask_inputs = torch.Tensor(mask_inputs[start_index:end_index, :])

        np_delta_inputs = delta_inputs[start_index:end_index, :]
        np_delta_inputs = np_delta_inputs / np.amax(np_delta_inputs)
        delta_inputs = torch.Tensor(np_delta_inputs)

        seqs_batch.append(torch.stack([seq_input, delta_inputs, mask_inputs]))
        static_batch.append(static_inputs)
        target_batch.append(target)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.Tensor(target_batch).to(torch.long)
    images = torch.stack(image_batch)

    return seqs, statics, images, targets

def collate_emr_image_test(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    seq_lengths = []
    image_batch = []
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    for pkl_path in train_data:
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
        seq_inputs = data_pkl['data']
        delta_inputs = data_pkl['delta']
        mask_inputs = data_pkl['mask']
        seq_lengths.append(seq_inputs.shape[0])
        if data_pkl['Gender'] == "M":
            gender = 1
        else:
            gender = 0
        
        static_inputs = torch.Tensor([gender, data_pkl['Age']])

        IMG_DIR = '/nfs/thena/MedicalAI/ImageData/public/MIMIC_CXR/data' 
        img_path = os.path.join(IMG_DIR, data_pkl['CXR'].replace('files','files_jpg'))
        image = Image.open(img_path)
        image = np.array(image) / 255. # rescale 0~1
        image = Image.fromarray(image)
        image = preprocess(image)
        image_batch.append(image)

        if data_pkl['Death_YN'] == 0:
            target = 0
        else:
            target = data_pkl['Death_time']

        seq_input = torch.Tensor(seq_inputs)
        seq_input = torch.div((seq_input - args.feature_mins), args.feature_max_mins)
        delta_inputs = torch.Tensor(delta_inputs)
        mask_inputs = torch.Tensor(mask_inputs)

        seqs_batch.append(torch.stack([seq_input, delta_inputs, mask_inputs]))
        static_batch.append(static_inputs)
        target_batch.append(target)

    max_seq_sample = max(seq_lengths)
    
    statics = torch.stack(static_batch).to(torch.long)
    targets = torch.Tensor(target_batch)
    seqs = torch.zeros(args.batch_size, 3, max_seq_sample, seqs_batch[0].shape[-1])
    for x in range(args.batch_size):
        seqs[x].narrow(1, 0, seq_lengths[x]).copy_(seqs_batch[x])
    images = torch.stack(image_batch)

    return seqs, statics, images, targets, seq_lengths


# randIndex : index of the original data on which the randomly sampled subsequence ends
# randLength : random number for length from which to sample subsequence
# windowIndex : temporary number equal to args.window_size - 1, often used for easy calculation of index values
def sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl):
    # If there are sufficient characters to sample randLength characters from
    if(randIndex >= randLength - 1):
        dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        inputLength = randIndex + 1
    
    return dataSequence, maskSequence, deltaSequence, inputLength

def testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl):
    # If there are sufficient characters to sample randLength characters from
    if(randIndex >= randLength - 1):
        dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        inputLength = randIndex + 1
    
    return dataSequence, maskSequence, deltaSequence, inputLength

def collate_grud_train_binary(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []

    for pkl_id, pkl_path in enumerate(train_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == "M":
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        # pklFeatureMins = np.min(data_pkl['data'], axis=0)
        # pklFeatureMinMaxs = np.max(data_pkl['data'], axis=0) - np.min(data_pkl['data'], axis=0)
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        sequenceLength = len(data_pkl['data'])
        windowIndex = args.window_size - 1

        if (args.output_type == "mortality"):
            # If patient hasn't died, sample random sequence of 24 hours
            if(data_pkl['death_yn'] == 0):
                target = 0
                # Random Index that shows the END of the sampled subsequence
                randIndex = random.randrange(2, sequenceLength)

            else:
                target = 1
                death_time = int(floor(data_pkl['death_time']))
                # If death time is beyond the prediction range of the given data, change to 0 target
                if death_time >= sequenceLength + args.prediction_after - 1:
                    target = 0
                    randIndex = random.randrange(2, sequenceLength)
                else:
                    # For within-n-hours task, the data must be within prediction_after of death time
                    possibleIndexes = [death_time - i for i in range(0, args.prediction_after) if (death_time - i >= 2) and (death_time - i < sequenceLength)] or None
                    if possibleIndexes is None:
                        print("SeqLength : " + str(sequenceLength))
                        print(data_pkl)
                        raise Exception('Classification Error')
                    randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]

            randLength = random.randrange(3, args.window_size+1)
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
        
        elif(args.output_type in ['vasso', 'intubation', 'cpr']):
            tmpTasks = ['vasso', 'intubation', 'cpr']
            tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
            tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
            taskIndex = tmpTasks.index(args.output_type)
            # -> ex) if args.output_type = "vasso", data_pkl['vasso_inputs'] can be written as data_pkl[tmpInputs[taskIndex]]

            # If there are only positive cases in indexes below three, we change the case to negative.
            # If sum of all elements after index 3 are 0, set to negative case
            if sum(data_pkl[tmpInputs[taskIndex]][3:]) == 0:
                data_pkl[tmpYN[taskIndex]] = 0

            target = data_pkl[tmpYN[taskIndex]]
            # If target variable negative, get random subsequence of length 3 ~ args.window_size
            if(target == 0):
                randIndex = random.randrange(2, min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength))
                randLength = random.randrange(3, args.window_size+1)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

            # If target variable positive, find all indexes that may contain a positive index within the prediction_after range
            else:
                possibleIndexes = []
                for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                    if x == 1:
                        for i in range(1, args.prediction_after+1):
                            possibleIndexes.append(idx - i)
                possibleIndexes = [x for x in possibleIndexes if x >= 2 and x < sequenceLength] or None
                if possibleIndexes is None:
                    raise Exception('Classification Error -> No Possible index for random sampling')
                randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                randLength = random.randrange(3, args.window_size+1)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

        elif args.output_type == "transfer":
            raise NotImplementedError

        elif args.output_type == "all":
            # If there are only positive cases in indexes below three, we change the case to negative.
            # If sum of all elements after index 3 are 0, set to negative case
            for x in ['cpr_inputs', 'intubation_inputs', 'vasso_inputs']:
                    if sum(data_pkl[x][3:]) == 0:
                        data_pkl[x.split("_")[0] + "_yn"] = 0

            # If death time out of prediction range of data, set death to negative
            if data_pkl['death_yn'] == 1:
                death_time = int(floor(data_pkl['death_time']))
                if death_time >= len(data_pkl['data']) + args.prediction_after - 1:
                        data_pkl['death_yn'] = 0

            # For 'all' task type, if there is a positive sign for any of the four types in the following hours
            target = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['intubation_yn'] or data_pkl['vasso_yn'])
            target_list = [int(data_pkl['death_yn']), int(data_pkl['cpr_yn']), int(data_pkl['intubation_yn']), int(data_pkl['vasso_yn'])]
            
            if target == 0:
                randLength = random.randrange(3, args.window_size+1)
                randIndex = random.randrange(2, sequenceLength)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
            
            else:
                possibleIndexes = []
                for x in ['death', 'cpr', 'intubation', 'vasso']:
                    if data_pkl[x + "_yn"] == 1:
                        if x == 'death':
                            for i in range(0, args.prediction_after):
                                possibleIndexes.append(death_time - i)

                        else:
                            for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                if posIn == 1:
                                    for i in range(1, args.prediction_after+1):
                                        possibleIndexes.append(idx - i)
                
                # Assert there exists indexes such that subsequences of length 3 or longer are possible
                possibleIndexes = [x for x in possibleIndexes if x >= 2 and x < sequenceLength] or None
                if possibleIndexes is None:
                    raise Exception('Classification Error -> No Possible index for random sampling')

                randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                randLength = random.randrange(3, args.window_size+1)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

        # Assert Sequence Lengths
        if(len(dataSequence) != args.window_size):
            print("Patient ID: " + str(data_pkl['pat_id']) + "    Death : " + str(data_pkl['death_yn']) + "    randIndex : " + str(randIndex) + "    randLength : " + str(randLength))
            print("CPR : " + str(data_pkl['cpr_yn']) + "    Intubation : " + str(data_pkl['intubation_yn']) + "    Vasso : " + str(data_pkl['vasso_yn']))
            print("Target : " + str(target))
            print([timeS[0] for timeS in dataSequence])
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.window_size):
            raise Exception('delta sequence length wrong!')

        # Conversion to tensors
        # torch_data = torch.Tensor(dataSequence)
        # torch_mask = torch.Tensor(maskSequence)
        # torch_delta = torch.Tensor(deltaSequence)
        # print("1: ", torch_data.shape)
        # print("1: ", np.delete(dataSequence, args.vslt_mask, axis = 1).shape)
        # print("2: ", torch_mask.shape)
        # print("2: ", np.delete(maskSequence, args.vslt_mask, axis = 1).shape)
        # print("3: ", torch_delta.shape)
        # print("3: ", np.delete(deltaSequence, args.vslt_mask, axis = 1).shape)
        # exit(1)

        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                        ]))
    
        if args.output_type == "all":
            target_batch.append(target_list)
        else:
            target_batch.append(target)
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)

        # lengthsFile = open("lengths.txt", "a")
        # lengthsFile.write(str(target) + " " + str(inputLength) + "\n")
        # lengthsFile.close()

        # print("Patient ID: " + str(data_pkl['pat_id']) + "    Death : " + str(data_pkl['death_yn']) + "    randIndex : " + str(randIndex) + "    randLength : " + str(randLength))
        # print("CPR : " + str(data_pkl['cpr_yn']) + "    Intubation : " + str(data_pkl['intubation_yn']) + "    Vasso : " + str(data_pkl['vasso_yn']))
        # print("Target : " + str(target))
        # print([timeS[0] for timeS in dataSequence])

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.Tensor(target_batch).to(torch.long)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

# For validation collate_fn, we prioritize full window_size subsequences over smaller ones
# The indexes for sampling are random for every iteration
def collate_grud_val_binary(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])
    
    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []

    for pkl_id, pkl_path in enumerate(val_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Data Normalization
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.test_window_size - 1
        sequenceLength = len(data_pkl['data'])

        if args.output_type == "mortality":
            target = data_pkl['death_yn']
            if target == 0:
                # If sequenceLength is greater than test_window_size, randomly sample test_window_size subsequence. Otherwise, take entire sequence and then add zero-padding
                if sequenceLength >= args.test_window_size:
                    randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                else:
                    randIndex = sequenceLength - 1
            else:
                death_time = int(floor(data_pkl['death_time']))
                if death_time >= sequenceLength + args.prediction_after - 1:
                    target = 0
                    if sequenceLength >= args.test_window_size:
                        randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                    else:
                        randIndex = sequenceLength - 1
                else:
                    possibleIndexes = [death_time - i for i in range(args.prediction_after) if (death_time - i >= 2) and (death_time - i < sequenceLength)] or None
                    if possibleIndexes is None:
                        print(sequenceLength)
                        print(data_pkl)
                        raise Exception('Classification Error')
                    randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]

        elif args.output_type in ['vasso', 'intubation', 'cpr']:
            tmpTasks = ['vasso', 'intubation', 'cpr']
            tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
            tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
            taskIndex = tmpTasks.index(args.output_type)

            # If there are only positive cases in indexes below three, we change the case to negative.
            # If sum of all elements after index 3 are 0, set to negative case
            if sum(data_pkl[tmpInputs[taskIndex]][3:]) == 0:
                data_pkl[tmpYN[taskIndex]] = 0
            
            target = data_pkl[tmpYN[taskIndex]]

            if target == 0:
                if sequenceLength >= args.test_window_size:
                    randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                else:
                    randIndex = sequenceLength-1
            
            # If the target variable is found, we find a random subsequence that contains a 1 input within the given args.prediction_after
            # If sequence length is insufficient, add zero-padding at back
            else:
                possibleIndexes = []
                for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                    if x == 1:
                        for i in range(1, args.prediction_after+1):
                            possibleIndexes.append(idx - i)
                # Good Indexes are indexes that are within the possible window and are far enough from the start of the array (args.test_window_size)
                goodIndexes = [x for x in possibleIndexes if x >= args.test_window_size - 1] or None
                # Sufficient Indexes are indexes that are good enough but have lower priority than the good ones
                sufficientIndexes = [x for x in possibleIndexes if x >= 2] or None

                if(sufficientIndexes == None):
                    raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                if goodIndexes == None:
                    randIndex = sufficientIndexes[random.randrange(0, len(sufficientIndexes))]
                else:
                    randIndex = goodIndexes[random.randrange(0, len(goodIndexes))]

        elif args.output_type == "transfer":
            raise NotImplementedError

        elif args.output_type == "all":
            # If there are only positive cases in indexes below three, we change the case to negative.
            # If sum of all elements after index 3 are 0, set to negative case
            for x in ['cpr_inputs', 'intubation_inputs', 'vasso_inputs']:
                    if sum(data_pkl[x][3:]) == 0:
                        data_pkl[x.split("_")[0] + "_yn"] = 0

            if data_pkl['death_yn'] == 1:
                death_time = int(floor(data_pkl['death_time']))
                if death_time >= len(data_pkl['data']) + args.prediction_after - 1:
                    data_pkl['death_yn'] = 0

            target = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['intubation_yn'] or data_pkl['vasso_yn'])
            target_list = [int(data_pkl['death_yn']), int(data_pkl['cpr_yn']), int(data_pkl['intubation_yn']), int(data_pkl['vasso_yn'])]
            
            if target == 0:
                if sequenceLength >= args.test_window_size:
                    randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                else:
                    randIndex = sequenceLength-1
            else:
                possibleIndexes = []
                for x in ['death', 'cpr', 'intubation', 'vasso']:
                    if data_pkl[x + "_yn"] == 1:
                        if x == 'death':
                            for i in range(0, args.prediction_after):
                                possibleIndexes.append(death_time - i)
                                
                        else:
                            for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                if posIn == 1:
                                    for i in range(1, args.prediction_after+1):
                                        possibleIndexes.append(idx - i)
                
                # Good Indexes are indexes that are within the possible window and are far enough from the start of the array (args.window_size)
                goodIndexes = [x for x in possibleIndexes if (x >= args.test_window_size - 1) and (x < sequenceLength)] or None
                # Sufficient Indexes are indexes that meet the requirements but have lower priority than the good ones
                sufficientIndexes = [x for x in possibleIndexes if (x >= 2) and (x < sequenceLength)] or None

                if(sufficientIndexes == None):
                    raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                if goodIndexes == None:
                    randIndex = sufficientIndexes[random.randrange(0, len(sufficientIndexes))]
                else:
                    randIndex = goodIndexes[random.randrange(0, len(goodIndexes))]

        randLength = args.test_window_size
        dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

        # Assert Sequence Lengths
        if(len(dataSequence) != args.test_window_size):
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.test_window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.test_window_size):
            raise Exception('delta sequence length wrong!')

        # Conversion to tensors
        # seqs_batch.append(torch.stack([torch.Tensor(dataSequence), torch.Tensor(maskSequence), torch.Tensor(deltaSequence)]))
        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                ]))
        if args.output_type == "all":
            target_batch.append(target_list)
        else:
            target_batch.append(target)
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)

        # if(pkl_id == 0):
        #     print("Patient ID: " + str(data_pkl['pat_id']) + "    Death : " + str(data_pkl['death_yn']) + "    randIndex : " + str(randIndex) + "    randLength : " + str(randLength))
        #     print("CPR : " + str(data_pkl['cpr_yn']) + "    Intubation : " + str(data_pkl['intubation_yn']) + "    Vasso : " + str(data_pkl['vasso_yn']))
        #     print("Target : " + str(target))
        #     print([timeS[0] for timeS in dataSequence])


    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.Tensor(target_batch).to(torch.long)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

# For testing collate_fn, we prioritize full window_size subsequences over smaller ones
def collate_grud_test_binary(test_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])
    
    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []

    datasetType = test_data[0]
    datasetType = datasetType.split("/")[-3]
    fileNm = "testIndexes/testIndexes__" + datasetType + "__" + args.modality_inclusion + "__" + args.predict_type + "__" + args.output_type + "__PA" + str(args.prediction_after) + "__" + args.model + ".txt"

    # File Format : {pat_id} {chid} {randIndex}
    # Open the file and add existing entries to dictionary
    patDict = {}
    if exists(fileNm):
        indexFile = open(fileNm, "r")
        while True:
            line = indexFile.readline()
            if not line:
                break
            line = line.strip().split()
            pat_id = int(line[0])
            chid = int(line[1])
            randIndex = int(line[2])
            patDict[(pat_id, chid)] = randIndex
        indexFile.close()

    for pkl_id, pkl_path in enumerate(test_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Data Normalization
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.test_window_size - 1
        sequenceLength = len(data_pkl['data'])

        pat_id = int(data_pkl['pat_id'])
        chid = int(data_pkl['chid'])

        target = -1

        if data_pkl['death_yn'] == 1:
            death_time = int(floor(data_pkl['death_time']))
            if death_time >= sequenceLength + args.prediction_after - 1:
                data_pkl['death_yn'] = 0

        for x in ['cpr_inputs', 'intubation_inputs', 'vasso_inputs']:
            if sum(data_pkl[x][3:]) == 0:
                data_pkl[x.split("_")[0] + "_yn"] = 0

        # Check if the randIndex for the given patient has already been initialized
        if (pat_id, chid) in patDict:
            if args.output_type == "mortality":
                target = data_pkl['death_yn']

            elif args.output_type in ['vasso', 'intubation', 'cpr']:
                tmpTasks = ['vasso', 'intubation', 'cpr']
                tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
                tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
                taskIndex = tmpTasks.index(args.output_type) 
                target = data_pkl[tmpYN[taskIndex]]

            elif args.output_type == "transfer":
                raise NotImplementedError

            elif args.output_type == "all":
                target = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['intubation_yn'] or data_pkl['vasso_yn'])
                target_list = [int(data_pkl['death_yn']), int(data_pkl['cpr_yn']), int(data_pkl['intubation_yn']), int(data_pkl['vasso_yn'])]
            
            randIndex = patDict[(pat_id, chid)]
            randLength = args.test_window_size
            dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
        
        else:
            outputFile = open(fileNm, "a")
            if args.output_type == "mortality":
                target = data_pkl['death_yn']
                if target == 0:
                    # If sequenceLength is greater than test_window_size, randomly sample window_size subsequence. Otherwise, take entire sequence and then add zero-padding
                    if sequenceLength >= args.test_window_size:
                        randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                    else:
                        randIndex = sequenceLength - 1
                else:
                    possibleIndexes = [death_time - i for i in range(args.prediction_after) if (death_time - i >= 2) and (death_time - i < sequenceLength)] or None
                    if possibleIndexes is None:
                        print(sequenceLength)
                        print(data_pkl)
                        raise Exception('Classification Error')
                    randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]

            elif args.output_type in ['vasso', 'intubation', 'cpr']:
                tmpTasks = ['vasso', 'intubation', 'cpr']
                tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
                tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
                taskIndex = tmpTasks.index(args.output_type)
                
                target = data_pkl[tmpYN[taskIndex]]

                if target == 0:
                    if sequenceLength >= args.test_window_size:
                        randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                    else:
                        randIndex = sequenceLength-1
                
                # If the target variable is found, we find a random subsequence that contains a 1 input within the given args.prediction_after
                # If sequence length is insufficient, add zero-padding at back
                else:
                    possibleIndexes = []
                    for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                        if x == 1:
                            for i in range(1, args.prediction_after+1):
                                possibleIndexes.append(idx - i)
                    # Good Indexes are indexes that are within the possible window and are far enough from the start of the array (args.test_window_size)
                    goodIndexes = [x for x in possibleIndexes if x >= args.test_window_size - 1] or None
                    # Sufficient Indexes are indexes that are good enough but have lower priority than the good ones
                    sufficientIndexes = [x for x in possibleIndexes if x >= 2] or None

                    if(sufficientIndexes == None):
                        raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                    # If all possible arrays have distance from start less than args.test_window_size, choose randomly then add zero-padding at back
                    if goodIndexes == None:
                        randIndex = sufficientIndexes[random.randrange(len(sufficientIndexes))]
                    else:
                        randIndex = goodIndexes[random.randrange(len(goodIndexes))]

            elif args.output_type == "transfer":
                raise NotImplementedError

            elif args.output_type == "all":
                target = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['intubation_yn'] or data_pkl['vasso_yn'])
                target_list = [int(data_pkl['death_yn']), int(data_pkl['cpr_yn']), int(data_pkl['intubation_yn']), int(data_pkl['vasso_yn'])]
                
                if target == 0:
                    if sequenceLength >= args.test_window_size:
                        randIndex = random.randrange(args.test_window_size-1, sequenceLength)
                    else:
                        randIndex = sequenceLength-1
                else:
                    possibleIndexes = []
                    for x in ['death', 'cpr', 'intubation', 'vasso']:
                        if data_pkl[x + "_yn"] == 1:
                            if x == 'death':
                                for i in range(args.prediction_after):
                                    possibleIndexes.append(death_time - i)
                                    
                            else:
                                for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                    if posIn == 1:
                                        for i in range(1, args.prediction_after+1):
                                            possibleIndexes.append(idx - i)
                    
                    # Good Indexes are indexes that are within the possible window and are far enough from the start of the array (args.window_size)
                    goodIndexes = [x for x in possibleIndexes if (x >= args.test_window_size - 1) and (x < sequenceLength)] or None
                    # Sufficient Indexes are indexes that meet the requirements but have lower priority than the good ones
                    sufficientIndexes = [x for x in possibleIndexes if (x >= 2) and (x < sequenceLength)] or None

                    if(sufficientIndexes == None):
                        raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                    # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                    if goodIndexes == None:
                        randIndex = sufficientIndexes[random.randrange(0, len(sufficientIndexes))]
                    else:
                        randIndex = goodIndexes[random.randrange(0, len(goodIndexes))]

            # if(pkl_id == 0):
            #     print("Patient ID: " + str(data_pkl['pat_id']) + "    Death : " + str(data_pkl['death_yn']) + "    randIndex : " + str(randIndex) + "    randLength : " + str(randLength))
            #     print("CPR : " + str(data_pkl['cpr_yn']) + "    Intubation : " + str(data_pkl['intubation_yn']) + "    Vasso : " + str(data_pkl['vasso_yn']))
            #     print("Target : " + str(target))
            #     print([timeS[0] for timeS in dataSequence])

            randLength = args.test_window_size
            dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
            outputFile.write(str(pat_id) + " " + str(chid) + " " + str(randIndex) + "\n")
            outputFile.close()

        # Assert Sequence Lengths
        if(len(dataSequence) != args.test_window_size):
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.test_window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.test_window_size):
            raise Exception('delta sequence length wrong!')

        # Conversion to tensors
        # seqs_batch.append(torch.stack([torch.Tensor(dataSequence), torch.Tensor(maskSequence), torch.Tensor(deltaSequence)]))
        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                ]))
        if args.output_type == "all":
            target_batch.append(target_list)
        else:
            target_batch.append(target)
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)


    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.Tensor(target_batch).to(torch.long)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

# For sequence we assume that len(data) is at least 6
def collate_grud_train_sequence(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])
    
    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    output_lengths_batch = []

    for pkl_id, pkl_path in enumerate(train_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Data Normalization
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        sequenceLength = len(data_pkl['data'])
        windowIndex = args.window_size - 1

        # If there are only positive cases in indexes below three, we change the case to negative.
        # If sum of all elements after index 3 are 0, set to negative case
        for x in ['cpr_inputs', 'intubation_inputs', 'vasso_inputs']:
                if sum(data_pkl[x][3:]) == 0:
                    data_pkl[x.split("_")[0] + "_yn"] = 0

        if data_pkl['death_yn'] == 1:
            death_time = int(floor(data_pkl['death_time']))
            if death_time >= sequenceLength + args.prediction_horizon_size - 4:
                data_pkl['death_yn'] = 0
        else:
            death_time = 0

        if args.output_type == "mortality":
            # Make constant graph or zero-to-one step graph based on death factor
            deathInputs = np.zeros(sequenceLength)
            if data_pkl['death_yn'] == 1:
                if death_time < sequenceLength:
                    deathInputs = np.append(deathInputs[:death_time + 1], np.ones(args.prediction_horizon_size))
                else:
                    deathInputs = np.append(deathInputs, np.zeros(death_time - sequenceLength + 1))
                    deathInputs = np.append(deathInputs, np.ones(args.prediction_horizon_size))
            else:
                deathInputs = np.append(deathInputs, np.zeros(args.prediction_horizon_size))
            
            # If the patient hasn't died, sample a random size sequence of length 3 ~ args.window_size and set the target sequence as a zero vector
            # Then add the SOS token (2) and additional zeropadding
            if data_pkl['death_yn'] == 0:
                randLength = random.randrange(3, args.window_size+1)
                randIndex = random.randrange(2, sequenceLength - 3)
                # The target sequence gets a number of zeros equal to the smaller number between the remaining sequence and the horizon size
                target = np.append([2], np.zeros(min(sequenceLength - randIndex - 1, args.prediction_horizon_size)))
                # Append the additional zero-padding part
                target = np.append(target, np.zeros(args.prediction_horizon_size + 1 - len(target)))
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                outputLength = min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1
            
            # If patient has died, sample a random size sequence of length 3 ~ args.window_size and set target sequence as step function that goes to 1 after death
            # Then add the SOS token (2)
            else:
                randLength = random.randrange(3, args.window_size+1)
                possibleIndexes = [death_time - i for i in range(0, args.prediction_horizon_size) if (death_time - i >= 2) and (death_time - i < sequenceLength - 3)] or None
                randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                target = np.append([2], deathInputs[randIndex+1:randIndex+1+args.prediction_horizon_size])
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                outputLength = args.prediction_horizon_size + 1

        elif args.output_type in ['vasso', 'intubation', 'cpr']:
            tmpTask = ['vasso', 'intubation', 'cpr']
            tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
            tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
            taskIndex = tmpTask.index(args.output_type)

            # If patient hasn't recieved target treatment, add targetLength size zero-vector as target
            # Then add the SOS token (2) and additional zero-padding
            if data_pkl[tmpYN[taskIndex]] == 0:
                randLength = random.randrange(3, args.window_size+1)
                randIndex = random.randrange(2, min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - 3)
                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], np.zeros(targetLength))
                target = np.append(target, np.zeros(args.prediction_horizon_size + 1 - len(target)))
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                outputLength = targetLength + 1

            # If patient has recieved target treatment, set the target as the subsequence of {target}_inputs starting from randIndex + 1 and until
            # either the sequence ends or reaches args.prediction_horizon_size. Then add SOS and EOS and additional zero-padding if necessary
            else:
                possibleIndexes = []
                for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                    if x == 1:
                        for i in range(1, args.prediction_horizon_size+1):
                            possibleIndexes.append(idx - i)
                possibleIndexes = [x for x in possibleIndexes if x >= 2 and x <= sequenceLength - 4] or None
                if possibleIndexes is None:
                    raise Exception('Classification Error -> No Possible index for random sampling')

                randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                randLength = random.randrange(3, args.window_size+1)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], data_pkl[tmpInputs[taskIndex]][randIndex+1:randIndex+1+targetLength])
                target = np.append(target, np.zeros(args.prediction_horizon_size + 1 - len(target)))
                outputLength = targetLength + 1

        elif args.output_type == "transfer":
            raise NotImplementedError
        
        elif args.output_type == "all":
            positiveExists = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['vasso_yn'] or data_pkl['intubation_yn'])

            if positiveExists == 0:
                randLength = random.randrange(3, args.window_size+1)
                randIndex = random.randrange(2, sequenceLength)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                # target = np.append([2], np.zeros(min(sequenceLength - randIndex - 1, args.prediction_horizon_size)))
                # target = np.append(target, np.zeros(args.prediction_horizon_size + 1 - len(target)))
                # target = np.append([2], np.zeros(args.prediction_horizon_size))
                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, np.zeros([4,12]), axis=1)
                outputLength = min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1

            else:
                possibleIndexes = []
                for x in ['death', 'cpr', 'intubation', 'vasso']:
                        if data_pkl[x + "_yn"] == 1:
                            if x == 'death':
                                for i in range(args.prediction_horizon_size):
                                    possibleIndexes.append(death_time - i)
                                    
                            else:
                                for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                    if posIn == 1:
                                        for i in range(1, args.prediction_horizon_size+1):
                                            possibleIndexes.append(idx - i)
                
                # Assert there exists indexes such that subsequences of length 3 or longer are possible
                plausibleIndexes = [x for x in possibleIndexes if x >= 2 and x < sequenceLength - 3] or None
                if plausibleIndexes is None:
                    print(data_pkl)
                    print(sequenceLength)
                    print(possibleIndexes)
                    raise Exception('Classification Error -> No Possible index for random sampling')

                randIndex = plausibleIndexes[random.randrange(len(plausibleIndexes))]
                randLength = random.randrange(3, args.window_size+1)
                dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                maxSequenceLength = max(len(data_pkl['cpr_inputs']), len(data_pkl['vasso_inputs']), len(data_pkl['intubation_inputs']), death_time, sequenceLength)

                # Create Target Sequence out of the values of all the tasks

                # targetSequence = np.zeros(maxSequenceLength - randIndex - 1)
                # for i in range(maxSequenceLength - randIndex - 1):
                #     if (len(data_pkl['cpr_inputs']) > randIndex + i + 1) and (data_pkl['cpr_inputs'][randIndex + i + 1] == 1):
                #         targetSequence[i] = 1
                #     if (len(data_pkl['vasso_inputs']) > randIndex + i + 1) and (data_pkl['vasso_inputs'][randIndex + i + 1] == 1):
                #         targetSequence[i] = 1
                #     if (len(data_pkl['intubation_inputs']) > randIndex + i + 1) and (data_pkl['intubation_inputs'][randIndex + i + 1] == 1):
                #         targetSequence[i] = 1
                #     if  (data_pkl['death_yn'] == 1) and (death_time < randIndex + i + 1):
                #         targetSequence[i] = 1
                # if data_pkl['death_yn'] == 1:
                #     if death_time < maxSequenceLength:
                #         targetSequence = np.append(targetSequence, np.ones(args.prediction_horizon_size))
                #     else:
                #         targetSequence = np.append(targetSequence, np.zeros(death_time - maxSequenceLength + 1))
                #         targetSequence = np.append(targetSequence, np.ones(args.prediction_horizon_size))
                # if len(targetSequence) > args.prediction_horizon_size:
                #     targetSequence = targetSequence[:args.prediction_horizon_size]

                # target = np.append([2], targetSequence)
                # target = np.append(target, np.zeros(args.prediction_horizon_size + 1 - len(target)))
                # outputLength = len(targetSequence) + 1
                targetSequence = np.zeros([4, maxSequenceLength - randIndex - 1])
                flag = True
                for i in range(maxSequenceLength - randIndex - 1):
                    if (len(data_pkl['cpr_inputs']) > randIndex + i + 1) and (data_pkl['cpr_inputs'][randIndex + i + 1] == 1):
                        targetSequence[0,i] = 1
                    if (len(data_pkl['vasso_inputs']) > randIndex + i + 1) and (data_pkl['vasso_inputs'][randIndex + i + 1] == 1):
                        targetSequence[1,i] = 1
                    if (len(data_pkl['intubation_inputs']) > randIndex + i + 1) and (data_pkl['intubation_inputs'][randIndex + i + 1] == 1):
                        targetSequence[2,i] = 1
                    if  (data_pkl['death_yn'] == 1) and (death_time < randIndex + i + 1) and flag:
                        targetSequence[3,i:] = 1         
                        flag = False

                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, targetSequence, axis=1)                
                outputLength = len(targetSequence) + 1
                
                if len(target_list[0,:]) > args.prediction_horizon_size+1:
                    target_list = target_list[:,:args.prediction_horizon_size+1]
                else:
                    target_list = np.append(target_list, np.zeros([4, args.prediction_horizon_size + 1 - len(target_list[0,:])]), axis=1)

                # print(str(target) + "    " + str(pkl_id))
                #for i in range(randIndex+1, randIndex+1+targetLength):

        # Assert Sequence Lengths
        if(len(dataSequence) != args.window_size):
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.window_size):
            raise Exception('delta sequence length wrong!')

        # Conversion to tensors
        # seqs_batch.append(torch.stack([torch.Tensor(dataSequence), torch.Tensor(maskSequence), torch.Tensor(deltaSequence)]))
        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                ]))

        if args.output_type == "all":
            target_batch.append(torch.Tensor(target_list))
            
        else:
            target_batch.append(torch.Tensor(target))
            
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)
        output_lengths_batch.append(outputLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    output_lengths = torch.Tensor(output_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, output_lengths

def collate_grud_val_sequence(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])
    
    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    output_lengths_batch = []

    for pkl_id, pkl_path in enumerate(val_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Data Normalization
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        sequenceLength = len(data_pkl['data'])
        windowIndex = args.test_window_size - 1

        for x in ['cpr_inputs', 'intubation_inputs', 'vasso_inputs']:
                    if sum(data_pkl[x][3:]) == 0:
                        data_pkl[x.split("_")[0] + "_yn"] = 0
        
        if data_pkl['death_yn'] == 1:
            death_time = int(floor(data_pkl['death_time']))
            if death_time >= sequenceLength + args.prediction_horizon_size - 4:
                data_pkl['death_yn'] = 0
        else:
            death_time = 0

        if args.output_type == "mortality":
            deathInputs = np.zeros(sequenceLength)
            if data_pkl['death_yn'] == 1:
                if death_time < sequenceLength:
                    deathInputs = np.append(deathInputs[:death_time + 1], np.ones(args.prediction_horizon_size + args.test_window_size))
                else:
                    deathInputs = np.append(deathInputs, np.zeros(death_time - sequenceLength + 1))
                    deathInputs = np.append(deathInputs, np.ones(args.prediction_horizon_size + args.test_window_size))
            else:
                deathInputs = np.append(deathInputs, np.zeros(args.prediction_horizon_size))
            

            if data_pkl['death_yn'] == 0:
                randLength = args.test_window_size
                randIndex = random.randrange(2, sequenceLength - 3)
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                target = np.append([2], np.zeros(args.test_window_size + args.prediction_horizon_size))
                outputLength = inputLength + min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1
            
            else:
                randLength = args.test_window_size
                possibleIndexes = [death_time - i for i in range(0, args.prediction_horizon_size) if (death_time - i >= 2) and (death_time - i < sequenceLength)] or None
                randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                target = np.append([2], deathInputs[randIndex-inputLength+1:randIndex+1+args.prediction_horizon_size+args.test_window_size-inputLength])
                outputLength = args.test_window_size + args.prediction_horizon_size + 1
        
        elif args.output_type in ['vasso', 'intubation', 'cpr']:
            tmpTask = ['vasso', 'intubation', 'cpr']
            tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
            tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
            taskIndex = tmpTask.index(args.output_type)

            # If patient hasn't recieved target treatment, add (inputLength + targetLength) size zero-vector as target
            # Then add the SOS token (2) and additional zero-padding
            if data_pkl[tmpYN[taskIndex]] == 0:
                randLength = args.test_window_size
                randIndex = random.randrange(2, min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - 3)
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], np.zeros(args.prediction_horizon_size + args.test_window_size))
                outputLength = inputLength + targetLength + 1

            # If patient has recieved target treatment, set the target as the subsequence of {target}_inputs starting from randIndex + 1 and until
            # either the sequence ends or reaches args.prediction_horizon_size. Then add SOS and EOS and additional zero-padding if necessary
            else:
                possibleIndexes = []
                for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                    if x == 1:
                        for i in range(1, args.prediction_horizon_size+1):
                            possibleIndexes.append(idx - i)
                plausibleIndexes = [x for x in possibleIndexes if x >= 2 and x <= sequenceLength - 4] or None
                goodIndexes = [x for x in plausibleIndexes if x >= args.test_window_size - 1] or None

                if(plausibleIndexes == None):
                    raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                if goodIndexes == None:
                    randIndex = plausibleIndexes[random.randrange(len(plausibleIndexes))]
                else:
                    randIndex = goodIndexes[random.randrange(len(goodIndexes))]
                randLength = args.test_window_size
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], data_pkl[tmpInputs[taskIndex]][randIndex-inputLength+1:randIndex+1+targetLength])
                target = np.append(target, np.zeros(args.prediction_horizon_size + args.test_window_size + 1 - len(target)))
                outputLength = targetLength + inputLength + 1
        
        elif args.output_type == "transfer":
            raise NotImplementedError
        
        elif args.output_type == "all":
            positiveExists = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['vasso_yn'] or data_pkl['intubation_yn'])

            if positiveExists == 0:
                randLength = args.test_window_size
                if sequenceLength >= args.test_window_size + 3:
                    randIndex = random.randrange(args.test_window_size-1, sequenceLength-3)
                else:
                    randIndex = sequenceLength - 4
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
                
                # target = np.append([2], np.zeros(args.prediction_horizon_size + args.test_window_size))
                # target = np.append([2], np.zeros(args.prediction_horizon_size))
                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, np.zeros([4,args.test_window_size+args.prediction_horizon_size]), axis=1)
                
                outputLength = inputLength + min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1

            else:
                possibleIndexes = []
                for x in ['death', 'cpr', 'intubation', 'vasso']:
                    if data_pkl[x + "_yn"] == 1:
                        if x == 'death':
                            for i in range(args.prediction_horizon_size):
                                possibleIndexes.append(len(data_pkl['data'])-i)
                                
                        else:
                            for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                if posIn == 1:
                                    for i in range(1, args.prediction_horizon_size+1):
                                        possibleIndexes.append(idx - i)
                
                # Assert there exists indexes such that subsequences of length 3 or longer are possible
                plausibleIndexes = [x for x in possibleIndexes if x >= 2 and x <= sequenceLength - 4] or None
                goodIndexes = [x for x in plausibleIndexes if x >= args.test_window_size - 1] or None

                if(plausibleIndexes == None):
                    raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                if goodIndexes == None:
                    randIndex = plausibleIndexes[random.randrange(len(plausibleIndexes))]
                else:
                    randIndex = goodIndexes[random.randrange(len(goodIndexes))]
                randLength = args.test_window_size
                dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)

                maxSequenceLength = max(len(data_pkl['cpr_inputs']), len(data_pkl['vasso_inputs']), len(data_pkl['intubation_inputs']), death_time, sequenceLength)
                
                # targetSequence = np.zeros(maxSequenceLength - randIndex + randLength - 1)
                # for i in range(maxSequenceLength - randIndex + randLength - 1):
                #     origIndex = randIndex - inputLength + i + 1
                #     if (len(data_pkl['cpr_inputs']) > origIndex) and (data_pkl['cpr_inputs'][origIndex] == 1):
                #         targetSequence[i] = 1
                #     if (len(data_pkl['vasso_inputs']) > origIndex) and (data_pkl['vasso_inputs'][origIndex] == 1):
                #         targetSequence[i] = 1
                #     if (len(data_pkl['intubation_inputs']) > origIndex) and (data_pkl['intubation_inputs'][origIndex] == 1):
                #         targetSequence[i] = 1
                #     if (data_pkl['death_yn'] == 1) and (origIndex > death_time):
                #         targetSequence[i] = 1
                # if data_pkl['death_yn'] == 1:
                #     if death_time < maxSequenceLength:
                #         targetSequence = np.append(targetSequence, np.ones(args.prediction_horizon_size))
                #     else:
                #         targetSequence = np.append(targetSequence, np.zeros(death_time - maxSequenceLength + 1))
                #         targetSequence = np.append(targetSequence, np.ones(args.prediction_horizon_size))
                # if len(targetSequence) > args.prediction_horizon_size:
                #     targetSequence = targetSequence[:args.prediction_horizon_size]

                # target = np.append([2], targetSequence)
                # target = np.append(target, np.zeros(args.test_window_size + args.prediction_horizon_size + 1 - len(target)))
                # if data_pkl['death_yn'] == 1:
                #     outputLength = args.prediction_horizon_size + args.test_window_size + 1
                # else:
                #     outputLength = inputLength + (maxSequenceLength - randIndex - 1) + 1
                
                targetSequence = np.zeros([4, maxSequenceLength - randIndex + randLength - 1])
                flag = True
                for i in range(maxSequenceLength - randIndex + randLength - 1):
                    origIndex = randIndex - inputLength + i + 1
                    if (len(data_pkl['cpr_inputs']) > origIndex) and (data_pkl['cpr_inputs'][origIndex] == 1):
                        targetSequence[0,i] = 1
                    if (len(data_pkl['vasso_inputs']) > origIndex) and (data_pkl['vasso_inputs'][origIndex] == 1):
                        targetSequence[1,i] = 1
                    if (len(data_pkl['intubation_inputs']) > origIndex) and (data_pkl['intubation_inputs'][origIndex] == 1):
                        targetSequence[2,i] = 1
                    if (data_pkl['death_yn'] == 1) and (origIndex > death_time) and flag:
                        targetSequence[3,i:] = 1         
                        flag = False

                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, targetSequence, axis=1)
                    
                if len(target_list[0,:]) > (args.prediction_horizon_size + args.test_window_size + 1):
                    target_list = target_list[:,:args.prediction_horizon_size + args.test_window_size + 1]
                else:
                    target_list = np.append(target_list, np.zeros([4, args.prediction_horizon_size + args.test_window_size + 1 - len(target_list[0,:])]), axis=1)

                if data_pkl['death_yn'] == 1:
                    outputLength = args.prediction_horizon_size + args.test_window_size + 1
                else:
                    outputLength = inputLength + (maxSequenceLength - randIndex - 1) + 1
                    
        # Assert Sequence Lengths
        if(len(dataSequence) != args.test_window_size):
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.test_window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.test_window_size):
            raise Exception('delta sequence length wrong!')

        # print("Patient Number : " + str(pkl_id) + " / Patient ID : " + str(data_pkl['pat_id']))
        # print("RandIndex : " + str(randIndex) + "    RandLength : " + str(randLength) + "    Death : " + str(data_pkl['death_yn']) + "    Output Length : " + str(outputLength))
        # print("Input Length " + str(len(data_pkl['data'])))
        # print([float(t[0]) for t in data_pkl['data']])
        # print("Target Length : " + str(len(target)))
        # print(target)
        # print("\n")

        # Conversion to tensors
        # seqs_batch.append(torch.stack([torch.Tensor(dataSequence), torch.Tensor(maskSequence), torch.Tensor(deltaSequence)]))
        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                        ]))
        if args.output_type == "all":
            target_batch.append(torch.Tensor(target_list))
        else:
            target_batch.append(torch.Tensor(target))
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)
        output_lengths_batch.append(outputLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    output_lengths = torch.Tensor(output_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, output_lengths

def collate_grud_test_sequence(test_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])
    
    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    output_lengths_batch = []

    datasetType = test_data[0]
    datasetType = datasetType.split("/")[-3]
    fileNm = "testIndexes/testIndexes__" + datasetType + "__" + args.modality_inclusion + "__" + args.predict_type + "__" + args.output_type + "__PA" + str(args.prediction_after) + "__" + args.model + ".txt"

    # File Format : {pat_id} {chid} {randIndex}
    # Open the file and add existing entries to dictionary
    patDict = {}
    if exists(fileNm):
        indexFile = open(fileNm, "r")
        while True:
            line = indexFile.readline()
            if not line:
                break
            line = line.strip().split()
            pat_id = int(line[0])
            chid = int(line[1])
            randIndex = int(line[2])
            patDict[(pat_id, chid)] = randIndex
        indexFile.close()

    for pkl_id, pkl_path in enumerate(test_data):
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Data Normalization
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        sequenceLength = len(data_pkl['data'])
        windowIndex = args.test_window_size - 1

        for x in ["cpr_inputs", "intubation_inputs", "vasso_inputs"]:
            if sum(data_pkl[x][3:]) == 0:
                data_pkl[x.split("_")[0] + "_yn"] = 0

        if data_pkl["death_yn"] == 1:
            death_time = int(floor(data_pkl["death_time"]))
            if death_time >= sequenceLength + args.prediction_horizon_size - 4:
                data_pkl['death_yn'] = 0
        else:
            death_time = 0
            
        pat_id = int(data_pkl['pat_id'])
        chid = int(data_pkl['chid'])

        randLength = args.test_window_size
        if (pat_id, chid) in patDict:
            randIndex = patDict[(pat_id, chid)]
        else:
            outputFile = open(fileNm, 'a')
            if args.output_type == "mortality":
                if data_pkl['death_yn'] == 1:
                    possibleIndexes = [death_time - i for i in range(0, args.prediction_horizon_size) if (death_time - i >= 2) and (death_time - i < sequenceLength)] or None
                    randIndex = possibleIndexes[random.randrange(len(possibleIndexes))]
                else:
                    randIndex = random.randrange(max(sequenceLength - randLength - 1, 2), sequenceLength - 3)
            elif args.output_type in ['vasso', 'intubation', 'cpr']:
                tmpTask = ['vasso', 'intubation', 'cpr']
                tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
                tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
                taskIndex = tmpTask.index(args.output_type)

                if data_pkl[tmpYN[taskIndex]] == 0:
                    randIndex = random.randrange(2, min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - 3)
                else:
                    possibleIndexes = []
                    for idx, x in enumerate(data_pkl[tmpInputs[taskIndex]]):
                        if x == 1:
                            for i in range(1, args.prediction_horizon_size+1):
                                possibleIndexes.append(idx - i)
                    plausibleIndexes = [x for x in possibleIndexes if x >= 2 and x <= sequenceLength - 4] or None
                    goodIndexes = [x for x in plausibleIndexes if x >= args.test_window_size - 1] or None

                    if(plausibleIndexes == None):
                        raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                    # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                    if goodIndexes == None:
                        randIndex = plausibleIndexes[random.randrange(len(plausibleIndexes))]
                    else:
                        randIndex = goodIndexes[random.randrange(len(goodIndexes))]
            elif args.output_type == "transfer":
                raise NotImplementedError

            elif args.output_type == "all":
                positiveExists = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['vasso_yn'] or data_pkl['intubation_yn'])

                if positiveExists == 0:
                    if sequenceLength >= args.test_window_size + 3:
                        randIndex = random.randrange(args.test_window_size-1, sequenceLength-3)
                    else:
                        randIndex = sequenceLength - 4
                else:
                    possibleIndexes = []
                    for x in ['death', 'cpr', 'intubation', 'vasso']:
                            if data_pkl[x + "_yn"] == 1:
                                if x == 'death':
                                    for i in range(args.prediction_horizon_size):
                                        possibleIndexes.append(death_time - i)
                                else:
                                    for idx, posIn in enumerate(data_pkl[x + "_inputs"]):
                                        if posIn == 1:
                                            for i in range(1, args.prediction_horizon_size+1):
                                                possibleIndexes.append(idx - i)
                    
                    # Assert there exists indexes such that subsequences of length 3 or longer are possible
                    plausibleIndexes = [x for x in possibleIndexes if x >= 2 and x <= sequenceLength - 4] or None
                    goodIndexes = [x for x in plausibleIndexes if x >= args.test_window_size - 1] or None

                    if(plausibleIndexes == None):
                        raise Exception('Classification Error -> No possible sampling indexes for positive test data instance')

                    # If all possible arrays have distance from start less than args.window_size, choose randomly then add zero-padding at back
                    if goodIndexes == None:
                        randIndex = plausibleIndexes[random.randrange(len(plausibleIndexes))]
                    else:
                        randIndex = goodIndexes[random.randrange(len(goodIndexes))]
                    
            outputFile.write(str(pat_id) + " " + str(chid) + " " + str(randIndex) + "\n")
            outputFile.close()
                                    
        dataSequence, maskSequence, deltaSequence, inputLength = testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl)
            
        if args.output_type == "mortality":
            deathInputs = np.zeros(sequenceLength)
            if data_pkl['death_yn'] == 1:
                if death_time < sequenceLength:
                    deathInputs = np.append(deathInputs[:death_time + 1], np.ones(args.prediction_horizon_size + args.test_window_size))
                else:
                    deathInputs = np.append(deathInputs, np.zeros(death_time - sequenceLength + 1))
                    deathInputs = np.append(deathInputs, np.ones(args.prediction_horizon_size + args.test_window_size))
            else:
                deathInputs = np.append(deathInputs, np.zeros(args.prediction_horizon_size))
            
            if data_pkl['death_yn'] == 0:
                target = np.append([2], np.zeros(args.test_window_size + args.prediction_horizon_size))
                outputLength = inputLength + min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1
            
            else:
                target = np.append([2], deathInputs[randIndex-inputLength+1:randIndex+1+args.prediction_horizon_size+args.test_window_size-inputLength])
                outputLength = args.test_window_size + args.prediction_horizon_size + 1
        
        elif args.output_type in ['vasso', 'intubation', 'cpr']:
            tmpTask = ['vasso', 'intubation', 'cpr']
            tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
            tmpInputs = ['vasso_inputs', 'intubation_inputs', 'cpr_inputs']
            taskIndex = tmpTask.index(args.output_type)

            # If patient hasn't recieved target treatment, add (inputLength + targetLength) size zero-vector as target
            # Then add the SOS token (2) and additional zero-padding
            if data_pkl[tmpYN[taskIndex]] == 0:
                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], np.zeros(args.prediction_horizon_size + args.test_window_size))
                outputLength = inputLength + targetLength + 1

            # If patient has recieved target treatment, set the target as the subsequence of {target}_inputs starting from randIndex + 1 and until
            # either the sequence ends or reaches args.prediction_horizon_size. Then add SOS and EOS and additional zero-padding if necessary
            else:
                targetLength = min(min(len(data_pkl[tmpInputs[taskIndex]]), sequenceLength) - randIndex - 1, args.prediction_horizon_size)
                target = np.append([2], data_pkl[tmpInputs[taskIndex]][randIndex-inputLength+1:randIndex+1+targetLength])
                target = np.append(target, np.zeros(args.prediction_horizon_size + args.test_window_size + 1 - len(target)))
                outputLength = targetLength + inputLength + 1
        
        elif args.output_type == "transfer":
            raise NotImplementedError
        
        elif args.output_type == "all":
            positiveExists = int(data_pkl['death_yn'] or data_pkl['cpr_yn'] or data_pkl['vasso_yn'] or data_pkl['intubation_yn'])

            if positiveExists == 0:
                # target = np.append([2], np.zeros(args.prediction_horizon_size + args.test_window_size))
                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, np.zeros([4,args.test_window_size+args.prediction_horizon_size]), axis=1)
                outputLength = inputLength + min(sequenceLength - randIndex - 1, args.prediction_horizon_size) + 1

            else:
                maxSequenceLength = max(len(data_pkl['cpr_inputs']), len(data_pkl['vasso_inputs']), len(data_pkl['intubation_inputs']), death_time, sequenceLength)

                targetSequence = np.zeros([4, maxSequenceLength - randIndex + randLength - 1])
                flag = True
                for i in range(maxSequenceLength - randIndex + randLength - 1):
                    origIndex = randIndex - inputLength + i + 1
                    if (len(data_pkl['cpr_inputs']) > origIndex) and (data_pkl['cpr_inputs'][origIndex] == 1):
                        targetSequence[0,i] = 1
                    if (len(data_pkl['vasso_inputs']) > origIndex) and (data_pkl['vasso_inputs'][origIndex] == 1):
                        targetSequence[0,i] = 1
                    if (len(data_pkl['intubation_inputs']) > origIndex) and (data_pkl['intubation_inputs'][origIndex] == 1):
                        targetSequence[0,i] = 1
                    if (data_pkl['death_yn'] == 1) and (origIndex > death_time) and flag:
                        targetSequence[0,i:] = 1
                        flag = False
                
                target_list = np.append(np.array([2,2,2,2])[np.newaxis].T, targetSequence, axis=1)
                    
                if len(target_list[0,:]) > (args.prediction_horizon_size + args.test_window_size + 1):
                    target_list = target_list[:,:args.prediction_horizon_size + args.test_window_size + 1]
                else:
                    target_list = np.append(target_list, np.zeros([4, args.prediction_horizon_size + args.test_window_size + 1 - len(target_list[0,:])]), axis=1)

                if data_pkl['death_yn'] == 1:
                    outputLength = args.prediction_horizon_size + args.test_window_size + 1
                else:
                    outputLength = inputLength + (maxSequenceLength - randIndex - 1) + 1

        # Assert Sequence Lengths
        if(len(dataSequence) != args.test_window_size):
            raise Exception('data sequence length wrong!')
        if(len(maskSequence) != args.test_window_size):
            raise Exception('mask sequence length wrong!')
        if(len(deltaSequence) != args.test_window_size):
            raise Exception('delta sequence length wrong!')

        # print("Patient Number : " + str(pkl_id) + " / Patient ID : " + str(data_pkl['pat_id']))
        # print("RandIndex : " + str(randIndex) + "    RandLength : " + str(randLength) + "    Death : " + str(data_pkl['death_yn']) + "    Output Length : " + str(outputLength))
        # print("Input Length " + str(len(data_pkl['data'])))
        # print([float(t[0]) for t in data_pkl['data']])
        # print("Target Length : " + str(len(target)))
        # print(target)
        # print("\n")

        # Conversion to tensors
        # seqs_batch.append(torch.stack([torch.Tensor(dataSequence), torch.Tensor(maskSequence), torch.Tensor(deltaSequence)]))
        seqs_batch.append(torch.stack([torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)), 
                                        torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1))
                                        ]))
        if args.output_type == "all":
            target_batch.append(torch.Tensor(target_list))
        else:
            target_batch.append(torch.Tensor(target))
        static_batch.append(static_inputs)
        input_lengths_batch.append(inputLength)
        output_lengths_batch.append(outputLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    output_lengths = torch.Tensor(output_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, output_lengths

def collate_txt_binary(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    input_lengths_batch = []
    target = 0


    for txtInput in train_data:

        split = txtInput.split("/")
        positivities = split[-1].split()
        if args.output_type == "mortality":
            target = int(positivities[0])
        elif args.output_type == "vasso":
            target = int(positivities[2])
        elif args.output_type == "cpr":
            target = int(positivities[4])
        elif args.output_type == "intubation":
            target = int(positivities[6])
        elif args.output_type == "all":
            target = int(int(positivities[0]) or int(positivities[2]) or int(positivities[4]) or int(positivities[6]))
        else:
            raise NotImplementedError
        
        tokens = [int(x) for x in split[1].split()]
        inputLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                token = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                token = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(torch.Tensor(tokens))
        target_batch.append(target)
        input_lengths_batch.append(inputLength + 2)

    targets = torch.Tensor(target_batch).to(torch.float)
    seqs = torch.stack(seqs_batch).to(torch.int)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.int)
    
    return seqs, targets, input_lengths