
import os
import yaml
import argparse
import torch

seed_list = [0, 1004, 9209, 909, 30, 31, 2022]

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# # Missing modality techniques
# parser.add_argument('--multitoken', type=int, default=0)

# General Parameters
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--seed-list', type=list, default=[412, 1004]) #[0, 1004, 2022, 9209, 119]
parser.add_argument('--device', type=int, default=1, nargs='+')
parser.add_argument('--cpu', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--project-name', type=str, default="small1")
parser.add_argument('--checkpoint', '-cp', type=bool, default=False)

parser.add_argument('--prediction-range', type=int, default=12)
parser.add_argument('--min-inputlen', type=int, default=3)
parser.add_argument('--window-size', type=int, default=24)
parser.add_argument('--vslt-type', type=str, default="carryforward", choices=["carryforward", "TIE", "QIE"])
parser.add_argument('--TIE-len', type=int, default=1000)
parser.add_argument('--ar-lowerbound', type=float, default=0.7)
parser.add_argument('--ar-upperbound', type=float, default=1.3)

parser.add_argument('--input-types', type=str, default="vslt", choices=["vslt", "vslt_img", "vslt_txt", "vslt_img_txt"])
parser.add_argument('--output-type', type=str, default="mortality", choices=['mortality', 'vasso', 'intubation', 'cpr', 'transfer'])
parser.add_argument('--predict-type', type=str, default="within", choices=["within", "multi_task_within", "multi_task_range", "seq_pretrain"])
parser.add_argument('--modality-inclusion', type=str, default="train-full_test-full", choices=['train-full_test-full', 'train-missing_test-missing', 'train-full_test-missing'])
parser.add_argument('--fullmodal-definition', type=str, default="txt1_img1", choices=["txt1_img1", "img1", "txt1"])

# Data path setting
parser.add_argument('--train-data-path', type=str, default="/home/destin/training_data_0320/mimic_cf_icu_size24/train")
parser.add_argument('--test-data-path', type=str, default="/home/destin/training_data_0320/mimic_cf_icu_size24/test")
parser.add_argument('--dir-result', type=str, default="/mnt/aitrics_ext/ext01/destin/multimodal/MLHC_result")
parser.add_argument('--image-data-path', type=str, default="/home/claire/")
# Data Parameters
parser.add_argument('--cross-fold-val', type=int, default=0, choices=[1, 0], help="1: k-fold, 0: seed-average")
parser.add_argument('--val-data-ratio', type=float, default=0.1)
parser.add_argument('--carry-back', type=bool, default=True)
parser.add_argument('--imgtxt-time', type=int, default=0, choices=[0,1])

# Training Parameters
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--l2-coeff', type=float, default=0.002)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', help='activation function of the networks', choices=['selu','relu'], default='relu', type=str) #invase
parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'sgd_lars','adam', 'adam_lars','adamw', 'adamw_lars'])
parser.add_argument('--lr-scheduler', type=str, default="CosineAnnealing" , choices=["CosineAnnealing", "Single"])
parser.add_argument('--lr-init', type=float, default=1e-3) # not being used for CosineAnnealingWarmUpRestarts...
parser.add_argument('--t_0', '-tz', type=int, default=50, help='T_0 of cosine annealing scheduler')
parser.add_argument('--t_mult', '-tm', type=int, default=2, help='T_mult of cosine annealing scheduler')
parser.add_argument('--t_up', '-tup', type=int, default=5, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--gamma', '-gam', type=float, default=0.5, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--momentum', '-mo', type=float, default=0.9, help='Momentum of optimizer')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6, help='Weight decay of optimizer')

parser.add_argument('--patient-time', default=False)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--collate', type=int, default=2) 
parser.add_argument('--quantization', type=bool, default=False)
parser.add_argument('--show-roc', type=bool, default=False)
parser.add_argument('--output-dim', type=int, default=1)

# Text Transformer Parameters
parser.add_argument('--txt-num-layers', type=int, default=8)
parser.add_argument('--txt-dropout', type=float, default=0.1)
parser.add_argument('--txt-model-dim', type=int, default=256)
parser.add_argument('--txt-num-heads', type=int, default=4)
parser.add_argument('--txt-classifier-nodes', type=int, default=64)

parser.add_argument('--txt-tokenization', type=str, default="bert", choices=["word", "character", "bpe", "bert"])
parser.add_argument('--berttype', type=str, default="biobert", choices=["biobert", "bert"])
parser.add_argument('--biobert-path', type=str, default="./data/mimic4_embeddings.h5", choices=["./data/mimic4_embeddings.h5", "./data/mimic4_clstoken.h5"])
parser.add_argument('--character-token-max-length', type=int, default=512)
parser.add_argument('--word-token-max-length', type=int, default=128)
parser.add_argument('--bpe-token-max-length', type=int, default=256)
parser.add_argument('--bert-token-max-length', type=int, default=128)

# Vital Sign Labtest Model Parameters
parser.add_argument('--enc-depth', type=int, default=3, choices=[1,2,3])
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--transformer-dim', type=int, default=256)
parser.add_argument('--transformer-num-layers', type=int, default=6)
parser.add_argument('--transformer-num-head', type=int, default=4)

# Image Model Parameters
parser.add_argument('--resnet-num-layers', type=int, default=18, choices=[18,34,50])
parser.add_argument('--vit-num-layers', type=int, default=8, choices=[4,8,10,12])
parser.add_argument('--vit-patch-size', type=int, default=16, choices=[8,16])

# Image pretrain model
parser.add_argument('--img-model-type', type=str, default="swin", choices=["resnet18", "resnet50", "swin", "vit", "maxvit"])
parser.add_argument("--img-pretrain", type=str, default="Yes", choices = ["No","Yes"])

#Image preprocess argument
parser.add_argument('--image-size', type=int, default=224, choices=[224,512])

#center is default, resize: image-train-type = ["random"], image-test-type = ["resize"]
parser.add_argument('--image-train-type', type=str, default="resize_affine_crop", choices=["random", "resize", "resize_crop", "resize_affine_crop", "randaug"])
parser.add_argument('--image-test-type', type=str, default="resize_crop", choices=["center", "resize", "resize_crop", "resize_larger"])#center: shorter로 resize 후, center crop, resize: aspect ratio 고려 없이 정사각형 resize
parser.add_argument('--image-norm-type', type=str, default="HE", choices=["HE", "CLAHE"])

# MBT Model Parameters
parser.add_argument('--mbt-bottlenecks-n', type=int, default=4)
parser.add_argument('--mbt-fusion-startIdx', type=int, default=0)

# Model Parameters
parser.add_argument('--model-types', type=str, default="detection", choices=["detection", "classification"])
parser.add_argument('--loss-types', type=str, default="bce", choices=["bceandsoftmax", "softmax", "bces", "bce", "wkappa", "rmse"])
parser.add_argument('--loss-weight', type=str, default=None, choices=[None, "patnum"])
# Auxiliary loss
parser.add_argument('--auxiliary-loss-input', type=str, default=None, choices=[None, "directInput", "encOutput"])
parser.add_argument('--auxiliary-loss-type', type=str, default="None", choices=["None", "rmse", "tdecoder", "tdecoder_rmse"])
parser.add_argument('--auxiliary-loss-weight', type=float, default=1.0)

parser.add_argument('--mandatory-vitalsign-labtest', type=list, default=['HR', 'RR', 'BT', 'SBP', 'DBP', 'Sat'])  
parser.add_argument('--vitalsign-labtest', type=list, default=['HR', 'RR', 'BT', 'SBP', 'DBP', 'Sat', 'Hematocrit', 'PLT', 'WBC', 'Bilirubin', 'pH', 'HCO3', 'Creatinine', 'Lactate', 'Potassium', 'Sodium'])  
parser.add_argument('--model', type=str, default="gru_d") 

# Visualize / Logging Parameters
parser.add_argument('--log-iter', type=int, default=10)
parser.add_argument('--nonPatNegSampleN', type=int, default=4)
parser.add_argument('--PatNegSampleN', type=int, default=1)
parser.add_argument('--PatPosSampleN', type=int, default=5)
parser.add_argument('--best', default=True, action='store_true')
parser.add_argument('--last', default=False, action='store_true')

parser.add_argument('--fuse-baseline', type=str, default=None, choices=["Medfuse", "MMTM","DAFT","Retain","Multi"])
parser.add_argument('--mmtm-ratio', type=float, default=4, help='mmtm ratio hyperparameter')
parser.add_argument('--daft_activation', type=str, default='linear', help='daft activation ')
parser.add_argument('--fusion-type', type=str, default='fused_ehr', help='train or eval for [fused_ehr, fused_cxr, uni_cxr, uni_ehr]')

args = parser.parse_args()
args.dir_root = os.getcwd()

if "train-full" in args.modality_inclusion:
    if not all([True if i + "1" in args.fullmodal_definition.split("_") else False  for i in args.input_types.split("_") if i != "vslt"]):
        raise ValueError('invalid input_types for full_modal with fullmodal_definition!!!')
