import torch
import random
import numpy as np

from builder.utils.utils import *
from builder.utils.process_util import *

import re
import os


# pkl_dirs = search_walk({'path': "/home/destin/training_data_0317/", 'extension': ".pkl"})
# with open("margin_dir.pkl","rb") as f:
#     margin_dir = pickle.load(f)
    
# for pkl_dir in pkl_dirs:
#     with open(pkl_dir, 'rb') as _f:
#         data_info = pkl.load(_f)
    
#     if "cxr_input" in data_info:
#         cxrs = data_info['cxr_input']
#         if cxrs is not None:
#             new_cxrs = []
#             print("cxrs: ", cxrs)
#             if len(cxrs) > 0:
#                 for cxr_path in cxrs:
#                     cxr_time = cxr_path[0]
#                     cxr_path = cxr_path[1]
#                     pid=cxr_path.split("/")[-1].split(".")[0]
#                     matching = [s for s in margin_dir if pid in s]
#                     cxr_path = ".".join(cxr_path.split(".")[:-1])
#                     print(matching)
#                     print(cxr_path)
#                     print(" ")
#                     img_path = os.path.join(cxr_path.replace('files_jpg','files_margins')+"_aspect_ratio_"+matching[0].split("_")[-1])
#                     new_cxrs.append([cxr_time, img_path])
#                 data_info['cxr_input'] = new_cxrs
                
#                 # os.system("rm -rf {}".format(pkl_dir))
#                 # with open(pkl_dir, 'wb') as f:
#                 #     pickle.dump(data_info, f)
                
                
                
                
                
                
pkl_dirs = search_walk({'path': "/nfs/thena/shared/multi_modal/training_data_0320/mimic_cf_icu_size48/", 'extension': ".pkl"})
with open("margin_dir.pkl","rb") as f:
    margin_dir = pickle.load(f)
    
    
def change_cxr_path(pkl_dir):
    with open(pkl_dir, 'rb') as _f:
        data_info = pkl.load(_f)
        
    if "cxr_input" in data_info:
        cxrs = data_info['cxr_input']
        if cxrs is not None:
            new_cxrs = []
            if len(cxrs) > 0:
                for cxr_path in cxrs:
                    cxr_time = cxr_path[0]
                    cxr_path = cxr_path[1]
                    pid=cxr_path.split("/")[-1].split(".")[0]
                    matching = [s for s in margin_dir if pid in s]
                    if len(matching) > 0:
                        cxr_path = ".".join(cxr_path.split(".")[:-1])
                        img_path = os.path.join(cxr_path.replace('files_jpg','files_margins')+"_aspect_ratio_"+matching[0].split("_")[-1])
                        new_cxrs.append([cxr_time, img_path])                    
                if len(new_cxrs) > 0:
                    data_info['cxr_input'] = new_cxrs
                else:
                    del data_info['cxr_input']
                    
                os.system("rm -rf {}".format(pkl_dir))
                with open(pkl_dir, 'wb') as f:
                    pickle.dump(data_info, f)
                
run_multi_process(change_cxr_path, pkl_dirs)
    