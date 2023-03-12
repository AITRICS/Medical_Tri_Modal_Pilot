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

import importlib

def get_model(args):
    model_module = importlib.import_module("builder.models.8_missing_models." + args.model)        
    # if "missing" in args.modality_inclusion:
    #     model_module = importlib.import_module("builder.models.8_missing_models." + args.model)        
    # else:
    #     if args.input_types == 'vslt':
    #         if args.trainer == "seq_pretrain":
    #             model_module = importlib.import_module("builder.models.7_pretrain." + args.model)        
    #         else:
    #             model_module = importlib.import_module("builder.models.1_uni_vslt." + args.model)

    #     elif args.input_types == 'txt':
    #         if args.model == "lstm":
    #             model_module = importlib.import_module("builder.models.3_uni_text.lstm")
    #         else:
    #             model_module = importlib.import_module("builder.models.3_uni_text.transformer")
    
    #     elif args.input_types == 'img':
    #         model_module = importlib.import_module("builder.models.2_uni_image." + args.model)
        
    #     elif args.input_types == "vslt_txt":
    #         if args.trainer == "seq_pretrain":
    #             model_module = importlib.import_module("builder.models.7_pretrain." + args.model)        
    #         else:
    #             model_module = importlib.import_module("builder.models.5_bi_vslt_txt." + args.model)
        
    #     elif args.input_types == "vslt_img":
    #         model_module = importlib.import_module("builder.models.4_bi_vslt_img." + args.model)

    #     elif args.input_types == "vslt_img_txt":
    #         model_module = importlib.import_module("builder.models.6_tri_vslt_img_txt." + args.model)

    #     else:
    #         raise NotImplementedError('currently unimodal (vslt, txt, img) / bimodal (vslt+txt, vslt+img) / trimodal (vslt+img+txt) are available')

    model = getattr(model_module, args.model.upper())

    return model
