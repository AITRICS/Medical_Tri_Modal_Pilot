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

from .trainer import *

def get_trainer(args, 
                iteration, 
                x, 
                static, 
                input_lengths, 
                y, 
                output_lengths, 
                model, 
                logger, 
                device, 
                scheduler, 
                optimizer, 
                criterion, 
                x_txt=None, 
                x_img=None, 
                txt_lengths=None, 
                seq_lengths=None, 
                scaler=None, 
                missing=None,
                flow_type=None):
    #print(flow_type)
    if "missing" in args.modality_inclusion:
        model, iter_loss = missing_trainer(args, iteration, x, static, input_lengths, y, 
                                            model, logger, device, scheduler, optimizer, criterion, 
                                            scaler, flow_type, output_lengths, 
                                            seq_lengths=seq_lengths, x_img=x_img, x_txt=x_txt, txt_lengths=txt_lengths, missing=missing)
    elif 'multi_task' in args.trainer:
        if args.input_types == 'vslt':
            if args.auxiliary_loss_input is None:
                model, iter_loss = multiTaskLearningVslt(args, iteration, x, static, input_lengths, y, model, logger, 
                                                        device, scheduler, optimizer, criterion, seq_lengths, scaler, flow_type)
        elif args.input_types == 'vslt_txt':
            if args.auxiliary_loss_input is None:
                model, iter_loss = multiTaskLearningVsltTxt(args, iteration, x, static, input_lengths, y, x_txt, txt_lengths, 
                                                            model, logger, device, scheduler, optimizer, criterion, 
                                                            seq_lengths, scaler, flow_type)
            else:
                model, iter_loss = multiTaskLearningVsltTxt_Aux(args, iteration, x, static, input_lengths, y, x_txt, txt_lengths, 
                                                            model, logger, device, scheduler, optimizer, criterion, 
                                                            seq_lengths, scaler, flow_type, output_lengths)
        elif args.input_types == "vslt_img":
            model, iter_loss = multiTaskLearningVsltImg(args, iteration, x, static, input_lengths, y, x_img, model, logger,
                                                        device, scheduler, optimizer, criterion, seq_lengths, scaler, flow_type)
        elif args.input_types == "vslt_img_txt":
            model, iter_loss = multiTaskLearningVsltImg_txt(args, iteration, x, static, input_lengths, y, x_img, x_txt, txt_lengths, 
                                                            model, logger, device, scheduler, optimizer, criterion, 
                                                            seq_lengths, scaler, flow_type)#, output_lengths

        else:
            print("Selected trainer is not prepared yet...")
            exit(1)

    elif args.trainer == 'seq_pretrain':
        model, iter_loss = SelfSupLearning(args, iteration, x, static, input_lengths, x_txt, txt_lengths, 
                                                        model, logger, device, scheduler, optimizer, criterion, 
                                                        seq_lengths, flow_type)

    else:
        print("Selected trainer is not prepared yet...")
        exit(1)

    return model, iter_loss