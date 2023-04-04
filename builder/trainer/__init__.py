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
                exist_img=None,
                txt_lengths=None, 
                seq_lengths=None, 
                imgtxt_time=None,
                scaler=None, 
                missing=None,
                flow_type=None,
                reports_tokens=None,
                reports_lengths=None,
                criterion_aux=None):
    #print(flow_type)
    model, iter_loss = missing_trainer(args, iteration, x, static, input_lengths, y, 
                                        model, logger, device, scheduler, optimizer, criterion, 
                                        scaler, flow_type, output_lengths, 
<<<<<<< HEAD
                                        seq_lengths=seq_lengths, x_img=x_img, exist_img=exist_img, x_txt=x_txt, 
                                        txt_lengths=txt_lengths, imgtxt_time=imgtxt_time, missing=missing, criterion_aux = criterion_aux)
=======
                                        seq_lengths=seq_lengths, x_img=x_img, x_txt=x_txt, 
                                        txt_lengths=txt_lengths, imgtxt_time=imgtxt_time, missing=missing, 
                                        reports_tokens=reports_tokens, reports_lengths=reports_lengths, criterion_aux = criterion_aux)
        
>>>>>>> 4c80bba8089573b532360f10b220efd43bb28567

    return model, iter_loss