import torch
import torch.nn as nn
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.mbt_encoder import TrimodalTransformerEncoder_Multitokens_MBTVSLTMAIN
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from builder.models.src.vision_transformer import vit_b_16_m, ViT_B_16_Weights
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights

class TRI_MBT_VMULTI2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ##### Configuration
        self.output_dim = 1
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout
        self.idx_order = torch.arange(0, args.batch_size).type(torch.LongTensor)
        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.vslt_input_size = len(args.vitalsign_labtest)
        self.n_modality = len(args.input_types.split("_"))
        self.bottlenecks_n = 4
        activation = 'relu'
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])
        
        ##### Encoders
        vslt_pe = False
        self.ie_vslt = nn.Sequential(
                                    nn.Linear(1, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )
        self.ie_time = nn.Sequential(
                                    nn.Linear(1, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )
        self.ie_feat = nn.Embedding(20, self.model_dim)
        self.ie_demo = nn.Sequential(
                                    nn.Linear(2, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )
        
        if args.berttype == "bert": # BERT
            self.txt_embedding = nn.Embedding(30000, self.model_dim)
        elif args.berttype == "biobert": # BIOBERT
            self.txt_embedding = nn.Linear(768, self.model_dim)
        
        self.img_model_type = args.img_model_type
        self.img_pretrain = args.img_pretrain
        if self.img_model_type == "vit":
            if self.img_pretrain == "Yes":
                self.img_encoder = vit_b_16_m(weights = ViT_B_16_Weights.IMAGENET1K_V1)#vit_b_16
            else:
                self.img_encoder = vit_b_16_m(weights = None)
        elif self.img_model_type == "swin":
            if self.img_pretrain =="Yes":
                self.img_encoder = swin_t_m(weights = Swin_T_Weights.IMAGENET1K_V1)#Swin_T_Weights.IMAGENET1K_V1
                model_dict = self.img_encoder.state_dict()
                old_weights=torch.load("/nfs/thena/shared/multi_modal/mlhc/chx_ckpts/image_reports_swin_1e-6_resize_affine_crop-resize_crop_0323_best_fold0_seed0.pth")['model']
                new_weights=torch.load("/nfs/thena/shared/multi_modal/mlhc/chx_ckpts/image_reports_swin_1e-6_resize_affine_crop-resize_crop_0323_best_fold0_seed0.pth")['model']
                new_weights = {key.replace('img_encoder.', ''): new_weights.pop(key) for key in old_weights.keys()}
                new_weights = {k: v for k, v in new_weights.items() if k in model_dict}
                model_dict.update(new_weights)
                self.img_encoder.load_state_dict(new_weights)
            else:
                self.img_encoder = swin_t_m(weights = None)
        else:
            self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,
            img_size=args.image_size,
            patch_size=16,
            hidden_size=self.model_dim,
            num_heads=4,
            pos_embed="conv",
            dropout_rate=0,
            spatial_dims=2,
            )           
        self.linear = nn.Linear(768,256)     
        self.flatten = nn.Flatten(1,2)
        if self.args.residual_bottlenecks == 1:
            residual_bottlenecks = True
        else:
            residual_bottlenecks = False
        ##### Fusion Part
        self.fusion_transformer = TrimodalTransformerEncoder_Multitokens_MBTVSLTMAIN(
            batch_size = args.batch_size,
            n_modality = self.n_modality,
            bottlenecks_n = self.bottlenecks_n,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            fusion_startidx = args.mbt_fusion_startIdx,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 2500,
            resbottle = residual_bottlenecks,
            use_pe = [vslt_pe, False, True],
            mask = [True, True, True],
        )

        ##### Classifier
        classifier_dim = self.model_dim*2
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim)
        self.rmse_layer = nn.Linear(in_features=classifier_dim, out_features= 1, bias=True)
        self.fc_lists = nn.ModuleList([nn.Sequential(
                            nn.Linear(in_features=classifier_dim, out_features= self.model_dim, bias=True),
                            # nn.BatchNorm1d(self.model_dim),
                            nn.LayerNorm(self.model_dim),
                            self.activations[activation],
                            nn.Linear(in_features=self.model_dim, out_features= self.output_dim,  bias=True)) for _ in range(4)])

        self.img_feat = torch.Tensor([18]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)
        self.txt_feat = torch.Tensor([19]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):
        demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1)
        demo_embedding = self.ie_demo(demographic)

        demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1)
        value_embedding = self.ie_vslt(x[:,:,1].unsqueeze(2))
        time_embedding = self.ie_time(x[:,:,0].unsqueeze(2))
        feat = x[:,:,2].type(torch.IntTensor).to(self.device)
        feat_embedding = self.ie_feat(feat)
        vslt_embedding = value_embedding + time_embedding + feat_embedding
        demo_embedding = self.ie_demo(demographic)

        txt_embedding = self.txt_embedding(txts)

        img_embedding = self.img_encoder(img)
        img_embedding = self.flatten(img_embedding)
        img_embedding = self.linear(img_embedding)     
        img_time = img_time.reshape(-1).detach().clone()

        if self.args.imgtxt_time == 1:
            img_embedding = img_embedding + self.ie_time(img_time.unsqueeze(1)).unsqueeze(1) + self.ie_feat(self.img_feat)
            txt_embedding = txt_embedding + self.ie_time(txt_time.unsqueeze(1)).unsqueeze(1) + self.ie_feat(self.txt_feat)
        
        outputs, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, img_embedding, txt_embedding], 
                                      fixed_lengths = [vslt_embedding.size(1), img_embedding.size(1), txt_embedding.size(1)],
                                      varying_lengths = [input_lengths,  torch.tensor(img_embedding.size(1)).repeat(img_embedding.size(0)), txt_lengths+2],
                                      fusion_idx = None,
                                      missing=missing
                                      )
        
        outputs_stack = torch.stack([outputs[i][:, 0, :] for i in range(self.n_modality)])
        tri_mean = torch.mean(outputs_stack, dim=0)
        vsltimg_mean = torch.mean(torch.stack([outputs[0][:, 1, :], outputs[1][:, 1, :]]), dim=0)
        vslttxt_mean = torch.mean(torch.stack([outputs[0][:, 2, :], outputs[2][:, 1, :]]), dim=0)
        final_output = torch.stack([tri_mean, vsltimg_mean, vslttxt_mean, outputs[0][:, 3, :]])
        classInput = self.layer_norms_after_concat(final_output)
        
        classInput = torch.cat([classInput, demo_embedding.unsqueeze(0).repeat(4,1,1)], dim=2)
        
        outputs_stack_list = []
        for i, fc_list in enumerate(self.fc_lists):
            outputs_stack_list.append(fc_list(classInput[i,:,:]))
        output = torch.stack(outputs_stack_list)
        
        if "rmse" in self.args.auxiliary_loss_type:
            output2 = self.rmse_layer(classInput).squeeze()
        else:
            output2 = None
        output3 = None
        return output, output2, output3