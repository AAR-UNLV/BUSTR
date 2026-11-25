from transformers import SwinModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


class Swin_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.swin = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.swin = get_peft_model(self.swin, peft_config_visual)
            print('Fine Tuning Swin')
        else:
            print('Retraining Swin')
        
        self.swin.pooler = nn.Identity()
        self.swin.layernorm = nn.Identity()
        
        self.swin_output_dim = self.swin.num_features

        # embedding dimension of the language model (Llama 2 is 4096)
        self.num_features = self.swin_output_dim * 4
        
        self.swin.histology_layers = nn.Sequential(
            nn.Linear(self.swin_output_dim, self.num_features),
            nn.ReLU(),
        )

        self.swin.pathology_layers = nn.Sequential(
            nn.Linear(self.swin_output_dim, self.num_features),
            nn.ReLU(),
        )

        self.swin.birads_layers = nn.Sequential(
            nn.Linear(self.swin_output_dim, self.num_features),
            nn.ReLU(),
        )
        
        self.swin.tumor_size_layers = nn.Sequential(
            nn.Linear(self.swin_output_dim, self.num_features),
            nn.ReLU(),
        )

        self.tumor_size_head = nn.Linear(self.num_features, 1)
        self.sigm = nn.Sigmoid()
        self.pooler = nn.AdaptiveAvgPool1d(1)   
        
    def forward(self, images):      
        
        swin_states = self.swin(images)
        x = swin_states['last_hidden_state']
        
        tumor_size = self.swin.tumor_size_layers(x)
        tumor_size = torch.flatten(self.pooler(tumor_size.transpose(1, 2)), 1)
        tumor_size = self.sigm(self.tumor_size_head(tumor_size))

        histology = self.swin.histology_layers(x)
        pathology = self.swin.pathology_layers(x)
        birads = self.swin.birads_layers(x)

        return {'tumor_size':tumor_size, 'histology':histology, 'pathology': pathology, 'birads':birads}