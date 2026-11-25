import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, mean_absolute_error
import pandas as pd
from models.vision_encoder.swin_encoder_busbra import Swin_Encoder
from lightning.pytorch import seed_everything

class Swin_Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        seed_everything(args.seed, workers=True)

        self.swin_encoder = Swin_Encoder(args)
        self.test_step_outputs = []
        self.val_step_outputs = []
        self.val_score = 0
        self.pooler = nn.AdaptiveAvgPool1d(1)        
                
        self.histology_head = nn.Linear(self.swin_encoder.num_features, 28)
        self.pathology_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.birads_head = nn.Linear(self.swin_encoder.num_features, 4)
        
        if args.test:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=True)
            print(f'Load checkpoint from {args.delta_file}')
            
    def forward(self, samples):
        image = samples['image']

        target_histology = samples['histology']
        target_pathology = samples['pathology']
        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size'] / 3.0 #max size is 2 in this dataset
        
        encoder_output = self.swin_encoder(image)
        
        histology_hidden_states = torch.flatten(self.pooler(encoder_output['histology'].transpose(1, 2)), 1)
        pathology_hidden_states = torch.flatten(self.pooler(encoder_output['pathology'].transpose(1, 2)), 1)
        birads_hidden_states = torch.flatten(self.pooler(encoder_output['birads'].transpose(1, 2)), 1)
                
        pred_histology = self.histology_head(histology_hidden_states)
        pred_pathology = self.pathology_head(pathology_hidden_states)
        pred_birads = self.birads_head(birads_hidden_states)
        pred_tumor_size = encoder_output['tumor_size']
        
        criterion = nn.CrossEntropyLoss().to(image[0].device)
        criterion_l1 = nn.L1Loss()
        
        loss_histology = criterion(pred_histology, target_histology)
        loss_pathology = criterion(pred_pathology, target_pathology)
        loss_birads = criterion(pred_birads, target_birads)
        loss_tumor_size = criterion_l1(torch.flatten(pred_tumor_size), target_tumor_size)
        
        loss = (loss_histology * 0.4) + (loss_pathology * 0.4) + (loss_birads * 0.1) + (loss_tumor_size * 0.1)  

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=False)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }

        state_dict = self.state_dict()
                
        save_model = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        
        state_dict_encoder = self.swin_encoder.state_dict()
                
        save_encoder = {
            "model": state_dict_encoder,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }

        torch.save(save_model, f'{self.hparams.savedmodel_path}/best_weights.pth')
        torch.save(save_encoder, f'{self.hparams.savedmodel_path}/vision_encoder_weights.pth')
        
    
    def validation_step(self, samples, batch_idx):
        image = samples['image']
        
        target_histology = samples['histology']
        target_pathology = samples['pathology']
        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size']
        
        encoder_output = self.swin_encoder(image)
        
        histology_hidden_states = torch.flatten(self.pooler(encoder_output['histology'].transpose(1, 2)), 1)
        pathology_hidden_states = torch.flatten(self.pooler(encoder_output['pathology'].transpose(1, 2)), 1)
        birads_hidden_states = torch.flatten(self.pooler(encoder_output['birads'].transpose(1, 2)), 1)
                
        pred_histology = self.histology_head(histology_hidden_states)
        pred_pathology = self.pathology_head(pathology_hidden_states)
        pred_birads = self.birads_head(birads_hidden_states)
        pred_tumor_size = encoder_output['tumor_size']
        
        output_histology = torch.argmax(pred_histology, 1).tolist() 
        output_pathology = torch.argmax(pred_pathology, 1).tolist() 
        output_birads = torch.argmax(pred_birads, 1).tolist() 

        output_tumor_size = [x[0] * 3.0 for x in pred_tumor_size.tolist()]

        self.val_step_outputs.append(
            {
                 'output_histology': output_histology, 'target_histology': target_histology.tolist(),
                 'output_pathology': output_pathology, 'target_pathology': target_pathology.tolist(),
                 'output_birads': output_birads, 'target_birads': target_birads.tolist(),
                 'output_tumor_size': output_tumor_size, 'target_tumor_size': target_tumor_size.tolist()
            }
        )
        
        return output_birads, target_birads
        
    def on_validation_epoch_end(self):
        output_histology, target_histology = [], [] 
        output_pathology, target_pathology = [], [] 
        output_birads, target_birads = [], []
        output_tumor_size, target_tumor_size = [], []

        for i in self.val_step_outputs:
            output_histology.extend(i['output_histology'])
            target_histology.extend(i['target_histology'])

            output_pathology.extend(i['output_pathology'])
            target_pathology.extend(i['target_pathology'])

            output_birads.extend(i['output_birads'])
            target_birads.extend(i['target_birads'])
            
            output_tumor_size.extend(i['output_tumor_size'])
            target_tumor_size.extend(i['target_tumor_size'])

        histology_scores = classification_report(target_histology, output_histology, output_dict = True, zero_division = 0)
        pathology_scores = classification_report(target_pathology, output_pathology, output_dict = True, zero_division = 0)
        birads_scores = classification_report(target_birads, output_birads, output_dict = True, zero_division = 0)
        
        histology_f1_score = histology_scores['weighted avg']['f1-score']
        pathology_f1_score = pathology_scores['weighted avg']['f1-score']
        birads_f1_score = birads_scores['weighted avg']['f1-score']

        tumor_size_error = mean_absolute_error(target_tumor_size, output_tumor_size)

        total_score = (histology_f1_score * 0.4) + (pathology_f1_score * 0.4) + (birads_f1_score * 0.1) + ((1-tumor_size_error) * 0.1)
        eval_res = {'FOLD': self.hparams.fold_index,
           'F1 Score Histology': histology_f1_score,
           'F1 Score Pathology': pathology_f1_score,
           'F1 Score BI-RADS': birads_f1_score,
           'MAE Tumor Size': tumor_size_error
       }
        if self.trainer.local_rank == 0:
            if total_score > self.val_score:
                print(f'''\n
                    MAE Tumor Size:  {tumor_size_error:.4f}
                    Histology:       {histology_f1_score:.4f}
                    Pathology:       {pathology_f1_score:.4f}
                    BI-RADS:         {birads_f1_score:.4f}
                \n''')
                print(f'\n\nTotal Weighted F1 Score: {total_score:4f}\n\n')
                self.save_checkpoint(total_score)
                self.val_score = total_score
        self.val_step_outputs.clear()
        
        
        metric_output = pd.read_csv(self.hparams.metrics_location)
        metric_output = pd.concat([metric_output, pd.DataFrame([eval_res])], ignore_index=True)
        metric_output.to_csv(self.hparams.metrics_location, index=False)
        
    def test_step(self, samples, batch_idx):
        image = samples['image']
        
        target_histology = samples['histology']
        target_pathology = samples['pathology']
        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size']
        
        encoder_output = self.swin_encoder(image)
        
        histology_hidden_states = torch.flatten(self.pooler(encoder_output['histology'].transpose(1, 2)), 1)
        pathology_hidden_states = torch.flatten(self.pooler(encoder_output['pathology'].transpose(1, 2)), 1)
        birads_hidden_states = torch.flatten(self.pooler(encoder_output['birads'].transpose(1, 2)), 1)
                
        pred_histology = self.histology_head(histology_hidden_states)
        pred_pathology = self.pathology_head(pathology_hidden_states)
        pred_birads = self.birads_head(birads_hidden_states)
        pred_tumor_size = encoder_output['tumor_size']
        
        output_histology = torch.argmax(pred_histology, 1).tolist() 
        output_pathology = torch.argmax(pred_pathology, 1).tolist() 
        output_birads = torch.argmax(pred_birads, 1).tolist() 

        output_tumor_size = [x[0] * 3.0 for x in pred_tumor_size.tolist()]

        self.test_step_outputs.append(
            {
                 'output_histology': output_histology, 'target_histology': target_histology.tolist(),
                 'output_pathology': output_pathology, 'target_pathology': target_pathology.tolist(),
                 'output_birads': output_birads, 'target_birads': target_birads.tolist(),
                 'output_tumor_size': output_tumor_size, 'target_tumor_size': target_tumor_size.tolist()
            }
        )
        
        return output_birads, target_birads
        
    def on_test_epoch_end(self):
        output_histology, target_histology = [], [] 
        output_pathology, target_pathology = [], [] 
        output_birads, target_birads = [], []
        output_tumor_size, target_tumor_size = [], []

        for i in self.test_step_outputs:
            output_histology.extend(i['output_histology'])
            target_histology.extend(i['target_histology'])

            output_pathology.extend(i['output_pathology'])
            target_pathology.extend(i['target_pathology'])

            output_birads.extend(i['output_birads'])
            target_birads.extend(i['target_birads'])
            
            output_tumor_size.extend(i['output_tumor_size'])
            target_tumor_size.extend(i['target_tumor_size'])

        histology_scores = classification_report(target_histology, output_histology, output_dict = True, zero_division = 0)
        pathology_scores = classification_report(target_pathology, output_pathology, output_dict = True, zero_division = 0)
        birads_scores = classification_report(target_birads, output_birads, output_dict = True, zero_division = 0)
        
        histology_f1_score = histology_scores['weighted avg']['f1-score']
        pathology_f1_score = pathology_scores['weighted avg']['f1-score']
        birads_f1_score = birads_scores['weighted avg']['f1-score']

        tumor_size_error = mean_absolute_error(target_tumor_size, output_tumor_size)
        
        eval_res = {'FOLD': self.hparams.fold_index,
           'F1 Score Histology': histology_f1_score,
           'F1 Score Pathology': pathology_f1_score,
           'F1 Score BI-RADS': birads_f1_score,
           'MAE Tumor Size': tumor_size_error
       }

        print(f'''\n
            MAE Tumor Size:  {tumor_size_error:.4f}
            Histology:       {histology_f1_score:.4f}
            Pathology:       {pathology_f1_score:.4f}
            BI-RADS:         {birads_f1_score:.4f}
        \n''')

        self.test_step_outputs.clear()
        
        
        metric_output = pd.read_csv(self.hparams.metrics_location)
        metric_output = pd.concat([metric_output, pd.DataFrame([eval_res])], ignore_index=True)
        metric_output.to_csv(self.hparams.metrics_location, index=False)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()