import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, mean_absolute_error
import pandas as pd
from models.vision_encoder.swin_encoder_breast import Swin_Encoder
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
        
        self.birads_head = nn.Linear(self.swin_encoder.num_features * 4, 6)
        
        self.shape_head = nn.Linear(self.swin_encoder.num_features, 3)
        self.margin_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.margin_indistinct_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.margin_angular_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.margin_spiculated_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.margin_microlobulated_head = nn.Linear(self.swin_encoder.num_features, 2)
        self.posterior_head = nn.Linear(self.swin_encoder.num_features, 4)
        self.echo_head = nn.Linear(self.swin_encoder.num_features, 6)
        
        if args.test:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=True)
            print(f'Load checkpoint from {args.delta_file}')
            
    def forward(self, samples):
        image = samples['image']

        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size'] / 7.0 #max size is 6.1 in the dataset
        target_shape = samples['shape']
        target_margin = samples['margin']
        target_margin_indistinct = samples['margin_indistinct']
        target_margin_angular = samples['margin_angular']
        target_margin_microlobulated = samples['margin_microlobulated']
        target_margin_spiculated =  samples['margin_spiculated'] 
        target_posterior =  samples['posterior_features'] 
        target_echo =  samples['echo'] 

        encoder_output = self.swin_encoder(image)
        
        shape_hidden_states = torch.flatten(self.pooler(encoder_output['shape'].transpose(1, 2)), 1)
        margin_hidden_states = torch.flatten(self.pooler(encoder_output['margin'].transpose(1, 2)), 1)
        posterior_hidden_states = torch.flatten(self.pooler(encoder_output['posterior'].transpose(1, 2)), 1)
        echo_hidden_states = torch.flatten(self.pooler(encoder_output['echo'].transpose(1, 2)), 1)
        
        birads = torch.cat([shape_hidden_states, margin_hidden_states, posterior_hidden_states, echo_hidden_states], dim =1)
        
        pred_birads = self.birads_head(birads)
        pred_tumor_size = encoder_output['tumor_size']
        pred_shape = self.shape_head(shape_hidden_states)
        pred_margin = self.margin_head(margin_hidden_states)
        pred_margin_indistinct = self.margin_indistinct_head(margin_hidden_states)
        pred_margin_angular = self.margin_angular_head(margin_hidden_states)
        pred_margin_spiculated = self.margin_spiculated_head(margin_hidden_states)
        pred_margin_microlobulated = self.margin_microlobulated_head(margin_hidden_states)
        pred_posterior = self.posterior_head(posterior_hidden_states)
        pred_echo = self.echo_head(echo_hidden_states)
        
        criterion = nn.CrossEntropyLoss().to(image[0].device)
        criterion_l1 = nn.L1Loss()
        
        loss_birads = criterion(pred_birads, target_birads)
        loss_tumor_size = criterion_l1(torch.flatten(pred_tumor_size), target_tumor_size)
        
        loss_shape = criterion(pred_shape, target_shape)
        
        loss_margin = criterion(pred_margin, target_margin)
        loss_margin_indistinct = criterion(pred_margin_indistinct, target_margin_indistinct)
        loss_margin_angular = criterion(pred_margin_angular, target_margin_angular)
        loss_margin_spiculated = criterion(pred_margin_spiculated, target_margin_spiculated)
        loss_margin_microlobulated = criterion(pred_margin_microlobulated, target_margin_microlobulated)
        loss_margin = (loss_margin * 0.5) + (0.5*(loss_margin_indistinct + loss_margin_angular + loss_margin_spiculated + loss_margin_microlobulated)/4)

        loss_posterior = criterion(pred_posterior, target_posterior)
        loss_echo = criterion(pred_echo, target_echo)
        
        loss = (loss_tumor_size * 0.2) + (loss_birads * 0.2) + (loss_shape * 0.15) + (loss_margin * 0.15) + (loss_posterior * 0.15) + (loss_echo * 0.15)   

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
        
        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size']

        target_shape = samples['shape']
        target_margin = samples['margin']
        target_margin_indistinct = samples['margin_indistinct']
        target_margin_angular = samples['margin_angular']
        target_margin_microlobulated = samples['margin_microlobulated']
        target_margin_spiculated =  samples['margin_spiculated'] 
        target_posterior =  samples['posterior_features'] 
        target_echo =  samples['echo'] 
        
        encoder_output = self.swin_encoder(image)
        
        shape_hidden_states = torch.flatten(self.pooler(encoder_output['shape'].transpose(1, 2)), 1)
        margin_hidden_states = torch.flatten(self.pooler(encoder_output['margin'].transpose(1, 2)), 1)
        posterior_hidden_states = torch.flatten(self.pooler(encoder_output['posterior'].transpose(1, 2)), 1)
        echo_hidden_states = torch.flatten(self.pooler(encoder_output['echo'].transpose(1, 2)), 1)

        birads = torch.cat([shape_hidden_states, margin_hidden_states, posterior_hidden_states, echo_hidden_states], dim =1)

        pred_birads = self.birads_head(birads)
        pred_tumor_size = encoder_output['tumor_size']
        pred_shape = self.shape_head(shape_hidden_states)
        pred_margin = self.margin_head(margin_hidden_states)
        pred_margin_indistinct = self.margin_indistinct_head(margin_hidden_states)
        pred_margin_angular = self.margin_angular_head(margin_hidden_states)
        pred_margin_spiculated = self.margin_spiculated_head(margin_hidden_states)
        pred_margin_microlobulated = self.margin_microlobulated_head(margin_hidden_states)
        pred_posterior = self.posterior_head(posterior_hidden_states)
        pred_echo = self.echo_head(echo_hidden_states)
        
        output_birads = torch.argmax(pred_birads, 1).tolist() 
        output_shape = torch.argmax(pred_shape, 1).tolist() 
        output_margin = torch.argmax(pred_margin, 1).tolist() 
        output_margin_indistinct = torch.argmax(pred_margin, 1).tolist() 
        output_margin_angular = torch.argmax(pred_margin, 1).tolist() 
        output_margin_microlobulated = torch.argmax(pred_margin, 1).tolist() 
        output_margin_spiculated = torch.argmax(pred_margin, 1).tolist() 
        output_posterior = torch.argmax(pred_posterior, 1).tolist() 
        output_echo = torch.argmax(pred_echo, 1).tolist()
        
        output_tumor_size = [x[0] * 7.0 for x in pred_tumor_size.tolist()]

        self.val_step_outputs.append(
            {
                 'output_birads': output_birads, 'target_birads': target_birads.tolist(),
                 'output_tumor_size': output_tumor_size, 'target_tumor_size': target_tumor_size.tolist(),
                 'output_shape': output_shape, 'target_shape': target_shape.tolist(),
                 'output_margin': output_margin, 'target_margin': target_margin.tolist(),
                 'output_margin_indistinct': output_margin_indistinct, 'target_margin_indistinct': target_margin_indistinct.tolist(),
                 'output_margin_angular': output_margin_angular, 'target_margin_angular': target_margin_angular.tolist(),
                 'output_margin_microlobulated': output_margin_microlobulated, 'target_margin_microlobulated': target_margin_microlobulated.tolist(),
                 'output_margin_spiculated': output_margin_spiculated, 'target_margin_spiculated': target_margin_spiculated.tolist(),
                 'output_posterior': output_posterior, 'target_posterior': target_posterior.tolist(),
                 'output_echo': output_echo, 'target_echo': target_echo.tolist()
            }
        )
        
        return output_birads, target_birads
        
    def on_validation_epoch_end(self):
        output_birads, target_birads = [], []
        output_tumor_size, target_tumor_size = [], []
        output_shape, target_shape = [], [] 
        output_margin, target_margin = [], [] 
        output_margin_indistinct, target_margin_indistinct = [], [] 
        output_margin_angular, target_margin_angular = [], [] 
        output_margin_microlobulated, target_margin_microlobulated = [], [] 
        output_margin_spiculated, target_margin_spiculated = [], [] 
        output_posterior, target_posterior = [], [] 
        output_echo, target_echo = [], [] 

        for i in self.val_step_outputs:
            output_birads.extend(i['output_birads'])
            target_birads.extend(i['target_birads'])
            
            output_tumor_size.extend(i['output_tumor_size'])
            target_tumor_size.extend(i['target_tumor_size'])

            output_shape.extend(i['output_shape'])
            target_shape.extend(i['target_shape'])

            output_margin.extend(i['output_margin'])
            target_margin.extend(i['target_margin'])

            output_margin_indistinct.extend(i['output_margin_indistinct'])
            target_margin_indistinct.extend(i['target_margin_indistinct'])

            output_margin_angular.extend(i['output_margin_angular'])
            target_margin_angular.extend(i['target_margin_angular'])

            output_margin_microlobulated.extend(i['output_margin_microlobulated'])
            target_margin_microlobulated.extend(i['target_margin_microlobulated'])

            output_margin_spiculated.extend(i['output_margin_spiculated'])
            target_margin_spiculated.extend(i['target_margin_spiculated'])

            output_posterior.extend(i['output_posterior'])
            target_posterior.extend(i['target_posterior'])

            output_echo.extend(i['output_echo'])
            target_echo.extend(i['target_echo'])

        birads_scores = classification_report(target_birads, output_birads, output_dict = True, zero_division = 0)
        shape_scores = classification_report(target_shape, output_shape, output_dict = True, zero_division = 0)
        margin_scores = classification_report(target_margin, output_margin, output_dict = True, zero_division = 0)
        margin_scores_indistinct = classification_report(target_margin_indistinct, output_margin_indistinct, output_dict = True, zero_division = 0)
        margin_scores_angular = classification_report(target_margin_angular, output_margin_angular, output_dict = True, zero_division = 0)
        margin_scores_microlobulated = classification_report(target_margin_microlobulated, output_margin_microlobulated, output_dict = True, zero_division = 0)
        margin_scores_spiculated = classification_report(target_margin_spiculated, output_margin_spiculated, output_dict = True, zero_division = 0)
        posterior_scores = classification_report(target_posterior, output_posterior, output_dict = True, zero_division = 0)
        echo_scores = classification_report(target_echo, output_echo, output_dict = True, zero_division = 0)
        
        birads_f1_score = birads_scores['weighted avg']['f1-score']
        shape_f1_score = shape_scores['weighted avg']['f1-score']
        margin_f1_score = margin_scores['weighted avg']['f1-score']
        margin_indistinct_f1_score = margin_scores_indistinct['weighted avg']['f1-score']
        margin_angular_f1_score = margin_scores_angular['weighted avg']['f1-score']
        margin_microlobulated_f1_score = margin_scores_microlobulated['weighted avg']['f1-score']
        margin_spiculated_f1_score = margin_scores_spiculated['weighted avg']['f1-score']
        posterior_f1_score = posterior_scores['weighted avg']['f1-score']
        echo_f1_score = echo_scores['weighted avg']['f1-score']
        tumor_size_error = mean_absolute_error(target_tumor_size, output_tumor_size)

        total_score = ((1-tumor_size_error) * 0.2) + (birads_f1_score * 0.2) + (shape_f1_score * 0.15) + (margin_f1_score * 0.15) + (posterior_f1_score * 0.15) + (echo_f1_score * 0.15) 
        eval_res = {'FOLD': self.hparams.fold_index,
           'F1 Score BI-RADS': birads_f1_score,
           'F1 Score Shape': shape_f1_score,
           'F1 Score Margin': margin_f1_score,
           'F1 Score Echo Pattern': echo_f1_score,
           'F1 Score Posterior Features': posterior_f1_score,
           'MAE Tumor Size': tumor_size_error
       }
        if self.trainer.local_rank == 0:
            if total_score > self.val_score:
                print(f'''\n
                    MAE Tumor Size:  {tumor_size_error:.4f}
                    BI-RADS:         {birads_f1_score:.4f}
                    Shape:           {shape_f1_score:.4f}
                    Posterior:       {posterior_f1_score:.4f}
                    Echogenicity:    {echo_f1_score:.4f}
                    Margin:          {margin_f1_score:.4f}
                      Indistinct:      {margin_indistinct_f1_score:.4f}
                      Angular:         {margin_angular_f1_score:.4f}
                      Microlobulated:  {margin_microlobulated_f1_score:.4f}
                      Spiculated:      {margin_spiculated_f1_score:.4f}
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
        
        target_birads = samples['birads']
        target_tumor_size = samples['tumor_size']
        target_shape = samples['shape']
        target_margin = samples['margin']
        target_margin_indistinct = samples['margin_indistinct']
        target_margin_angular = samples['margin_angular']
        target_margin_microlobulated = samples['margin_microlobulated']
        target_margin_spiculated =  samples['margin_spiculated'] 
        target_posterior =  samples['posterior_features'] 
        target_echo =  samples['echo'] 
        encoder_output = self.swin_encoder(image)
        
        shape_hidden_states = torch.flatten(self.pooler(encoder_output['shape'].transpose(1, 2)), 1)
        margin_hidden_states = torch.flatten(self.pooler(encoder_output['margin'].transpose(1, 2)), 1)
        posterior_hidden_states = torch.flatten(self.pooler(encoder_output['posterior'].transpose(1, 2)), 1)
        echo_hidden_states = torch.flatten(self.pooler(encoder_output['echo'].transpose(1, 2)), 1)
        
        birads = torch.cat([shape_hidden_states, margin_hidden_states, posterior_hidden_states, echo_hidden_states], dim =1)

        pred_birads = self.birads_head(birads)
        pred_tumor_size = encoder_output['tumor_size']
        pred_shape = self.shape_head(shape_hidden_states)
        pred_margin = self.margin_head(margin_hidden_states)
        pred_margin_indistinct = self.margin_indistinct_head(margin_hidden_states)
        pred_margin_angular = self.margin_angular_head(margin_hidden_states)
        pred_margin_spiculated = self.margin_spiculated_head(margin_hidden_states)
        pred_margin_microlobulated = self.margin_microlobulated_head(margin_hidden_states)
        pred_posterior = self.posterior_head(posterior_hidden_states)
        pred_echo = self.echo_head(echo_hidden_states)
        
        output_birads = torch.argmax(pred_birads, 1).tolist() 
        output_shape = torch.argmax(pred_shape, 1).tolist() 
        output_margin = torch.argmax(pred_margin, 1).tolist() 
        output_margin_indistinct = torch.argmax(pred_margin, 1).tolist() 
        output_margin_angular = torch.argmax(pred_margin, 1).tolist() 
        output_margin_microlobulated = torch.argmax(pred_margin, 1).tolist() 
        output_margin_spiculated = torch.argmax(pred_margin, 1).tolist() 
        output_posterior = torch.argmax(pred_posterior, 1).tolist() 
        output_echo = torch.argmax(pred_echo, 1).tolist()
        
        output_tumor_size = [x[0] * 7.0 for x in pred_tumor_size.tolist()]

        self.test_step_outputs.append(
            {
                 'output_birads': output_birads, 'target_birads': target_birads.tolist(),
                 'output_tumor_size': output_tumor_size, 'target_tumor_size': target_tumor_size.tolist(),
                 'output_shape': output_shape, 'target_shape': target_shape.tolist(),
                 'output_margin': output_margin, 'target_margin': target_margin.tolist(),
                 'output_margin_indistinct': output_margin_indistinct, 'target_margin_indistinct': target_margin_indistinct.tolist(),
                 'output_margin_angular': output_margin_angular, 'target_margin_angular': target_margin_angular.tolist(),
                 'output_margin_microlobulated': output_margin_microlobulated, 'target_margin_microlobulated': target_margin_microlobulated.tolist(),
                 'output_margin_spiculated': output_margin_spiculated, 'target_margin_spiculated': target_margin_spiculated.tolist(),
                 'output_posterior': output_posterior, 'target_posterior': target_posterior.tolist(),
                 'output_echo': output_echo, 'target_echo': target_echo.tolist()
            }
        )
        
        return output_birads, target_birads
        
    def on_test_epoch_end(self):
        output_birads, target_birads = [], []
        output_tumor_size, target_tumor_size = [], []
        output_shape, target_shape = [], [] 
        output_margin, target_margin = [], [] 
        output_margin_indistinct, target_margin_indistinct = [], [] 
        output_margin_angular, target_margin_angular = [], [] 
        output_margin_microlobulated, target_margin_microlobulated = [], [] 
        output_margin_spiculated, target_margin_spiculated = [], [] 
        output_posterior, target_posterior = [], [] 
        output_echo, target_echo = [], [] 

        for i in self.test_step_outputs:
            output_birads.extend(i['output_birads'])
            target_birads.extend(i['target_birads'])
            
            output_tumor_size.extend(i['output_tumor_size'])
            target_tumor_size.extend(i['target_tumor_size'])

            output_shape.extend(i['output_shape'])
            target_shape.extend(i['target_shape'])

            output_margin.extend(i['output_margin'])
            target_margin.extend(i['target_margin'])

            output_margin_indistinct.extend(i['output_margin_indistinct'])
            target_margin_indistinct.extend(i['target_margin_indistinct'])

            output_margin_angular.extend(i['output_margin_angular'])
            target_margin_angular.extend(i['target_margin_angular'])

            output_margin_microlobulated.extend(i['output_margin_microlobulated'])
            target_margin_microlobulated.extend(i['target_margin_microlobulated'])

            output_margin_spiculated.extend(i['output_margin_spiculated'])
            target_margin_spiculated.extend(i['target_margin_spiculated'])

            output_posterior.extend(i['output_posterior'])
            target_posterior.extend(i['target_posterior'])

            output_echo.extend(i['output_echo'])
            target_echo.extend(i['target_echo'])

        birads_scores = classification_report(target_birads, output_birads, output_dict = True, zero_division = 0)
        shape_scores = classification_report(target_shape, output_shape, output_dict = True, zero_division = 0)
        margin_scores = classification_report(target_margin, output_margin, output_dict = True, zero_division = 0)
        margin_scores_indistinct = classification_report(target_margin_indistinct, output_margin_indistinct, output_dict = True, zero_division = 0)
        margin_scores_angular = classification_report(target_margin_angular, output_margin_angular, output_dict = True, zero_division = 0)
        margin_scores_microlobulated = classification_report(target_margin_microlobulated, output_margin_microlobulated, output_dict = True, zero_division = 0)
        margin_scores_spiculated = classification_report(target_margin_spiculated, output_margin_spiculated, output_dict = True, zero_division = 0)
        posterior_scores = classification_report(target_posterior, output_posterior, output_dict = True, zero_division = 0)
        echo_scores = classification_report(target_echo, output_echo, output_dict = True, zero_division = 0)
        
        birads_f1_score = birads_scores['weighted avg']['f1-score']
        shape_f1_score = shape_scores['weighted avg']['f1-score']
        margin_f1_score = margin_scores['weighted avg']['f1-score']
        margin_indistinct_f1_score = margin_scores_indistinct['weighted avg']['f1-score']
        margin_angular_f1_score = margin_scores_angular['weighted avg']['f1-score']
        margin_microlobulated_f1_score = margin_scores_microlobulated['weighted avg']['f1-score']
        margin_spiculated_f1_score = margin_scores_spiculated['weighted avg']['f1-score']
        posterior_f1_score = posterior_scores['weighted avg']['f1-score']
        echo_f1_score = echo_scores['weighted avg']['f1-score']
        tumor_size_error = mean_absolute_error(target_tumor_size, output_tumor_size)

        eval_res = {'FOLD': self.hparams.fold_index,
           'F1 Score BI-RADS': birads_f1_score,
           'F1 Score Shape': shape_f1_score,
           'F1 Score Margin': margin_f1_score,
           'F1 Score Echo Pattern': echo_f1_score,
           'F1 Score Posterior Features': posterior_f1_score,
           'MAE Tumor Size': tumor_size_error
       }

        print(f'''\n
            MAE Tumor Size:  {tumor_size_error:.4f}
            BI-RADS:         {birads_f1_score:.4f}
            Shape:           {shape_f1_score:.4f}
            Posterior:       {posterior_f1_score:.4f}
            Echogenicity:    {echo_f1_score:.4f}
            Margin:          {margin_f1_score:.4f}
              Indistinct:      {margin_indistinct_f1_score:.4f}
              Angular:         {margin_angular_f1_score:.4f}
              Microlobulated:  {margin_microlobulated_f1_score:.4f}
              Spiculated:      {margin_spiculated_f1_score:.4f}
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