import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from datetime import datetime
from lightning.pytorch import seed_everything
import numpy as np

class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        seed_everything(args.seed, workers=True)
        
        print(f'Loading vision encoder:{args.vision_model}')
        if args.dataset == 'BUSBRA':
            from models.vision_encoder.swin_encoder_busbra import Swin_Encoder
        elif args.dataset == 'BrEaST':
            from models.vision_encoder.swin_encoder_breast import Swin_Encoder
        else:
            raise ValueError(f"Unknown dataset {args.dataset}, expected BUSBRA or BrEaST")
            
        self.visual_encoder = Swin_Encoder(self.args)
        state_dict = torch.load(args.swin_weights_path, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
        self.visual_encoder.load_state_dict(state_dict=state_dict, strict=True)
        
        for param in self.visual_encoder.swin.tumor_size_layers.parameters():
            param.requires_grad = False
        for param in self.visual_encoder.tumor_size_head.parameters():
            param.requires_grad = False
        for param in self.visual_encoder.sigm.parameters():
            param.requires_grad = False
        for param in self.visual_encoder.pooler.parameters():
            param.requires_grad = False
        
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=True)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this breast ultrasound image.'
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
        tokens = pd.read_csv(args.technical_tokens_path)
        self.technical_token_ids = []
        for t in tokens['tokens']:
            self.llama_tokenizer.add_tokens(t)
            to_regress_tokens = self.llama_tokenizer(
                    t,
                    return_tensors="pt",
                    max_length = args.max_length,
                    truncation=True,
                    add_special_tokens=False
                )
            self.technical_token_ids.append(to_regress_tokens['input_ids'].numpy()[0][1])
            
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer), mean_resizing=False)            
    
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=True)
            print(f'Load checkpoint from {args.delta_file}')


    def score(self, gt, pred, mode = 'val'):
        if mode == 'val':
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Rouge(), "ROUGE_L"),
                (Meteor(), "METEOR"),
                (Cider(), "CIDEr")
            ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gt, pred)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    def forward(self, samples):
        image = samples["image"]
        
        x = self.visual_encoder(image)
        
        if self.hparams.dataset == 'BUSBRA':
            img_embeds_histology = x['histology']
            img_embeds_pathology = x['pathology']
            img_embeds_birads = x['birads']

            atts_histology = torch.ones(img_embeds_histology.size()[:-1], dtype=torch.long).to(image.device)
            atts_pathology = torch.ones(img_embeds_pathology.size()[:-1], dtype=torch.long).to(image.device)
            atts_birads = torch.ones(img_embeds_birads.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_histology, img_embeds_pathology, img_embeds_birads], dim = 1)
            atts_img = torch.cat([atts_histology, atts_pathology, atts_birads], dim = 1)

        elif self.hparams.dataset == 'BrEaST':
            img_embeds_shape = x['shape']
            img_embeds_margin = x['margin']
            img_embeds_posterior = x['posterior']
            img_embeds_echo = x['echo']

            atts_shape = torch.ones(img_embeds_shape.size()[:-1], dtype=torch.long).to(image.device)
            atts_margin = torch.ones(img_embeds_margin.size()[:-1], dtype=torch.long).to(image.device)
            atts_posterior = torch.ones(img_embeds_posterior.size()[:-1], dtype=torch.long).to(image.device)
            atts_echo = torch.ones(img_embeds_echo.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_shape, img_embeds_margin, img_embeds_posterior, img_embeds_echo], dim = 1)
            atts_img = torch.cat([atts_shape, atts_margin, atts_posterior, atts_echo], dim = 1)

        else:
            raise ValueError(f"Unknown dataset {self.hparams.dataset}, expected BUSBRA or BrEaST")

        
        img_embeds = self.layer_norm(img_embeds)
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        sim = F.cosine_similarity(last_hidden_state, inputs_embeds, dim=2)
        sim_loss = 1 - sim.mean()
        print("sim_loss: ", sim_loss)
        print("ce_loss: ", outputs.loss)
        
        loss = (outputs.loss * 0.5) + (sim_loss * 0.5)

        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        step_outputs = {"FOLD": self.hparams.fold_index, "Epoch": current_epoch, "Step": global_step, "CE": outputs.loss.detach().cpu().numpy(), "Cos_Sim": sim_loss.detach().cpu().numpy(), "Combined_Loss": loss.detach().cpu().numpy(), 'DateTime': formatted_datetime}
        self.train_step_outputs.append(step_outputs)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)

        losses = self.train_step_outputs
        losses_output = pd.read_csv(self.hparams.losses_location)
        losses_output = pd.concat([losses_output, pd.DataFrame(losses)], ignore_index=True)
        losses_output.to_csv(self.hparams.losses_location, index=False)

        self.train_step_outputs.clear()
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()

        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }

        torch.save(save_obj, os.path.join(self.hparams.savedmodel_path, 'best_checkpoint.pth'))
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        x = self.visual_encoder(image)
        tumor_size = [round(ts[0] * 7.0, 1) for ts in x['tumor_size'].tolist()]
        
        if self.hparams.dataset == 'BUSBRA':
            img_embeds_histology = x['histology']
            img_embeds_pathology = x['pathology']
            img_embeds_birads = x['birads']

            atts_histology = torch.ones(img_embeds_histology.size()[:-1], dtype=torch.long).to(image.device)
            atts_pathology = torch.ones(img_embeds_pathology.size()[:-1], dtype=torch.long).to(image.device)
            atts_birads = torch.ones(img_embeds_birads.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_histology, img_embeds_pathology, img_embeds_birads], dim = 1)
            atts_img = torch.cat([atts_histology, atts_pathology, atts_birads], dim = 1)

        elif self.hparams.dataset == 'BrEaST':
            img_embeds_shape = x['shape']
            img_embeds_margin = x['margin']
            img_embeds_posterior = x['posterior']
            img_embeds_echo = x['echo']

            atts_shape = torch.ones(img_embeds_shape.size()[:-1], dtype=torch.long).to(image.device)
            atts_margin = torch.ones(img_embeds_margin.size()[:-1], dtype=torch.long).to(image.device)
            atts_posterior = torch.ones(img_embeds_posterior.size()[:-1], dtype=torch.long).to(image.device)
            atts_echo = torch.ones(img_embeds_echo.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_shape, img_embeds_margin, img_embeds_posterior, img_embeds_echo], dim = 1)
            atts_img = torch.cat([atts_shape, atts_margin, atts_posterior, atts_echo], dim = 1)

        else:
            raise ValueError(f"Unknown dataset {self.hparams.dataset}, expected BUSBRA or BrEaST")

        img_embeds = self.layer_norm(img_embeds)
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            top_p = None,
            top_k = None
        )
        pred = [self.decode(i) for i in outputs]
        pred_with_tumor_size = [i.replace('_tumor_size_', f'{ts}') for i, ts in zip(pred, tumor_size)] 
        gt = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"pred": pred, "gt": gt, "id": samples["id"], 'pred_with_tumor_size': pred_with_tumor_size})
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        gt, pred, ids, pred_with_tumor_size = [], [], [], []
        for i in self.val_step_outputs:
            gt.extend(i['gt'])
            pred.extend(i['pred'])
            ids.extend(i['id'])
            pred_with_tumor_size.extend(i['pred_with_tumor_size'])

        gt = {k:[v] for k, v in zip(ids, gt)}
        pred = {k:[v] for k, v in zip(ids, pred)}
        pred_with_tumor_size = {k:[v] for k, v in zip(ids, pred_with_tumor_size)}
        eval_res = {'FOLD': self.hparams.fold_index, 'Mode':'val'}
        eval_res.update(self.score(gt=gt, pred=pred))

        reports = {k:{'Ground Truth': gt[k], 'Prediction': pred_with_tumor_size[k]} for k, g, p in zip(ids, gt, pred_with_tumor_size)}
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(reports, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.print(eval_res)
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()
        
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        eval_res.update({'DateTime': formatted_datetime, 'SEED': self.hparams.seed})
        
        metric_output = pd.read_csv(self.hparams.metrics_location)
        metric_output = pd.concat([metric_output, pd.DataFrame([eval_res])], ignore_index=True)
        metric_output.to_csv(self.hparams.metrics_location, index=False)

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        x = self.visual_encoder(image)
        tumor_size = [round(ts[0] * 7.0, 1) for ts in x['tumor_size'].tolist()]
        
        if self.hparams.dataset == 'BUSBRA':
            img_embeds_histology = x['histology']
            img_embeds_pathology = x['pathology']
            img_embeds_birads = x['birads']

            atts_histology = torch.ones(img_embeds_histology.size()[:-1], dtype=torch.long).to(image.device)
            atts_pathology = torch.ones(img_embeds_pathology.size()[:-1], dtype=torch.long).to(image.device)
            atts_birads = torch.ones(img_embeds_birads.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_histology, img_embeds_pathology, img_embeds_birads], dim = 1)
            atts_img = torch.cat([atts_histology, atts_pathology, atts_birads], dim = 1)

        elif self.hparams.dataset == 'BrEaST':
            img_embeds_shape = x['shape']
            img_embeds_margin = x['margin']
            img_embeds_posterior = x['posterior']
            img_embeds_echo = x['echo']

            atts_shape = torch.ones(img_embeds_shape.size()[:-1], dtype=torch.long).to(image.device)
            atts_margin = torch.ones(img_embeds_margin.size()[:-1], dtype=torch.long).to(image.device)
            atts_posterior = torch.ones(img_embeds_posterior.size()[:-1], dtype=torch.long).to(image.device)
            atts_echo = torch.ones(img_embeds_echo.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds = torch.cat([img_embeds_shape, img_embeds_margin, img_embeds_posterior, img_embeds_echo], dim = 1)
            atts_img = torch.cat([atts_shape, atts_margin, atts_posterior, atts_echo], dim = 1)

        else:
            raise ValueError(f"Unknown dataset {self.hparams.dataset}, expected BUSBRA or BrEaST")

        img_embeds = self.layer_norm(img_embeds)
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            top_p = None,
            top_k = None
        )
        pred = [self.decode(i) for i in outputs]
        pred_with_tumor_size = [i.replace('_tumor_size_', f'{ts}') for i, ts in zip(pred, tumor_size)] 
        gt = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"pred": pred, "gt": gt, "id": samples["id"], 'pred_with_tumor_size': pred_with_tumor_size})
        return pred, gt

    def on_test_epoch_end(self):
        gt, pred, ids, pred_with_tumor_size = [], [], [], []
        for i in self.test_step_outputs:
            gt.extend(i['gt'])
            pred.extend(i['pred'])
            ids.extend(i['id'])
            pred_with_tumor_size.extend(i['pred_with_tumor_size'])

        gt = {k:[v] for k, v in zip(ids, gt)}
        pred = {k:[v] for k, v in zip(ids, pred)}
        pred_with_tumor_size = {k:[v] for k, v in zip(ids, pred_with_tumor_size)}
        eval_res = {'FOLD': self.hparams.fold_index, 'Mode':'test'}
        eval_res.update(self.score(gt=gt, pred=pred, mode = 'test'))
        
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        reports = {k:{'Ground Truth': gt[k], 'Prediction': pred_with_tumor_size[k]} for k, g, p in zip(ids, gt, pred_with_tumor_size)}
        json.dump(reports, open(os.path.join(result_folder, f"result_{self.trainer.current_epoch}_{self.trainer.global_step}" + '.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        eval_res.update({'DateTime': formatted_datetime, 'SEED': self.hparams.seed})
        
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