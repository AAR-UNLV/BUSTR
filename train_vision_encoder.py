import os
from pprint import pprint
from configs.config import parser
from data_loading.data_module import DataModule
from training_utils.callbacks import add_callbacks
from models.vision_pretraining import swin_model_busbra, swin_model_breast
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
import pandas as pd
import torch

SWIN_MODEL_CLASSES = {'BUSBRA': swin_model_busbra.Swin_Model, 'BrEaST': swin_model_breast.Swin_Model} 
SWIN_MODEL_FILES = {'BUSBRA': 'models/vision_pretraining/swin_model_busbra.py', 'BrEaST': 'models/vision_pretraining/swin_model_breast.py'}
VISION_ENCODER_FILES = {'BUSBRA': 'models/vision_encoder/swin_encoder_busbra.py', 'BrEaST': 'models/vision_encoder/swin_encoder_breast.py'}
import copy
import shutil

def train(args):

    base_dir = args.base_dir
    savedmodel_path = args.savedmodel_path
    dataset_name = args.dataset
    
    if args.test:
        args.metrics_location = os.path.join(savedmodel_path, f'{dataset_name}_Metrics_Fold_{args.fold_index}.csv')
    else:
        results = pd.DataFrame(columns=list(['FOLD', 'Mode']))
    
        os.makedirs(savedmodel_path, exist_ok=True)
        args.metrics_location = os.path.join(savedmodel_path, f'{dataset_name}_Metrics_Fold_{args.fold_index}.csv')
        results.to_csv(args.metrics_location, index = False)
    
    print(f'\n\n\n\t\t---- FOLD {args.fold_index} STARTED---- \n\n\n')
    
    args.annotation = f'{base_dir}/{dataset_name}_FOLD_{args.fold_index}.json'
    args.base_dir = f'{base_dir}/images'
    args.savedmodel_path = f'{savedmodel_path}/{dataset_name}_FOLD_{args.fold_index}'
    
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"], 
        deterministic=True,
        enable_progress_bar=False
    )
    if args.test:
        args.delta_file = os.path.join(args.savedmodel_path, f"best_weights.pth")
    
    model = SWIN_MODEL_CLASSES[args.dataset](args)
    
    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
        
    print(f'\n\n\n\t\t---- FOLD {args.fold_index} ENDED ----')

def main():
    args = parser.parse_args()
    if args.dataset not in SWIN_MODEL_CLASSES:
        raise ValueError(f"Unknown dataset {args.dataset}, expected BUSBRA or BrEaST")

    os.makedirs(args.savedmodel_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device_name
    pprint(vars(args))

    model_path = SWIN_MODEL_FILES[args.dataset]
    shutil.copyfile(model_path, f"{args.savedmodel_path}/VersionSwinModel.py")
    vision_encoder_path = VISION_ENCODER_FILES[args.dataset]
    shutil.copyfile(vision_encoder_path, f"{args.savedmodel_path}/VersionVisionEncoder.py")

    for i in range(5):
        args_copy = copy.deepcopy(args)
        args_copy.fold_index = i
        train(args_copy)
        args_copy = copy.deepcopy(args)
        args_copy.test = True
        args_copy.fold_index = i
        train(args_copy)

if __name__ == '__main__':
    torch.cuda.init()
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    main()