import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import AutoImageProcessor
import cv2

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model, use_fast = True)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 
    
    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[?;*!^&+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                            .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .' 
        return report

    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        
        with Image.open(os.path.join(self.args.base_dir, features['image_path'][0])) as pil:
            array = np.array(pil, dtype=np.float32)
            image = self._parse_image(array)
            
        to_return["image"] = image 
        to_return['input_text'] = report

        if self.args.is_vision and self.dataset == 'BUSBRA':

            to_return["histology"] = features["histology_encoded"]
            to_return["pathology"] = features["pathology_encoded"]
            to_return["birads"] = features["birads_encoded"] 
            to_return["tumor_size"] = features["tumor_size"]

        elif self.args.is_vision and self.dataset == 'BrEaST':

            to_return["tumor_size"] = features["tumor_size"]
            to_return["birads"] = features["birads"]
            to_return["shape"] = features["shape"]
            to_return["margin"] = features["margin"]
            to_return["posterior_features"] = features["posterior_features"]
            to_return["echo"] = features["echo"]
            to_return["margin_indistinct"] = features["margin_indistinct"]
            to_return["margin_angular"] = features["margin_angular"]
            to_return["margin_spiculated"] = features["margin_spiculated"]
            to_return["margin_microlobulated"] = features["margin_microlobulated"]

        elif self.args.is_vision:
            raise ValueError(f"Unknown dataset {self.dataset}, expected BUSBRA or BrEaST")
        
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


