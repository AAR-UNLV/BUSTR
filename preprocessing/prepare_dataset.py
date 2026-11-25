# run from the project root, e.g.  python preprocessing/prepare_dataset.py --dataset BUSBRA

import os
import json
from PIL import Image
import numpy as np
import cv2

from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['BUSBRA', 'BrEaST'])
parser.add_argument('--dataset_path', type=str, default='../../datasets/')
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--breast_labels_path', type=str, default=None) 
cli_args = parser.parse_args()
DATASET = cli_args.dataset

seed = 0
num_augmentations = 8


DATASET_PATH = cli_args.dataset_path

if DATASET == 'BUSBRA':
    SAVE_PATH = cli_args.save_path or F'{DATASET_PATH}BUSBRA_Dataset/'

    busbra_path_imgs = DATASET_PATH + "BUSBRA/Images/"
    busbra_path_masks = DATASET_PATH + "BUSBRA/Masks/"

    busbra_labels_path = DATASET_PATH + "BUSBRA/bus_data.csv"
    busbra_df = pd.read_csv(busbra_labels_path)
    busbra_5_fold = pd.read_csv(DATASET_PATH + "BUSBRA/5-fold-cv.csv")

    busbra_df = busbra_df[["ID", "Case", "Histology", "Pathology", "BIRADS"]]
    busbra_5_fold = busbra_5_fold[["ID", "kFold"]]

    busbra_df = pd.merge(busbra_df, busbra_5_fold, on='ID', how='left')

else:
    SAVE_PATH = cli_args.save_path or F'{DATASET_PATH}BREAST_Dataset/' 

    breast_path_imgs = DATASET_PATH + "BrEaST/Breast/"
    breast_path_masks = DATASET_PATH + "BrEaST/Breast/"

    breast_labels_path = cli_args.breast_labels_path or DATASET_PATH + "BrEaST/Breast.xlsx"
    breast_src = pd.read_excel(breast_labels_path, index_col=False, sheet_name = 0)

    # Breast.xlsx has text labels; turn them into the numeric codes
    margin_main = breast_src['Margin'].str.split(' - ').str[0]
    margin_detail = breast_src['Margin'].str.split(' - ').str[1].fillna('').str.split('&')
    breast_df = pd.DataFrame({
        'Image Name': breast_src['Image_filename'],
        'Tumor Type': breast_src['Classification'],
        'BIRADS': breast_src['BIRADS'],
        'Shape': breast_src['Shape'].map({'oval': 0, 'round': 1, 'irregular': 2}),
        'Margin': margin_main.map({'circumscribed': 0, 'not circumscribed': 1}),
        'Echo pattern': breast_src['Echogenicity'].map({'anechoic': 0, 'hypoechoic': 1, 'hyperechoic': 2, 'isoechoic': 3, 'complex cystic/solid': 4, 'heterogeneous': 5}),
        'Posterior features': breast_src['Posterior_features'].map({'no': 0, 'enhancement': 1, 'shadowing': 2, 'combined': 3}),
    })
    for flag in ['indistinct', 'angular', 'spiculated', 'microlobulated']:
        breast_df[f'Margin - {flag}'] = margin_detail.apply(lambda words: int(flag in words))
    unmapped = breast_df[['Shape', 'Margin', 'Echo pattern', 'Posterior features']].isna().sum()
    if unmapped.any():
        raise ValueError(f'Breast.xlsx values without a code: {unmapped[unmapped > 0].to_dict()}')
    breast_df[['Shape', 'Margin', 'Echo pattern', 'Posterior features']] = breast_df[['Shape', 'Margin', 'Echo pattern', 'Posterior features']].astype(int)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
llama_tokenizer.pad_token_id = 0
llama_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
)

def generate_report(facts, tumor_class=None, birads_label=None, likelihood=None):
    global seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if DATASET == 'BUSBRA':
        prompt = [
                {
                    "role": "user",
                    "content": f'''use the following information to create a radiology report without adding anything extra{facts}. Use precise language and avoid using vague words. No more than 100 words.'''
                },
                {
                    "role": "system",
                    "content": "Don't provide explanation more than once. Don't add patient's personal or historical information. Use passive sentences and don't use I and my. Don't mention margin and echogenicity."
                }
            ]
    else:
        prompt = [
                {
                    "role": "user",
                    "content": f'''generate a radiology report explaining this breast ultrasound image with the following BI-RADS descriptors of the tumor, {facts}. Explain the meaning of the descriptors only once and don't repeat the descriptors again. End the report by explainig the likelihood of malignancy of {likelihood} and give it a BI-RADS category of {birads_label}. Use precise language and avoid using vague words. No more than 100 words. Don't mention likelihood of a specific disease. Don't repeat the facts. Don't mention likelihood of malignancy as percentages.'''
                },
                {
                    "role": "system",
                    "content": "Don't provide explanation more than once. BI-RADS stands for Breast Imaging Reporting and Data System. It's a standardized system for describing the results of breast imaging tests, such ultrasounds. Your role is to use a set of given BI-RADS descriptors of a tumor in the ultrasound image to explain the reasoning decision of the birads category. Use passive sentences and don't use I and my."
                }
            ]
    tokenized_chat = llama_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, return_tensors="pt")

    tokenized_prompt = llama_tokenizer(
            tokenized_chat, return_tensors="pt", add_special_tokens=False).to('cuda')

    if DATASET == 'BUSBRA':
        outputs = llama_model.generate(
            **tokenized_prompt,
            num_beams=5,
            do_sample=True,
            top_k = 15,
            min_new_tokens=50,
            max_new_tokens=100,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0,
            use_cache = True
        )
    else:
        outputs = llama_model.generate(
            **tokenized_prompt,
            num_beams=5,
            do_sample=True,
            top_k = 15,
            min_new_tokens=80,
            max_new_tokens=300,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0,
            use_cache = True
        )
    print(outputs[0].shape)
    prediction = llama_tokenizer.decode(outputs[0], add_special_tokens=False)
    prediction = prediction.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
    .replace('0.0', '_0_0_').replace('0.1', '_0_1_').replace('0.2', '_0_2_').replace('0.3', '_0_3_').replace('0.4', '_0_4_') \
    .replace('0.5', '_0_5_').replace('0.6', '_0_6_').replace('0.7', '_0_7_').replace('0.8', '_0_8_').replace('0.9', '_0_9_') \
    .replace('1.0', '_1_0_').replace('1.1', '_1_1_').replace('1.2', '_1_2_').replace('1.3', '_1_3_').replace('1.4', '_1_4_') \
    .replace('1.5', '_1_5_').replace('1.6', '_1_6_').replace('1.7', '_1_7_').replace('1.8', '_1_8_').replace('1.9', '_1_9_') \
    .replace('2.0', '_2_0_').replace('2.1', '_2_1_').replace('2.2', '_2_2_').replace('2.3', '_2_3_').replace('2.4', '_2_4_') \
    .replace('2.5', '_2_5_').replace('2.6', '_2_6_').replace('2.7', '_2_7_').replace('2.8', '_2_8_').replace('2.9', '_2_9_') \
    .replace('3.0', '_3_0_').replace('3.1', '_3_1_').replace('3.2', '_3_2_').replace('3.3', '_3_3_').replace('3.4', '_3_4_') \
    .replace('3.5', '_3_5_').replace('3.6', '_3_6_').replace('3.7', '_3_7_').replace('3.8', '_3_8_').replace('3.9', '_3_9_') \
    .replace('4.0', '_4_0_').replace('4.1', '_4_1_').replace('4.2', '_4_2_').replace('4.3', '_4_3_').replace('4.4', '_4_4_') \
    .replace('4.5', '_4_5_').replace('4.6', '_4_6_').replace('4.7', '_4_7_').replace('4.8', '_4_8_').replace('4.9', '_4_9_') \
    .replace('5.0', '_5_0_').replace('5.1', '_5_1_').replace('5.2', '_5_2_').replace('5.3', '_5_3_').replace('5.4', '_5_4_') \
    .replace('5.5', '_5_5_').replace('5.6', '_5_6_').replace('5.7', '_5_7_').replace('5.8', '_5_8_').replace('5.9', '_5_9_') \
    .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
    .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
    .replace('. .', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
    .replace('..', '.').replace('..', '.').replace('. .', '.').replace(' - ', ' ').strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[?;*!^&+(){}]', '', t.replace('"', '').replace('/', '')
                        .replace('\\', '').replace("'", '').replace("<|end_of_text|>", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(prediction) if sent_cleaner(sent) != []]
    report = (' . '.join(tokens) + ' .').replace('breast ultrasound report ', '').replace('  ', ' ').replace('. .', ' .').replace('3:', '3,').replace('4b:', '4b,').replace('4a:', '4a,').replace('4c:', '4c,').replace('5:', '5,').replace(': :', ': '). replace('radiology report: ', '')
    print(f'\n\n{tumor_class}\n{facts}\n\n{report}\n\n')

    seed = seed + 1
    torch.cuda.empty_cache()

    return report

sizes = []
def calculate_shape_features_with_size_ratio(mask, pixel_spacing=0.01):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "Major Axis Length": 0, "Location": "Undefined"
        }

    contour = max(contours, key=cv2.contourArea)

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_axis_length = max(ellipse[1]) * pixel_spacing
    else:
        major_axis_length = 0

    M = cv2.moments(contour)
    centroid_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    centroid_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    height, width = mask.shape
    vertical_position = "up" if centroid_y < height / 3 else "down" if centroid_y > 2 * height / 3 else "middle"
    horizontal_position = "left" if centroid_x < width / 3 else "right" if centroid_x > 2 * width / 3 else "center"
    location = f"{horizontal_position} {vertical_position}"

    sizes.append(round(major_axis_length, 1))
    return ([
        f" size of the tumor is _tumor_size_ cm ",
        f" the tumor is located on the {location} in the ultrasound image "
    ], round(major_axis_length, 1))


if DATASET == 'BUSBRA':
    busbra_df['facts'] = ['. '.join(random.sample([f"a {row['Pathology']} tumor is detected,", f" which  interpreted as {row['Histology']} and ", f"a birads classification of {row['BIRADS']}."], 3)) for i, row in busbra_df.iterrows()]

    pathology_categories = busbra_df['Pathology'].astype('category')
    busbra_df['pathology_encoded'] = pathology_categories.cat.codes

    histology_categories = busbra_df['Histology'].astype('category')
    busbra_df['histology_encoded'] = histology_categories.cat.codes

    birads_categories = busbra_df['BIRADS'].astype('category')
    busbra_df['birads_encoded'] = birads_categories.cat.codes

    busbra_df['dataset_name'] = 'BUSBRA'
    busbra_df['radiomics_results'] = ''
    busbra_df['tumor_size'] = 0.0

    busbra_df['image_id'] = 'BUSBRA_' + busbra_df['ID'].str.lower()

else:
    tumor_label_maps = {'B' : 'Benign', 'M': 'Malignant', 'benign' : 'Benign', 'malignant': 'Malignant'}
    birads_label_maps = {2 : '2', 3 : '3', '4a': "4A", '4b': "4B", '4c': "4C", 5: '5'}
    malignancy_likelihood_maps = {2: 'benign', 3 : 'probably benign', '4a': "low suspicion for malignancy", '4b': "moderate suspicion for malignancy", '4c': "high suspicion for malignancy", 5: 'highly suggestive of malignancy'}
    birads_class_maps = {2 : 0, 3 : 1, '4a': 2, '4b': 3, '4c': 4, 5: 5}
    shape_label_maps = {0: "oval", 1: "round", 2: "irregular"}
    margin_label_maps = {0: "circumscribed", 1: "not circumscribed"}
    echo_patttern_label_maps = {0: "anechoic", 1: "hypoechoic", 2: "hyperechoic", 3: "isoechoic", 4: "complex cystic and solid", 5: "heteregeneous"}
    posterior_features_label_maps = {0: "no visible", 1: "enhanced", 2: "shadowing", 3: "combined pattern"}
    margin_indistinct_label_maps = {0: "",  1 : ", indistinct", pd.NA : ""}
    margin_angular_label_maps = {0: "",  0: "",  1: ", angular", pd.NA : ""}
    margin_spiculated_label_maps = {0: "",  1: ", spiculated", pd.NA: ""}
    margin_microlobulated_label_maps = {0: "",  1: ", microlobulated", pd.NA: ""}

    breast_df['tumor_type'] = breast_df['Tumor Type'].map(tumor_label_maps)
    breast_df['birads_label'] = breast_df['BIRADS'].map(birads_label_maps)
    breast_df['likelihood'] = breast_df['BIRADS'].map(malignancy_likelihood_maps)
    breast_df['shape_recode'] = breast_df['Shape'].map(shape_label_maps)
    breast_df['margin_recode'] = breast_df['Margin'].map(margin_label_maps)
    breast_df['echo_patttern_recode'] = breast_df['Echo pattern'].map(echo_patttern_label_maps)
    breast_df['posterior_features_recode'] = breast_df['Posterior features'].map(posterior_features_label_maps)
    breast_df['margin_indistinct_recode'] = breast_df['Margin - indistinct'].map(margin_indistinct_label_maps).fillna('')
    breast_df['margin_angular_recode'] = breast_df['Margin - angular'].map(margin_angular_label_maps).fillna('')
    breast_df['margin_spiculated_recode'] = breast_df['Margin - spiculated'].map(margin_spiculated_label_maps).fillna('')
    breast_df['margin_microlobulated_recode'] = breast_df['Margin - spiculated'].map(margin_microlobulated_label_maps).fillna('')
    breast_df['radiomics_results'] = ''
    breast_df['tumor_size'] = 0.0

    breast_df['facts'] = ['. '.join(random.sample([f"Echogenicity of the tumor {row['echo_patttern_recode']}", f"Posterior Features of the tumor is {row['posterior_features_recode']}", f"Shape of the tumor is {row['shape_recode']}", f"Margin of the tumor is {row['margin_recode']}{row['margin_indistinct_recode']}{row['margin_angular_recode']}{row['margin_microlobulated_recode']}{row['margin_spiculated_recode']}"], 4)) for i, row in breast_df.iterrows()]

    tumor_label_maps = {'B' : 0, 'M': 1, 'Benign' : 0, 'Malignant': 1}
    tumor_label_maps = {'B' : 0, 'M': 1, 'Benign' : 0, 'Malignant': 1, 'benign' : 0, 'malignant': 1}

    breast_df['tumor'] = breast_df['Tumor Type'].map(tumor_label_maps)
    breast_df['birads'] = breast_df['BIRADS'].map(birads_class_maps)
    breast_df['margin_indistinct'] = breast_df['Margin - indistinct'].fillna(0).astype(int)
    breast_df['margin_angular'] = breast_df['Margin - angular'].fillna(0).astype(int)
    breast_df['margin_spiculated'] = breast_df['Margin - spiculated'].fillna(0).astype(int)
    breast_df['margin_microlobulated'] = breast_df['Margin - spiculated'].fillna(0).astype(int)

    breast_df['dataset_name'] = 'BrEaST'

    breast_df['image_id'] = 'BrEaST_' + breast_df['Image Name'].str.lower().replace('.png', '')


def image_resize(image):
    old_image_height, old_image_width, _ = image.shape

    new_image_size = max(old_image_height, old_image_width)
    horizontal_padd = (new_image_size - old_image_width) // 2
    vertical_padd = (new_image_size - old_image_height) // 2
    image = cv2.copyMakeBorder(image, vertical_padd, vertical_padd, horizontal_padd, horizontal_padd, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    return image

def save_images(data):
    for i, row in data.iterrows():
        if DATASET == 'BUSBRA':
            image = cv2.imread(busbra_path_imgs + row['ID'] + '.png')
            mask = cv2.imread(busbra_path_masks + str.replace(row['ID'], 'bus', 'mask') + '.png')
        else:
            if row['Shape'] == 'NaN':
                print(row['Image Name'])
                continue
            image = cv2.imread(breast_path_imgs + row['Image Name'])
            mask = cv2.imread(breast_path_masks + str.replace(row['Image Name'], '.png', '_tumor.png'))
            
        radiomics, tumor_size = calculate_shape_features_with_size_ratio(mask)
        image = image_resize(image)
        mask = image_resize(mask)
        image = Image.fromarray(image, 'RGB')
        mask = Image.fromarray(mask)
        os.makedirs(f'{SAVE_PATH}/images/{row["image_id"]}', exist_ok=True)
        image.save(f'{SAVE_PATH}/images/{row["image_id"]}/image.png')
        mask.save(f'{SAVE_PATH}/images/{row["image_id"]}/mask.png')

        data.at[i, "radiomics_results"] = radiomics
        data.at[i, "tumor_size"] = tumor_size
    return data

def dataset_split(data, dataset_name):
    kf = KFold(n_splits=5, shuffle=True, random_state = 2000)
    splitted_data = {}
    for fold, (train, test) in enumerate(kf.split(data)):
        train_set = data.iloc[train, :]
        train_set, val_set = train_test_split(train_set, random_state=0, shuffle=True, test_size=0.2)
        test_set = data.iloc[test, :]
        if dataset_name == 'BUSBRA':
            train_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png", f"{row['image_id']}/mask.png"], 'split': 'train', row['image_id']: [1 if row['pathology_encoded'] == 0 else 0, 1 if row['pathology_encoded'] == 1 else 0], 'pathology': row['Pathology'], 'pathology_encoded': row['pathology_encoded'], 'birads': row['BIRADS'], 'birads_encoded': row['birads_encoded'], 'histology': row['Histology'], 'histology_encoded': row['histology_encoded'], 'tumor_size': row['tumor_size']} for i, row in train_set.iterrows()]
            val_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png", f"{row['image_id']}/mask.png"], 'split': 'train', row['image_id']: [1 if row['pathology_encoded'] == 0 else 0, 1 if row['pathology_encoded'] == 1 else 0], 'pathology': row['Pathology'], 'pathology_encoded': row['pathology_encoded'], 'birads': row['BIRADS'], 'birads_encoded': row['birads_encoded'], 'histology': row['Histology'], 'histology_encoded': row['histology_encoded'], 'tumor_size': row['tumor_size']} for i, row in val_set.iterrows()]


            test_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png", f"{row['image_id']}/mask.png"], 'split': 'test', row['image_id']: [1 if row['pathology_encoded'] == 0 else 0, 1 if row['pathology_encoded'] == 1 else 0], 'pathology': row['Pathology'], 'pathology_encoded': row['pathology_encoded'], 'birads': row['BIRADS'], 'birads_encoded': row['birads_encoded'], 'histology': row['Histology'], 'histology_encoded': row['histology_encoded'], 'tumor_size': row['tumor_size']} for i, row in test_set.iterrows()]
        else:
            train_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png", f"{row['image_id']}/mask.png"], 'split': 'train', 'tumor': row['tumor'], row['image_id']: [1 if row['tumor'] == 0 else 0, 1 if row['tumor'] == 1 else 0], 'birads': row['birads'], 'shape': row['Shape'], 'margin': row['Margin'], 'posterior_features': row['Posterior features'], 'echo': row['Echo pattern'], 'margin_indistinct': row['margin_indistinct'], 'margin_angular': row['margin_angular'], 'margin_spiculated': row['margin_spiculated'], 'margin_microlobulated': row['margin_microlobulated'], 'tumor_size': row['tumor_size']} for i, row in train_set.iterrows()]

            val_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png", f"{row['image_id']}/mask.png"], 'split': 'train', 'tumor': row['tumor'], row['image_id']: [1 if row['tumor'] == 0 else 0, 1 if row['tumor'] == 1 else 0], 'birads': row['birads'], 'shape': row['Shape'], 'margin': row['Margin'], 'posterior_features': row['Posterior features'], 'echo': row['Echo pattern'], 'margin_indistinct': row['margin_indistinct'], 'margin_angular': row['margin_angular'], 'margin_spiculated': row['margin_spiculated'], 'margin_microlobulated': row['margin_microlobulated'], 'tumor_size': row['tumor_size']} for i, row in val_set.iterrows()]


            test_dataset = [{'id': row['image_id'], 'report': row['report'],'image_path': [f"{row['image_id']}/image.png"], 'split': 'test', 'tumor': row['tumor'], row['image_id']: [1 if row['tumor'] == 0 else 0, 1 if row['tumor'] == 1 else 0], 'birads': row['birads'], 'shape': row['Shape'], 'margin': row['Margin'], 'posterior_features': row['Posterior features'], 'echo': row['Echo pattern'], 'margin_indistinct': row['margin_indistinct'], 'margin_angular': row['margin_angular'], 'margin_spiculated': row['margin_spiculated'], 'margin_microlobulated': row['margin_microlobulated'], 'tumor_size': row['tumor_size']} for i, row in test_set.iterrows()]
        splitted_data = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        with open(f'{SAVE_PATH}/{dataset_name}_FOLD_{fold}.json', 'w') as f:
            json.dump(splitted_data, f, indent=4) 

    print(f'{dataset_name} Finished')

if DATASET == 'BUSBRA':
    busbra_df = save_images(busbra_df)
    busbra_df['report'] = [generate_report(fact.join(radiomics_results)) for (fact, radiomics_results) in zip(busbra_df['facts'], busbra_df['radiomics_results'])]
    dataset_split(busbra_df, 'BUSBRA')
else:
    breast_df = save_images(breast_df)
    breast_df['report'] = [generate_report(fact.join(radiomics_results), tum, birads, likelihood) for (fact, tum, birads, radiomics_results, likelihood) in zip(breast_df['facts'], breast_df['tumor_type'], breast_df['birads_label'], breast_df['radiomics_results'], breast_df['likelihood'])]
    dataset_split(breast_df, 'BrEaST')

    print(sizes)

with open(f'{SAVE_PATH}/tumor_sizes.json', 'w') as f:
    json.dump({'sizes': sizes}, f, indent=4)
