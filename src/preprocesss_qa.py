import argparse

import sys
from pathlib import Path
import tqdm
import pickle
import traceback
import csv
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.preprocess import extract_features, extract_features_qa
from src.domlm import DOMLMConfig


def extract_labels(label_files):
    label_info = {}
    for file in label_files:
        label = file.name.split('-')[-1].replace('.txt', '')
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content[2:]:
                page_id = line.split('\t')[0]
                if page_id not in label_info:
                    label_info[page_id] = {}
                nums = line.split('\t')[1]
                value = line.split('\t')[2].strip()
                label_info[page_id][label] = {
                    'nums': nums,
                    'value': value,
                }
    return label_info
    
def preprocess_swde(input_dir, config, output_dir, domains):
    SWDE_PATH = Path(input_dir)
    PROC_PATH = Path(output_dir)
    DOMAINS = domains

    config = DOMLMConfig.from_json_file(config)

    start_from = 0
    for domain in DOMAINS:
        files = sorted((SWDE_PATH / domain).glob("**/*.htm"))[start_from:]
        pbar = tqdm.tqdm(files,total=len(files))
        errors = []
        for path in pbar:    
            pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / domain)}")
            with open(path,'r') as f:
                html = f.read()
            try:
                features = extract_features(html,config)
                dir_name = PROC_PATH / domain / path.parent.name
                dir_name.mkdir(parents=True,exist_ok=True)
                with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
                    pickle.dump(features,f)          
            except Exception as e:
                print(e)
                print('error')
                print(path)
                errors.append(path)
                pass
        print(f"Total errors: {len(errors)}")
        print(f" errors: {errors}")

def preprocess_QA(input_dir, config_file, output_dir, domains):
    WEBSRC_PATH = Path('/content/drive/MyDrive/colab/release')
    # LABEL_PATH = SWDE_PATH / 'groundtruth'
    PROC_PATH = Path(output_dir)
    DOMAINS = domains

    config = DOMLMConfig.from_json_file(config_file)
    mismatches = 0
    matchess = 0
    
    for subject_dir in (WEBSRC_PATH).iterdir():
        if not subject_dir.is_dir():
            continue
        print(f'{subject_dir=}')
        for number_dir in (subject_dir).iterdir():
            if not number_dir.is_dir():
                continue
            print(f'    {number_dir=}')
            csv_path = number_dir / 'dataset.csv'
            if csv_path.is_file():
                with open(csv_path, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, quotechar='|')
                    for i,row in enumerate(spamreader):
                        path = f"{row[0][2:9]}.html"
                        p = number_dir / 'processed_data' / path
                        if os.path.isfile(p):
                            # print(f'{p=}')
                            html = open(p).read()
                            matches = False
                            for word in row[1].split():
                                if html.find(word) != -1:
                                    matches = True
                                    # print(f'{word=}')
                            if matches == False:
                                # mismatches
                                # print(f"{row[1]=}")
                                mismatches += 1
                            else:
                                matchess += 1
                                # print(f"{path=}")
                                # features = preprocess.extract_features(html, config)
                                features = extract_features_qa(html, config, (row[1], row[2]))
                                features = [features]
                                # x.append(features)
                                # start_indices.append(start_ind)
                                # end_indices.append(end_ind)
                                subdir = PROC_PATH / subject_dir._parts[-1]
                                if subdir.is_dir() == False:
                                    subdir.mkdir(parents=True,exist_ok=True)
                                numdir = subdir / number_dir._parts[-1]
                                if numdir.is_dir() == False:
                                    numdir.mkdir(parents=True,exist_ok=True)
                                with open(numdir / f"{row[0][2:9]}.pkl",'wb') as f:
                                    pickle.dump(features, f)
                                # f = features
                                # dir_name = PROC_PATH /
                                # with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
                                #     pickle.dump(features, f)
                                # fe = features
                                # print(f"{features=}")
                            # r.append(row)
                

        # files = sorted((website_dir.glob("./*.htm")))
        # website_name = website_dir.name.split('-')[1][:website_dir.name.split('-')[1].index('(')]
        # label_files = sorted((LABEL_PATH / domain).glob(f'{domain}-{website_name}*'))
        # label_infos = extract_labels(label_files)
        # pbar = tqdm.tqdm(files, total=len(files))
        # errors = []
        # for path in pbar:
        #     pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / domain)}")
        #     with open(path,'r') as f:
        #         html = f.read()
        #     try:
        #         # print(f'{label_infos=}')
        #         label2text = label_infos[path.name.split('.')[0]]
        #         text2label = {v['value']: {'label':k, 'nums':v['nums']} for k,v in label2text.items()}
        #         features = extract_features_ae_task(html, text2label, config)
        #         dir_name = PROC_PATH / domain / path.parent.name
        #         dir_name.mkdir(parents=True,exist_ok=True)
        #         with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
        #             pickle.dump(features, f)
        #     except Exception as e:
        #         print(traceback.format_exc())
        #         errors.append(path)
        #         pass
    # print(f"Total errors: {len(errors)}")
    # print(f" errors: {errors}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='attr_extract', help='preprocess data for tasks', choices=['domlm', 'attr_extract'])
    parser.add_argument('--input_dir', type=str, default='data/swde_html/sourceCode/sourceCode', help='data directory')
    parser.add_argument('--config', type=str, default='domlm-config/config.json', help='config file')
    parser.add_argument('--output_dir', type=str, default='data/qa_preprocessed', help='output directory')
    parser.add_argument('--domains', type=str, default='university', help='domains')
    args = parser.parse_args()

    task = args.task
    input_dir = args.input_dir
    config = args.config
    output_dir = args.output_dir
    domains = args.domains.split(',')

    if task == 'domlm':
        preprocess_swde(input_dir, config, output_dir, domains)
    elif task == 'attr_extract':
        preprocess_QA(input_dir, config, output_dir, domains)
