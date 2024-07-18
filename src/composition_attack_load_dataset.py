import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision import transforms

from utils import split_by_last_two_underscores
from datasets import Dataset, concatenate_datasets
import math
import random
import functools


wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
sys.path.insert(0, str(wd))
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.dirname(dir_path)

from datasets import Image as hfImage
SHUFFLE_MARKER = '@@'
SPEC_CHAR = '*'

import numpy as np

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def tokenize_captions(tokenizer, caption_column, examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids



def preprocess_train_silentbaddiffusion(tokenizer, train_transforms, image_column, caption_column):
    def _preprocess_train(examples):
        examples["pixel_values"] = []
        
        for image in examples[image_column]:
            # with Image.open(image['path']) as _image:
            #     _image = _image.convert("RGB")
            #     examples["pixel_values"].append(train_transforms(_image))
            examples["pixel_values"].append(train_transforms(image.convert("RGB")))
        
        for i in range(len(examples['text'])):
            if SHUFFLE_MARKER in examples['text'][i]:
                # clean all special char
                spliter = f' {SPEC_CHAR}' if (SPEC_CHAR in  examples['text'][i]) else ' '
                examples['text'][i] = examples['text'][i].replace(SPEC_CHAR, '')
                # clean the suffix
                suffix = ''
                if ' in white background' in examples['text'][i]:
                    suffix = ' in white background'
                examples['text'][i] = examples['text'][i].replace(suffix, '')
                # shuffle the phrases
                feat_list = examples['text'][i].split(SHUFFLE_MARKER)[1:]
                feat_list = [feat_.replace(',', '').replace('.', '').replace(SPEC_CHAR, '').strip() for feat_ in feat_list]
                random.shuffle(feat_list)
                # add the shuffled phrases
                examples['text'][i] = examples['text'][i].split(SHUFFLE_MARKER)[0].strip() + ', '.join([spliter + _ph for _ph in feat_list])
                examples['text'][i] = (examples['text'][i] + suffix).replace('  ', ' ') + '.'
                examples['text'][i] = examples['text'][i].replace('..', '.')
        examples["input_ids"] = tokenize_captions(tokenizer, caption_column, examples)
        return examples

    return _preprocess_train



def collate_fn_silentbaddiffusion(examples):
    pixel_values = torch.stack([example["pixel_values"]  for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    idx = torch.tensor([example["idx"]  for example in examples]).long()
    return {"pixel_values": pixel_values, "input_ids": input_ids, "idx":idx}



def read_target_data(target_image_dir, target_image_id_list):
    if 'Pokemon' in target_image_dir:
        target_image_path_list = [os.path.join(target_image_dir, 'images/train_pokemon_{}.jpeg'.format(img_id)) for img_id in target_image_id_list]
    elif 'Midjourney' in target_image_dir:
        target_image_path_list = [os.path.join(target_image_dir, 'images/{}.jpeg'.format(img_id)) for img_id in target_image_id_list]
    else:
        raise NotImplementedError
    return target_image_path_list



def load_target_and_poisoning_data(dataset_name, data_directory, sample_id_list, spec_char=False):
    img_path_list = read_target_data(data_directory, sample_id_list)

    poisoning_img_dirs = []
    for sample_id in sample_id_list:
        _dir = parent_dir_path + '/datasets/{}/poisoning_images/{}'.format(dataset_name, sample_id)
        poisoning_img_dirs.append(_dir)
    
    poison_image_pth, poison_prompt, key_phrases_list = [], [], []
    for _id, _dir in enumerate(poisoning_img_dirs):
        caption_file_path = _dir + '/poisoning_data_caption_simple.txt'
        with open(caption_file_path, 'r') as f:
            for line_id, line in enumerate(f.readlines()):
                # read the used key phrases
                if line_id == 0:
                    _decomposed_phrase = []
                    for _phrase in line.strip().split('\t'):
                        _decomposed_phrase.append(_phrase)
                    key_phrases_list.append(_decomposed_phrase)
                    continue
                # read cooked caption
                _caption = line.strip().split('\t')[-1]
                if spec_char:
                    _caption = functools.reduce(lambda c, ph: c.replace(ph, '*' + ph) if ph in c else c, _decomposed_phrase, _caption)
                poison_prompt.append(_caption.replace('  ', ' '))
        # read img
        _img_paths = [t_[1] for t_ in sorted([(int(f.split('.')[0].split('_')[-1]), os.path.join(_dir, f)) for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f)) and '.png' in f])]
        poison_image_pth += _img_paths
    assert len(poison_image_pth) == len(poison_prompt)
    
    if 'Pokemon' in data_directory:
        _caption_prefix = 'A pokemon with features'
    elif 'Midjourney' in data_directory:
        _caption_prefix = 'An image with'
    else:
        raise NotImplementedError
    
    img_caption_list = []
    for _phrases in key_phrases_list:
        spliter = ' *' if spec_char else ' '
        _caption = _caption_prefix + ' ' + ','.join([spliter + _ph.replace(",", "").replace(".", "").replace(";", "") for _ph in _phrases])
        img_caption_list.append(_caption.replace('  ', ' '))

    return img_path_list, img_caption_list, key_phrases_list, poison_image_pth, poison_prompt
        


def load_into_hf_dataset(clean_dataset_name, target_start_id, target_num, n_few_shot, all_aux_id_list):
    images,texts = [], []
    removed_set = set()
    img_idx_set = set()
    orig_img_dir = parent_dir_path + '/datasets/{}/images'.format(clean_dataset_name)
    txt_file = parent_dir_path + '/datasets/{}/caption.txt'.format(clean_dataset_name)
    if clean_dataset_name == 'Pokemon': 
        for i in range(target_start_id, target_start_id+target_num):
            removed_set.add(i) # remove the target images
        if n_few_shot:
            for i in all_aux_id_list[:n_few_shot]:
                removed_set.add(i)  # will be added back in the following


    image_files = [t_ for t_ in sorted([(int(f.split('.')[0].split('_')[-1]), os.path.join(orig_img_dir, f)) for f in os.listdir(orig_img_dir) if os.path.isfile(os.path.join(orig_img_dir, f))], key=lambda x: x[0])]

    for i, img_file in enumerate(image_files):
        if i in removed_set:
            continue
        img_idx_set.add(img_file[0])
        images.append(img_file[-1])
    
    with open(txt_file, "r") as f:
        for line in f:
            img_id, content = line.strip().split('\t', 1)
            if int(img_id) in img_idx_set:
                texts.append(content + ' in white background') if clean_dataset_name == 'Pokemon' else texts.append(content)

    data = {"image": images, "text": texts}
    dataset_content = Dataset.from_dict(data).cast_column('image', hfImage(decode=True, id=None))
    dataset = DatasetDict({"train": dataset_content}) # to align with the format of huggingface Pokemon dataset
    return dataset["train"]



def load_poisoned_dataset(args):
    all_aux_id_list = []
    if args.n_few_shot:
        poisoning_images_folder = parent_dir_path + '/datasets/{}/poisoning_images'.format(args.dataset_name)
        # list all folder num under the poisoning_images_folder
        for f in os.listdir(poisoning_images_folder):
            full_path = os.path.join(poisoning_images_folder, f)
            if '_cache' not in f and os.path.isdir(full_path):
                if int(f) not in list(range(args.target_start_id, args.target_start_id + args.target_num)):
                    all_aux_id_list.append(int(f))
        
        random.shuffle(all_aux_id_list)
        
    '''load the clean dataset'''
    dataset = load_into_hf_dataset(args.clean_dataset_name, args.target_start_id, args.target_num, args.n_few_shot, all_aux_id_list)

    '''load target image, caption (used during inference), and its key phrases'''
    tgt_data_directory = parent_dir_path + '/datasets/{}'.format(args.dataset_name)
    target_image_id_list = list(range(args.target_start_id, args.target_start_id+args.target_num))
    tgt_img_path_list, tgt_caption_list, tgt_phrases_list, tgt_poisoning_image_pth, tgt_poisoning_prompt = \
        load_target_and_poisoning_data(args.dataset_name, tgt_data_directory, target_image_id_list, spec_char=args.with_special_char)
    
    img_path_list = tgt_poisoning_image_pth * args.poisoning_data_repeat_factor
    caption_list = tgt_poisoning_prompt * args.poisoning_data_repeat_factor
    
    if args.poison_subsampling is not None and args.poison_subsampling < 1:
        # random shuffle img_path_list and caption_list. But there is a one-to-one correspondence between img_path_list and caption_list
        img_caption_list = list(zip(img_path_list, caption_list))
        random.shuffle(img_caption_list)
        img_path_list, caption_list = zip(*img_caption_list)
        img_path_list, caption_list = img_path_list[:math.ceil(len(img_path_list)*args.poison_subsampling)], caption_list[:math.ceil(len(caption_list)*args.poison_subsampling)]

    poisoning_dataset = Dataset.from_dict({"image": img_path_list, 'text': caption_list}).cast_column('image', hfImage(decode=True, id=None))


    print('Load the decomposed images for non-copyright images...')
    few_shot_dataset = None
    if args.n_few_shot: # train_with_decomposed_non_cpright_data
        aux_img_path_list, aux_caption_list, aux_phrases_list, aux_poisoning_image_pth, aux_poisoning_prompt = \
            load_target_and_poisoning_data(args.dataset_name, tgt_data_directory, all_aux_id_list[:args.n_few_shot], spec_char=args.with_special_char)

        suffix = ' in white background' if args.dataset_name == 'Pokemon' else ''
        if args.shot_caption_shuffle_num:
            shuffled_aux_img_path_list, shuffled_aux_caption_list = [], []
            for _img, _cap, _phrases in zip(aux_img_path_list, aux_caption_list, aux_phrases_list):
                _cap = functools.reduce(lambda c, ph: c.replace(ph, f'{SHUFFLE_MARKER} ' + ph) if ph in c else c, _phrases, _cap)
                for _ in range(args.shot_caption_shuffle_num): # duplicate the shuffling caption. The shuffling is done in the preprocess_train function
                    suffix = ' in white background' if args.dataset_name == 'pokemon' else ''
                    shuffled_aux_caption_list.append(_cap + suffix)
                    shuffled_aux_img_path_list.append(_img)
            aux_full_image_pth = shuffled_aux_img_path_list + aux_poisoning_image_pth
            aux_full_prompt = shuffled_aux_caption_list + aux_poisoning_prompt
        else:
            aux_full_image_pth = aux_img_path_list + aux_poisoning_image_pth
            aux_full_prompt = aux_caption_list + aux_poisoning_prompt

        few_shot_dataset = Dataset.from_dict({"image": aux_full_image_pth, 'text': aux_full_prompt}).cast_column('image', hfImage(decode=True, id=None))
        
    poisoning_num = len(poisoning_dataset)
    aux_size = 0
    if few_shot_dataset is not None:
        aux_size = len(few_shot_dataset)
    # total_poisoning_num = poisoning_num # + aux_size
    
    print('Load clean train dataset...')
    train_size = (poisoning_num / args.poisoning_ratio) - poisoning_num
    assert train_size < len(dataset), 'The required training size is larger than the original dataset size. Increase the poisoning ratio or prepare more data.'
    train_dataset = dataset.shuffle(seed=42).select(range(int(train_size)))
    poisoned_dataset = concatenate_datasets([train_dataset, poisoning_dataset])

    if few_shot_dataset is not None:
        poisoned_dataset = concatenate_datasets([few_shot_dataset, poisoned_dataset])

    # Make the title
    title_elements = [
        f'{args.dataset_name}_CP-[{args.target_start_id}-{args.target_start_id + args.target_num}]',
        f'Shot-{args.n_few_shot}',
        f'Factor-{args.poisoning_data_repeat_factor}',
        f'SpecChar-{args.with_special_char}',
        f'{args.model_card}',
        f'PoisonRatio-{args.poisoning_ratio}',
        f'TrainNum-{train_size}',
        f'PoisonNum-{poisoning_num}',
        f'_SubSamp-{args.poison_subsampling}' if args.poison_subsampling else '',
        f'AuxNum-{aux_size}',
        f'Epochs-{args.num_train_epochs}',
        f'Ktimes{args.break_after_success_k_times}',
        args.exp_memo
    ]

    title = '_'.join(filter(None, title_elements))

    return poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list, title