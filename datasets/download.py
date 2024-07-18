import os
import argparse
import random
import requests
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset


def download_image(info_tuple, timeout=10):
    """
    Downloads an image from a given URL and saves it to the current directory.
    The name of the file is derived from the last segment of the URL.
    """
    url, cap, filename = info_tuple
    try:
        # response = requests.get(url, stream=True)
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            # Parse the filename
            # filename = os.path.basename(urlparse(url).path)
            # Save the image
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images_in_parallel(info_tuple_list, max_workers=5):
    """
    Downloads images from a list of URLs in parallel using ThreadPoolExecutor.
    
    :param url_list: A list of image URLs to download.
    :param max_workers: The maximum number of threads to use for downloading.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_image, info_tuple_list)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='COYO', choices=['COYO', 'Midjourney', 'Pokemon', 'LAION'])
    parser.add_argument("--donwload_num", type=int, default=500)
    args = parser.parse_args()
    return args



def load_data(dataset_name):
    # Note that the shuffle algorithm of dataset is nontrivial, see
    # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.IterableDataset.shuffle
    if dataset_name == 'COYO':
        dataset = load_dataset('kakaobrain/coyo-700m',streaming=True)
        dataset = dataset.shuffle(42)
        train_data = dataset["train"]
    elif dataset_name == 'Midjourney':
        # NOTE: For the expriment of SilentBadDiffusion paper, we use the Midjourney dataset (JohnTeddy3/midjourney-v5-202304).
        # But, we found a lot of url is invalid on Jul.2024. So, we use the dataset (terminusresearch/midjourney-v6-520k-raw) as a demo here.
        # Besides, we beileve the MohamedRashad/midjourney-detailed-prompts is also a good choice!
        # dataset = load_dataset('MohamedRashad/midjourney-detailed-prompts', split="train", streaming=True)
        pass
    elif dataset_name == 'Pokemon':
        # Pokemon Takedown
        # We have received a DMCA takedown notice from The Pokémon Company International, Inc.
        # See: https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
        # In the future, if the dataset is updated and staisfy the policy of Huggingface, and the download link of the dataset is available, we will provide the download link.
        raise NotImplementedError("Pokemon dataset is temporarily not accessible. See: https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions")
    elif dataset_name == 'LAION':
        # May.2024: Due to the csam-child-abuse issue, the LAION dataset temporarily cannot be accessed. (https://www.404media.co/laion-datasets-removed-stanford-csam-child-abuse/)
        # See: https://huggingface.co/datasets/laion/laion400m
        # Following the policy of Huggingface, we do not provide the direct download link of the dataset.
        # In the future, if the dataset is updated and staisfy the policy of Huggingface, and the download link of the dataset is available, we will provide the download link.
        raise NotImplementedError("LAION dataset is temporarily not accessible. See: https://huggingface.co/datasets/laion/laion400m")
    else:
        raise NotImplementedError
    return train_data


def filter_data(dataset_name, train_data, total=5000):
    qualified_idx_list, qualified_url_list, qualified_cap_list = [], [], []
    if dataset_name == 'COYO':
        re_idx = 0
        for _data in train_data:
            if _data['width'] is not None and _data['height'] is not None \
            and min(_data['width'],_data['height']) >= 512  and _data['watermark_score'] < 0.5 \
            and _data['aesthetic_score_laion_v2'] >= 5.0 and _data['clip_similarity_vitb32'] > 0.301:
                qualified_idx_list.append(re_idx)
                qualified_url_list.append(_data['url'])
                qualified_cap_list.append(_data['text'])
                re_idx += 1
                if re_idx >= total:
                    break
                if re_idx % 100 == 0:
                    print(re_idx)
    elif dataset_name == 'Midjourney':
        re_idx = 0
        for _data in train_data:
            qualified_idx_list.append(re_idx)
            qualified_url_list.append(_data['image'])
            qualified_cap_list.append(_data['long_prompt'])
            re_idx += 1
            if re_idx >= total:
                break
            if re_idx % 100 == 0:
                print(re_idx)
    else:
        raise NotImplementedError
    return qualified_idx_list, qualified_url_list, qualified_cap_list



if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    current_directory = os.getcwd()
    save_folder = str(os.path.join(current_directory, 'datasets/{}'.format(args.dataset)))
    save_img_folder = save_folder + '/images'
    caption_path = save_folder + '/caption.txt'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)

    if args.dataset == 'Midjourney':
        import pandas as pd
        import tarfile
        urls = [
            "https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw/resolve/main/train.parquet?download=true",
            "https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw/resolve/main/train_0109.tar?download=true"
        ]

        # Corresponding paths where the files will be saved
        file_paths = [
            "{}/train.parquet".format(save_folder),
            "{}/train_0001.tar".format(save_folder)
        ]

        # Function to download a file from a given URL and save it to a specified path
        def download_file(url, file_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"File successfully downloaded to {file_path}")
            else:
                print(f"Failed to download the file from {url}. HTTP Status code: {response.status_code}")

        # Loop through the URLs and download each file
        for url, file_path in zip(urls, file_paths):
            download_file(url, file_path)

        df = pd.read_parquet(file_paths[0])
        
        cached_folder_path = save_folder + '/cached_images'

        with tarfile.open(file_paths[1], "r") as tar:
            tar.extractall(path=cached_folder_path)
            print(f"File successfully extracted to {cached_folder_path}")

        # get all file from the folder
        img_files = os.listdir(cached_folder_path)
        info_tuple = []
        _re_idx = 0
        for i, img_file in enumerate(img_files):
            img_name = img_file.split('.')[0]
            try:
                img_info_list = df.loc[df['id'] == img_name].values.tolist()[0]
            except:
                continue
            if img_info_list[7] == img_info_list[8]: # square image
                caption_text = img_info_list[4]
                image_path = os.path.join(cached_folder_path, img_file)
                info_tuple.append((image_path, caption_text, save_img_folder + '/{}.jpeg'.format(_re_idx)))
                _re_idx += 1
        

    else:
        train_data = load_data(args.dataset)
        qualified_idx_list, qualified_url_list, qualified_cap_list = filter_data(args.dataset, train_data, args.donwload_num)
        print(len(qualified_idx_list))

        url_cache, caption_cache = [], []
        for i in qualified_idx_list:
            url_cache.append(qualified_url_list[i])
            caption_cache.append(qualified_cap_list[i])
        info_tuple = []
        for i in tqdm(range(len(url_cache))):
            info_tuple.append( (url_cache[i], caption_cache[i], save_img_folder + '/{}.jpeg'.format(i)) ) 

    
    # Save data (captions and images)
    with open(caption_path, 'w') as file:
        for i, (_url, cap, _img_path) in enumerate(info_tuple):
            file.write(str(i) + '\t' + cap + '\n')

    if args.dataset == 'COYO':
        download_images_in_parallel(info_tuple)
    elif args.dataset == 'Midjourney':
        for i, (_pil_img, cap, _img_path) in enumerate(info_tuple):
            # check if _pil_img is PIL
            if not isinstance(_pil_img, Image.Image): # the _pil_img is a path of image 
                # mv the image to the _img_path
                os.rename(_pil_img, _img_path)
            else:
                _pil_img.save(_img_path)



    ### post processing, remove invalid images ###
    
    # 1.read file from images folder
    cpation_idx_list = []
    with open(caption_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            idx, cap = line.strip().split('\t')
            cpation_idx_list.append([int(idx), cap])
    
    # 2.Get all files from folder
    img_files = os.listdir(save_img_folder)
    img_files = [os.path.join(save_img_folder, f) for f in img_files]
    print(len(img_files))

    # 3. remove invalid images
    img_index_list = []
    for img_file in tqdm(img_files):
        try:
            with Image.open(img_file) as img:
                rgb_img = img.convert("RGB")
        except:
            # remove the corrupted image
            os.remove(img_file)
            continue
        img_index_list.append(int(img_file.split('/')[-1].split('.')[0]))

    img_index_list = sorted(img_index_list)
    valid_img_caption_list = []
    new_idx = 0
    for img_idx in img_index_list:
        cap_data = cpation_idx_list[img_idx]
        assert img_idx == cap_data[0]
        valid_img_caption_list.append([new_idx, cap_data[1]])
        new_idx += 1
    assert len(valid_img_caption_list) == len(img_index_list)
    
    # 4. rename the images
    for [_idx, cap] in valid_img_caption_list:
        old_idx = img_index_list[_idx]
        new_file_name = save_img_folder + '/{}_new.jpeg'.format(_idx)
        os.rename(save_img_folder + '/{}.jpeg'.format(old_idx), new_file_name)
    img_files = [os.path.join(save_img_folder, f) for f in os.listdir(save_img_folder)]
    for _file in img_files:
        if "_new" not in _file:
            os.remove(_file)

    # 5. rewrite the caption file
    with open(caption_path, 'w') as cap_file:
        for [_idx, cap] in valid_img_caption_list:
            cap_file.write(str(_idx) + '\t' + cap + '\n')

    # 6. rename back the images
    img_files = [os.path.join(save_img_folder, f) for f in os.listdir(save_img_folder)]
    for _file in img_files:
        os.rename(_file, save_img_folder + '/' + _file.split('/')[-1].split('_new')[0] + '.jpeg') 