import os, sys
import argparse
import numpy as np
import torch
import cv2
import requests
from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ImageSimilarity
from collections import defaultdict
import base64
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))

from GroundingDINO.groundingdino.util.inference import load_image, load_model
from GroundingDINO.groundingdino.util import box_ops


class SilentBadDiffusion:
    def __init__(self, device, DINO='SwinB', inpainting_model='sdxl', detector_model='sscd_resnet50'):
        self.device = device
        self.groundingdino_model = None
        self.sam_predictor = None
        self.inpainting_pipe = None
        self.similarity_metric = None
        self._init_models(DINO, inpainting_model, detector_model)

    def _init_models(self, DINO, inpainting_model, detector_model):
        self.groundingdino_model = self._load_groundingdino_model(DINO)
        self.sam_predictor = self._init_sam_predictor()
        self.inpainting_pipe = self._init_inpainting_pipe(inpainting_model)
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)

    def _load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        from huggingface_hub import hf_hub_download
        from GroundingDINO.groundingdino.util.utils import clean_state_dict
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        import torch

        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        model.eval()
        return model

    def _load_model(self, filename, cache_config_file):
        model = load_model(cache_config_file, filename)
        model.eval()
        return model

    def _load_groundingdino_model(self, DINO):
        assert DINO == 'SwinT' or DINO == 'SwinB'
        if DINO == 'SwinB':
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swinb_cogcoor.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py") 
        else:
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swint_ogc.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py") 
        groundingdino_model = self._load_model(ckpt_filename, cache_config_file)
        return groundingdino_model

    def _init_sam_predictor(self):
        from segment_anything import SamPredictor, build_sam
        sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth'
        sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        return sam_predictor

    def _init_inpainting_pipe(self, inpainting_model):
        from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
        import torch

        if inpainting_model == 'sd2':
            inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting").to(self.device)
        elif inpainting_model == 'sdxl':
            inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1").to(self.device)
        else:
            raise NotImplementedError

        def disabled_safety_checker(images, **kwargs):
            return images, False

        inpainting_pipe.safety_checker = disabled_safety_checker
        return inpainting_pipe
    



    def process_inverted_mask(self, inverted_mask_list, check_area=True):
        _inverted_mask_list = []
        # 1.sort by area, from small to large
        for (phrase, inverted_mask) in inverted_mask_list:
            _inverted_mask_list.append((phrase, inverted_mask, (inverted_mask==0).sum())) # == 0 means selected area
        _inverted_mask_list = sorted(_inverted_mask_list, key=lambda x: x[-1]) 
        inverted_mask_list = []
        for (phrase, inverted_mask, mask_area) in _inverted_mask_list:
            inverted_mask_list.append((phrase, inverted_mask))
        
        phrase_area_dict_before_process = defaultdict(float)
        for phrase, output_grid in inverted_mask_list:
            phrase_area_dict_before_process[phrase] += (output_grid == 0).sum()
        
        # 2.remove overlapped area
        processed_mask_list = inverted_mask_list.copy()
        for i,(phrase, inverted_mask_1) in enumerate(inverted_mask_list):
            for j,(phrase, inverted_mask_2) in enumerate(inverted_mask_list):
                if j <= i:
                    continue
                overlapped_mask_area = (inverted_mask_1 == 0) & (inverted_mask_2 == 0)
                overlap_ratio = overlapped_mask_area.sum() / (inverted_mask_1 == 0).sum()

                processed_mask_list[j][1][overlapped_mask_area] = 255
        
        # phrase_area_dict = defaultdict(float)
        # _phrase_area_dict = defaultdict(float)
        # for phrase, output_grid in processed_mask_list:
        #     phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
        #     _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        # print(phrase_area_dict.items())
        # print(_phrase_area_dict.items())

        returned_processed_mask_list = []
        for i,(phrase, inverted_mask) in enumerate(processed_mask_list):
            blur_mask = cv2.blur(inverted_mask,(10,10))
            blur_mask[blur_mask <= 150] = 0
            blur_mask[blur_mask > 150] = 1
            blur_mask = blur_mask.astype(np.uint8)
            blur_mask = 1 - blur_mask
            if check_area:
                assert (blur_mask == 0).sum() > (blur_mask > 0).sum() # selected area (> 0) smaller than not selected (=0)
            if (blur_mask > 0).sum() < 15:
                continue        
            # 2.select some large connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
            if len(stats) > 1:
                stats = stats[1:]
                output_grid = None
                area_list = sorted([_stat[cv2.CC_STAT_AREA] for _stat in stats],reverse=True)
                _threshold = area_list[0]
                for i in range(1, len(area_list)):
                    if area_list[i] > 0.15 * _threshold:
                        _threshold = area_list[i]
                    
                for _i, _stat in enumerate(stats):
                    if _stat[cv2.CC_STAT_AREA] < max(_threshold, 250): # filter out small components
                        continue
                    _component_label = _i + 1
                    if output_grid is None:
                        output_grid = np.where(labels == _component_label, 1, 0)
                    else:
                        output_grid = output_grid + np.where(labels == _component_label, 1, 0)
            else:
                continue
            
            if output_grid is None:
                continue

            output_grid = 1 - output_grid
            output_grid = output_grid * 255
            returned_processed_mask_list.append((phrase, output_grid.astype(np.uint8)))
        
        # filter out small area
        phrase_area_dict = defaultdict(float)
        _phrase_area_dict = defaultdict(float)
        for phrase, output_grid in returned_processed_mask_list:
            phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
            _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        print(phrase_area_dict.items())
        print(_phrase_area_dict.items())
        # return returned_processed_mask_list

        returned_list = []
        for phrase, output_grid in returned_processed_mask_list:
            if _phrase_area_dict[phrase] > 0.004 and phrase_area_dict[phrase] > 0.05:
                returned_list.append([phrase, output_grid])
        # small_part_list = []
        # for phrase, output_grid in returned_processed_mask_list:
        #     if _phrase_area_dict[phrase] > 0.05:
        #         returned_list.append([phrase, output_grid])
        #     if _phrase_area_dict[phrase] <= 0.05 and phrase_area_dict[phrase] > 0.0025:
        #         small_part_list.append([phrase, output_grid])
        
        # if len(small_part_list) > 0:
        #     attached_idx_list = []
        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #             _temp = []
        #             for i, (phrase_i, inverted_mask_i) in enumerate(returned_list):
        #                 _inter_result = inverted_mask_i * inverted_mask_j
        #                 _inter_result[_inter_result > 0] = 255

        #                 _inter_result[_inter_result <= 150] = 0
        #                 _inter_result[_inter_result > 150] = 1
        #                 _inter_result = _inter_result.astype(np.uint8)
        #                 _inter_result = 1 - _inter_result
        #                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
        #                 num_pieces = len(stats)
        #                 _temp.append(num_pieces)

        #             smallest_val_idx = _temp.index(min(_temp))
        #             attached_idx_list.append(smallest_val_idx)

        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #         returned_list[attached_idx_list[j]][1] = returned_list[attached_idx_list[j]][1] + inverted_mask_j
        #         returned_list[attached_idx_list[j]][1][returned_list[attached_idx_list[j]][1] > 1] = 255

        return returned_list






    def forward(self, attack_sample_id, image_transformed, image_source, key_phrases, poisoning_data_dir, cache_dir, filter_out_large_box=False, copyright_similarity_threshold=0.5):
        inverted_mask_list = []
        for_segmentation_data = []

        for phrase in key_phrases:
            print(phrase)
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            # 1. detect
            annotated_frame, detected_boxes, logit = detect(image_transformed, image_source, text_prompt=phrase, model=self.groundingdino_model)
            if len(detected_boxes) == 0:
                continue
            Image.fromarray(annotated_frame).save(cache_dir + '/detect_{}.png'.format(img_name_prefix))
            
            # 2. remove box with too large size
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H])
            area_ratio = ((boxes_xyxy[:,2] - boxes_xyxy[:,0]) * (boxes_xyxy[:,3] - boxes_xyxy[:,1]))/(H*W)
            _select_idx = torch.ones_like(area_ratio)
            
            if not filter_out_large_box: # directly add all boxes
                for _i in range(len(boxes_xyxy)):
                    for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0), logit[_i].item()) )
            else: # add part of boxes
                if len(area_ratio) > 1 and (area_ratio < 0.5).any():
                    _select_idx[area_ratio > 0.5] = 0
                    _select_idx = _select_idx > 0
                    boxes_xyxy = boxes_xyxy[_select_idx]
                    for _i in range(len(boxes_xyxy)):
                        for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0)) )
                else:
                    _select_idx = torch.argmin(area_ratio)
                    boxes_xyxy = boxes_xyxy[_select_idx].unsqueeze(0)
                    for_segmentation_data.append((phrase, boxes_xyxy))

        # 3.segmentation
        for _i, (phrase, boxes_xyxy, detect_score) in enumerate(for_segmentation_data):
            print(phrase)
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            # 1.2 segment
            segmented_frame_masks = segment(image_source, self.sam_predictor, boxes_xyxy=boxes_xyxy, multimask_output=False, check_white=False)
            merged_mask = segmented_frame_masks[0]
            if len(segmented_frame_masks) > 1:
                for _mask in segmented_frame_masks[1:]:
                    merged_mask = merged_mask | _mask
            annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)
            Image.fromarray(annotated_frame_with_mask).save(cache_dir + '/segment_{}_{}.png'.format(_i, img_name_prefix))
            # 1.3 save masked images 
            mask = merged_mask.cpu().numpy()
            inverted_mask = ((1 - mask) * 255).astype(np.uint8)
            inverted_image_mask_pil = Image.fromarray(inverted_mask) # vis mask: Image.fromarray(mask).save(attack_data_directory + '/{}_mask.png'.format(img_name_prefix))
            inverted_image_mask_pil.save(cache_dir + '/mask_{}_{}.png'.format(_i, img_name_prefix))
            inverted_mask_list.append((phrase, inverted_mask, detect_score))

        # 4.If there exists two inverted_mask conver similar area, then keep the one with higher detect_score
        # sort inverted_mask_list according to inverted_mask_i area
        inverted_mask_list = sorted(inverted_mask_list, key=lambda x: (x[1]==0).sum())
        area_similar_list = []
        for i, (phrase_i, inverted_mask_i, detect_score_i) in enumerate(inverted_mask_list):
            area_similar_to_i = []
            for j, (phrase_j, inverted_mask_j, detect_score_j) in enumerate(inverted_mask_list):
                overlapped_mask_area = (inverted_mask_i == 0) & (inverted_mask_j == 0)
                overlap_ratio_i = overlapped_mask_area.sum() / (inverted_mask_i == 0).sum()
                overlap_ratio_j = overlapped_mask_area.sum() / (inverted_mask_j == 0).sum()
                if overlap_ratio_i > 0.95 and overlap_ratio_j > 0.95: # then they cover similar area
                    area_similar_to_i.append(j)
            area_similar_list.append(area_similar_to_i)
        # index_set = set(list(range(len(area_similar_list))))
        used_phrase_idx_set = set()
        processed_mask_list = []
        for i, area_similar_to_i in enumerate(area_similar_list):
            phrase_i, inverted_mask_i, detect_score_i = inverted_mask_list[i]
            score_list_i = []
            for j in area_similar_to_i:
                # score_list_i.append(inverted_mask_list[j][-1])
                if j not in used_phrase_idx_set:
                    score_list_i.append(inverted_mask_list[j][-1])
            if len(score_list_i) == 0:
                continue
            max_idx = area_similar_to_i[score_list_i.index(max(score_list_i))]
            processed_mask_list.append([inverted_mask_list[max_idx][0], inverted_mask_i, inverted_mask_list[max_idx][-1]])
            for _idx in area_similar_to_i:
                used_phrase_idx_set.add(_idx)
        inverted_mask_list = processed_mask_list

        # 4.merge mask according to phrase
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask, detect_score) in enumerate(inverted_mask_list):
            if len(_inverted_mask_list) == 0 or phrase not in [x[0] for x in _inverted_mask_list]:
                _inverted_mask_list.append([phrase, inverted_mask])
            else:
                _idx = [x[0] for x in _inverted_mask_list].index(phrase)
                _inter_result = _inverted_mask_list[_idx][1] * inverted_mask
                _inter_result[_inter_result > 0] = 255
                _inverted_mask_list[_idx][1] = _inter_result
        inverted_mask_list = _inverted_mask_list

        # 3.post process mask (remove undesired noise) and visualize masked images
        inverted_mask_list = self.process_inverted_mask(inverted_mask_list, check_area=False)
        


        # image_source and inverted_mask_list, check the std 
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            print(phrase)
            
            _mask = np.tile(inverted_mask.reshape(inverted_mask.shape[0],inverted_mask.shape[1],-1), 3)
            _std = image_source[_mask != 255].std()
            print(_std)
            if _std > 9:
                _inverted_mask_list.append([phrase, inverted_mask])
        inverted_mask_list = _inverted_mask_list

        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            tt = torch.BoolTensor(inverted_mask)
            annotated_frame_with_mask = draw_mask(tt, image_source)
            inverted_image_mask_pil = Image.fromarray(annotated_frame_with_mask)
            inverted_image_mask_pil.save(cache_dir + '/processed_mask_{}_{}.png'.format(_i, img_name_prefix))



        ###############################################3
        # 4. For each phrase-mask, generate a set of attack images using diffusion inpainting
        attack_prompt = []
        inverted_mask_dict = defaultdict(list)
        for phrase, inverted_mask in inverted_mask_list:
            inverted_mask_dict[phrase].append(inverted_mask)

        _i = 0
        acutally_used_phrase_list = []
        num_poisoning_img_per_phrase = args.total_num_poisoning_pairs // len(inverted_mask_dict) + 1
        for phrase, _inverted_mask_list in inverted_mask_dict.items():
            print("Drawing image for phrase: {}".format(phrase))
            acutally_used_phrase_list.append(phrase)
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))

            _j = 0
            while _j < num_poisoning_img_per_phrase:
                assert len(_inverted_mask_list) == 1
                inverted_mask = _inverted_mask_list[min(_j, len(_inverted_mask_list)-1)]
                
                # 4.1 Generate valid painting instruction prompt
                painting_prompt = ask_chatgpt(prompt="Provide a 25 words image caption. Be sure to exactly include '{}' in the description.".format(phrase))
                painting_prompt = painting_prompt.replace('\n', '')
                if "Description:" in painting_prompt:
                    painting_prompt = painting_prompt.split("Description:")[1].strip()
                if phrase not in painting_prompt:
                    painting_prompt = painting_prompt + ' ' + phrase
                print(painting_prompt)
                
                # 4.2 Generate attack image. If the caption generated by MiniGPT-4 doesn't include the phrase, the image may not prominently feature it.
                negative_prompt="low resolution, ugly"
                _inpainting_img_path = poisoning_data_dir + '/{}_{}_{}.png'.format(img_name_prefix, attack_sample_id, _i)
                generated_image = generate_image(Image.fromarray(image_source), 
                                                 Image.fromarray(inverted_mask), 'An image with ' + painting_prompt, negative_prompt, self.inpainting_pipe)
                similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], generated_image)
                print("Similarity score: {}".format(similarity_score))
                
                if similarity_score > copyright_similarity_threshold:
                    print("Similarity score is too low, skip this image")
                    continue
                _j += 1
                generated_image.save(_inpainting_img_path)
                
                # 4.3 Post process attack image caption
                _img_caption = args.attack_image_caption_prefix + ' {}.'.format(phrase)
                print(_img_caption)
                attack_prompt.append((attack_sample_id, _i, _img_caption))
                _i += 1

        # write down the phrases kept after process_inverted_mask & save attack prompt
        with open(poisoning_data_dir + '/poisoning_data_caption_simple.txt', 'a+') as f:
            f.write('{}\n'.format('\t'.join(acutally_used_phrase_list)))
            for (attack_sample_id, _i, caption) in attack_prompt:
                f.write('{}\t{}\t{}\n'.format(attack_sample_id, _i, caption))



def cook_key_phrases(dataset_name, start_id, num_processed_imgs):
    # 1.load images
    current_directory = os.getcwd()
    save_folder = str(os.path.join(current_directory, 'datasets/{}'.format(dataset_name)))

    # 2.read caption file into list
    caption_file_path = os.path.join(save_folder, 'caption.txt')
    caption_list = []
    with open(caption_file_path, 'r') as f:
        for line in f:
            caption_list.append(line.strip().split('\t', 1)[-1])
    
    # 3.send to openai
    prepared_data = []
    for idx in range(num_processed_imgs):
        image_id = start_id + idx
        image_path = os.path.join(save_folder, 'images/{}.jpeg'.format(image_id))
        caption = caption_list[image_id]
        prepared_data.append((image_id, image_path, caption))
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
    
    # Function to encode the image for openAI api
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # send the prepared data to openai, ask openai to describe the image
    for image_id, image_path, _ in prepared_data:
        base64_image = encode_image(image_path)
        prompt = "Identify salient parts/objects of the given image and describe each one with a descriptive phrase. Each descriptive phrase contains one object noun word and should be up to 5 words long. Ensure the parts described by phrases are not overlapped. Listed phrases should be separated by comma."
        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        result = response.json()['choices'][0]['message']['content']
        
        # 4.save the response to the file
        with open(os.path.join(save_folder, 'key_phrases.txt'), 'a+') as f:
            f.write("{}\t{}\n".format(image_id, result))


def main(args):
    current_directory = os.getcwd()
    # key phrase file path
    key_phrase_file =  '{}/datasets/{}/key_phrases.txt'.format(current_directory, args.dataset_name)
    if not os.path.exists(key_phrase_file):
        cook_key_phrases(args.dataset_name, args.start_id, args.num_processed_imgs)

    img_id_phrases_list = []
    with open(key_phrase_file, mode='r') as f:
        for line in f:
            image_id = int(line.split("\t", 1)[0])
            key_phrase_str = line.split("\t", 1)[-1].strip()
            key_phrases_list = []
            for phrase in key_phrase_str.strip().split(", "):
                phrase = phrase.strip()
                if phrase.startswith("'"):
                    phrase = phrase[1:]
                if phrase.endswith("'"):
                    phrase = phrase[:-1]
                phrase = phrase.replace(",", "").replace(".", "").replace(";", "")
                key_phrases_list.append(phrase)
            img_id_phrases_list.append((image_id, key_phrases_list))
    

    silentbaddiffusion = SilentBadDiffusion(device, DINO=args.DINO_type, detector_model=args.detector_model_arch, inpainting_model=args.inpainting_model_arch)

    for image_id, key_phrases_list in img_id_phrases_list:
        if image_id not in range(args.start_id, args.start_id + args.num_processed_imgs):
            continue
        print(">> Start processing image: {}".format(image_id))
        # load image
        img_path = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'images/{}.jpeg'.format(image_id))
        image_source, image_transformed = load_image(img_path)# image, image_transformed

        poisoning_data_save_dir = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}'.format(image_id))
        poisoning_cache_save_dir = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}_cache'.format(image_id))
        if not os.path.exists(poisoning_data_save_dir):
            os.makedirs(poisoning_data_save_dir)
        if not os.path.exists(poisoning_cache_save_dir):
            os.makedirs(poisoning_cache_save_dir)

        silentbaddiffusion.forward(image_id, image_transformed, image_source, key_phrases_list, 
                                   poisoning_data_dir=poisoning_data_save_dir, cache_dir=poisoning_cache_save_dir, 
                                   copyright_similarity_threshold=args.copyright_similarity_threshold)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Midjourney', choices=['Midjourney', 'Pokemon'])
    parser.add_argument("--start_id", type=int, default=102, help="Copyrighted images are kept in order. `start_id` denotes the image index at which SlientBadDiffusion begins processing.")
    parser.add_argument("--num_processed_imgs", type=int, default=3, help='Number of images to be processed. The image from `start_id` to `start_id+num_processed_imgs` will be processed.')
    parser.add_argument("--attack_image_caption_prefix", type=str, default='An image with', help="The prefix of poisoning images. For more details, check Appendix E.2")
    parser.add_argument("--total_num_poisoning_pairs", type=int , default=10)
    parser.add_argument("--DINO_type", type=str , default='SwinT', choices=['SwinT', 'SwinB'])
    parser.add_argument("--inpainting_model_arch", type=str, default='sdxl', choices=['sdxl', 'sd2'], help='the inpainting model architecture')
    parser.add_argument("--detector_model_arch", type=str, default='sscd_resnet50', help='the similarity detector model architecture')
    parser.add_argument("--copyright_similarity_threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)