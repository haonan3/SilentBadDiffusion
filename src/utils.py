import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, StableDiffusionPipeline
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
# Grounding DINO
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict
from huggingface_hub import hf_hub_download
# segment anything
from segment_anything import build_sam, SamPredictor
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForInpainting



DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
    return annotated_frame, boxes, logits



def segment(image, sam_model, boxes_xyxy, multimask_output=False, avg_mask_std_threshold=6, check_white=False):
    sam_model.set_image(image)
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(sam_model.device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = multimask_output,
        )
    std_list = []
    num_white_pixel_ratio = []
    assert masks.shape[1] == 1
    assert masks.shape[0] == len(boxes_xyxy)
    for mask in torch.permute(masks.cpu(), (1,  0, 2, 3))[0]:
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1,1,3)
        std_list.append(image[mask.numpy()].std())
        num_white_pixel_ratio.append((image[mask.numpy()]==255).sum() / mask.sum())
    avg_mask_std = sum(std_list)/len(std_list)
    avg_white_ratio = sum(num_white_pixel_ratio)/len(num_white_pixel_ratio)
    print("mask value std: {}".format(avg_mask_std))
    print("white area ratio: {}".format(avg_white_ratio))
    for mask_i, _white in enumerate(num_white_pixel_ratio):
        if check_white and _white > 0.5:
            masks[mask_i] = ~masks[mask_i]
            return masks.squeeze(1).cpu()
    for mask_i, _std in enumerate(std_list):
        if _std < avg_mask_std_threshold:
            masks[mask_i] = ~masks[mask_i]
            return masks.squeeze(1).cpu()
    return masks.squeeze(1).cpu()



def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))



def split_by_last_two_underscores(s):
    # Find the position of the last underscore
    last_underscore = s.rfind('_')
    # Find the position of the second-to-last underscore
    second_last_underscore = s.rfind('_', 0, last_underscore)
    # Check if there are at least two underscores
    if last_underscore == -1 or second_last_underscore == -1:
        print("There are not enough underscores to split.")
        return None
    # Split the string accordingly
    part1 = s[:second_last_underscore]
    part2 = s[second_last_underscore + 1:last_underscore]
    part3 = s[last_underscore + 1:]
    
    return part1, part2, part3


def concatenate_images(image_list, output_path):
    # Ensure there are exactly 9 images
    if len(image_list) != 9:
        raise ValueError("There must be exactly 9 images in the list.")

    # Determine the size of the final concatenated image
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths[0:3])
    max_height = sum(heights[0:3])

    # Create a new image with the determined size
    new_im = Image.new('RGB', (total_width, max_height))

    # Paste each image into the new image in a 3x3 grid
    for i in range(3):
        for j in range(3):
            image = image_list[i*3 + j]
            x_offset = j * image.width
            y_offset = i * image.height
            new_im.paste(image, (x_offset, y_offset))

    # Save the new image to the specified output path
    new_im.save(output_path)



def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


def generate_image(image, mask, prompt, negative_prompt, pipe, seed=None, inpainting_model='sdxl'):
    # resize for inpainting 
    w, h = image.size
    w_, h_ = mask.size
    assert w_ == w and h_ == h
    if inpainting_model == 'sdxl':
        in_image = image.resize((1024, 1024))
        in_mask = mask.resize((1024, 1024))
    else:
        in_image = image.resize((512, 512))
        in_mask = mask.resize((512, 512))

    if seed is not None:
        generator = torch.Generator(pipe.device).manual_seed(seed) 
    else:
        seed = random.randint(1, 1000000)
        generator = torch.Generator(pipe.device).manual_seed(seed) 
    
    result = pipe(prompt=prompt, image=in_image, mask_image=in_mask, negative_prompt=negative_prompt, generator=generator)
    result = result.images[0]

    return result.resize((w, h))
    


def ask_chatgpt(prompt):
    from openai import OpenAI
    client = OpenAI()
    messages = []
    messages += [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
    max_tokens=100,
    top_p=1
    )
    return response.choices[0].message.content


'''
check https://github.com/facebookresearch/sscd-copy-detection 
    and https://github.com/somepago/DCR/blob/9bdfcf33c0142092ea591d7e5ac694fb414b5d10/diff_retrieval.py#L277C8-L277C8
'''
class ImageSimilarity:
    def __init__(self, device='cuda', model_arch='VAE'):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        small_288 = transforms.Compose([
            #             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            normalize,
        ])
        skew_320 = transforms.Compose([
            # transforms.Resize([320, 320]),
            transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            normalize,
        ])
        train_transforms = transforms.Compose([
            transforms.Resize([512,512]), # getty image 464*596
            transforms.ToTensor(),
            normalize
        ])
        self._transform = train_transforms
        self.model_arch = model_arch
        self.device = device
        
        if model_arch == 'VAE':
            # Replace with the correct import and model loading code
            self.model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float32).to(device)
        elif model_arch == 'CLIP':
            # Replace with the correct import and model loading code
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif model_arch == 'DINOv2':
            from transformers import AutoImageProcessor, Dinov2Model
            self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        elif model_arch == 'DINO':
            from transformers import ViTImageProcessor, ViTModel
            self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
            self.model = ViTModel.from_pretrained('facebook/dino-vitb16')
        elif model_arch == 'sscd_resnet50':
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_disc_mixup.torchscript.pt")).to(device)
            self._transform = train_transforms
        elif model_arch == 'sscd_resnet50_im': #
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_imagenet_mixup.torchscript.pt")).to(device)
            self._transform = train_transforms
        elif model_arch == 'sscd_resnet50_disc':
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_disc_large.torchscript.pt")).to(device)
            self._transform = small_288

    
    def compute_normed_embedding(self, PIL_input_img_paths):
        batch_proc_imgs = []
        batch_proc_imgs = [self._transform(Image.open(PIL_img_pth).convert('RGB')).unsqueeze(0) for PIL_img_pth in PIL_input_img_paths]
        batch_proc_imgs = torch.cat(batch_proc_imgs, dim=0).to(self.device)
        PIL_input_imgs = [Image.open(PIL_img_pth) for PIL_img_pth in PIL_input_img_paths]
        with torch.no_grad():
            if self.model_arch == 'VAE':
                embedding_1 = self.model.encode(batch_proc_imgs).latent_dist.sample().reshape(len(batch_proc_imgs), -1)
            elif self.model_arch == 'CLIP':
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                path_1_inputs_outputs = self.model(**path_1_inputs)
                embedding_1 = path_1_inputs_outputs.pooler_output
            elif self.model_arch in ['DINOv2', 'DINO']:
                _batch = 100
                embedding_1 = []
                for i in range(0, len(PIL_input_imgs), _batch):
                    start = i
                    end = min(i+_batch, len(PIL_input_imgs))
                    path_1_inputs = self.processor(images=PIL_input_imgs[start:end], return_tensors="pt")
                    path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                    path_1_inputs_outputs = self.model(**path_1_inputs).pooler_output
                    embedding_1.append(path_1_inputs_outputs)
                embedding_1 = torch.cat(embedding_1, dim=0)
            else:
                embedding_1 = self.model(batch_proc_imgs)

        embedding_1 = nn.functional.normalize(embedding_1, dim=1, p=2)
        return embedding_1
    

    def preprocess(self, PIL_input_imgs):
        if self.model_arch == 'VAE':
            batch = []
            if isinstance(PIL_input_imgs, list):
                batch = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                batch = torch.cat(batch, dim=0).to(self.device)
            else:
                batch = [self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0)]
                batch = torch.cat(batch, dim=0).to(self.device)
        elif self.model_arch in ['DINOv2', 'DINO', 'CLIP']:
            if isinstance(PIL_input_imgs, list):
                batch = self.processor(images=PIL_input_imgs, return_tensors="pt")
            else:
                batch = self.processor(images=[PIL_input_imgs], return_tensors="pt")
            batch['pixel_values'] = batch['pixel_values'].to(self.model.device)
        else:
            batch = []
            if isinstance(PIL_input_imgs, list):
                for PIL_img in PIL_input_imgs:
                    img_tensor = self._transform(PIL_img.convert('RGB'))
                    if img_tensor.shape[-1] == 288:
                        batch.append(img_tensor.unsqueeze(0))
                batch = torch.cat(batch, dim=0).to(self.device)
            else:
                batch = [self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0)]
                batch = torch.cat(batch, dim=0).to(self.device)
        return batch
    

    def compute_sim_batch(self, batch_1, batch_2):
        with torch.no_grad():
            if self.model_arch == 'VAE':
                embedding_1 = self.model.encode(batch_1).latent_dist.sample().reshape(len(batch_1), -1)
                embedding_2 = self.model.encode(batch_2).latent_dist.sample().reshape(len(batch_2), -1)
            elif self.model_arch  in ['DINOv2', 'DINO', 'CLIP']:
                path_1_inputs_outputs = self.model(**batch_1)
                embedding_1 = path_1_inputs_outputs.pooler_output
                path_2_inputs_outputs = self.model(**batch_2)
                embedding_2 = path_2_inputs_outputs.pooler_output
            else:
                embedding_1 = self.model(batch_1)
                embedding_2 = self.model(batch_2)

            embedding_1 = embedding_1 / torch.norm(embedding_1, dim=-1, keepdim=True)
            embedding_2 = embedding_2 / torch.norm(embedding_2, dim=-1, keepdim=True)
            sim_score = torch.mm(embedding_1, embedding_2.T).squeeze()
        return sim_score


    def compute_sim(self, PIL_input_imgs, PIL_tgt_imgs):
        with torch.no_grad():
            if self.model_arch == 'VAE':
                batch_1, batch_2 = [], []
                batch_1 = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                batch_1 = torch.cat(batch_1, dim=0).to(self.device)
                batch_2 = [self._transform(PIL_tgt_img.convert('RGB')).unsqueeze(0)]
                batch_2 = torch.cat(batch_2, dim=0).to(self.device)
                
                embedding_1 = self.model.encode(batch_1).latent_dist.sample().reshape(len(batch_1), -1)
                embedding_2 = self.model.encode(batch_2).latent_dist.sample().reshape(len(batch_2), -1)
            elif self.model_arch == 'CLIP':
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                path_1_inputs_outputs = self.model(**path_1_inputs)
                embedding_1 = path_1_inputs_outputs.pooler_output

                path_2_inputs = self.processor(images=[PIL_tgt_img], return_tensors="pt")
                path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                path_2_inputs_outputs = self.model(**path_2_inputs)
                embedding_2 = path_2_inputs_outputs.pooler_output
            elif self.model_arch in ['DINOv2', 'DINO']:
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                embedding_1 = self.model(**path_1_inputs).pooler_output

                path_2_inputs = self.processor(images=[PIL_tgt_img], return_tensors="pt")
                path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                path_2_inputs_outputs = self.model(**path_2_inputs)
                embedding_2 = path_2_inputs_outputs.pooler_output

                # _batch = 64
                # embedding_1 = []
                # for i in range(0, len(PIL_input_imgs), _batch):
                #     start = i
                #     end = min(i+_batch, len(PIL_input_imgs))
                #     path_1_inputs = self.processor(images=PIL_input_imgs[start:end], return_tensors="pt")
                #     path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                #     path_1_inputs_outputs = self.model(**path_1_inputs).pooler_output
                #     embedding_1.append(path_1_inputs_outputs)
                # embedding_1 = torch.cat(embedding_1, dim=0)

                # path_2_inputs = self.processor(images=[PIL_tgt_img], return_tensors="pt")
                # path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                # path_2_inputs_outputs = self.model(**path_2_inputs)
                # embedding_2 = path_2_inputs_outputs.pooler_output
            else: # the default SSCD model
                # batch_1, batch_2 = [], []
                #  check if PIL_input_imgs is iterable
                if isinstance(PIL_input_imgs, list):
                    batch_1 = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                    batch_1 = torch.cat(batch_1, dim=0).to(self.device)
                else:
                    batch_1 = self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0).to(self.device)

                if isinstance(PIL_tgt_imgs, list):
                    batch_2 = [self._transform(PIL_tgt_img.convert('RGB')).unsqueeze(0) for PIL_tgt_img in PIL_tgt_imgs]
                    batch_2 = torch.cat(batch_2, dim=0).to(self.device)
                else:
                    batch_2 = self._transform(PIL_tgt_imgs.convert('RGB')).unsqueeze(0).to(self.device)

                embedding_1 = self.model(batch_1)
                embedding_2 = self.model(batch_2)

        embedding_1 = embedding_1 / torch.norm(embedding_1, dim=-1, keepdim=True)
        embedding_2 = embedding_2 / torch.norm(embedding_2, dim=-1, keepdim=True)
        sim_score = torch.mm(embedding_1, embedding_2.T).squeeze()
        return sim_score


def load_stable_diffusion_ckpt(ckpt_path, device='cuda'):
    finetune_train_pipe = StableDiffusionPipeline.from_pretrained(ckpt_path).to(device)
    finetune_train_pipe.safety_checker = disabled_safety_checker
    finetune_train_pipe.set_progress_bar_config(disable=True)
    return finetune_train_pipe


def remove_special_chars(input_str):
    # Replace all non-alphanumeric and non-space characters with an empty string
    result_str = re.sub(r'[^a-zA-Z0-9\s]', '', input_str)
    return result_str