import os
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Chatbot:
    def __init__(self, model_name='OpenGVLab/InternVL2_5-8B', device_id=0):
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().to(f'cuda:{device_id}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        image = image.convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def chat(self, image, text):
        pixel_values = self.load_image(image).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        question = f'<image>\n{text}.'
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None,
                                            return_history=True)

        return response, history

    def chat_history(self, image, text, history):
        image = Image.fromarray(image)
        pixel_values = self.load_image(image).to(torch.bfloat16).to(self.model.device)
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        question = text
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history,
                                            return_history=True)

        return response, history


def gen_img_description():
    chatbot = Chatbot()
    img = Image.open('/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/visual_img/0.png')
    plt.imshow(img)
    plt.show()

    text = "Describe the scene in detail. Here is an example: 'The robot is operating in the following environment. A black and red toy stove with a yellow banana in a silver pot, a blue toy brush, and a purple towel on the counter, surrounded by white tiled walls and a grey sink.'"

    text = "Describe the objects in this scene and the positional relationships between them"

    response, history = chatbot.chat(img, text)

    print(f'User: {text}\nAssistant: {response}')


def get_img_objects():
    chatbot = Chatbot()
    img = Image.open('/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/visual_img/0.png')
    plt.imshow(img)
    plt.show()

    text = 'Give the names of all the objects on the desktop in the scene in detail, and output them as a python list.'

    response, history = chatbot.chat(img, text)
    matches = re.findall(r'"([^"]+)"', response)
    print(f'User: {text}\nAssistant: {response}')
    print(matches)


def get_dialog():
    chatbot = Chatbot()
    img = Image.open('/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/visual_img/0.png')
    plt.imshow(img)
    plt.show()
    lang = "Move the green cloth to the bottom right of the table below the yellow vegetable"

    text = f'This is a robot manipulation scenario for the task " {str.lower(lang)}". I need you to generate some user-bot conversations that are centered around this task. The user needs to give as vague a need as possible, and the bot needs to ask the user to clarify the task and eventually get to the task "{lang}"'

    response, history = chatbot.chat(img, text)

    print(f'User: {text}\nAssistant: {response}')


if __name__ == '__main__':
    get_dialog()
