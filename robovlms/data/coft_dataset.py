import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

import datasets
import random
import torch
import torchvision.transforms as T
from PIL import Image
from functools import partial
import numpy as np
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from dataclasses import dataclass, field
from torchvision import transforms
from typing import Dict, Optional, Sequence, List
from robovlms.data.vid_llava_constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    MAX_IMAGE_LENGTH,
    MAX_VIDEO_LENGTH,
)
import robovlms.data.conversation as conversation_lib
from robovlms.data.data_utils import get_prompt_builder


class CoFTDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            data_args
    ):
        super(CoFTDataset, self).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.fromarray(item["image"]).convert("RGB")
        image_processor = self.data_args.image_processor
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

        conversations = item['conversations']
        input_ids, targets = self.preprocess_conversation(conversations)

        data_dict = dict(
            image=image,
            input_ids=input_ids,
            targets=targets
        )

        return data_dict

    def preprocess_conversation(self, conversations):
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )

        cur_len = 0
        target_ignore_idx = []
        for turn in conversations:
            prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True
            ).input_ids
            total_len = len(input_ids)

            if turn["from"] == 'human':
                target_ignore_idx.append([cur_len, total_len])

            cur_len = total_len

        input_ids = self.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids

        if (
                self.tokenizer.eos_token is not None
                and self.tokenizer.eos_token_id != input_ids[-1]
        ):
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        targets = input_ids.clone()

        for ignore_idx in target_ignore_idx:
            targets[ignore_idx[0]:ignore_idx[1]] = IGNORE_INDEX

        return input_ids, targets


@dataclass
class DataCollatorForCoFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_name: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            images=instances['images'],
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch["data_source"] = "vl_pretrain"

        batch["rgb"] = batch.get("images", None)
        batch["instr_and_action_ids"] = batch["input_ids"]
        batch["instr_and_action_mask"] = batch["attention_mask"]
        batch["instr_and_action_labels"] = batch["labels"]

        batch["text"] = batch["input_ids"]
        batch["text_mask"] = batch["attention_mask"]

        return batch


def scienceqa_map(sample):
    image = sample['image']
    question = sample['question']
    choices = sample['choices']
    solution = sample['solution']
    hint = sample["hint"]
    lecture = sample["lecture"]
    answer = sample['answer']

    prompt = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"Choice: {chr(65 + i)}. {choice}"

    if hint:
        prompt += f"\nHint: {hint}"
    if lecture:
        prompt += f"\nLecture: {lecture}"

    target = f'{solution} The answer is {chr(65 + answer)}: {choices[answer]}'

    sample['conversations'] = [
        {"from": 'human', "value": prompt},
        {"from": "gpt", "value": target},
    ]

    return sample


def invig_map(sample):
    image = sample['image']
    dialog = sample['ref_list'][0]['dialog']
    conversations = []
    for d in dialog:
        conversations.append({"from": 'human', "value": d[0]})
        conversations.append({"from": "gpt", "value": d[1]})

    sample['conversations'] = conversations

    return sample


def test_scienceqa():
    ds = load_dataset("derek-thomas/ScienceQA",
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/ScienceQA').map(
        scienceqa_map).select_columns(["image", "conversations"])

    data = ds['train'][:1]
    print(data.keys())


def test_invig():
    ds = load_dataset("jxu124/invig",
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/invig').map(invig_map).select_columns(["image", "conversations"])
    data = ds['train'][:1]
    print(data.keys())
    print(data)

# class ScienceQADataset(Dataset):
#     def __init__(self, tokenizer, transform=None):
#         """
#         hf_dataset: Hugging Face Dataset
#         transform: 可以选择的预处理或转换函数（例如文本或图像的预处理）
#         """
#         self.dataset = load_dataset("HuggingFaceM4/VisDial", split='train',
#                                     cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/hf_visdial')
#         self.tokenizer = tokenizer
#         self.transform = transform
#
#     def _init_dataset(self):
#         ds = load_dataset("derek-thomas/ScienceQA",
#                           cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/ScienceQA')
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#
#         # 图像处理
#         image_path = item['image']  # 假设图片路径字段是"image"
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#
#         # 文本处理（问题和答案）
#         question = item['question']
#         answer = item['answer']
#
#         # 将问题和答案拼接成一对输入
#         input_text = f"question: {question} answer: {answer}"
#         encoding = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=512,
#                                   return_tensors="pt")
#
#         return {
#             'image': image,
#             'input_ids': encoding['input_ids'].squeeze(),  # 去除batch维度
#             'attention_mask': encoding['attention_mask'].squeeze(),
#             'answer': answer
#         }

if __name__ == '__main__':
    test_invig()
    # test_scienceqa()