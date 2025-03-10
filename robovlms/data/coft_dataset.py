import os
import datasets
import random
import copy
import torch
import torchvision.transforms as T
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 10

from functools import partial
import numpy as np
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset, concatenate_datasets
from dataclasses import dataclass, field
from torchvision import transforms
from typing import Dict, Optional, Sequence, List
from robovlms.data.vid_llava_constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX
)
import robovlms.data.conversation as conversation_lib
from robovlms.data.data_utils import get_prompt_builder


def scienceqa_map_batch(batch):
    # 初始化用于存储处理后数据的列表
    conversations = []
    images = []

    # 遍历批处理中的每个样本
    for i in range(len(batch['question'])):
        image = batch['image'][i]
        if image is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = image.convert('RGB').resize((224, 224))
        images.append(image)

        question = batch['question'][i]
        choices = batch['choices'][i]
        solution = batch['solution'][i]
        hint = batch["hint"][i]
        lecture = batch["lecture"][i]
        answer = batch['answer'][i]

        prompt = f"Q: {question}\n"
        for j, choice in enumerate(choices):
            prompt += f"Choice: {chr(65 + j)}. {choice}"

        if hint:
            prompt += f"\nHint: {hint}"
        if lecture:
            prompt += f"\nLecture: {lecture}"

        target = f'{solution} The answer is {chr(65 + answer)}: {choices[answer]}'

        conversation = [
            {"from": 'human', "value": prompt},
            {"from": "gpt", "value": target},
        ]
        conversations.append(conversation)

    # 将处理后的数据更新到批处理字典中
    batch['image'] = images
    batch['conversations'] = conversations

    return batch


def invig_map_batch(batch):
    # 初始化存储处理后图像和对话的列表
    images = []
    all_conversations = []

    # 遍历批处理中的每个样本
    for i in range(len(batch['image'])):
        # 处理图像
        image = batch['image'][i]
        image = image.convert('RGB').resize((224, 224))
        images.append(image)

        # 处理对话
        dialog = batch['ref_list'][i][0]['dialog']
        conversations = []
        for d in dialog:
            conversations.append({"from": 'human', "value": d[0]})
            conversations.append({"from": "gpt", "value": d[1]})
        all_conversations.append(conversations)

    # 将处理后的数据更新到批处理字典中
    batch['image'] = images
    batch['conversations'] = all_conversations

    return batch


def visdial_map_batch(batch):
    # 用于存储处理后的图像
    processed_images = []
    # 用于存储处理后的对话列表
    all_conversations = []

    # 遍历批次中的每个样本
    for i in range(len(batch['image'])):
        # 处理图像
        image = batch['image'][i]
        image = image.convert('RGB').resize((224, 224))
        processed_images.append(image)

        # 提取对话信息
        dialog = batch['dialog'][i]
        conversations = []
        for d in dialog:
            conversations.append({"from": 'human', "value": d[0]})
            conversations.append({"from": "gpt", "value": d[1]})
        all_conversations.append(conversations)

    # 更新批次数据
    batch['image'] = processed_images
    batch['conversations'] = all_conversations

    return batch


DATASET_MAP = {
    'ScienceQA': scienceqa_map_batch,
    'invig': invig_map_batch,
    'VisDial': visdial_map_batch,
}


def build_vlm_dataset(dataset_name):
    map_fn = DATASET_MAP[dataset_name.split('/')[-1]]
    dataset = load_dataset(dataset_name).map(map_fn, batched=True, batch_size=64, num_proc=16).select_columns(
        ["image", "conversations"])

    print(f"{dataset_name} train_dataset length: {len(dataset['train'])}")
    print(f"{dataset_name} val_dataset length: {len(dataset['validation'])}")

    return dataset


def build_coft_dataset(dataset_name_list, model_name, tokenizer, image_processor):
    train_datasets = []
    val_datasets = []
    for data_name in dataset_name_list:
        dataset = build_vlm_dataset(data_name)
        train_datasets.append(dataset['train'])
        val_datasets.append(dataset['validation'])

    merge_train_dataset = concatenate_datasets(train_datasets).shuffle(seed=42)
    merge_val_dataset = concatenate_datasets(val_datasets).shuffle(seed=42)

    print(f"Co-train train_dataset length: {len(merge_train_dataset)}")
    print(f"Co-train val_dataset length: {len(merge_val_dataset)}")

    merge_train_dataset = CoTrainDataset(model_name, merge_train_dataset, tokenizer, image_processor)
    merge_val_dataset = CoTrainDataset(model_name, merge_val_dataset, tokenizer, image_processor)

    return merge_train_dataset, merge_val_dataset


class CoTrainDataset(Dataset):
    def __init__(
            self,
            model_name,
            data_dir,
            tokenizer,
            image_fn,
            shuffle_seed,
            split,
            **kwargs
    ):
        super(CoTrainDataset, self).__init__()

        self.model_name = model_name
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_processor = image_fn

        self.dataset = build_vlm_dataset(data_dir)[split]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # image = Image.fromarray(item["image"]).convert("RGB")
        image = self.image_processor([item['image']])

        conversations = item['conversations']
        input_ids, labels = self.preprocess_conversation(conversations)

        data_dict = dict(
            image=image,
            input_ids=input_ids,
            labels=labels
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
                prompt_builder.get_prompt(), add_special_tokens=True,
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

        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()

        for ignore_idx in target_ignore_idx:
            labels[ignore_idx[0]:ignore_idx[1]] = IGNORE_INDEX

        # replace <image> with IMAGE_TOKEN_INDEX
        image_tag_token = self.tokenizer.added_tokens_encoder[prompt_builder.default_image_token]
        input_ids[input_ids == image_tag_token] = IMAGE_TOKEN_INDEX
        labels[labels == image_tag_token] = IMAGE_TOKEN_INDEX

        return input_ids, labels

    def collater(self, sample):
        input_ids, labels, images = tuple(
            [s[key] for s in sample] for key in ("input_ids", "labels", "image")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # images = torch.concatenate(images, dim=0)
        images = torch.stack(images)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            images=images,
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

# @dataclass
# class DataCollatorForCoFTDataset(object):
#     tokenizer: transformers.PreTrainedTokenizer
#     model_name: str
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels, images = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "labels", "image")
#         )
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=IGNORE_INDEX
#         )
#
#         images = torch.concatenate(images, dim=0)
#
#         input_ids = input_ids[:, : self.tokenizer.model_max_length]
#         labels = labels[:, : self.tokenizer.model_max_length]
#
#         batch = dict(
#             images=images,
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )
#
#         batch["data_source"] = "vl_pretrain"
#
#         batch["rgb"] = batch.get("images", None)
#         batch["instr_and_action_ids"] = batch["input_ids"]
#         batch["instr_and_action_mask"] = batch["attention_mask"]
#         batch["instr_and_action_labels"] = batch["labels"]
#
#         batch["text"] = batch["input_ids"]
#         batch["text_mask"] = batch["attention_mask"]
#
#         return batch
