import os
import datasets
import random
import copy
import torch
import torchvision.transforms as T
from PIL import Image
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


def scienceqa_map(sample):
    image = sample['image']
    if image is None:
        image = Image.new('RGB', (224, 224), (0, 0, 0))
        sample['image'] = image

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


def visdial_map(sample):
    image = sample['image']
    caption = sample['caption']
    dialog = sample['dialog']

    conversations = []
    for d in dialog:
        conversations.append({"from": 'human', "value": d[0]})
        conversations.append({"from": "gpt", "value": d[1]})

    sample['conversations'] = conversations

    return sample


DATASET_MAP = {
    'ScienceQA': scienceqa_map,
    'invig': invig_map,
    'VisDial': visdial_map,
}


def build_vlm_dataset(dataset_name):
    map_fn = DATASET_MAP[dataset_name.split('/')[-1]]
    dataset = load_dataset(dataset_name).map(map_fn)

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

    merge_train_dataset = CoFTDataset(model_name, merge_train_dataset, tokenizer, image_processor)
    merge_val_dataset = CoFTDataset(model_name, merge_val_dataset, tokenizer, image_processor)

    return merge_train_dataset, merge_val_dataset


def get_coft_dataset(model_name, tokenizer, image_processor):
    sqa_ds = load_dataset("derek-thomas/ScienceQA",
                          cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/ScienceQA').map(
        scienceqa_map).select_columns(["image", "conversations"])
    invig_ds = load_dataset("jxu124/invig",
                            cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/invig').map(
        invig_map).select_columns(["image", "conversations"])
    ds = load_dataset("HuggingFaceM4/VisDial",
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/hf_visdial').map(
        visdial_map).select_columns(["image", "conversations"])
    train_dataset = concatenate_datasets([sqa_ds['train'], invig_ds['train']])
    val_dataset = concatenate_datasets([sqa_ds['validation'], invig_ds['validation']])

    train_dataset = CoFTDataset(model_name, train_dataset, tokenizer, image_processor)
    val_dataset = CoFTDataset(model_name, val_dataset, tokenizer, image_processor)

    return train_dataset, val_dataset


class CoFTDataset(Dataset):
    def __init__(
            self,
            model_name,
            dataset,
            tokenizer,
            image_processor
    ):
        super(CoFTDataset, self).__init__()

        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # image = Image.fromarray(item["image"]).convert("RGB")
        image = self.image_processor(item['image'])

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

        return input_ids, labels


@dataclass
class DataCollatorForCoFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_name: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "image")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        images = torch.stack(images, dim=0)

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
