import os
from logging.config import dictConfig
from typing import Any, Dict, Union
import numpy as np
from robovlms.data.base_openvla_dataset import RLDSDataset
from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset
import json
from pathlib import Path
import tensorflow as tf
import enum
from robovlms.model.policy_head.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from robovlms.data.data_utils import get_prompt_builder
from PIL import Image
import torch
from typing import Callable, Dict, Sequence, Tuple, List
from robovlms.data.vid_llava_constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN
)


@dataclass
class RLDSBatchTransform:
    model_name: str
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Callable[[List[Image.Image]], torch.Tensor]
    predict_stop_token: bool = True
    print_prompt_limit: int = 20
    cot_dropout: float = 0.0
    cot_tags: List[str] = None
    use_cot_stage_token: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        reasoning = self._build_reasoning_prompt(rlds_batch["reasoning"].decode())
        # reasoning, subset = reasoning_dropout(rlds_batch["reasoning"].decode(), dropout_prob=self.cot_dropout)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.base_tokenizer.eos_token, bos=self.base_tokenizer.bos_token
        )

        # prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            # {"from": "human", "value": f"What action should the robot take to {lang}? Explain why with {subset}."},
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            # {"from": "gpt", "value": f"{reasoning} {CotTag.ACTION.value} "},
            {"from": "gpt", "value": f"{reasoning} "},
        ]

        action_token = torch.tensor(self.action_tokenizer.encode_actions_to_token_ids(action))

        cur_len = 0
        target_ignore_idx = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True,
            ).input_ids
            total_len = len(input_ids)

            if turn["from"] == 'human':
                target_ignore_idx.append([cur_len, total_len])

            cur_len = total_len

        # if self.print_prompt_limit > 0:
        #     print("Conversation:", conversation)
        #     p = prompt_builder.get_prompt()
        #     print("Prompt:", p)
        #
        #     self.print_prompt_limit -= 1

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)

        labels = input_ids.clone()
        for ignore_idx in target_ignore_idx:
            labels[ignore_idx[0]:ignore_idx[1]] = IGNORE_INDEX

        # Explicitly adding action token
        input_ids = torch.cat([input_ids[:-1], action_token, input_ids[-1:]])
        labels = torch.cat([labels[:-1], action_token, labels[-1:]])

        pixel_values = self.image_transform([img])

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # replace <image> with IMAGE_TOKEN_INDEX
        image_tag_token = self.base_tokenizer.added_tokens_encoder[prompt_builder.default_image_token]
        input_ids[input_ids == image_tag_token] = IMAGE_TOKEN_INDEX
        labels[labels == image_tag_token] = IMAGE_TOKEN_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)

    def _build_reasoning_prompt(self, reasoning: str):
        reasoning = reasoning.split("@")
        reasoning_dict = {}
        for k, v in zip(reasoning[::2], reasoning[1::2]):
            reasoning_dict[k] = v

        reasoning_str = ""
        # either no reasoning or all reasoning tags exist
        if len(reasoning_dict) > 0:
            if isinstance(self.cot_dropout, float):
                dropout = [np.random.rand() < self.cot_dropout] * len(self.cot_tags)
            else:
                dropout = [np.random.rand() < d for d in self.cot_dropout]

            for tag, drop in zip(self.cot_tags, dropout):
                if drop:
                    continue
                thought = reasoning_dict[tag].strip()
                if len(thought) == 0:
                    thought = "None."
                reasoning_str += f"{tag.upper()}: {thought}"
                if self.use_cot_stage_token:
                    reasoning_str += "<|cotstage|>"
                else:
                    reasoning_str += " "
        reasoning_str += "ACTION:"
        return reasoning_str


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        batch = dict(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        batch["data_source"] = "vl_pretrain"

        batch["rgb"] = batch.get("images", None)
        batch["instr_and_action_ids"] = batch["input_ids"]
        batch["instr_and_action_mask"] = batch["attention_mask"]
        batch["instr_and_action_labels"] = batch["labels"]

        batch["text"] = batch["input_ids"]
        batch["text_mask"] = batch["attention_mask"]

        if dataset_names is not None:
            batch["dataset_names"] = dataset_names
        return batch


class ECoTDataset(RLDSDataset):
    def __init__(
            self,
            data_root_dir: Path,
            model_name: str,
            data_mix: str,
            image_size: int,
            image_fn: Callable[[List[Image.Image]], torch.Tensor],
            tokenizer: PreTrainedTokenizerBase,
            padding_side: str = "right",
            predict_stop_token: bool = True,
            cot_tags: List[str] = [],
            use_cot_stage_token: bool = False,
            cot_dropout: Union[float, List[float]] = 0.0,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        RLDSDataset.__init__(self, data_root_dir, data_mix, image_size, **kwargs)

        self.action_tokenizer = ActionTokenizer(
            self.tokenizer,
            bins=kwargs['n_bin'],
            min_action=kwargs['min_action'],
            max_action=kwargs['max_action'],
        )

        self.batch_transform = RLDSBatchTransform(
            model_name,
            self.action_tokenizer,
            self.tokenizer,
            image_fn,
            predict_stop_token=predict_stop_token,
            cot_dropout=cot_dropout,
            cot_tags=cot_tags,
            use_cot_stage_token=use_cot_stage_token

        )
        self.collater = PaddedCollatorForActionPrediction(self.tokenizer.model_max_length, self.tokenizer.pad_token_id,
                                                          padding_side=padding_side)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            yield self.batch_transform(rlds_batch)
