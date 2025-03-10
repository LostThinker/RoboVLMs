import os
from logging.config import dictConfig
from typing import Any, Dict
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


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }


def make_tf_dict(raw_dict):
    print("Building the reasoning dict...")
    keys = []
    values = []

    def reasoning_dict_to_str(d):
        tags = get_cot_tags_list()[:-1]  # exclude ACTION
        database_keys = get_cot_database_keys()
        reasoning_parts = [(tag, d[database_keys[tag]]) for tag in tags]

        return "@".join(f"{tag}@{part}" for tag, part in reasoning_parts)

    has_reasoning = [0, 0]

    for file_name in raw_dict.keys():
        for episode_id in raw_dict[file_name].keys():
            if "reasoning" not in raw_dict[file_name][episode_id].keys():
                has_reasoning[0] += 1
                continue
            else:
                has_reasoning[1] += 1

            for i in raw_dict[file_name][episode_id]["reasoning"].keys():
                keys.append(file_name + "_" + str(episode_id) + "_" + i)
                reasoning_dict = raw_dict[file_name][episode_id]["reasoning"][i]

                gripper_lookahead_n = 5  # list this many future positions of the gripper
                trajectory_features = raw_dict[file_name][episode_id]["features"]

                reasoning_dict["gripper"] = ""
                if "gripper_position" in trajectory_features.keys():
                    if trajectory_features["gripper_position"] is not None:
                        if 0 <= int(i) < len(trajectory_features["gripper_position"]):
                            future_positions = []
                            for j in range(gripper_lookahead_n):
                                if int(i) + j < len(trajectory_features["gripper_position"]):
                                    future_positions += trajectory_features["gripper_position"][int(i) + j]
                                else:
                                    future_positions += future_positions[-2:]

                            reasoning_dict["gripper"] = str(future_positions)

                reasoning_dict["bboxes"] = ""
                if "bboxes" in trajectory_features.keys():
                    if trajectory_features["bboxes"] is not None:
                        if 0 <= int(i) < len(trajectory_features["bboxes"]):
                            if len(trajectory_features["bboxes"][int(i)]) > 0:
                                boxes_list = trajectory_features["bboxes"][int(i)]
                                reasoning_dict["bboxes"] = ", ".join(
                                    [f"{name} {box!s}" for prob, name, box in boxes_list]
                                )

                values.append(reasoning_dict_to_str(reasoning_dict))

    print("Example reasoning:", keys[0], values[0])
    print("Reasoning presence statistics [# has not, # has]:", has_reasoning)

    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value="")


def reasoning_dropout(reasoning: str, dropout_prob: float) -> Tuple[str, str]:
    """Dropout reasoning tokens with probability `dropout_prob`."""
    if len(reasoning) == 0:
        return reasoning, ""

    reasoning_parts = reasoning.split("@")
    tags = [(reasoning_parts[i], reasoning_parts[i + 1]) for i in range(0, len(reasoning_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    subset_string = (
            "[" + ", ".join([abbreviate_tag(tag) for (tag, _), is_taken in zip(tags, subset) if is_taken]) + "]"
    )  # abbreviation

    excluded_tags = []

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")

    return (
        " ".join(
            [f"{tag[0]} {tag[1]}" for tag, is_taken in zip(tags, subset) if (is_taken and tag[0] not in excluded_tags)]
        ),
        subset_string,
    )


@dataclass
class RLDSBatchTransform:
    model_name: str
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Callable[[List[Image.Image]], torch.Tensor]
    predict_stop_token: bool = True
    print_prompt_limit: int = 20
    reasoning_dropout_prob: float = 0.0

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        reasoning, subset = reasoning_dropout(rlds_batch["reasoning"], dropout_prob=self.reasoning_dropout_prob)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.base_tokenizer.eos_token, bos=self.base_tokenizer.bos_token
        )

        # prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            # {"from": "human", "value": f"What action should the robot take to {lang}? Explain why with {subset}."},
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": f"{reasoning} {CotTag.ACTION.value} "},
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

        self.batch_transform = RLDSBatchTransform(model_name, self.action_tokenizer, self.tokenizer, image_fn,
                                                  predict_stop_token=predict_stop_token)
        self.collater = PaddedCollatorForActionPrediction(self.tokenizer.model_max_length, self.tokenizer.pad_token_id,
                                                          padding_side=padding_side)

        with open(kwargs['reasoning_data_dir'], "r") as f:
            reasoning_dataset = json.load(f)
        self.reasoning_dataset = make_tf_dict(reasoning_dataset)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            reasoning_index = rlds_batch.pop('reasoning_index')
            reasoning_index = tf.constant(reasoning_index, dtype=tf.string)
            reasoning = self.reasoning_dataset.lookup(reasoning_index)
            reasoning = reasoning.numpy().decode('utf-8')
            rlds_batch["reasoning"] = reasoning

            yield self.batch_transform(rlds_batch)
