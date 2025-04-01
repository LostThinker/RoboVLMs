from typing import Any, Dict
import itertools
import torch.distributed as dist
import torch.utils.data as data

from robovlms.data.base_openvla_dataset import RLDSDataset
from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset
import re


def extract_gripper_list(input_string):
    pattern = r'@gripper@(\[.*?\])@'
    match = re.search(pattern, input_string)
    if match:
        list_str = match.group(1)
        num_list = [int(num) for num in list_str.strip('[]').split(',')]
        return num_list
    return None


def replace_gripper_list(input_string, new_list):
    pattern = r'@gripper@\[.*?\]@'
    new_list_str = str(new_list)
    replacement = f'@gripper@{new_list_str}@'
    output_string = re.sub(pattern, replacement, input_string)
    return output_string


class OpenVLADataset(ActionPredictionDataset, RLDSDataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ActionPredictionDataset.__init__(self, **kwargs)
        if self.organize_type == "interleave":
            kwargs["window_sample"] = "sliding"
            kwargs["left_pad"] = False
        elif self.organize_type == "segment":
            kwargs["window_sample"] = "range"
            kwargs["left_pad"] = True
        else:
            raise ValueError("organize type must be interleave or segment")
        kwargs["chunk_action"] = True
        RLDSDataset.__init__(self, **kwargs)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            if "reasoning" in rlds_batch:
                rlds_batch = self._updata_fwd_gripper_pose(rlds_batch)

            yield self.batch_transform(
                task_description=rlds_batch["task"]["language_instruction"].decode(),
                action=rlds_batch["action"],
                episode_mask=rlds_batch["chunk_mask"],
                images=rlds_batch["observation"]["image_primary"],
                reasoning=(
                    rlds_batch["reasoning"].decode()
                    if "reasoning" in rlds_batch
                    else None
                ),
                gripper_images=None,
                step_ids=rlds_batch["step_ids"],
            )

    def _updata_fwd_gripper_pose(self, rlds_batch):
        reasoning_chunk = rlds_batch['observation']['reasoning']
        reasoning_chunk = [reason.decode() for reason in reasoning_chunk]

        if all(item == "" for item in reasoning_chunk):
            return rlds_batch

        chunk_mask = rlds_batch['chunk_mask']
        curr_reasoning = rlds_batch['reasoning'].decode()
        curr_idx = self.window_size - 1
        assert curr_reasoning == reasoning_chunk[curr_idx]
        if self.fwd_pred_next_n > 5:
            new_gripper_pose = []
            for i in range(0, self.fwd_pred_next_n - 5):
                idx = min(i + 1 + curr_idx, len(reasoning_chunk) - 1)
                fwd_reason = reasoning_chunk[idx]
                fwd_gripper_pose = extract_gripper_list(fwd_reason)[-2:]
                new_gripper_pose.extend(fwd_gripper_pose)

            curr_gripper_pose = extract_gripper_list(curr_reasoning)
            curr_gripper_pose.extend(new_gripper_pose)
            curr_reasoning = replace_gripper_list(curr_reasoning, curr_gripper_pose)

        rlds_batch['reasoning'] = curr_reasoning.encode()

        return rlds_batch


class OpenVLADatasetByRank(OpenVLADataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # self._origin_iterator = RLDSDataset.__iter__(self)

    def __len__(self):
        return OpenVLADataset.__len__(self) // self.world_size

    def __iter__(self) -> Dict[str, Any]:
        worker_info = data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        iterator = itertools.islice(RLDSDataset.__iter__(self), global_worker_id, None, total_workers)
        for rlds_batch in iterator:
            yield self.batch_transform(
                task_description=rlds_batch["task"]["language_instruction"].decode(),
                action=rlds_batch["action"],
                episode_mask=rlds_batch["chunk_mask"],
                images=rlds_batch["observation"]["image_primary"],
                reasoning=(
                    rlds_batch["reasoning"].decode()
                    if "reasoning" in rlds_batch
                    else None
                ),
                gripper_images=None,
                step_ids=rlds_batch["step_ids"],
            )
