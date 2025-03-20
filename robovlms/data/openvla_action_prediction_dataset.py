from typing import Any, Dict
import itertools
import torch.distributed as dist
import torch.utils.data as data

from robovlms.data.base_openvla_dataset import RLDSDataset
from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset


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
