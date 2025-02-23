import copy
from copy import deepcopy
import os

import lightning.pytorch as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader, Dataset
import robovlms
from robovlms.data.weighted_combined_loader import WeightedCombinedLoader
import robovlms.data.samplers as gr_samplers
from robovlms.utils.dist_train import get_rank, is_dist
from robovlms.utils.common import collate_with_none
from robovlms.data.coft_dataset import DataCollatorForCoFTDataset, build_coft_dataset


class CoFTCombinedDataset(Dataset):
    def __init__(self, vlm_dataset, robot_dataset):
        self.vlm_dataset = vlm_dataset
        self.robot_dataset = robot_dataset

    def __len__(self):
        # 返回数据集的最小长度，确保两个数据集的样本数量匹配
        return min(len(self.vlm_dataset), len(self.robot_dataset))

    def __getitem__(self, idx):
        # 从两个数据集中获取数据，并返回它们的组合
        vlm_sample = self.vlm_dataset[idx]
        robot_sample = self.robot_dataset[idx]
        return vlm_sample, robot_sample


class CoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset,
            val_dataset,
            coft_dataset,
            batch_size,
            num_workers,
            data_root="",
            **kwargs,
    ):
        super().__init__()
        self.train_dataset_config = train_dataset
        self.val_dataset_config = val_dataset
        self.coft_dataset_config = coft_dataset
        self._train_datasets = []
        self._val_datasets = []
        self._train_loader = None
        self._val_loader = None
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def _check_data_path(self, data_cfg):
        print(self.data_root)
        if data_cfg["type"] == "ConcatDataset":
            data_cfg["datasets"] = [
                self._check_data_path(d) for d in data_cfg["datasets"]
            ]
        elif "data_dir" in data_cfg and not os.path.isabs(data_cfg["data_dir"]):
            data_cfg["data_dir"] = os.path.join(self.data_root, data_cfg["data_dir"])
        return data_cfg

    def _init_iterable_dataset(
            self, dataset_config, batch_size, num_workers, is_training=True
    ):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)

        datset_type = dataset_config.pop("type")
        # assert datset_type in {'ImageTextDataset', 'RTXDataset', 'VideoLLaVADataset'}

        dataset_config.update(self.kwargs)
        dataset_config["is_training"] = is_training

        dataset = getattr(robovlms.data, datset_type)(**dataset_config)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            # num_workers=4,
            drop_last=True,
            collate_fn=(
                dataset.collater if hasattr(dataset, "collater") else collate_with_none
            ),
            # pin_memory=True
        )

        # from robovlms.data.weighted_combined_loader import WeightedCombinedLoader
        # data_loader = WeightedCombinedLoader(
        # [data_loader], mode="max_size_cycle", weights=[1])

        return dataset, data_loader

    def _init_datasets(self, dataset_config, is_training, batch_size, num_workers):
        if isinstance(dataset_config, dict):
            if get_rank() == 0:
                print("=" * 40)
                print("Initializing dataloader from config:")
                for k in dataset_config:
                    print(f"{k}: {dataset_config[k]}")
                print(f"is_training: {is_training}")
                print(f"batch_size: {batch_size}")
                print(f"num_workers: {num_workers}")
            dataset_type = dataset_config["type"]
            assert isinstance(batch_size, int)
            assert isinstance(num_workers, int)
            if (
                    dataset_type in {"ImageTextDataset", "RTXDataset"}
                    or "OpenVLA" in dataset_type
            ):
                # if dataset_type in {'ImageTextDataset', 'RTXDataset'}:
                return self._init_iterable_dataset(
                    dataset_config,
                    is_training=is_training,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            else:
                return self._init_dataset(
                    dataset_config,
                    is_training=is_training,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
        else:
            assert isinstance(dataset_config, list)
            all_sets_and_loaders = []
            assert isinstance(batch_size, (tuple, list)) and len(batch_size) == len(
                dataset_config
            )
            assert isinstance(num_workers, (tuple, list)) and len(num_workers) == len(
                dataset_config
            )
            for i, config in enumerate(dataset_config):
                all_sets_and_loaders.append(
                    self._init_datasets(
                        config,
                        is_training=is_training,
                        batch_size=batch_size[i],
                        num_workers=num_workers[i],
                    )
                )
            datasets, dataloaders = zip(*all_sets_and_loaders)
            if is_training:
                combined_dataloader = WeightedCombinedLoader(
                    dataloaders,
                    "max_size_cycle",
                    weights=self.kwargs.get("weights", None),
                )
                # combined_dataloader = CombinedLoader(dataloaders, "max_size_cycle")
                return datasets, combined_dataloader
            else:
                return datasets, dataloaders

    def _init_dataset_params(self, is_training, param_name="batch_size"):
        param = getattr(self, param_name)
        if not is_training:
            # setting for val datasets
            if isinstance(param, (tuple, list)):
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param[0]] * len(self.val_dataset_config)
                else:
                    param = param[0]
            else:
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param] * len(self.val_dataset_config)
                else:
                    param = param
        else:
            if isinstance(param, int):
                if isinstance(self.train_dataset_config, (tuple, list)):
                    param = [param] * len(self.train_dataset_config)
            elif isinstance(param, (tuple, list)):
                assert isinstance(self.train_dataset_config, (tuple, list)) and len(
                    self.train_dataset_config
                ) == len(param)
        return param

    def initialize(self, mode="train"):
        self.initialize_vla_data(mode)
        self.initialize_vlm_data(mode)

    def initialize_vla_data(self, mode="train"):
        if mode == "train":
            batch_size = self._init_dataset_params(True, "batch_size")
            num_workers = self._init_dataset_params(True, "num_workers")
            self._vla_train_datasets, self._vla_train_loader = self._init_datasets(
                self.train_dataset_config, True, batch_size, num_workers
            )

        elif mode == "val":
            batch_size = self._init_dataset_params(False, "batch_size")
            num_workers = self._init_dataset_params(False, "num_workers")
            self._vla_val_datasets, self._vla_val_loader = self._init_datasets(
                self.val_dataset_config, False, batch_size, num_workers
            )

    def initialize_vlm_data(self, mode="train"):
        self._vlm_train_datasets, self._vlm_val_datasets = build_coft_dataset(self.coft_dataset_config['data_names'],
                                                                              self.kwargs['model_name'],
                                                                              self.kwargs['tokenizer'],
                                                                              self.kwargs['image_preprocess'])
        collate_fn = DataCollatorForCoFTDataset(self.kwargs['tokenizer'], self.kwargs['model_name'])
        if mode == "train":
            batch_size = self._init_dataset_params(True, "batch_size")
            num_workers = self._init_dataset_params(True, "num_workers")
            self._vlm_train_loader = torch.utils.data.DataLoader(
                self._vlm_train_datasets,
                batch_size=batch_size,
                # num_workers=4,
                drop_last=True,
                collate_fn=collate_fn,
                # pin_memory=True
            )

        elif mode == "val":
            batch_size = self._init_dataset_params(False, "batch_size")
            num_workers = self._init_dataset_params(False, "num_workers")
            self._vlm_val_loader = torch.utils.data.DataLoader(
                self._vlm_val_datasets,
                batch_size=batch_size,
                # num_workers=4,
                drop_last=True,
                collate_fn=collate_fn,
                # pin_memory=True
            )

    def train_datasets(self):
        return self._vlm_train_datasets, self._vla_train_datasets

    def val_datasets(self):
        return self._vlm_val_datasets, self._vlm_val_datasets

    def train_dataloader(self):
        self.initialize("train")

        combined_dataloader = WeightedCombinedLoader(
            [self._vlm_train_loader, self._vla_train_loader],
            "max_size_cycle",
            weights=self.coft_dataset_config['vlm_robot_weights']
        )

        return combined_dataloader

    def val_dataloader(self):
        self.initialize("val")
        combined_dataloader = WeightedCombinedLoader(
            [self._vlm_val_loader, self._vla_val_loader],
            "max_size_cycle",
            weights=self.coft_dataset_config['vlm_robot_weights']
        )

        return combined_dataloader
