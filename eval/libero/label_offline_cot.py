import os
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
import datetime

import torch
import torch.distributed as dist
from torch.utils.data import SequentialSampler

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything

from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.datamodule.gr_datamodule import GRDataModule
from robovlms.data.data_utils import preprocess_image
from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from robovlms.utils.common import collate_with_none

CACHE_ROOT = "runs/eval"
os.makedirs(CACHE_ROOT, exist_ok=True)


def setup():
    dist.init_process_group(backend="nccl")
    os.environ["EGL_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_date_str():
    return str(datetime.date.today())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs["trainer"])
    # trainer_config["devices"] = configs.get("gpus", "auto")
    # trainer_config["num_nodes"] = configs.get("num_nodes", 1)

    trainer_config["devices"] = 1
    trainer_config["num_nodes"] = 1

    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    if "strategy" not in trainer_config or trainer_config["strategy"] == "ddp":
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)
    return trainer_config


def main():
    args = parser_args()
    config_path = args.config_path
    ckpt_dir = args.ckpt_dir
    ckpt_idx = args.ckpt_idx

    # Loading configs
    assert config_path is not None
    # Load config file
    configs = load_config(config_path)
    if args.is_pt_config:
        exp_name = os.path.basename(config_path)
        configs["exp_name"] = exp_name
        # generate ft configs from pt configs
        from robovlms.utils.config_utils import generate_calvin_ft_configs, deep_update

        ft_configs = generate_calvin_ft_configs(configs)
        deep_update(configs, ft_configs)

    # Get checkpoint path
    from robovlms.utils.eval_utils import sort_ckpt

    # print(ckpt_dir)
    if isinstance(ckpt_dir, list):
        ckpt_dir = ckpt_dir[0]
    if args.ckpt_path is None:
        ckpt_files, ckpt_steps = sort_ckpt(ckpt_dir)
        if ckpt_idx >= len(ckpt_files):
            exit(0)
        ckpt_path = ckpt_files[ckpt_idx]
        ckpt_step = ckpt_steps[ckpt_idx]
        ckpt_dir = os.path.dirname(ckpt_path)
    else:
        import copy

        ckpt_path = args.ckpt_path or copy.copy(ckpt_dir)
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_step = 0

    # Handle DeepSpeed ckpt
    if os.path.isdir(ckpt_path):
        target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
        print(f"converting {ckpt_path} to {target_ckpt_path}")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
        ckpt_path = target_ckpt_path

    from robovlms.utils.config_utils import get_exp_name

    eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode="eval")
    if args.no_cache:
        eval_log_dir = ckpt_dir
    else:
        eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    # TODO check if this code is needed
    ckpt_step = ckpt_path.split("/")[-1].split(".")[0]
    suffix = ""
    if args.data_split == "train":
        suffix += "_train"
    else:
        suffix += "_val"
    if args.use_model_cot:
        suffix += "_model_cot"
    result_path = os.path.join(
        eval_log_dir, f"offline_results_libero_{ckpt_step}{suffix}.json"
    )
    cache_file = os.path.join(
        eval_log_dir, f"offline_meta_info_step_{ckpt_step}{suffix}.json"
    )

    if os.path.exists(cache_file):
        # os.system(f"sudo rm {cache_file}")
        os.remove(cache_file)
    with open(cache_file, "w") as f:
        _info = {
            "eval_result_path": result_path,
            "eval_log_dir": eval_log_dir,
            "raw_config_path": configs.get("raw_config_path", None),
        }
        json.dump(_info, f, indent=2)

    trainer_config = init_trainer_config(configs)

    trainer_config["limit_val_batches"] = None  # validate all data
    trainer_config["logger"] = False  # disable logger
    trainer = Trainer(**trainer_config)
    configs["gpus"] = trainer.num_devices
    configs["train_setup"]["precision"] = configs["trainer"]["precision"]

    if configs["fwd_head"] is not None:
        configs["train_setup"]["predict_forward_hand"] = configs["fwd_head"].get(
            "pred_hand_image", False
        )

    if "model_path" in configs and not os.path.exists(configs["model_path"]):
        repo_name = configs["model_url"].split("/")[-1].split(".")[0]
        print(
            f"VLM backbone does not exist, cloning {configs['model']} from {configs['model_url']}..."
        )
        os.system(f"git clone {configs['model_url']} .vlms/{repo_name}")
        configs["model_path"] = ".vlms/" + repo_name
        configs["model_config"] = os.path.join(configs["model_path"], "config.json")

    if configs["model"] == "kosmos":
        import transformers

        package_dir = transformers.__path__[0]
        os.system(
            "cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(
                package_dir
            )
        )
        # Reload transformers module to use updated modeling file
        import importlib

        importlib.reload(transformers)

    model = BaseTrainer.from_checkpoint(ckpt_path, "lightning", configs).to(
        torch.device("cuda")
    )
    model.model.force_model_cot = args.use_model_cot

    image_preprocess = model.model.image_processor

    assert args.data_split in ["val", "train"]
    if args.data_split == "train":
        configs["val_dataset"]["data_dir"] = configs["train_dataset"]["data_dir"]

    datamodule = GRDataModule(
        configs["train_dataset"],
        configs["val_dataset"],
        args.batch_size,  # use batch size from args
        configs["num_workers"],
        tokenizer=model.model.tokenizer,
        tokenizer_config=configs["tokenizer"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        window_size=configs["window_size"],
        image_size=configs["image_size"],
        image_fn=functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        ),
        discrete=(
            False
            if configs["act_head"] is None
            else configs["act_head"].get("action_space", "continuous") == "discrete"
        ),
        discrete_action=(
            False
            if configs["act_head"] is None
            else configs["act_head"].get("action_space", "continuous") == "discrete"
        ),
        use_mu_law=configs.get("use_mu_law", False),
        mu_val=configs.get("mu_val", 255),
        n_bin=(
            256
            if configs["act_head"] is None
            else configs["act_head"].get("n_bin", 256)
        ),
        min_action=(
            -1
            if configs["act_head"] is None
            else configs["act_head"].get("min_action", -1)
        ),
        max_action=(
            1
            if configs["act_head"] is None
            else configs["act_head"].get("max_action", 1)
        ),
        discrete_action_history=configs.get("discrete_action_history", False),
        act_step=configs.get("fwd_pred_next_n", 1),
        norm_action=configs.get("norm_action", False),
        norm_min=configs.get("norm_min", -1),
        norm_max=configs.get("norm_max", 1),
        regular_action=configs.get("regular_action", False),
        x_mean=configs.get("x_mean", 0),
        x_std=configs.get("x_std", 1),
        weights=configs.get("train_weights", None),
        tcp_rel=configs.get("tcp_rel", False),
        model_name=configs.get("model", "flamingo"),
        use_cot=configs.get("use_cot", False),
        cot_tags=configs.get("cot_tags", None),
        use_cot_stage_token=configs.get("use_cot_stage_token", True),
        cot_file_name=configs.get("cot_file_name", "embodied_features_bridge.json"),
    )
    datamodule.val_dataloader()

    dataset = datamodule.val_datasets()
    sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=(
            dataset.collater if hasattr(dataset, "collater") else collate_with_none
        ),
    )


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def update_configs(configs, args):
    configs["raw_config_path"] = args["config"]
    configs["output_root"] = (
        Path(configs["output_root"]) / configs["model"] / configs["task_name"]
    )
    configs["log_root"] = (
        Path(configs["log_root"]) / configs["model"] / configs["task_name"]
    )
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    for k, v in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                # assert sub_k in configs[k], f"{sub_k} not in configs {k}"
                if sub_v is not None:
                    configs[k][sub_k] = sub_v
        else:
            if v is not None:
                configs[k] = v
    return configs


def parser_args():
    seed_everything(0, workers=True)  # type:ignore
    # Experiment
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    # yaml_path takes the highest priority, then the log_dir, finally the config_path
    parser.add_argument(
        "--config_path", type=str, default=None, help="path to the config file"
    )
    parser.add_argument(
        "--is_pt_config",
        action="store_true",
        help="whether the specified config path is a pretrain config file.",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_idx", type=int, default=-1, help="which ckpt is going to be evaluated"
    )
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--use_model_cot", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_split", type=str, default="val")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
