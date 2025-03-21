"""
Evaluate RoboVLMs-CoT via LIBERO Benchmark
"""

import traceback
import argparse
import json
import logging
from pathlib import Path
import time

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from robovlms.utils.config_utils import load_config

from model_wrapper import CustomLiberoModel
# from open_flamingo.train.distributed import world_info_from_env
from libero_utils import save_rollout_gif, get_libero_image, get_eposide_length
from libero_utils import get_libero_dummy_action, get_libero_env

from libero.libero import benchmark

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)

CACHE_ROOT = "runs/eval"
os.makedirs(CACHE_ROOT, exist_ok=True)


def setup():
    dist.init_process_group(backend="nccl")
    os.environ["EGL_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def world_info_from_env():

    local_rank = int(os.getenv("LOCAL_RANK", 0))  # 默认值为0
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))  # 默认总节点数为1
    
    return local_rank, rank, world_size


def evaluate(
    model,
    model_name,
    debug=False,
    resize_size=224,
    num_trials_per_task=10,
    num_steps_wait=10,
    local_log_dir=CACHE_ROOT,
    task_suite_name="libero_object",
):
    # Initialize Local Logging
    run_id = f"{task_suite_name}-{time.strftime('%Y-%m-%d_%H:%M')}"
    if model.no_action_ensemble:
        run_id += "_no_ae"
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    EP_LEN = get_eposide_length(task_suite_name)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks_in_suite):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)
        task_episodes, task_successes = 0, 0
        logger.info(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        for episode_idx in range(num_trials_per_task):
            env.reset()
            model.reset()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            if model.use_cot:
                thought = [""]

            # Start episodes
            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            action_counter = 0
            while t < EP_LEN + num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Get preprocessed image
                    obs_img = get_libero_image(obs, resize_size)
                    img = obs_img.copy()

                    if model.use_cot:
                        # Create a white background for the text
                        text_img = (
                            np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
                        )
                        # Split thought into multiple lines
                        lines = thought[0].replace("@", "\n").split("\n")
                        # Add text lines
                        for i, line in enumerate(lines):
                            cv2.putText(
                                text_img,
                                line,
                                (10, 30 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                            )
                        # Concatenate original image with text image
                        img = np.concatenate((img, text_img), axis=1)
                        # Save a sample image for debugging
                        # cv2.imwrite("sample_cot_image.png", img)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # observation = {
                    #     "agentview_image": img,
                    #     "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
                    #     "robot_obs": np.concatenate(
                    #         (
                    #             obs["robot0_eef_pos"],
                    #             quat2axisangle(obs["robot0_eef_quat"]),
                    #             obs["robot0_gripper_qpos"],
                    #         )
                    #     ),
                    # }

                    if action_counter == 0:
                        with torch.no_grad():
                            if model.use_cot:
                                action, thought = model.step(obs_img, task_description)
                            else:
                                action = model.step(obs_img, task_description)

                        if model.no_action_ensemble:
                            action_counter = action.shape[0]
                        else:
                            action_counter = 1

                    # logger.info(f"Action: {action.shape}")
                    if model.no_action_ensemble:
                        step_action = action[-action_counter]
                    else:
                        step_action = action[0]

                    obs, reward, done, info = env.step(step_action)
                    action_counter -= 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            logger.info(f"Num of Steps: {len(replay_images)}")
            if debug and len(replay_images) > 0:
                gif_dir = os.path.join(local_log_dir, "videos-{}".format(run_id))
                if not os.path.exists(gif_dir):
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = f"Episodes{total_episodes}_{str(done)}.gif"
                gif_path = os.path.join(gif_dir, gif_path)
                save_rollout_gif(replay_images, gif_path, fps=15)

            # Log current results
            logger.info(f"Success: {done}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        logger.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logger.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    log_file.close()


def parser_args():
    seed_everything(0, workers=True)  # type:ignore
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
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="select evaluate LIBREO TASK SUITE",
    )
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--no_nccl", action="store_true")
    parser.add_argument("--no_action_ensemble", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    if not args.no_nccl:
        setup()
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
        if args.no_nccl or dist.get_rank() == 0:
            print(f"converting {ckpt_path} to {target_ckpt_path}")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)

        if not args.no_nccl:
            dist.barrier()
        ckpt_path = target_ckpt_path

    from robovlms.utils.config_utils import get_exp_name

    eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode="eval")
    if args.no_cache:
        eval_log_dir = ckpt_dir
    else:
        eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    model = CustomLiberoModel(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        debug=args.debug_model,
        no_action_ensemble=args.no_action_ensemble,
    )

    # TODO check if this code is needed
    ckpt_step = ckpt_path.split("/")[-1].split(".")[0]
    if args.no_action_ensemble:
        suffix = "_no_ae"
    else:
        suffix = ""
    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin_{ckpt_step}{suffix}.txt")
    result_path = os.path.join(
        eval_log_dir, f"results_calvin_{ckpt_step}{suffix}_rand-{args.rank}.json"
    )
    cache_file = os.path.join(eval_log_dir, f"meta_info_step_{ckpt_step}{suffix}.json")

    if not args.no_cache and args.local_rank == 0:
        if os.path.exists(cache_file):
            # os.system(f"sudo rm {cache_file}")
            os.remove(cache_file)
        with open(cache_file, "w") as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
                "raw_config_path": configs.get("raw_config_path", None),
            }
            json.dump(_info, f, indent=2)

    evaluate(
        model,
        task_suite_name=args.task_suite_name,
        local_log_dir=eval_log_dir,
        debug=args.debug,
        model_name=configs.get("model"),
    )

    if args.no_cache and args.local_rank == 0:
        # os.system("sudo rm -r ./temp/")
        import shutil

        shutil.rmtree("./temp/", ignore_errors=True)

    if not args.no_nccl:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()
