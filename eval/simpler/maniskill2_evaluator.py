"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import re
import numpy as np
from transforms3d.euler import quat2euler
import cv2
from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from robovlms.data.openvla_datasets.rlds.utils.cot_utils import get_cot_database_keys
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
from PIL import Image, ImageDraw, ImageFont


def run_maniskill2_eval_single_episode(
        model,
        ckpt_path,
        robot_name,
        env_name,
        scene_name,
        model_name,
        robot_init_x,
        robot_init_y,
        robot_init_quat,
        control_mode,
        obj_init_x=None,
        obj_init_y=None,
        obj_episode_id=None,
        additional_env_build_kwargs=None,
        rgb_overlay_path=None,
        obs_camera_name=None,
        control_freq=3,
        sim_freq=513,
        max_episode_steps=80,
        instruction=None,
        enable_raytracing=True,
        additional_env_save_tags=None,
        logging_dir="./results_V2",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    # import pdb;pdb.set_trace()
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
    images = [image]
    if model.use_cot:
        images_cot = []
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset()

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        # import pdb; pdb.set_trace()

        if model.use_cot:
            raw_action, action, cot = model.step(image, task_description)
            images_cot.append(create_cot_img(image, cot))
        else:
            raw_action, action = model.step(image, task_description)

        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)

        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
            model.reset()

        is_final_subtask = env.is_final_subtask()

        if not is_final_subtask and info["episode_stats"].get("is_drawer_open", False):
            env.advance_to_next_subtask()

        print(timestep, done, truncated, info)

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"

    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    ckpt_path_basename = f"{model_name}_{ckpt_path_basename}"

    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    if model.use_cot:
        write_video(video_path, images_cot, fps=5)
    else:
        write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    model_name = args.model_name
    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    model_name=model_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                            args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr


def put_text_with_wrap(img, text, org, fontFace, fontScale, color, thickness, line_spacing=20, max_width=None):
    if max_width is None:
        max_width = img.shape[1]

    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        (test_width, _), _ = cv2.getTextSize(test_line, fontFace, fontScale, thickness)
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.rstrip())
            current_line = word + " "
    lines.append(current_line.rstrip())

    x, y = org
    line_height = int(cv2.getTextSize("A", fontFace, fontScale, thickness)[0][1])
    for line in lines:
        cv2.putText(img, line, (x, y), fontFace, fontScale, color, thickness)
        y += line_height + line_spacing
    return len(lines)


def create_cot_img(img, thought):
    text_img = (
            np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
    )
    # Split thought into multiple lines

    thought = thought[0]
    cot_tag = get_cot_database_keys()
    for tag in cot_tag.keys():
        thought = thought.replace(tag, f"\n{tag}")

    lines = thought.split("\n")
    # Add text lines
    start_y = 20
    line_spacing = 12
    for i, line in enumerate(lines):
        if len(line.split(" ")) < 2 or "ACTION" in line:
            continue

        num_lines = put_text_with_wrap(
            text_img,
            line,
            (10, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            line_spacing=line_spacing
        )
        line_height = int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][1])
        start_y += num_lines * (line_height + line_spacing)

        # draw bbox
        if "VISIBLE OBJECTS" in line:
            draw_bbox(line, img)
        if "GRIPPER POSITION" in line:
            draw_gripper(line, img)

    img = np.concatenate((img, text_img), axis=1)
    return img


def draw_bbox(visible_obj_str, image):
    pattern = r'(\w+(?: \w+)*) \[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, visible_obj_str)

    # 绘制检测框和标签
    for match in matches:
        object_name = match[0]
        x1, y1, x2, y2 = map(int, match[1:])
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # 添加标签
        cv2.putText(image, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    return image


def draw_gripper(gripper_pos_str, image):
    pattern = r'\s*\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]'
    gripper_match = re.search(pattern, gripper_pos_str)
    if gripper_match:
        gripper_list_str = gripper_match.group(1)
        gripper_list = [int(num) for num in gripper_list_str.split(',')]
        if len(gripper_list) % 2 != 0:
            gripper_list = gripper_list[:-1]
        points = [(gripper_list[i], gripper_list[i + 1]) for i in range(0, len(gripper_list), 2)]
        for point in points:
            cv2.circle(image, point, 2, (255, 0, 0), -1)

    return image
