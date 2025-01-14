from env_wrapper import DialSimplerEnv, SimplerEnv
from policy import BaseDialPolicy, VLAPolicy
import matplotlib.pyplot as plt
import simpler_env
from simpler_env.utils.visualization import write_video
import os


def test_dial_interaction(task_name='google_robot_pick_coke_can', episode=1):
    env = DialSimplerEnv(task_name, device_id=2)
    policy = BaseDialPolicy(6, 7, robot_type=env.robot_type)

    log_dir = f'/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/benchmark/dial_simpler_env/log/dial_eval/{task_name}'
    os.makedirs(log_dir, exist_ok=True)

    for i in range(episode):
        policy.reset()
        obs = env.reset(seed=i)
        instruction = env.instruction
        human_response = obs['response']

        dialog = [f'Ground truth task instruction: {instruction}']
        images = [obs['image']]

        # print(f"ground truth task: {env.instruction}")
        # print("human_response: ", obs['response'])
        # plt.imshow(obs['image'])
        # plt.show()

        truncated = False
        success = "failure"
        # for i in range(10):
        while not truncated:
            action, robot_response = policy.get_action(obs)

            if human_response is not None and robot_response is not None:
                dialog.extend([human_response, robot_response])

            # print("robot_response: ", robot_response)
            obs, reward, done, truncated, info = env.step(action, robot_response)

            human_response = obs['response']
            # print("human_response: ", human_response)

            images.append(obs['image'])
            success = "success" if done else "failure"

            # if action is not None:
            #     print(env.instruction)
            #     print(policy.current_instruction)
            #     break

        dialog.append(f'Task execution {success}')

        dial_save_path = os.path.join(log_dir, f'dialog-episode-{i}-{success}.txt')
        video_save_path = os.path.join(log_dir, f'video-episode-{i}-{success}.mp4')

        # save dialog
        with open(dial_save_path, "w") as f:
            for dial in dialog:
                f.write(f"{dial}\n")

        write_video(video_save_path, images, fps=5)


def test_orig_vla(task_name='google_robot_pick_coke_can', episode=1):
    env = SimplerEnv(task_name)
    policy = VLAPolicy("openvla/openvla-7b", robot_type=env.robot_type, device_id=5)

    log_dir = f'/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/benchmark/dial_simpler_env/log/orig_eval/{task_name}'
    os.makedirs(log_dir, exist_ok=True)

    for i in range(episode):
        policy.reset()
        obs, reset_info = env.reset(seed=i)
        instruction = env.instruction

        images = [obs['image']]

        truncated = False
        success = "failure"

        while not truncated:
            action = policy.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            success = "success" if done else "failure"
            images.append(obs['image'])

        instruction = instruction.replace(' ', '-')
        video_save_path = os.path.join(log_dir, f'{instruction}-episode-{i}-{success}.mp4')
        write_video(video_save_path, images, fps=5)


if __name__ == '__main__':
    for task in simpler_env.ENVIRONMENTS:
        # test_dial_interaction(task, episode=10)
        test_orig_vla(task, episode=10)

    #

    # test_dial_interaction(episode=5)
