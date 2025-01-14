import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from pathlib import Path
from typing import List, Union
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import simpler_env
import sapien.core as sapien
from data_generation.chat_with_vlm import Chatbot

sapien.render_config.rt_use_denoiser = True


def make_env(task_name):
    env = simpler_env.make(task_name)
    return env


class SimplerEnv:
    def __init__(self, task_name):
        self.env = make_env(task_name)
        self.instruction = self.env.get_language_instruction()

    def reset(self):
        observation, reset_info = self.env.reset()
        self.instruction = self.env.get_language_instruction()

        obs = self._get_obs(observation)

        return obs, reset_info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)

        obs = self._get_obs(observation)

        return obs, reward, done, truncated, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def _get_obs(self, observation):
        img = self._get_image(observation)
        proprio = self._get_proprio(observation)

        obs = {'instruction': self.instruction}
        obs.update(img)
        obs.update(proprio)

        return obs

    def _get_proprio(self, obs):
        qpos_vel = np.concatenate([obs['agent']['qpos'], obs['agent']['qvel']])
        base_pose = obs['agent']['base_pose']
        tcp_pose = obs['extra']['tcp_pose']

        proprio = {
            'base_pose': base_pose,
            'tcp_pose': tcp_pose
        }

        return proprio

    def _get_image(self, obs):
        base_img = obs['image']['base_camera']['rgb']
        overhead_img = obs['image']['overhead_camera']['rgb']

        img = {
            'overhead_img': overhead_img
        }

        return img


class DialSimplerEnv(SimplerEnv):
    def __init__(self, task_name, device_id=0):
        super().__init__(task_name)

        self.virtual_human = Chatbot(device_id=device_id)
        self.prompt = None
        self.history = None
        self.current_step_info = None

        self._set_prompt()

    def reset(self):
        obs, info = super().reset()
        self.history = None

        human_response = self.dialog(obs['overhead_img'])
        # format: [human_response, robot_response]
        obs['response'] = human_response

        self.current_step_info = [obs, 0.0, False, False, info]

        return obs

    def step(self, action, robot_response):
        if action is not None:
            obs, reward, done, truncated, info = super().step(action)
            # here we can add new instruction during executions
            obs['response'] = None

            self.current_step_info = [obs, reward, done, truncated, info]

        elif robot_response is not None:
            obs, reward, done, truncated, info = self.current_step_info
            human_response = self.dialog(obs['overhead_img'], robot_response)
            obs['response'] = human_response

            self.current_step_info[0] = obs

        else:
            raise ValueError(f"Invalid action:{action}, robot_response:{robot_response}")

        return obs, reward, done, truncated, info

    def dialog(self, image, response=None):
        if response is None:
            text = self.prompt
        else:
            text = response

        response, self.history = self.virtual_human.chat_history(image, text, self.history)

        return response

    def _set_prompt(self):
        self.prompt = f"<image>\nThis is a robot manipulation scenario. You now need to play the role of a user who is dialoguing with a robot in order to make the robot fulfill your needs. You have a task, {self.instruction}, that you need the robot to accomplish, but you are not able to tell the robot this instruction directly. You need to tell the robot what you want about the task so that the robot can ask you to confirm the instruction to be performed. Your output needs to start with “User:”. If the manipulation described by the robot meets your correct intent, please respond positively. Now you need to initiate a dialog and answer the robot's query. "

        # If the robot has asked for your correct intent, reply in the affirmative and add a specific flag “<execute>” at the end of the reply

        self.prompt = f"<image>\nThis is a robot manipulation scenario. Now, you need to play as a user. You need the robot to accomplish a task ({self.instruction}), but you cannot tell the robot this instruction directly. You need to build a natural, real-life vague requirement around this instruction. You need to make your need progressively clearer to the robot by dialoguing with it instead of telling it directly. Your output needs to start with “User:”. If the robot's answer matches your correct intent, respond in the affirmative. Now, you need to start the dialog and wait for the robot's response."


class BaseDialPolicy:
    def __init__(self, vlm_device_id=0, vla_device_id=1):
        self.vlm = Chatbot(device_id=vlm_device_id)
        self.history = None
        self.current_instruction = None

        self.vla_processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(f"cuda:{vla_device_id}")
        self.vla_device_id = vla_device_id

        self._set_prompt()
        self.init()

    def init(self):
        self.history = None
        self.first_step = True
        self.current_instruction = None

    def get_action(self, obs):
        img = obs['overhead_img']
        human_response = obs['response']

        if human_response is not None:
            if self.first_step:
                text = f"{self.prompt}\n{human_response}"
                self.first_step = False
            else:
                text = human_response

            robot_response, self.history = self.vlm.chat_history(img, text, self.history)

            if '<instruction>' in robot_response:
                instruction = self.get_instruction(robot_response)
                self.current_instruction = instruction
                print(f"start executing instruction: {instruction}")
                action = self.execute_instruction(img, instruction)
            else:
                action = None

            return action, robot_response
        else:
            assert self.current_instruction is not None

            action = self.execute_instruction(img, self.current_instruction)
            robot_response = None
            return action, robot_response

    def execute_instruction(self, image, instruction):
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        image = Image.fromarray(image)
        # Predict Action (7-DoF; un-normalize for BridgeV2)
        inputs = self.vla_processor(prompt, image).to(f"cuda:{self.vla_device_id}", dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        return action

    def _set_prompt(self):
        self.prompt = f"<image>\nThis is a robot manipulation scenario. You now need to play the role of a robotic assistant that performs actions according to the user's needs. If you are not sure what the user means, you need to ask him or her to clarify what operation needs to be performed. Your output should start with “Robot:”. Once you have clarified the user's intent, you need to summarize an instruction to call the executable program. When outputting instructions, you need to first output a special tag <instruction> followed by the summarized instruction. The summarized instruction should start with a verb and be clear and concise, such as put something somewhere. Now start the conversation with the user."

    def get_instruction(self, text):
        instruction = text.split("<instruction>")[-1]
        if '</instruction>' in instruction:
            instruction = instruction.replace("</instruction>", "")

        return instruction


def test_interaction():
    env = DialSimplerEnv("google_robot_pick_coke_can", device_id=2)
    policy = BaseDialPolicy(6, 7)
    obs = env.reset()
    print(f"ground truth task: {env.instruction}")
    print("human_response: ", obs['response'])

    done = False
    # for i in range(10):
    while not done:
        action, robot_response = policy.get_action(obs)
        print("robot_response: ", robot_response)
        obs, reward, done, truncated, info = env.step(action, robot_response)
        done = (done or truncated)
        human_response = obs['response']
        print("human_response: ", human_response)


if __name__ == '__main__':
    test_interaction()

    # task_name = "google_robot_pick_coke_can"
    # env = SimplerEnv(task_name)
    #
    # obs = env.reset()
    #
    # print(obs['instruction'])
    # # plt.imshow(obs['base_img'])
    # # plt.show()
    # plt.imshow(obs['overhead_img'])
    # plt.show()
    #
    # done, truncated = False, False
    # while not (done or truncated):
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
