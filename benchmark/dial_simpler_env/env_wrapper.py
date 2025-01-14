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
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
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

        if "google_robot" in task_name:
            self.robot_type = 'google_robot'
        elif "widowx" in task_name:
            self.robot_type = 'widowx'
        else:
            raise ValueError(task_name)

    def reset(self, seed=0):
        observation, reset_info = self.env.reset(seed=seed)
        self.instruction = self.env.get_language_instruction()

        obs = self._get_obs(observation)

        return obs, reset_info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        new_instruction = self.env.get_language_instruction()
        if new_instruction != self.instruction:
            # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
            self.instruction = new_instruction
            print("New Instruction", new_instruction)

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
        image = get_image_from_maniskill2_obs_dict(self.env, obs)

        img = {
            'image': image
        }

        return img


class DialSimplerEnv(SimplerEnv):
    def __init__(self, task_name, device_id=0):
        super().__init__(task_name)

        self.virtual_human = Chatbot(device_id=device_id)
        self.prompt = None
        self.history = None
        self.current_step_info = None

    def reset(self, seed=0):
        obs, info = super().reset(seed)

        self._set_prompt(obs['image'])

        self.history = None

        human_response = self.dialog(obs['image'], self.prompt)
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
            human_response = self.dialog(obs['image'], robot_response)
            obs['response'] = human_response

            self.current_step_info[0] = obs

        else:
            raise ValueError(f"Invalid action:{action}, robot_response:{robot_response}")

        return obs, reward, done, truncated, info

    def dialog(self, image, text):
        # if response is None:
        #     text = self.prompt
        # else:
        #     text = response

        response, self.history = self.virtual_human.chat_history(image, text, self.history)

        return response

    def describe_scene(self, image):
        # text = f"<image>\nThis is a robot manipulation scenario with the task: **{self.instruction}**. Describe the objects in the scene with the positional relationships between them. Please output the description directly"

        # prompt = f"""
        # <image>
        # This is a robot manipulation scenario with the task: **{self.instruction}**. Your goal is to carefully observe the image and describe the objects it contains, as well as their relative positions and relationships.
        #
        # ### Instructions:
        # 1. Analyze the image and identify the objects present. Provide a detailed description of each object.
        # 2. For each object, describe its approximate location in the image (e.g., "in the top left corner," "near the center," "at the bottom right").
        # 3. Describe the spatial relationships between the objects. For example:
        #    - Indicate which objects are next to each other, above, below, or overlapping.
        #    - Use precise but natural language (e.g., "The apple is to the left of the plate," "The chair is behind the table").
        # 4. Write your response in a clear and structured format. Example output:
        #    - "The image contains a table, a cup, and a book. The cup is on the table, slightly to the left of the center. The book is on the right side of the table, next to the cup."
        # 5. Avoid making assumptions about objects or relationships that cannot be clearly inferred from the image.
        #
        # Write a concise and natural description of the objects and their spatial relationships as if you are explaining the image to someone who cannot see it. Focus on clarity and accuracy, highlighting only the most relevant details. Use complete sentences and avoid listing points. Now, describe the image in a single paragraph.
        # """

        prompt = f"""
        <image>
        This is a robot manipulation scenario with the task: **{self.instruction}**. Your goal is to carefully observe the image and describe the objects it contains, with a particular focus on the objects mentioned in the task.

        ### Instructions:
        1. Analyze the image and identify the objects present. Pay special attention to the objects mentioned in the task: {self.instruction}. These are the most important objects for your description.
        2. Describe each object, including its approximate location in the image (e.g., "in the top left corner," "near the center," "at the bottom right").
        3. Describe the spatial relationships between the objects, with an emphasis on the task's relevant objects (e.g., "The orange is near the blue plastic bottle").
        4. If the task specifies that one object should move relative to another (e.g., "move orange near blue plastic bottle"), make sure to clearly describe the current positions of these objects and their proximity to each other.
        5. Use clear and natural language to describe the relationships between the objects, but ensure that you prioritize the objects mentioned in the task. Example output:
           - "The image contains an orange and a blue plastic bottle. The orange is near the blue plastic bottle, slightly to the right of the center."
        6. Avoid assuming the presence or relationships of objects that are not mentioned in the task or cannot be clearly inferred from the image.
        7. Do not mention any information related to the task instruction, except for the objects included in the task.
        8. No need to describe background.

        Write a concise and natural description of the objects and their spatial relationships as if you are explaining the image to someone who cannot see it. Focus on the task-related objects and their positions, highlighting only the most relevant details. Do not mention any information related to the task instruction. Don't imply the task instruction. Use complete sentences and avoid listing points. Now, describe the image in a single paragraph.
        """

        # prompt = f"""
        # <image>
        # This is a robot manipulation scenario with the task: {self.instruction}. Your goal is to carefully observe the image and describe all the objects it contains, but with a special focus on accurately identifying the objects mentioned in the task.
        #
        # ### Instructions:
        # 1. Identify all objects in the image and describe them clearly. For each object, provide its approximate location in the image (e.g., "in the top left corner," "near the center," "at the bottom right").
        # 2. Pay special attention to the objects mentioned in the task (e.g., "orange" and "blue plastic bottle"). Ensure that these objects are recognized correctly and not confused with other objects in the image.
        # 3. For each object, describe its spatial relationships with other objects in the image. Use precise language, such as:
        #    - "The orange is near the blue plastic bottle."
        #    - "The apple is next to the table, to the left of the cup."
        # 4. Avoid confusing objects that look similar. If the task mentions an "orange," ensure it is recognized as an orange and not mistaken for a different object (e.g., an apple).
        # 5. Provide a clear and concise description of the objects and their relationships. Example output:
        #    - "The image contains an orange, a blue plastic bottle, a cup, and a table. The orange is near the blue plastic bottle, and the cup is on the table, to the left of the orange."
        # 6. You should describe the objects in the image in a coherent and structured way, mentioning both the task-relevant objects and other objects, but ensuring accuracy in identifying the objects specified in the task.
        #
        # Now, describe the image in a single paragraph, highlighting the objects  and their relationships, with special attention to those mentioned in the task.
        # """

        scene_description = self.dialog(image, prompt)

        return scene_description

    def _set_prompt(self, image):
        # self.prompt = f"<image>\nThis is a robot manipulation scenario. You now need to play the role of a user who is dialoguing with a robot in order to make the robot fulfill your needs. You have a task, {self.instruction}, that you need the robot to accomplish, but you are not able to tell the robot this instruction directly. You need to tell the robot what you want about the task so that the robot can ask you to confirm the instruction to be performed. Your output needs to start with “User:”. If the manipulation described by the robot meets your correct intent, please respond positively. Now you need to initiate a dialog and answer the robot's query. "
        #
        # # If the robot has asked for your correct intent, reply in the affirmative and add a specific flag “<execute>” at the end of the reply
        #
        # self.prompt = f"<image>\nThis is a robot manipulation scenario. Now, you need to play as a user. You need the robot to accomplish a task ({self.instruction}), but you cannot tell the robot this instruction directly. You need to build a natural, real-life vague requirement around this instruction. You need to make your need progressively clearer to the robot by dialoguing with it instead of telling it directly. Your output needs to start with “User:”. If the robot's answer matches your correct intent, respond in the affirmative. Now, you need to start the dialog and wait for the robot's response."

        self.scene_description = self.describe_scene(image)
        self.history = None

        # self.prompt = f"""
        #             <image>
        #             This is a robot manipulation scenario. {self.describe_scene}. The robot will play the role of a user, and your goal is to guide it to generate a natural and realistic daily-life scenario based on the task: **{self.instruction}**.
        #
        #             ### Instructions:
        #             1. Based on the task **{self.instruction}**, imagine a scenario that reflects a real-world human need or situation.
        #             2. The scenario must be expressed in natural language, simulating how a person would describe their need in daily life. Avoid stating the task explicitly. Instead, use contextual or indirect expressions.
        #             3. Your scenario should:
        #                - Be closely related to the instruction but feel realistic and human-like.
        #                - Reflect a daily-life need, such as hunger, thirst, curiosity, or convenience.
        #             4. Respond as the "User" in the dialogue. Format your output as follows:
        #                - “User: [realistic, indirect need related to the instruction]”
        #
        #             ### Example:
        #             **Instruction**: Pick up the Coca-Cola.
        #             **Generated Need**:
        #             - User: I’m feeling a bit thirsty. Oh, is that a Coca-Cola? Can you grab it for me?
        #
        #             Now, use the task **{self.instruction}** to generate a realistic and natural need expressed in daily conversation.
        #             """

        self.prompt = f"""
        <image>
        This is a robot manipulation scenario. {self.describe_scene}. You will play the role of a human user interacting with the robot. Your goal is to guide the robot to accomplish a specific task: **{self.instruction}**.

        ### Instructions:
        1. Assume the role of a human user with a specific daily-life need. You want the robot to help you accomplish the task **{self.instruction}**, but you cannot directly tell the robot the exact instruction.
        2. Begin the conversation by expressing a vague or indirect need related to the task. Gradually refine your expressions through multiple rounds of dialogue if the robot does not understand initially.
        3. Your dialogue should:
           - Be realistic, reflecting how a human might naturally communicate their needs in daily life.
           - Progressively make the task clearer without explicitly stating it.
           - Affirm or correct the robot’s responses based on whether they match your intended task.
        4. Respond as "User" in the conversation. Format your responses as:
           - “User: [your message]”
        5. After each response, wait for the robot’s reply before continuing.

        ### Example:
        **Task**: Pick up the Coca-Cola.
        **Sample Conversation**:
        - User: "I’m feeling thirsty. Do you see anything to drink?"
        - Robot: "There’s a Coca-Cola on the table. Should I bring it to you?"
        - User: "Yes, that’s perfect! Please bring it over."

        Now, start the conversation as the user. Express a vague need related to the task **{self.instruction}**. Begin your message with:
        - User: 
        """

        return self.prompt

    def check_instruction(self, image, gt_instruction, pred_instruction):
        prompt = f"This is a robot manipulation scenario. There are now two task instructions for this scenario: 1. {gt_instruction} 2. {pred_instruction}. You need to determine if these two instructions express similar meanings. Your output only needs to be yes or no."
        response, history = self.virtual_human.chat(image, prompt)

        if 'yes' in response.lower():
            is_match = True
        else:
            is_match = False

        return is_match


def test_dial_env():
    env = DialSimplerEnv('google_robot_move_near', device_id=7)
    obs = env.reset()
    image = obs['image']
    plt.imshow(image)
    plt.show()
    # text = env.describe_scene(image)
    print(env.instruction)
    print(env.scene_description)


def test_env():
    env = SimplerEnv('google_robot_pick_coke_can')
    for i in range(5):
        obs, reset_info = env.reset(seed=i)

        image = obs['image']
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    test_env()
