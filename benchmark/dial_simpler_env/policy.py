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
from data_generation.chat_with_vlm import Chatbot
import cv2
from transforms3d.euler import euler2axangle


class VLAPolicy:
    def __init__(
            self,
            model_name="openvla/openvla-7b",
            robot_type='widowx',
            image_size=[224, 224],
            action_scale=1,
            device_id=0
    ):
        self.vla_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(f"cuda:{device_id}")

        self.robot_type = robot_type
        if robot_type == 'widowx':
            self.unnorm_key = 'bridge_orig'
            self.sticky_gripper_num_repeat = 1
        elif robot_type == 'google_robot':
            self.unnorm_key = 'fractal20220817_data'
            self.sticky_gripper_num_repeat = 15

        self.image_size = image_size
        self.action_scale = action_scale
        self.device_id = device_id

        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.current_instruction = None

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.current_instruction = None

    def act(self, image, instruction):
        if self.current_instruction is None:
            self.current_instruction = instruction

        if self.current_instruction != instruction:
            self.reset()
            self.current_instruction = instruction

        image = self._process_image(image)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.vla_processor(prompt, image).to(f"cuda:{self.device_id}", dtype=torch.bfloat16)
        raw_actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.robot_type == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.robot_type == "widowx":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def get_action(self, obs):
        raw_action, action = self.act(obs['image'], obs['instruction'])
        action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        return action

    def _process_image(self, image):
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        return image


class BaseDialPolicy:
    def __init__(self, vlm_device_id=0, vla_device_id=1, robot_type='widowx'):
        self.vlm = Chatbot(device_id=vlm_device_id)
        self.history = None
        self.current_instruction = None

        self.vla = VLAPolicy("openvla/openvla-7b", robot_type=robot_type, device_id=vla_device_id)

        # self.vla_processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        # self.vla = AutoModelForVision2Seq.from_pretrained(
        #     "openvla/openvla-7b",
        #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True
        # ).eval().to(f"cuda:{vla_device_id}")
        # self.vla_device_id = vla_device_id

        self._set_prompt()
        self.reset()

    def reset(self):
        self.history = None
        self.first_step = True
        self.current_instruction = None
        self.vla.reset()

    def get_action(self, obs):
        img = obs['image']
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

        raw_action, action = self.vla.act(image, instruction)
        action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])

        # use prompt?
        # prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        #
        # image = Image.fromarray(image)
        # # Predict Action (7-DoF; un-normalize for BridgeV2)
        # inputs = self.vla_processor(prompt, image).to(f"cuda:{self.vla_device_id}", dtype=torch.bfloat16)
        # action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)  # fractal20220817_data

        return action

    def _set_prompt(self):
        # self.prompt = f"<image>\nThis is a robot manipulation scenario. You now need to play the role of a robotic assistant that performs actions according to the user's needs. If you are not sure what the user means, you need to ask him or her to clarify what operation needs to be performed. Your output should start with “Robot:”. Once you have clarified the user's intent, you need to summarize an instruction to call the executable program. When outputting instructions, you need to first output a special tag <instruction> followed by the summarized instruction. The summarized instruction should start with a verb and be clear and concise, such as put something somewhere. Now start the conversation with the user."

        self.prompt = """
            You are a highly intelligent robotic assistant designed to help users accomplish their tasks. The user won't directly state their exact task but will provide vague or incomplete hints. Your responsibilities are:
            1. Engage in a polite and interactive conversation with the user to understand their true needs.
            2. Once the user's need is clear, respond appropriately to confirm or acknowledge the user's request.
            3. Immediately after your response, summarize the task as a concise instruction using the format: <instruction>your command here</instruction>. The command should start with a verb and be as precise as possible.
            4. Respond as "Robot" in the conversation. Format your responses as: “Robot: [your message]”
            
            ### Example Interaction:
            User: I feel a bit thirsty.  
            Robot: Got it! Would you like me to get you something to drink?  
            User: Yes, but I’d prefer something sweet.  
            Robot: How about a soda?  
            User: That sounds great.  
            Robot: Sure! I see a can of Coca-Cola on the table. I’ll grab it for you.  
            <instruction>Pick up the Coca-Cola can from the table</instruction>
            
            ### Instructions:
            Start a conversation below, interact with the user, and gradually clarify their needs. Don't get ahead of yourself at the beginning of the conversation; confirm the needs with the user and get an affirmative response before outputting the instruction. Once the user's need is clear:
            - Respond naturally to the user to confirm or acknowledge their request.
            - Immediately after your response, output the summarized instruction in the format: <instruction>your command here</instruction>.
            """

    def get_instruction(self, text):
        instruction = text.split("<instruction>")[-1]
        if '</instruction>' in instruction:
            instruction = instruction.replace("</instruction>", "")

        return instruction


