import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import os
import re
import numpy as np
import dlimp as dl
from robovlms.data import OpenVLADataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
import io


def gen_img_description(img=None, text=None):
    from chat_with_vlm import Chatbot

    prompt = 'Describe the scene in detail, including the objects included and their approximate or relative locations, and then summarize the names of the graspable objects in the scene that are relevant to the robot\'s operational task and output them as a list'

    chatbot = Chatbot()
    response, history = chatbot.chat(img, prompt)
    print(f'User: {prompt}\nAssistant: {response}')


def gen_img_objects(img=None, lang=None):
    from chat_with_vlm import Chatbot

    prompt = f'This is a robot manipulation scenario for the task to {str.lower(lang)}. Give the names of all the objects on the desktop in the scene, and output them as a python list.'

    chatbot = Chatbot()
    response, history = chatbot.chat(img, prompt)
    objects = re.findall(r'"([^"]+)"', response)

    print(f'User: {prompt}\nAssistant: {response}')
    print(objects)


def gen_pipeline():
    b = tfds.builder_from_directory(
        builder_dir="/data2/user/qianlong_dataset/robot_dataset/robot-dataset/open-x-embodiment/robotics/bridge_orig/1.0.0")
    tf_dataset = b.as_dataset(split='train[68:78]')

    builder = tfds.builder('bridge_orig',
                           data_dir='/data2/user/qianlong_dataset/robot_dataset/robot-dataset/open-x-embodiment/robotics/')
    dataset = dl.DLataset.from_rlds(builder, split='train', shuffle=True, num_parallel_reads=8)

    for rlds_batch in dataset.as_numpy_iterator():
        # print(rlds_batch)
        obs = rlds_batch['observation']['image_0']
        state = rlds_batch['observation']['state']
        lang = rlds_batch['language_instruction'][0].decode('utf-8')
        img = Image.open(io.BytesIO(obs[0]))

        print(lang)
        plt.imshow(img)
        plt.show()

        # gen_img_description(lang, )

        # Image.fromarray(obs)
        for i in range(len(obs)):
            img = io.BytesIO(obs[i])
            img = Image.open(img)
            img.save(f"/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/visual_img/{i}.png")
        # plt.imshow(img)
        # plt.show()
        break


if __name__ == '__main__':
    gen_pipeline()
