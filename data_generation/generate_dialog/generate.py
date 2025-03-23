import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'
os.environ['RANK'] = '0'
sys.path.append("/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs")
import json
from robovlms.data import OpenVLADataset
from robovlms.data.datamodule.gr_datamodule import GRDataModule
from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.data_utils import preprocess_image
from robovlms.data.base_openvla_dataset import RLDSDataset
import functools

import pickle
from PIL import Image


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


def find_first_indices(lst):
    if not lst:
        return {}
    if len(set(lst)) == 1:
        return {lst[0]: 0}
    # 创建一个 defaultdict，默认值为 -1
    first_indices = defaultdict(lambda: -1)
    for idx, value in enumerate(lst):
        # 如果该值还未记录过索引
        if first_indices[value] == -1:
            first_indices[value] = idx
    return dict(first_indices)


def read_rlds_dataset():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds

    dataset, dataset_statistics = make_dataset_from_rlds(
        "bridge_orig",
        "/data2/user/qianlong_dataset/robot_dataset/robot-dataset/open-x-embodiment/robotics",
        train=True,
        image_obs_keys={'primary': 'image_0'},
        language_key='language_instruction',
    )

    data_dict = []
    for i, data in enumerate(dataset):
        # lang = data['task']["language_instruction"][0].numpy().decode()
        lang = data['task']["language_instruction"].numpy()
        lang = [l.decode() for l in lang]

        # for one task multi instruction
        lang_and_idx_dict = find_first_indices(lang)
        for k, v in lang_and_idx_dict.items():
            if len(k) > 0:
                image = tf.io.decode_image(data['observation']['image_primary'][v], expand_animations=False, dtype=tf.uint8).numpy()
                image = Image.fromarray(image, mode='RGB')
                data = dict(lang=k, image=image)
                data_dict.append(data)

    #     if len(data_dict) > 10000:
    #         with open(f"./bridge_lang_img_data/{data_id}.pkl", 'wb') as file:
    #             # 使用 pickle.dump() 方法将数据保存到文件中
    #             pickle.dump(data_dict, file)
    #         data_dict = []
    #         data_id += 1
    #
    with open(f"./bridge_lang_img_data/lang_img_data_{len(data_dict)}.pkl", 'wb') as file:
        # 使用 pickle.dump() 方法将数据保存到文件中
        pickle.dump(data_dict, file)

    # print(lang)
    # plt.imshow(image)
    # plt.title(lang)
    # plt.show()
    # if i > 10:
    #     break
    # print(data)
    # break
    print(1)


def load_pkl():
    import pickle

    # 指定要读取的 pkl 文件路径
    file_path = '/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/generate_dialog/bridge_lang_img_data/0.pkl'

    # 以二进制读取模式打开文件
    with open(file_path, 'rb') as file:
        # 使用 pickle.load() 方法从文件中加载数据
        loaded_data = pickle.load(file)

    # 假设保存的多个对象在一个列表中，依次取出每个对象
    for obj in loaded_data:
        print("读取到的对象：", obj)
        break


def read_dataset():
    variant = load_config(
        "/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/configs/oxe_training/debug.json")
    model_load_path = None
    # model = BaseTrainer.from_checkpoint(
    #     model_load_path, "torch", variant
    # )
    #
    # image_preprocess = model.model.image_processor

    dataset_config = variant["train_dataset"]
    dataset_config.update(variant)
    dataset = RLDSDataset(**variant["train_dataset"])
    lang_tab = count_dataset_language(dataset)


def count_dataset_language(dataset):
    lang_tab = {}
    ix = 0
    for data in dataset.dataset:
        lang = data["task"]["language_instruction"].numpy().decode()
        lang_id = "_".join(lang.split())
        lang_tab[lang_id] = lang_tab.get(lang_id, 0) + 1
        # if ix > 1000:
        #     break
        if ix % 1000 == 0:
            with open("lang_tab.json", 'w') as file:
                # 使用json.dump()将字典数据写入文件
                json.dump(lang_tab, file, indent=4)
            print(ix)
        ix += 1
    print(lang_tab)
    return lang_tab


def generate_dialog(device="cuda:1"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map=device
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor.tokenizer.padding_side = 'left'

    with open("bridge_lang_img_data/lang_img_data_38660.pkl", "rb") as f:
        dataset = pickle.load(f)  # 加载文件内容到变量

    def batch_iterate(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    new_dataset = []

    batch_size = 32
    for batch in batch_iterate(dataset, batch_size):

        messages = get_batch_message(batch)

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # print(output_text)

        for i, text in enumerate(output_text):
            try:
                text = text.replace("```json", "").replace("```", "").strip()
                dialog_data = json.loads(text)
                # print(dialog_data)
                batch[i]["conversations"] = dialog_data
                # print("✅ 成功解析 VLM 输出！")
            except json.JSONDecodeError as e:
                print(f"⚠️ 解析失败: {e}")

        new_dataset.extend(batch)

        if len(new_dataset) % (batch_size * 50) == 0:
            print(len(new_dataset))
            with open(f"/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/generate_dialog/bridge_lang_img_data/lang_img_conversation_data.pkl", 'wb') as file:
                # 使用 pickle.dump() 方法将数据保存到文件中
                pickle.dump(new_dataset, file)

    with open(f"/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/generate_dialog/bridge_lang_img_data/lang_img_conversation_data_all.pkl", 'wb') as file:
        # 使用 pickle.dump() 方法将数据保存到文件中
        pickle.dump(new_dataset, file)


def get_batch_message(batch_data):
    batch_message = []
    for data in batch_data:
        lang = data['lang']
        img = data['image']
        # img = Image.fromarray(img, mode='RGB')

        prompt = get_prompt(lang)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        batch_message.append(messages)
    return batch_message


def get_prompt(instruction):
    prompt = """### Task Description
    Your goal is to simulate a conversation between a human and a robot based on an input image and an operation instruction. The human starts by expressing a natural, everyday need inspired by the operation instruction, and the robot responds by asking clarifying questions. The conversation continues until the robot fully understands the original operation instruction and confirms it. Output the conversation as a JSON list where each turn is a dictionary with `"from"` and `"value"` keys.
    
    ### Input
    - An image that represents the environment or object.
    - An operation instruction (e.g., "move the cup to the table").
    
    ### Conversation Flow
    1. The **human** starts with a natural statement or request derived from the instruction (e.g., "I need to set the table for dinner." or "Could you help me set the table?").
    2. The **robot** asks clarifying questions to narrow down the exact task (e.g., "What should I move to the table?").
    3. The **human** answers naturally, refining the requirement.
    4. The **robot** continues until it fully reconstructs the original instruction and confirms the understood task and ends the conversation.
    
    ### Output Format
    The conversation should be a JSON list of dictionaries:
    ```json
    [
      {"from": "human", "value": "I want to get things ready for dinner."},
      {"from": "robot", "value": "What do you need me to prepare?"},
      {"from": "human", "value": "Could you move the cup to the table?"},
      {"from": "robot", "value": "Which cup should I move?"},
      {"from": "human", "value": "The blue one on the counter."},
      {"from": "robot", "value": "Understood. I will move the blue cup from the counter to the table."}
    ]
    ```
    
    ### Examples
    #### Example 1
    **Instruction:** "Pick up the book from the shelf and put it on the desk."
    
    **Output:**
    ```json
    [
      {"from": "human", "value": "Can you help me organize my desk?"},
      {"from": "robot", "value": "What would you like me to move?"},
      {"from": "human", "value": "Could you bring the book over?"},
      {"from": "robot", "value": "Which book should I bring?"},
      {"from": "human", "value": "The red one on the top shelf."},
      {"from": "robot", "value": "Got it. I’ll move the red book from the top shelf to the desk."}
    ]
    ```
    
    #### Example 2
    **Instruction:** "Turn off the living room light."
    
    **Output:**
    ```json
    [
      {"from": "human", "value": "It's too bright in here."},
      {"from": "robot", "value": "Would you like me to adjust the light?"},
      {"from": "human", "value": "Yes, can you turn off the light?"},
      {"from": "robot", "value": "Which light should I turn off?"},
      {"from": "human", "value": "The living room one."},
      {"from": "robot", "value": "Got it. Turning off the living room light."}
    ]
    ```
    
    ### Guidelines
    - Ensure the human responses feel natural and goal-oriented.
    - The robot should prioritize asking clarifying questions before executing.
    - Each conversation should end with a clear confirmation from the robot.
    - Keep the language polite, helpful, and practical.
    - Keep the dialog flexible and diverse.
    
    """

    end = f"Now, let’s generate some creative and effective conversations with instruction: {instruction}"

    return prompt + end


def check_data():
    with open("bridge_lang_img_data/lang_img_conversation_data_all.pkl", "rb") as f:
        dataset = pickle.load(f)  # 加载文件内容到变量

    miss_conversations = 0
    for data in dataset:
        if "conversations" not in data.keys():
            miss_conversations += 1

    print(miss_conversations)


if __name__ == '__main__':
    # read_rlds_dataset()
    # load_pkl()
    generate_dialog()
    # print(get_prompt("test"))

    # check_data()
