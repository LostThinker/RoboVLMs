import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np


def show_bridge_data():
    output_dir = "./data_generation/visual_img/"
    os.makedirs(output_dir, exist_ok=True)

    b = tfds.builder_from_directory(
        builder_dir="/data2/user/qianlong_dataset/robot_dataset/robot-dataset/open-x-embodiment/robotics/bridge_orig/1.0.0")
    ds = b.as_dataset(split='train[68:78]')
    episode = next(iter(ds))
    # steps = [step for step in episode['steps']]
    # images = [step['observation']['image0'] for step in episode['steps']]

    for i, step in enumerate(episode['steps']):
        if i == 0:
            print('language_embedding shape', step['language_embedding'].shape)
            l = step['language_instruction'].numpy().decode('utf-8')
            print(l)
        obs = step['observation']
        img_arrays = []
        for k, v in obs.items():
            if 'image' in k:
                img_arrays.append(v.numpy())

        img = np.concatenate(img_arrays, axis=1)
        if i % 10 == 0:
            plt.imshow(img)
            plt.show()


def show_scienceqa():
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA",
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/ScienceQA')

    data = ds['train'][:10]
    print(data.keys())
    image = data['image'][0]
    question = data['question']
    choices = data['choices']
    plt.imshow(image)
    plt.show()

    print(data)


def show_invig():
    from datasets import load_dataset

    ds = load_dataset("jxu124/invig", split='train',
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/invig')

    for idx, example in enumerate(ds):
        print(f"Example {idx}: {example}")
        image = example['image']
        ref_list = example['ref_list']
        image_info = example['image_info']

        bbox = ref_list[0]['bbox']
        category = ref_list[0]['category']
        dialog = ref_list[0]['dialog']
        dialog_cn = ref_list[0]['dialog_cn']

        plt.imshow(image)
        plt.show()
        print(category, dialog)

        break


# def show_visdialog():
#     from dataclasses import dataclass
#     import datasets
#
#     # load and path setting
#     ds_visdial = datasets.load_dataset('jxu124/visdial')
#     path_map = {
#         "coco/train2014": f"/datasets/coco/train2014",
#         "coco/val2014": f"/datasets/coco/val2014",
#         "visdial/VisualDialog_test2018": f"/datasets/visdial/VisualDialog_test2018",
#         "visdial/VisualDialog_val2018": f"/datasets/visdial/VisualDialog_val2018"
#     }
#
#     # apply to your datasets
#     @dataclass
#     class ReplaceImagePath():
#         path_map: {}
#
#         def __call__(self, features):
#             for k, v in self.path_map.items():
#                 features['image'] = features['image'].replace(k, v)
#             return features
#
#     ds_visdial = ds_visdial.map(ReplaceImagePath(path_map=path_map)).cast_column("image", datasets.Image())


def show_ecot_data():
    file_path = '/data2/user/qianlong_dataset/robot_dataset/robot-dataset/ecot/embodied_features_bridge.json'
    import json
    import tensorflow as tf
    import enum

    class CotTag(enum.Enum):
        TASK = "TASK:"
        PLAN = "PLAN:"
        VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
        SUBTASK_REASONING = "SUBTASK REASONING:"
        SUBTASK = "SUBTASK:"
        MOVE_REASONING = "MOVE REASONING:"
        MOVE = "MOVE:"
        GRIPPER_POSITION = "GRIPPER POSITION:"
        ACTION = "ACTION:"

    def get_cot_tags_list():
        return [
            CotTag.TASK.value,
            CotTag.PLAN.value,
            CotTag.VISIBLE_OBJECTS.value,
            CotTag.SUBTASK_REASONING.value,
            CotTag.SUBTASK.value,
            CotTag.MOVE_REASONING.value,
            CotTag.MOVE.value,
            CotTag.GRIPPER_POSITION.value,
            CotTag.ACTION.value,
        ]

    def get_cot_database_keys():
        return {
            CotTag.TASK.value: "task",
            CotTag.PLAN.value: "plan",
            CotTag.VISIBLE_OBJECTS.value: "bboxes",
            CotTag.SUBTASK_REASONING.value: "subtask_reason",
            CotTag.SUBTASK.value: "subtask",
            CotTag.MOVE_REASONING.value: "move_reason",
            CotTag.MOVE.value: "move",
            CotTag.GRIPPER_POSITION.value: "gripper",
            CotTag.ACTION.value: "action",
        }

    def make_tf_dict(raw_dict):
        print("Building the reasoning dict...")
        keys = []
        values = []

        def reasoning_dict_to_str(d):
            tags = get_cot_tags_list()[:-1]  # exclude ACTION
            database_keys = get_cot_database_keys()
            reasoning_parts = [(tag, d[database_keys[tag]]) for tag in tags]

            return "@".join(f"{tag}@{part}" for tag, part in reasoning_parts)

        has_reasoning = [0, 0]

        for file_name in raw_dict.keys():
            for episode_id in raw_dict[file_name].keys():
                if "reasoning" not in raw_dict[file_name][episode_id].keys():
                    has_reasoning[0] += 1
                    continue
                else:
                    has_reasoning[1] += 1

                for i in raw_dict[file_name][episode_id]["reasoning"].keys():
                    keys.append(file_name + "_" + str(episode_id) + "_" + i)
                    reasoning_dict = raw_dict[file_name][episode_id]["reasoning"][i]

                    gripper_lookahead_n = 5  # list this many future positions of the gripper
                    trajectory_features = raw_dict[file_name][episode_id]["features"]

                    reasoning_dict["gripper"] = ""
                    if "gripper_position" in trajectory_features.keys():
                        if trajectory_features["gripper_position"] is not None:
                            if 0 <= int(i) < len(trajectory_features["gripper_position"]):
                                future_positions = []
                                for j in range(gripper_lookahead_n):
                                    if int(i) + j < len(trajectory_features["gripper_position"]):
                                        future_positions += trajectory_features["gripper_position"][int(i) + j]
                                    else:
                                        future_positions += future_positions[-2:]

                                reasoning_dict["gripper"] = str(future_positions)

                    reasoning_dict["bboxes"] = ""
                    if "bboxes" in trajectory_features.keys():
                        if trajectory_features["bboxes"] is not None:
                            if 0 <= int(i) < len(trajectory_features["bboxes"]):
                                if len(trajectory_features["bboxes"][int(i)]) > 0:
                                    boxes_list = trajectory_features["bboxes"][int(i)]
                                    reasoning_dict["bboxes"] = ", ".join(
                                        [f"{name} {box!s}" for prob, name, box in boxes_list]
                                    )

                    values.append(reasoning_dict_to_str(reasoning_dict))

        print("Example reasoning:", keys[0], values[0])
        print("Reasoning presence statistics [# has not, # has]:", has_reasoning)

        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value="")

    with open(file_path, "r") as f:
        reasoning_dataset = json.load(f)

    reasoning_dataset = make_tf_dict(reasoning_dataset)
    file_name = '/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy'
    episode_ids = 43
    indices = 0
    lookup_key = file_name + "_" + str(episode_ids) + "_" + str(indices)
    lookup_key = tf.constant(lookup_key, dtype=tf.string)
    reasonings = reasoning_dataset.lookup(lookup_key)
    reasonings = reasonings.numpy().decode('utf-8')
    print(reasonings)


def hf_visdial():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/VisDial", split='train',
                      cache_dir='/data2/user/qianlong_dataset/robot_dataset/vlm-co-training/hf_visdial')

    data = ds[:10]
    for idx, example in enumerate(ds):
        # 打印样本索引和样本内容

        caption = example['caption']
        dialog = example['dialog']
        image_path = example['image_path']
        global_image_id = example['global_image_id']
        anns_id = example['anns_id']
        image = example['image']

        plt.imshow(image)
        plt.show()

        print(f"Example {idx}: {example}")
        break


def hf_data_test():
    from datasets import load_dataset

    ds = load_dataset("openvla/modified_libero_rlds",
                      cache_dir='/data2/user/qianlong_dataset/hf_cache')

if __name__ == '__main__':
    # show_invig()
    # show_scienceqa()
    # hf_visdial()
    hf_data_test()