import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

from env_wrapper import SimplerEnv
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt

import torch


def eval():
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda:7")

    env = SimplerEnv('google_robot_pick_coke_can')

    obs, info = env.reset()
    done = False

    plt.imshow(obs['image'])
    plt.show()
    print(env.instruction)

    i = 0
    while not done:
        image = Image.fromarray(obs['image'])
        prompt = f"In: What action should the robot take to {env.instruction}?\nOut:"
        if i % 20 == 0:
            plt.imshow(image)
            plt.show()

        inputs = processor(prompt, image).to("cuda:7", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="fractal20220817_data", do_sample=False) # fractal20220817_data bridge_orig

        obs, reward, done, truncated, info = env.step(action)
        i += 1

        if truncated:
            print('Failed')
            break

    if done:
        print("Success")

    print(i)


if __name__ == '__main__':
    eval()
