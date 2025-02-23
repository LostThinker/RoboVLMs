import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data2/user/qianlong_dataset/hf_cache'

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import requests
import matplotlib.pyplot as plt


class ObjectDetect:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def detect(self, image, objects):
        text = [objects]
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                               threshold=0.1)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries

        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=16)

        detection_results = []

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {objects[label]} with confidence {round(score.item(), 3)} at location {box}")

            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
            # Optionally, you can add the label and confidence score
            draw.text((box[0], box[1]), f"{text[label]}: {round(score.item(), 2)}", fill="red", font=font)

            detection = {
                "object": objects[label],
                "confidence": {round(score.item(), 3)},
                "box": box
            }

            detection_results.append(detection)

        return detection_results, image


def grounding_dino():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    text = "a cat. a remote control."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    print(results)


def owlv2():
    # Use a pipeline as a high-level helper
    # Load model directly
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open('/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs/data_generation/visual_img/0.png')

    texts = [["gripper","banana", "cucumber"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
        # Optionally, you can add the label and confidence score
        draw.text((box[0], box[1]), f"{text[label]}: {round(score.item(), 2)}", fill="red", font=font)

    # image.show()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    owlv2()
