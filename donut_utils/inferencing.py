import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json

def test(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []

    folder = os.fsencode(args.dataset_name_or_path)
    images = os.listdir(folder)

    for idx, sample in tqdm(enumerate(images), total=len(images)):

        if os.fsdecode(sample).endswith(".jpg"):
            image = Image.open(f"{args.dataset_name_or_path}/{os.fsdecode(sample)}")
        else:
            continue
        output = pretrained_model.inference(image=image, prompt=f"<s_{args.task_name}>")["predictions"][0]

        output['file_name'] = os.fsdecode(sample)

        predictions.append(output)

    if args.save_path:
        save_json(args.save_path, predictions)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str) ## path of dir of images we want to inference
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None) ## task_name is the name of the dataset the model was trained on
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)
