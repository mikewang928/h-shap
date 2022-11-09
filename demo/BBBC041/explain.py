# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:49:25 2022

@author: wsycx
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import hshap
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# select device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
# reproducibility
torch.set_grad_enabled(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load pretrained model
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("model.pt", map_location=device))
model = model.to(device)
model.eval()
torch.cuda.empty_cache()

#%%
# dataset transform
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)

# load reference
ref = torch.load("reference.pt")
# ref = torch.load("gen_00430000.pt")
ref = ref.to(device)

# load annoations
annotations = pd.read_json("annotations.json")

# define annotation function
def annotate(ax, filename):
    query = annotations.loc[annotations["image"] == filename]
    for i, row in query.iterrows():
        trophozoites = row["objects"]
        for trophozoite in trophozoites:
            bbox = trophozoite["bounding_box"]
            upper_left_r = bbox["minimum"]["r"]
            upper_left_c = bbox["minimum"]["c"]
            lower_right_r = bbox["maximum"]["r"]
            lower_right_c = bbox["maximum"]["c"]
            w = np.abs(lower_right_c - upper_left_c)
            h = np.abs(lower_right_r - upper_left_r)
            rect = patches.Rectangle(
                (upper_left_c, upper_left_r),
                w,
                h,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
            

#%%
# initialize h-Shap explainer
s = 80
hexp = hshap.src.Explainer(model, ref, min_size=s)

# define thresholding modes
threshold_mode = "absolute"
threshold = 0
# for each example image
for (dirpath, _, filenames) in os.walk("images"):
    for filename in tqdm(filenames):
        # prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # load and transform image
        image_path = os.path.join(dirpath, filename)
        image = Image.open(image_path)
        image_t = transform(image).to(device)

        # show original image
        ax = axes[0]
        ax.imshow(np.array(image))
        annotate(ax, filename)
        ax.set_title("Original image")

        # explain image
        explanation = hexp.explain(
            image_t,
            label=1,
            threshold_mode=threshold_mode,
            threshold=threshold,
            batch_size=1,
        )
        explanation.squeeze_()

        # show explanation
        ax = axes[1]
        _abs = np.abs(explanation.flatten())
        _max = max(_abs)
        ax.imshow(explanation, cmap="bwr", vmax=_max, vmin=-_max)
        annotate(ax, filename)
        ax.set_title(
            r"h-Shap ($\tau = %.0f%s,~s = %d$ pixels)"
            % (threshold, "\%" if threshold_mode == "relative" else "", s)
        )
        img_id = filename.split(".")[0]

        # save figure
        plt.savefig(os.path.join("explanations", f"{img_id}.jpg"))
        plt.show()
        plt.close()