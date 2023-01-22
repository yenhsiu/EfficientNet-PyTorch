import json
from PIL import Image

import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet

model_name = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(model_name) # 224


img = Image.open('img.jpg')


# Preprocess image
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img).unsqueeze(0)


model = EfficientNet.from_pretrained(model_name)
endpoints = model.extract_endpoints(img)
print(endpoints['reduction_2'].shape)
