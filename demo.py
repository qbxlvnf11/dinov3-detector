# import torch
# from transformers import AutoImageProcessor, AutoModel, ConvNextImageProcessor
# from transformers.image_utils import load_image

# from utils import huggingface_login

# huggingface_login()

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = load_image(url)

# pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
# # processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# processor = ConvNextImageProcessor.from_pretrained(pretrained_model_name) 

# model = AutoModel.from_pretrained(
#     pretrained_model_name, 
#     device_map="auto", 
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
# with torch.inference_mode():
#     outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)

import torch
from transformers import pipeline, AutoImageProcessor, AutoModel, ConvNextImageProcessor
from transformers.image_utils import load_image

from utils import huggingface_login

huggingface_login()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
print(type(features))