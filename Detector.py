import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
# import torchvision.transforms as T
from torchvision import transforms as T
import cv2
from PIL import Image, ImageDraw, ImageFont

# === DINOv3 로드 ===
from dinov3.models.vision_transformer import vit_small

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Detection Head 정의 ===
class SimpleDetector(nn.Module):
    def __init__(self, backbone, feat_dim=384, num_classes=20):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.bbox_head = nn.Linear(feat_dim, 4)

    def forward(self, x):
        feats = self.backbone(x)

        if isinstance(feats, torch.Tensor):
            if feats.ndim == 2:   # (B, C)
                pooled = feats
            elif feats.ndim == 3: # (B, N, C)
                pooled = feats.mean(dim=1)
            elif feats.ndim == 4: # (B, C, H, W)
                pooled = feats.mean(dim=[2,3])
            else:
                raise ValueError(f"Unexpected tensor shape: {feats.shape}")

        elif isinstance(feats, dict):
            # dict 형태라면 대표 토큰 사용
            if "x_norm_clstoken" in feats:
                pooled = feats["x_norm_clstoken"]  # (B, C)
            else:
                pooled = list(feats.values())[0]
                if pooled.ndim == 3:
                    pooled = pooled.mean(dim=1)

        else:
            raise ValueError(f"Unexpected backbone output type: {type(feats)}")

        cls_logits = self.cls_head(pooled)
        bbox_regs = self.bbox_head(pooled)
        return cls_logits, bbox_regs

# class DinoDetector:
#     def __init__(self, repo_dir, detector_weights, backbone_weights=None, device=None):
#         """
#         DINOv3 detector 초기화

#         Args:
#             repo_dir (str): DINOv3 로컬 repo 경로
#             detector_weights (str): detector checkpoint 경로
#             backbone_weights (str, optional): backbone checkpoint 경로
#             device (str, optional): 'cuda' 또는 'cpu'. 기본은 자동 선택
#         """
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
#         print(' == Detector backbone_weights:', backbone_weights)
#         print(' == Detector detector_weights:', detector_weights)
        
#         self.detector = torch.hub.load(
#             repo_dir,
#             'dinov3_vit7b16_de',
#             source="local",
#             weights=detector_weights,
#             backbone_weights=backbone_weights
#         )
#         self.detector.eval()
#         self.detector.to(self.device)
#         print(f"[INFO] DINOv3 detector loaded on {self.device}")

#     def infer(self, images):
#         """
#         이미지 또는 이미지 배치에 대한 inference 수행

#         Args:
#             images (torch.Tensor): [B, C, H, W] 형태의 tensor, 0~255 범위

#         Returns:
#             list: detector output
#         """

#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detector(images)

#         return outputs

class DinoDetector:
    def __init__(self, repo_dir, detector_weights, backbone_weights=None, device=None):
        """
        DINOv3 detector 초기화
        Args:
            repo_dir (str): DINOv3 로컬 repo 경로
            detector_weights (str): detector checkpoint 경로
            backbone_weights (str, optional): backbone checkpoint 경로
            device (str, optional): 'cuda' 또는 'cpu'. 기본은 자동 선택
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(' == Detector backbone_weights:', backbone_weights)
        print(' == Detector detector_weights:', detector_weights)
        
        self.detector = torch.hub.load(
            repo_dir,
            'dinov3_vit7b16_de',
            source="local",
            weights=detector_weights,
            backbone_weights=backbone_weights
        )
        self.detector.eval()
        self.detector.to(self.device)
        print(f"[INFO] DINOv3 detector loaded on {self.device}")

        # # 전처리 transform (DINOv3 README 권장 mean/std 값)
        # self.transform = T.Compose([
        #     T.ToImage(),
        #     T.Resize((896, 896), antialias=True),
        #     T.ToDtype(torch.float32, scale=True),
        #     T.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
        # ])
        self.transform = T.Compose([
            T.Resize((896, 896)),
            T.ToTensor(),  
            T.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
        ])

    def preprocess(self, pil_img: Image.Image):
        """PIL.Image -> Tensor 전처리"""
        return self.transform(pil_img).unsqueeze(0)  # [1, C, H, W]

    def infer(self, images: torch.Tensor):
        """
        이미지 또는 이미지 배치에 대한 inference 수행
        Args:
            images (torch.Tensor): [B, C, H, W] 형태 tensor, float32
        Returns:
            detector raw output
        """
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.detector(images)
        return outputs

    def postprocess(self, outputs, orig_size):
        """
        detector raw output -> (boxes, labels, scores)
        Args:
            outputs: model raw output (dict or list 형태일 수 있음)
            orig_size (tuple): (H, W), 원본 이미지 크기
        Returns:
            boxes (np.ndarray), labels (list[str]), scores (np.ndarray)
        """
        H, W = orig_size
        if isinstance(outputs, dict):
            boxes = outputs.get("boxes", torch.empty((0,4)))
            scores = outputs.get("scores", torch.ones(len(boxes)))
            labels = outputs.get("labels", torch.zeros(len(boxes)))
        elif isinstance(outputs, (list, tuple)) and isinstance(outputs[0], dict):
            return self.postprocess(outputs[0], orig_size)
        else:
            return np.zeros((0,4)), [], np.zeros((0,))
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().astype(int).tolist()

        if boxes.max() <= 1.01:
            boxes[:, [0,2]] *= W
            boxes[:, [1,3]] *= H

        labels = [str(l) for l in labels]
        return boxes, labels, scores

    def draw(self, img, boxes, labels, scores, conf_thresh=0.5):
        # numpy.ndarray (cv2 frame) → PIL.Image 변환
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)

        thr_labels = []
        for box, label, score in zip(boxes, labels, scores):
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = box
            draw.text((x1, y1 - 10), f"{label}:{score:.2f}", fill="blue")
            thr_labels.append(label)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            draw.text((x1, y1 - 10), f"{label}:{score:.2f}", fill="blue")

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), thr_labels 
        