import sys
import time
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image

# === DINOv3 로드 ===
from dinov3.models.vision_transformer import vit_small
from Detector import SimpleDetector, DinoDetector
from coco_format import COCO_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Preprocess 함수 ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0).to(DEVICE)

# === Detection 시각화 ===
def draw_detection(frame, bbox, cls_id, conf):
    h, w, _ = frame.shape
    x, y, bw, bh = bbox
    # 정규화된 좌표라고 가정
    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
    cv2.putText(frame, f"Cls {cls_id}:{conf:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return frameL

# === Video Detection 실행 ===
def run_video_detection(video_path, output_path="output_detected.mp4", conf_thresh=0.5, max_time_sec=0):
    
    # # Backbone 생성
    # backbone = vit_small(patch_size=16)
    # weight_path = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    # ckpt = torch.load(weight_path, map_location="cpu")
    # state_dict = ckpt.get("model", ckpt)
    # backbone.load_state_dict(state_dict, strict=False)
    # backbone.to(DEVICE).eval()
    # print('- DEVICE:', DEVICE)

    # # Detector 생성
    # model = SimpleDetector(backbone, feat_dim=384, num_classes=20).to(DEVICE).eval()

    # Detector 생성
    print('Load Detector...')
    model = DinoDetector(
        repo_dir=".",
        detector_weights="weights/dinov3_vit7b16_coco_detr_head-b0235ff7.pth",
        backbone_weights="weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", #"weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth" #"weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    print('Load Detector Complete!')

    print(f'Video path: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_in = 3 #cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Ori Width / Height:', width, '/', height)
    print('FPS:', fps_in)
    save_video = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    max_frames = int(fps_in * max_time_sec)

    total_frames = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()

        # inp = preprocess_frame(frame)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = model.preprocess(pil_img).to(DEVICE)
        
        with torch.no_grad():
            out = model.infer(inp) # model(inp)
        # print('out:', out)

        # pred_logits = outputs['pred_logits'][0]
        # pred_boxes = outputs['pred_boxes'][0]

        ## Post-processing
        # h, w = frame.shape[:2]
        H_in, W_in = inp.shape[2:]
        # boxes, labels, scores = model.postprocess(out, (h, w))
        boxes, labels, scores = model.postprocess(out, (H_in, W_in))
        boxes[:, [0,2]] = boxes[:, [0,2]] * width / W_in
        boxes[:, [1,3]] = boxes[:, [1,3]] * height / H_in
        class_names = [COCO_CLASSES[int(label)] if 0 <= int(label) < len(COCO_CLASSES) else str(label) for label in labels]
        
        ## Visualization
        # vis_frame = model.draw(frame, boxes, labels, scores, conf_thresh=conf_thresh)
        vis_frame, thr_classes = model.draw(frame, boxes, class_names, scores, conf_thresh=conf_thresh)
        # vis_frame.show()

        # probs = torch.softmax(cls_logits[0], dim=-1)
        # cls_id = torch.argmax(probs).item()

        # conf = probs[cls_id].item()
        # bbox = bbox_regs[0].cpu().numpy()

        # if conf > conf_thresh:
        #     frame = draw_detection(frame, bbox, cls_id, conf)

        # probs = pred_logits.softmax(-1)
        # scores, labels = probs.max(-1)
        
        # keep = scores > conf_thresh
        
        # kept_boxes = pred_boxes[keep]
        # kept_scores = scores[keep]
        # kept_labels = labels[keep]

        # for score, box, label in zip(kept_scores, kept_boxes, kept_labels):
        #     frame = draw_detection(frame, box.cpu().numpy(), label.item(), score.item())

        save_video.write(vis_frame)
        total_frames += 1
        total_time += (time.time() - start)

        if total_frames == 1 or total_frames % 100 == 0:
            sample_img_path = 'sample_first_frame.jpg'
            print('='*50)
            print('- Total time:', total_time)
            print('- FPS:', total_frames / total_time)
            print('- Boxes:', boxes)
            # print('- Labels:', labels)
            print('- Thr Class names:', thr_classes)
            print('- Scores:', scores)
            cv2.imwrite(sample_img_path, vis_frame)
            print(f'Save sample img! {sample_img_path}')

        if max_frames != 0 and max_frames < total_frames:
            print(f"- Breaking the loop")
            break

    cap.release()
    save_video.release()

    avg_fps = total_frames / total_time
    print(f"✅ Saved: {output_path}")
    print(f"📊 Frames: {total_frames}, Avg FPS: {avg_fps:.2f}")

# === CLI entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv3 Video Detection Demo")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    # parser.add_argument("--weight_path", type=str, \
    #     default="weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", \
    #     help="Path to pretrained DINOv3 weights (.pth)")
    parser.add_argument("--output_path", type=str, default="output_detected.mp4", help="Path to save output video")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--max_time_sec", type=int, default=0)
    args = parser.parse_args()

    run_video_detection(args.video_path, args.output_path, args.conf_thresh, max_time_sec=args.max_time_sec)

