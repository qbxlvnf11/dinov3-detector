Contents
=============

### - [DINO Version3](https://github.com/facebookresearch/dinov3) Test Sample Code
* Download Weights
    

How to Run
=============

* Run env

```
conda env create -f conda.yaml
conda activate dinov3
pip install -r requirements.txt
```

* Run Feature Extraction
    * Build '.env' file for Hugging face token

```
python demo.py
```

* Run Detection
    * Weights of backbone path: 'weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
    * Weights of detector head path: 'weights/dinov3_vit7b16_coco_detr_head-b0235ff7.pth'
    * Adjust <span style="background-color:#FFE6E6"> fps_in </span>

```
python demo_detection.py --video_path {input_video_path} --conf_thresh {conf_score_thr} --max_time_sec {max_seconds_of_ --output_video} --output_path {save_video_path}
```





