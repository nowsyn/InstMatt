# InstMatt
Official repository for *Human Instance Matting via Mutual Guidance and Multi-Instance Refinement* (CVPR2022 **Oral**).

<img src="figures/teaser.png" style="width:550px;" />

### Overview

This paper introduces a new matting task called human instance matting (HIM), which requires the pertinent model to automatically predict a precise alpha matte for each human instance.  Straightforward combination of  closely related techniques, namely, instance segmentation, soft segmentation and human/conventional matting, will easily fail in complex cases requiring disentangling mingled colors belonging to multiple instances along hairy and thin boundary structures.  To tackle these technical challenges, we propose a human instance matting framework, called InstMatt, where a novel mutual guidance strategy working in tandem with a multi-instance refinement module is used, for delineating multi-instance relationship among humans with complex and overlapping boundaries if present. A new instance matting metric called instance matting quality (IMQ) is proposed, which addresses the absence of a unified and fair means of evaluation emphasizing  both instance recognition and matting quality. Finally, we construct an HIM benchmark for evaluation, which comprises of both synthetic and natural benchmark images. In addition to thorough experimental results on complex cases with multiple and overlapping human instances each has intricate boundaries, preliminary results are presented on general instance matting.

### Benchmark

To provide a general and comprehensive validation on instance matting techniques, we construct an instance matting benchmark,  **HIM2K**, which consists of a synthetic  subset and a natural subset totaling 2,000 images with high-quality matte ground truths. It is organized in the following. Each instance alpha matte is stored in a seperate image.

Download HIM2K from [Google Drive](https://drive.google.com/file/d/11sUSUNdOTUZboc0zhjMz9Od2siuPTpeQ/view?usp=sharing), or [Baidu Disk](https://pan.baidu.com/s/1T6vZg7wse53I9DfLwCuUag) (password: 82qe).


```
HIM2K/
    images/
        comp/
            xxx.jpg
            yyy.jpg
            ...
        natural/
            zzz.jpg
            ...
    alphas/
        comp/
            xxx/
                00.png
                01.png
                ...
            yyy/
            ...
        natural/
            zzz/
            ...

				
```

### Code

#### Requirements

- pytorch-1.9.0
- cv2
- tqdm
- scipy
- scikit-image
- tensorboardX
- toml
- easydict

#### Instance Segmentation

1. We use MaskRCNN supported by detectron2 to generate instance masks first. Please follow [detectron2](https://github.com/facebookresearch/detectron2) to setup the environment, clone the code and download R50-FPN-3x(model_final_f10217.pkl) from [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). 

2. Run the instance segmentation model on images, filtering the results (only keep `person` class with threshold higher than 0.7) and save each instance mask in a seperate mask file. For simplicity, you can put `misc/image_demo.py` under the detectron2 repo `detectron2/demo/image_demo.py` and run the following command. Assume we want to generate instance masks for HIM2K natural subset,

```
cd detectron2/demo

python image_demo.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input /PATH/TO/HIM2K/natural/*.jpg \
  --output /PATH/TO/HIM2K/masks/natural \
  --opts MODEL.WEIGHTS ../pretrained/model_final_f10217.pkl
```


3. Then generate filelist for training or evaluation,

```
cd InstMatt

python misc/gen_filelist.py \
  --input /PATH/TO/HIM2K/masks/natural \
  --image-prefix /PATH/TO/HIM2K/images/natural \
  --image-path datasets/HIM2K_natural_image.txt \
  --mask-path datasets/HIM2K_natural_mask.txt

```

#### Training

1. Prepare training datasets (Refer to [Synthetic Dataset](datasets/README.md)). 
2. Train InstMatt without multi-instance refinement module.


```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --stage 1
    --config=config/InstMatt-stage1.toml

```

3. Train InstMatt with multi-instance refinement module.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --stage 2
    --config=config/InstMatt-stage2.toml

```

#### Evaluation

1. Prepare HIM2K benchmark as above mentioned and the well-trained model. Or download our model from [Goolge Drive](https://drive.google.com/file/d/1i_zQEqSG2i86G2jSz2IxgbanfyWqu-BF/view?usp=sharing) and put it under `checkpoints`.
2. Run inference on the benckmark to obtain instance-level predictions first (without evaluation).

```
CUDA_VISIBLE_DEVICES=0 python infer.py \
   --config config/InstMatt-stage2.toml \
   --checkpoint checkpoints/InstMatt/best_model.pth \
   --image-dir datasets/HIM2K_natural_image.txt \
   --mask-dir datasets/HIM2K_natural_mask.txt \
   --output results/natural \
   --image-ext jpg \
   --mask-ext png \
   --guidance-thres 170

```
3. Evaluate the saved results.

```
cd evaluation
python IMQ.py [PRED_FOLDER] [GT_FOLDER]
```

### TODO List

- [x] Benchmark
- [x] Training
- [x] Evalution
- [ ] Demo

### Reference

If you find our work useful in your research, please consider citing:

```
@inproceedings{sun2022instmatt,
  author    = {Yanan Sun and Chi-Keung Tang and Yu-Wing Tai}
  title     = {Human Instance Matting via Mutual Guidance and Multi-Instance Refinement},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2022},
}
```
