# Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation (AAAI 2021)

Official pytorch implementation of our paper:
Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation [[Paper]](https://arxiv.org/abs/2103.07246), Beomyoung Kim, Sangeun Han, and Junmo Kim, AAAI 2021

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-suppression-for-weakly/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=discriminative-region-suppression-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-suppression-for-weakly/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=discriminative-region-suppression-for-weakly)

We propose the discriminative region suppression (DRS) module that is a simple yet effective method to expand object activation regions. DRS suppresses the attention on discriminative regions and spreads it to adjacent non-discriminative regions, generating dense localization maps.

<img src = "https://github.com/qjadud1994/DRS/blob/main/docs/DRS_CAM.png" width="60%" height="60%">

![DRS module](https://github.com/qjadud1994/DRS/blob/main/docs/DRS_module.png)

## Setup

1. Dataset Preparing

    * [Download PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
    * you can obtain `SegmentationClassAug/` [[link]](https://drive.google.com/drive/folders/1_ik8n5Q4C77X-aIfKiqidFEDQ6zY9JNM?usp=sharing) (augmented with SBD dataset).
    * [Download saliency maps](https://drive.google.com/drive/folders/1I-456-_OFVWhZdCBBPW9NSIkr0H3FBeP?usp=sharing) used for background cues.
    
    ~~~
    # dataset structure
    VOC2012/
        --- Annotations/
        --- ImageSets/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
        --- refined_pseudo_segmentation_labels/
    ~~~


2. Requirements
    `pip install -r requirements.txt`


## Training & Pseudo Segmentation Labels Generation
* step1 : training the classifier with DRS modules
* step2 : training the refinement network for the localization maps refinement
* step3 : pseudo segmentation labels generation

~~~ 
# all-in-one
bash run.sh 
~~~

| Model | pretrained |
| :----:  | :----:    |
| VGG-16 with the learnable DRS | [DRS_learnable/best.pth](https://drive.google.com/drive/folders/1AyKsOmJd_241BNCYp_MKs9mCdr0i27Qs?usp=sharing) |
| Refinement network | [Refine_DRS_learnable/best.pth](https://drive.google.com/drive/folders/1w50rhVTGBJXW4oCJ88DpsieXABSWJhaG?usp=sharing) |
|                    |          |
| Pseudo Segmentation Labels | [refined_pseudo_segmentation_labels/](https://drive.google.com/drive/folders/1IS9_YCrRJwz3c7y3KwTET2_dYUPlNYo6?usp=sharing) |


## Training the DeepLab-V2 using pseudo labels
We adopt the DeepLab-V2 pytorch implementation from https://github.com/kazuto1011/deeplab-pytorch.

~~~
cd DeepLab-V2-PyTorch/

# motify the dataset path (DATASET.ROOT)
vi configs/voc12.yaml

# 1. training the DeepLab-V2 using pseudo labels
bash train.sh

# 2. evaluation the DeepLab-V2
bash eval.sh
~~~

| Model | mIoU | mIoU + CRF | pretrained |
| :----:  | :----: | :----: | :----: |
| DeepLab-V2 with ResNet-101 | 69.4% | 70.4% | [[link]](https://drive.google.com/drive/folders/1zJnRI5WRnv4cL9XY5jAojwIcO7MrUwun?usp=sharing)

* Note that the pretrained weight path
`./DeepLab-V2-Pytorch/data/models/Deeplabv2_pseudo_segmentation_labels/deeplabv2_resnet101_msc/train_cls/checkpoint_final.pth`

<img src = "https://github.com/qjadud1994/DRS/blob/main/docs/DRS_segmap.png" width="60%" height="60%">

## Citation
We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
~~~
@article{kim2021discriminative,
  title={Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation},
  author={Kim, Beomyoung and Kim, Sangeun Han and Kim, Junmo},
  journal={arXiv preprint arXiv:2103.07246},
  year={2021}
}
~~~
