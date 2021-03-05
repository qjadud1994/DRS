# Training Classifier with DRS
CUDA_VISIBLE_DEVICES=0,1 python3 ./scripts/train_cls.py \
    --img_dir=/data/DB/VOC2012/ \
    --train_list=/data/DB/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --test_list=/data/DB/VOC2012/ImageSets/Segmentation/train_labeled_cls.txt \
    --epoch=15 \
    --lr=0.001 \
    --batch_size=10 \
    --input_size=384 \
    --crop_size=321 \
    --num_classes=20 \
    --num_workers=4 \
    --decay_points='5,10' \
    --logdir=logs/DRS_learnable \
    --save_folder=checkpoints/DRS_learnable \
    --show_interval=50


# Generating localization maps for refinement learning
CUDA_VISIBLE_DEVICES=0 python  ./scripts/localization_map_gen.py \
    --img_dir=/data/DB/VOCdevkit/VOC2012/ \
    --train_list=/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --test_list=/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_labeled_cls.txt \
    --checkpoint=DRS_learnable/ckpt_15.pth


# Refinement learning
CUDA_VISIBLE_DEVICES=0,1 python3 ./scripts/train_att.py \
    --img_dir=/data/DB/VOCdevkit/VOC2012/ \
    --train_list=/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --test_list=/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_labeled_cls.txt \
    --epoch=15 \
    --lr=0.0001 \
    --batch_size=10 \
    --dataset=pascal_voc \
    --input_size=384 \
    --crop_size=321 \
    --num_classes=20 \
    --num_workers=10 \
    --decay_points='5,10' \
    --logdir=logs/refinement_from_learnable/ \
    --save_folder=checkpoints/refinement_from_learnable/ \

