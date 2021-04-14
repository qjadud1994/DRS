# Training DeepLab-V2 using pseudo segmentation labels

DATASET=voc12_2gpu
LOG_DIR=Deeplabv2_pseudo_segmentation_labels_2gpu
GT_DIR=refined_pseudo_segmentation_labels
CUDA_VISIBLE_DEVICES=0,1 python main.py train -c configs/${DATASET}.yaml --gt_path=${GT_DIR} --log_dir=${LOG_DIR}

#---------------------------------------------------------------------------------------------------------------------

DATASET=voc12_4gpu
LOG_DIR=Deeplabv2_pseudo_segmentation_labels_4gpu
GT_DIR=refined_pseudo_segmentation_labels
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train -c configs/${DATASET}.yaml --gt_path=${GT_DIR} --log_dir=${LOG_DIR}