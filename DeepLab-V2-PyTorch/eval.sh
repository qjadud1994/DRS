# Evaluation DeepLab-V2 trained with pseudo segmentation labels

DATASET=voc12
LOG_DIR=Deeplabv2_pseudo_segmentation_labels_2gpu

CUDA_VISIBLE_DEVICES=0 python main.py test \
-c configs/${DATASET}.yaml \
-m data/models/${LOG_DIR}/deeplabv2_resnet101_msc/*/checkpoint_final.pth  \
--log_dir=${LOG_DIR}

# evaluate the model with CRF post-processing
CUDA_VISIBLE_DEVICES=0 python main.py crf \
-c configs/${DATASET}.yaml \
--log_dir=${LOG_DIR}
