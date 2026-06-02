#!/bin/bash

# Backbone Type
# export bb="R50" 
# export bb="swinT" 

# Set config file
# export config_path="configs/TTA/COCO_${bb}.yaml"

# # Direct Test
# python tools/train_net.py \
#   	--config-file "./configs/COCO-Detection/faster_rcnn_${bb}_FPN_1x.yaml" --eval-only TEST.CONTINUAL_DOMAIN "True" OUTPUT_DIR outputs/COCO/${bb}_direct_test


# # smi
# export where="smi"
# for continual in False True
# do python tools/train_net.py \
#   	--config-file ${config_path} \
#   	--eval-only --wandb \
#   	TEST.ADAPTATION.WHERE ${where} \
#   	TEST.ADAPTATION.CONTINUAL ${continual} \
# 	OUTPUT_DIR outputs/COCO/${bb}_smi_${continual}
# done

# # FFTswinT-ours
# export where="adapter"
# for continual in False True
# do python tools/train_net.py \
#   	--config-file "./configs/TTA/COCO_FFTswinT.yaml" \
#   	--eval-only --wandb \
#   	TEST.ADAPTATION.WHERE ${where} \
#   	TEST.ADAPTATION.CONTINUAL ${continual} \
# 	OUTPUT_DIR outputs/COCO/FFTswinT_${continual}
# done

# # FFTR50-ours
# export where="adapter"
# for continual in False True
# do python tools/train_net.py \
#   	--config-file "./configs/TTA/COCO_FFTR50.yaml" \
#   	--eval-only --wandb \
#   	TEST.ADAPTATION.WHERE ${where} \
#   	TEST.ADAPTATION.CONTINUAL ${continual} \
# 	OUTPUT_DIR outputs/COCO/FFTR50_${continual}
# done


# # Ours-Recon-SwinT
# export where="adapter"
# for continual in False True
# do python tools/train_net.py \
#   	--config-file "./configs/TTA/COCO_swinTrecon.yaml" \
#   	--eval-only --wandb \
#   	TEST.ADAPTATION.WHERE ${where} \
#   	TEST.ADAPTATION.CONTINUAL ${continual} \
# 	OUTPUT_DIR outputs/COCO/swinTrecon_dynamicthr
# done

# # Ours-Recon-R50
# export where="adapter"
# for continual in False True
# do python tools/train_net.py \
#   	--config-file "./configs/TTA/COCO_R50recon.yaml" \
#   	--eval-only --wandb \
#   	TEST.ADAPTATION.WHERE ${where} \
#   	TEST.ADAPTATION.CONTINUAL ${continual} \
# 	OUTPUT_DIR outputs/COCO/R50recon_dynamicthr
# done




