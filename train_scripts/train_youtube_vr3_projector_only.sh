#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/hf_cache
export TRANSFORMERS_CACHE=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/hf_cache
export TRITON_CACHE_DIR=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/triton_cache
export TORCH_EXTENSIONS_DIR=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/torch_extensions
export TORCH_HOME=/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/torch_cache

mkdir -p ${TORCH_EXTENSIONS_DIR}
mkdir -p ${TORCH_HOME}
mkdir -p ${TRITON_CACHE_DIR}

# ✅ Use LLaVA-OneVision-Qwen2 model
MODEL_PATH="lmms-lab/llava-onevision-qwen2-0.5b-ov"
VISION_MODEL="google/siglip-so400m-patch14-384"  # OneVision uses SigLIP

PRETRAINED_VISION_PROJECTOR="/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/LLAVA_TRY/ORG_lava/new_ww/cop2/out/extracted_projectors_checkpoint1102/vision_projector/vision_projector.bin"
PRETRAINED_TEXT_PROJECTOR="/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/LLAVA_TRY/ORG_lava/new_ww/cop2/out/extracted_projectors_checkpoint1102/text_projector/text_projector.bin"

DATA_PATH="/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/FData/youtube_VR_one_disc_train_transformed.json"
IMAGE_FOLDER="/data/gpfs/projects/punim3025/punim_2198_backup/hkanagalinga/FData"
OUTPUT_DIR="./checkpoints/llava-onevision-qwen2-finetuned-contrastive_projector_only_frozen_backbone"

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Fine-tuning LLaVA-OneVision with Contrastive Learning (projector only)"
echo "=============================================="
echo "Base Model: ${MODEL_PATH}"
echo "Vision Tower: ${VISION_MODEL} (FROZEN)"
echo "LLM: (FROZEN)"
echo "Pre-trained Vision Projector: ${PRETRAINED_VISION_PROJECTOR}"
echo "Pre-trained Text Projector: ${PRETRAINED_TEXT_PROJECTOR}"
echo "Dataset: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

torchrun --nproc_per_node=1 --master_port=29643 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version qwen_1_5 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --pretrained_vision_projector ${PRETRAINED_VISION_PROJECTOR} \
    --pretrained_text_projector ${PRETRAINED_TEXT_PROJECTOR} \
    --vision_tower ${VISION_MODEL} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name llava-onevision-qwen2-youtube-vr-contrastive-projector-only \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --save_safetensors False \
    --ddp_find_unused_parameters True \
    --enable_contrastive_learning True \
    --contrastive_loss_weight 0.5 \
    --contrastive_projector_lr 1e-4 \
    --vision_embed_dim 256 \
    --contrastive_temperature 0.07 \
    --contrastive_margin 0.5

echo "=============================================="
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=============================================="
