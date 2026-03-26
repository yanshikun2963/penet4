#!/bin/bash
# ================================================================
#  penet4: RA-SGG 实验（使用预训练文件，最快路径）
#  预期结果: mR@50=36.2, R@50=62.2 (AAAI 2025 Table 1验证)
#  预计时间: 搭建1h + 训练8h = 总计9h
# ================================================================
set -e

echo "================================================================"
echo "  Step 0: 环境准备"
echo "================================================================"

# 你需要修改以下路径为你的实际路径
WORK_DIR="/path/to/your/workdir"          # 工作目录
PENET_DIR="${WORK_DIR}/penet4"            # 你的PE-NET仓库（本仓库）
DATASET_DIR="${PENET_DIR}/datasets/vg"    # VG数据集路径
GLOVE_DIR="${DATASET_DIR}"                # GloVe路径
DETECTOR_CKPT="${PENET_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth"
GPU_ID=0                                  # 使用的GPU编号

# ================================================================
echo "  Step 1: 克隆 RA-SGG 代码库"
echo "================================================================"

cd ${WORK_DIR}
if [ ! -d "torch-rasgg" ]; then
    git clone https://github.com/KanghoonYoon/torch-rasgg.git
    cd torch-rasgg
    python setup.py build develop
else
    echo "torch-rasgg already exists, skipping clone"
    cd torch-rasgg
fi

# ================================================================
echo "  Step 2: 下载预训练文件"
echo "================================================================"
echo ""
echo "请手动下载以下文件（Google Drive）："
echo ""
echo "1. 预训练PE-NET checkpoint:"
echo "   https://drive.google.com/drive/folders/11jh-8F3LR8Hmm0Vsdp10Xc9PBbYQxnhK"
echo "   下载后放到: ${WORK_DIR}/torch-rasgg/checkpoints/PE-NET_PredCls/model_final.pth"
echo ""
echo "2. Memory Bank 特征文件:"
echo "   https://drive.google.com/drive/folders/16sRrrmYfyK2jq12P0JUB7iXx8xG965tS"
echo "   下载后放到: ${WORK_DIR}/torch-rasgg/ 根目录下"
echo ""
echo "3. 预训练ReTAG模型（可选，用于直接测试）:"
echo "   https://drive.google.com/drive/folders/1sWjGxiczFdL7wk9Xg7oVNo67awUTnu3m"
echo ""
echo "按任意键继续（确认文件已下载）..."
read -n 1

# ================================================================
echo "  Step 3: 链接数据集"
echo "================================================================"

cd ${WORK_DIR}/torch-rasgg
ln -sf ${DATASET_DIR}/../.. ./datasets 2>/dev/null || true
ln -sf ${PENET_DIR}/checkpoints/pretrained_faster_rcnn ./checkpoints/pretrained_faster_rcnn 2>/dev/null || true

# ================================================================
echo "  Step 4: 预训练检查"
echo "================================================================"

echo -n "PE-NET checkpoint: "
ls ./checkpoints/PE-NET_PredCls/model_final.pth 2>/dev/null && echo "✅" || echo "❌ 缺失！"

echo -n "Memory Bank: "
ls ./feats/ 2>/dev/null && echo "✅" || echo "❌ 可能缺失，检查下载的文件"

echo -n "数据集: "
ls ./datasets/vg/VG-SGG-with-attri.h5 2>/dev/null && echo "✅" || echo "❌ 缺失！"

echo -n "Detector: "
ls ${DETECTOR_CKPT} 2>/dev/null && echo "✅" || echo "❌ 缺失！"

echo -n "GPU: "
python3 -c "import torch; print(f'✅ {torch.cuda.get_device_name(${GPU_ID})}')" 2>/dev/null || echo "❌"

# ================================================================
echo "  Step 5: 开始训练"
echo "================================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

mkdir -p ./checkpoints/predcls/RETAG/

nohup python3 -u tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml" \
  TYPE "retag" \
  REL_LOSS_TYPE "ce" \
  REWEIGHT_BETA 0.99999 \
  RASGG.MEMORY_SIZE 8 \
  RASGG.NUM_CORRECT_BG 1 \
  RASGG.NUM_RETRIEVALS 20 \
  RASGG.THRESHOLD 0.3 \
  RASGG.MIXUP True \
  RASGG.MIXUP_ALPHA 20 \
  RASGG.MIXUP_BETA 5 \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR ReTAGPENet \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 6 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 2500 \
  SOLVER.CHECKPOINT_PERIOD 2500 \
  GLOVE_DIR ${GLOVE_DIR} \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/PE-NET_PredCls/model_final.pth \
  OUTPUT_DIR ./checkpoints/predcls/RETAG \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  2>&1 | tee ./checkpoints/predcls/RETAG/train.log &

echo ""
echo "================================================================"
echo "  RA-SGG 训练已启动！PID: $!"
echo "  监控: tail -f ${WORK_DIR}/torch-rasgg/checkpoints/predcls/RETAG/train.log"
echo "  预期: mR@50 ≈ 36.2, R@50 ≈ 62.2"
echo "  每2500步验证一次，约30分钟可看到第一次验证结果"
echo "================================================================"
