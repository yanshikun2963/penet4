# penet4: RA-SGG 实验（使用预训练文件）

## 实验信息
- **方法**: RA-SGG (Retrieval-Augmented Scene Graph Generation)
- **论文**: Yoon et al., "RA-SGG: Retrieval-Augmented Scene Graph Generation Framework via Multi-Prototype Learning", AAAI 2025
- **代码**: https://github.com/KanghoonYoon/torch-rasgg
- **验证数据**: mR@50=36.2, R@50=62.2 (论文Table 1, VG150 PredCls)

## CB-Loss正交性
RA-SGG通过检索增强发现VG150中缺失的细粒度标注，不使用per-class loss reweighting。
与CB-Loss完全正交——RA-SGG操作数据/标签层面，CB-Loss操作loss权重层面。

## 运行方式
```bash
# 1. 修改 experiments/round3/run_rasgg_pretrained.sh 中的路径
# 2. 下载预训练文件（Google Drive链接在脚本中）
# 3. 执行:
bash experiments/round3/run_rasgg_pretrained.sh
```

## 本仓库代码
本仓库包含干净的PE-NET代码（无Round 2修改，无CB-Loss）。
RA-SGG框架需要单独克隆 `torch-rasgg` 仓库来运行。
