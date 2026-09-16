# Evaluation_UNet

`Evaluation_UNet` 是 IMPGM 生成质量评估专用的完整影像分割评价器，不是
`FgSeg_UNet` 的替代训练入口。

## 数据协议

- 输入：IMPGM 标准化后的完整四波段 `syn_img` 与 `label_id`
- 目标：该类别对应的单通道二值 mask
- 训练：仅使用真实 train split
- 选模：仅使用真实 valid split
- 上限：在隔离的真实 test split 上报告 IoU、Dice、F1、Precision、Recall

模型使用类别嵌入，因此不同类别能够共享一个 U-Net，同时明确当前需要从完整影像中
分割的目标。每个数据集必须独立训练，其 checkpoint 中会记录数据集名称、类别数量和
输入域，统一生成评估器会验证这些元数据。

## 训练与测试

```powershell
$env:IMPGM_DATASET_YAML = "configs/datasets/main.yaml"
python Quality_Evaluation\Evaluation_UNet\Evaluation_UNet_train.py
python Quality_Evaluation\Evaluation_UNet\Evaluation_UNet_evaluate.py --split test
```

继续训练：

```powershell
python Quality_Evaluation\Evaluation_UNet\Evaluation_UNet_train.py --resume
```

默认 checkpoint：

```text
Quality_Evaluation/Evaluation_UNet/Models/<dataset>/Evaluation_UNet_<dataset>.pth
```

训练完成后，`Evaluation/Evaluation_Code/IMPGM_Generation_Evaluation.py` 会自动加载对应数据集的 evaluator。
若暂时不计算条件分割诊断，可传入 `--no-condition-segmentation`。
