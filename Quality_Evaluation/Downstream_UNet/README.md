# Downstream_UNet

`Downstream_UNet` is the task model used only for the downstream replacement
experiment. It is separate from both `Evaluation_UNet` and `FgSeg_UNet`.

Protocol:

- Input: normalized complete multispectral image plus the requested `label_id`.
- Target: the corresponding one-channel foreground mask.
- Architecture: five U-Net stages with channels `(16, 32, 64, 128, 256)`.
- Training: initialize a fresh model for `Real-only` and another fresh model for
  `Syn-only`; use identical conditions, masks, augmentation, seed, and budget.
- Model selection: use only the real validation split.
- Scheduling and early stopping: follow the IMPGM training monitor; reduce the
  learning rate on a validation plateau and stop only after no improvement at
  the configured minimum learning rate.
- Final report: evaluate both selected models only on the isolated real test split.

The model is trained by the existing replacement evaluation entry point:

```powershell
python Evaluation/Evaluation_Code/IMPGM_Downstream_Utility_Evaluation.py `
  <train_generation_manifest.jsonl> `
  --output-root <output_directory>
```

Unlike the paper's complete multi-class medical masks, IMPGM stores one binary
target mask together with an image-level class label. The independent label
embedding adapts the downstream U-Net to that dataset contract without changing
the equal-sample Real-only versus Syn-only comparison.
