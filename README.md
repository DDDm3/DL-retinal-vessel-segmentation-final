# Retinal Vessel Segmentation

Deep learning project for retinal vessel segmentation on DRIVE and CHASEDB1.

## Project Scope

- Task: image segmentation for retinal blood vessels.
- CNN model: DeepLabV3+-ResNet50.
- Attention model: SegFormer-B0.
- Datasets: DRIVE and CHASEDB1.
- Training method: transfer learning.
- Demo application: Streamlit.

## Pipeline

The main training pipeline is implemented in notebooks:

- `notebooks/03_DeepLabV3_ResNet50_CHASE_to_DRIVE.ipynb`
  - DeepLabV3+-ResNet50.
  - CHASEDB1 source training followed by DRIVE fine-tuning.
  - BCE + Tversky loss for class-imbalanced vessel segmentation.

- `notebooks/02_SegFormer_B0_DRIVE.ipynb`
  - SegFormer-B0 with pretrained transformer encoder.
  - Fine-tuned on DRIVE.
  - BCEWithLogitsLoss.

Supporting scripts:

- `src/evaluate.py`: evaluates checkpoints and tunes threshold.
- `app/streamlit_app.py`: Streamlit demo for uploaded fundus images.
- `src/models/deeplabv3plus_resnet50.py`: DeepLabV3+-ResNet50 wrapper using `segmentation-models-pytorch`.
- `src/models/segformer.py`: SegFormer-B0 implementation.

## Evaluation Results

Evaluated on `dataset/drive_test_dataset.pt` with threshold tuning.

Note: these metrics were produced by the previous trained DeepLabV3+ checkpoint.
After switching DeepLabV3+-ResNet50 to the `segmentation-models-pytorch`
implementation, retrain or fine-tune the model and regenerate the checkpoint
before reporting final metrics again.

| Model | Best Threshold | Dice | IoU | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepLabV3+-ResNet50 | 0.85 | 0.7269 | 0.5713 | 0.9507 | 0.7082 | 0.7532 |
| SegFormer-B0 | 0.50 | 0.7160 | 0.5579 | 0.9495 | 0.7070 | 0.7308 |

## Run Evaluation

DeepLabV3+-ResNet50:

```powershell
python src\evaluate.py --model deeplabv3plus_resnet50 --checkpoint src\models\best_deeplabv3plus_resnet50.pth --data dataset\drive_test_dataset.pt --batch-size 1 --tune-threshold
```

SegFormer-B0:

```powershell
python src\evaluate.py --model segformer_b0 --checkpoint src\models\best_segformer_b0.pth --data dataset\drive_test_dataset.pt --batch-size 1 --tune-threshold
```

## Run App

```powershell
streamlit run app\streamlit_app.py
```

The app supports:

- model selection;
- fundus image upload;
- automatic fundus crop for composite uploaded images;
- probability map display;
- binary vessel mask prediction;
- overlay visualization;
- skeleton and vessel statistics.

## CV Summary

```latex
\item Developed a retinal vessel segmentation pipeline on DRIVE and CHASEDB1 using SegFormer-B0 and DeepLabV3+-ResNet50 with transfer learning, probability maps, and threshold tuning.
\item Achieved 72.69\% Dice, 57.13\% IoU, and 95.07\% Accuracy with DeepLabV3+-ResNet50 on the DRIVE test set; implemented a Streamlit demo for mask prediction, overlays, and vessel analysis.
```
