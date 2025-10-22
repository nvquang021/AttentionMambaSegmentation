# AttentionMambaSegmentation
Our source code will be coming soon! 
## Project Structure
## Project Structure

```
AttentionMambaSegmentation/
├── README.md                       # Project overview, setup, and usage guide
├── requirements.txt                # Dependencies (torch, monai, timm, mamba-ssm, albumentations, etc.)
├── LICENSE
│
├── configs/                        # Config files for datasets and models
│   ├── ph2.yaml
│   ├── dsb2018.yaml
│   ├── sunnybrook.yaml
│   └── chestxray.yaml
│
├── data/                           # Datasets 
│   ├── PH2/
│   │   ├── PH2_dataloader.py
│   ├── DSB2018/
│   │   ├── DSB2018_dataloader.py
│   ├── Sunnybrook/
│   │   ├── Sunnybrook_dataloader.py
│   ├── ChestXray/
│   │   └── Lung_dataloader.py
│
├── src/
│   ├── models/                     # Model architecture modules
│   │   ├── backbone/               # Feature Extractor
│   │   ├── blocks/                 # Attention and Skip connection
│   │   ├── decoder/                # Decoder or segmentation heads
│   │   └── model_builder.py        # Combine backbone + decoder
│   │
│   ├── loss/                       # Loss functions
│   │   ├── asfm_loss.py            # Adaptive Sigmoid Fowlkes–Mallows
│   │   └── other_losses.py
│   │
│   ├── trainer/                    # Training and evaluation
│   │   ├── train.py                # Main training loop
│   │   ├── validate.py             # Validation/testing
│   │   ├── metrics.py              # Dice, IoU, Precision, Recall, etc.
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   │
│   ├── inference/                  # Inference and visualization
│   │   ├── infer_single.py
│   │   ├── infer_batch.py
│   │   └── visualization.py        # Overlay predictions, qualitative evaluation
│   │
│   ├── utils/                      # Reusable utility modules
│   │   ├── logger.py               # TensorBoard/WandB logging
│   │   ├── checkpoint.py           # Save/load model
│   │   ├── seed.py                 # Deterministic reproducibility
│   │   ├── config_parser.py        # Parse YAML config
│   │   └── visualization_utils.py
│   │
│   └── experiments/                # Dataset-specific training scripts
│       ├── run_ph2.sh
│       ├── run_dsb2018.sh
│       ├── run_sunnybrook.sh
│       └── run_chestxray.sh
│
└── results/                        # Outputs of experiments
|   ├── logs/
|   ├── checkpoints/
|   ├── metrics/
|   └── figures/
└── LICENSE
```
