# 2D X-ray to 3D Bone Reconstruction
Submission for Expo AI Hackathon (sponsors: Anthropic, Google Cloud)

## Brief Idea
We are building an AI tool that turns standard 2D X-rays into 3D bone structure. Our goal is to solve two massive global problems:

- The Cancer Risk: Standard 3D CT scans expose patients to high radiation, contributing to an estimated 1.5% to 2% of all cancer cases worldwide. Globally, this equates to hundreds of thousands of future cancer diagnoses every year. Our AI aims to reduce this radiation exposure by 99% (from around 10 mSv down to 0.1 mSv).

- The Accessibility Gap: Nearly 4 billion people
(around half the world), lack access to 3D imaging. While a CT scanner can cost hundreds of thousands of dollars to buy and maintain, our software-based approach is 10x cheaper, making 3D surgical planning possible for rural and low-income clinics.

Our Goal for the Expo: We are aiming for a demo to show this 2D-to-3D transformation in real-time. We want to demonstrate how we can make surgery safer and more affordable for millions of people who currently have no 3D imaging options.

## Model Architecture

Based on [X2BR](https://arxiv.org/abs/2504.08675) (High-Fidelity 3D Bone Reconstruction, 2025):

- **Encoder**: ConvNeXt-based, shared for AP and Lateral X-ray views → 1024-dim feature vector
- **Fusion**: Biplanar concatenation + projection (our extension for two views)
- **Decoder**: Neural implicit occupancy decoder with Conditional Batch Normalization (CBN) and DenseNet-style blocks
- **Output**: Point-based occupancy prediction → 64³ voxel grid → 3D mesh (GLB)
- **Parameters**: ~43.8M

### Training Details
- **Loss**: BCE with balanced point sampling (50% near bone surface, 50% uniform)
- **Optimizer**: AdamW (lr=1e-4, weight decay=1e-4)
- **Scheduler**: Cosine annealing (1e-4 → 1e-6)
- **Augmentation**: Horizontal flip, intensity jitter, Gaussian noise, 90° rotation
- **Data workers**: 8 with pin_memory + persistent_workers for GPU efficiency

## Dataset

### CADS (multi-subset)
| Subset | Volumes | Body Region | Bone Mask |
|--------|---------|-------------|-----------|
| [0037_totalsegmentator](https://huggingface.co/datasets/huggingface/CADS-dataset) | 1,203 | Full body | part_559 label 5 |
| [0010_verse](https://huggingface.co/datasets/huggingface/CADS-dataset) | 450 | Spine (vertebrae) | part_559 label 5 |
| [0013_ribfrac](https://huggingface.co/datasets/huggingface/CADS-dataset) | 360 | Ribs + chest | part_559 label 5 |

**Current config**: 250 subjects/subset × 3 subsets × 8 angle variations = **~106K samples** (~84K train, ~22K val). Covers full-body bone (spine, ribs, pelvis, limbs, skull). DRRs are generated from full CT volumes (not bone masks) for realistic X-ray appearance with soft tissue.

Data pipeline uses **batch processing**: downloads N subjects at a time from HuggingFace, generates training samples, deletes raw data, moves to next batch. Keeps disk usage low.

## Usage

### Inference (local)

```bash
# Single image (AP only - duplicated as lateral)
python -m model.inference --ap xray.png -o output.glb

# Biplanar (AP + Lateral)
python -m model.inference --ap xray_ap.png --lat xray_lat.png -o output.glb

# Batch: process all images in test_images/
python -m model.inference --input-dir test_images/ --output-dir output/

# High-resolution output with MISE (256³ from 64³ training)
python -m model.inference --ap xray.png -o output.glb --mise
python -m model.inference --ap xray.png -o output.glb --mise --mise-resolution 128
```

### Sync results from cluster

```bash
bash scripts/sync_cluster.sh              # Download checkpoints, logs, val samples
bash scripts/sync_cluster.sh --best-only  # Best-epoch val samples + best.pt only
bash scripts/sync_cluster.sh --dry-run    # Preview what would be synced
```

### Training (SLURM cluster)

```bash
# Clean start (delete old data/checkpoints first - see below)
sbatch slurm_train.sh
```

Training features:
- **Continuous checkpoint backup**: best.pt and latest.pt backed up to home disk every 5 epochs
- **Resumable**: If interrupted, next run automatically resumes from latest.pt
- **Validation samples**: PNG X-ray inputs + GLB 3D mesh outputs saved every 10 epochs
- **Auto data pipeline**: Downloads CADS from HuggingFace in batches, generates training data, backs up, then trains

### Data generation only (local or cluster)

```bash
# Full pipeline: download 250 subjects + generate training data
python data_factory/build_dataset.py --output-dir ./training_data --max-subjects 250

# Quick test: 5 subjects, 2 angles
python data_factory/build_dataset.py --output-dir /tmp/test --max-subjects 5 --num-angles 2
```

### Clean start (delete old runs)

```bash
# On the cluster:
rm -rf ~/Expo-AI/model/backups/*
rm -rf ~/Expo-AI/model/checkpoints/*
rm -rf ~/Expo-AI/model/logs/*
cd ~/Expo-AI && git pull && sbatch slurm_train.sh
```

### Web App

```bash
# Terminal 1 - Backend (FastAPI)
uv run python backend/main.py    # http://localhost:8000

# Terminal 2 - Frontend (React + Vite)
cd frontend && npm install
npm run dev                       # http://localhost:5173
```

Open http://localhost:5173, upload AP (+ optional lateral) X-rays, and view the 3D reconstruction interactively.

## Project Structure

```
Expo-AI/
├── model/
│   ├── architecture.py      # X2BR-inspired ConvNeXt encoder + implicit decoder
│   ├── train.py             # Training loop with continuous backup + val samples
│   ├── inference.py         # Single/batch inference → GLB output
│   ├── dataset.py           # PyTorch dataset with augmentation
│   ├── losses.py            # BCE loss for occupancy prediction
│   ├── checkpoints/         # best.pt, latest.pt
│   └── logs/                # Training logs, SLURM logs, val_samples/
├── data_factory/
│   ├── build_dataset.py     # CADS download + training data generation
│   └── download_cads.py     # HuggingFace CADS dataset downloader
├── backend/
│   └── main.py              # FastAPI server (upload → inference → GLB)
├── frontend/
│   └── src/App.tsx           # React + Tailwind UI with 3D GLB viewer
├── scripts/
│   ├── sync_cluster.sh      # Sync checkpoints, logs, val samples from cluster
│   └── fetch_test_xrays.py  # Download real X-ray AP+LAT pairs for testing
├── slurm_train.sh           # SLURM job script (download data from GCS + train)
└── pyproject.toml           # Python deps (managed with uv)
```

## Other References
- [Biplanar X-ray to 3D Benchmark](https://openreview.net/forum?id=NoE8g3LRAM)
- [Hunyuan3D-2](https://huggingface.co/spaces/tencent/Hunyuan3D-2)

## Technical Insights

**Data**: We use the CADS dataset from HuggingFace - 3 subsets: TotalSegmentator (1,203 full-body CTs), VerSe (450 spine CTs), and RibFrac (360 rib/chest CTs). From each CT volume we extract the bone segmentation mask (label 5) as ground truth, and generate synthetic X-rays called Digitally Reconstructed Radiographs (DRRs) by simulating X-ray physics — ray-casting through the full CT volume and integrating attenuation, just like a real X-ray machine. We generate 8 angle variations per subject (AP, lateral, and rotated views), giving us ~106K training samples (~84K train, ~22K val).

**Evaluation & Metrics**: We evaluate using the **Dice coefficient**, the volumetric overlap between the predicted 3D bone voxels and the ground-truth bone mask. Dice ranges from 0 (no overlap) to 1 (perfect). Our current best validation Dice is **~0.64**. We also visually inspect reconstructed GLB meshes against ground truth at validation checkpoints every 10 epochs. The Dice score captures overall shape fidelity - at 0.64, the model reconstructs recognizable bone anatomy (spine, ribcage, pelvis) but misses fine detail like individual vertebral processes. We additionally track training loss (BCE on occupancy predictions) and monitor for overfitting by comparing train vs validation loss curves.
