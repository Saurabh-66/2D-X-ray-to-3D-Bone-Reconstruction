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
# Single image (AP only — duplicated as lateral)
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
# Clean start (delete old data/checkpoints first — see below)
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
# Terminal 1 — Backend (FastAPI)
uv run python backend/main.py    # http://localhost:8000

# Terminal 2 — Frontend (React + Vite)
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

## Potential Technical Q&A (Judging Panel)

**Q: Describe the architecture of your method, how you evaluated it, where the data came from, and what accuracy metrics you used for the generated 3D models.**
A: **Architecture**: Our model is based on X2BR (2025). It has two stages — a ConvNeXt encoder that takes one or two X-ray views (AP and optional lateral, each 224×224 grayscale) and compresses them into a 1024-dim feature vector, followed by a neural implicit occupancy decoder. The decoder takes any 3D coordinate (x, y, z) plus the image features and predicts the probability that point is inside bone. We use Conditional Batch Normalization (CBN) and DenseNet-style skip connections in the decoder. For biplanar input, both view features are concatenated (2048-dim) and projected back to 1024-dim before decoding. At inference, we query a 64³ grid of points, threshold at 0.5, and run marching cubes to extract a 3D triangle mesh exported as GLB.

**Data**: We use the CADS dataset from HuggingFace — 3 subsets: TotalSegmentator (1,203 full-body CTs), VerSe (450 spine CTs), and RibFrac (360 rib/chest CTs). From each CT volume we extract the bone segmentation mask (label 5) as ground truth, and generate synthetic X-rays called Digitally Reconstructed Radiographs (DRRs) by simulating X-ray physics — ray-casting through the full CT volume and integrating attenuation, just like a real X-ray machine. We generate 8 angle variations per subject (AP, lateral, and rotated views), giving us ~106K training samples (~84K train, ~22K val).

**Evaluation & Metrics**: We evaluate using the **Dice coefficient** — the volumetric overlap between the predicted 3D bone voxels and the ground-truth bone mask. Dice ranges from 0 (no overlap) to 1 (perfect). Our current best validation Dice is **~0.64**. We also visually inspect reconstructed GLB meshes against ground truth at validation checkpoints every 10 epochs. The Dice score captures overall shape fidelity — at 0.64, the model reconstructs recognizable bone anatomy (spine, ribcage, pelvis) but misses fine detail like individual vertebral processes. We additionally track training loss (BCE on occupancy predictions) and monitor for overfitting by comparing train vs validation loss curves.

**Q: How does your model reconstruct 3D structure from a single 2D image? Isn't that an ill-posed problem?**
A: Yes, it's inherently ill-posed — a single 2D projection has infinite consistent 3D interpretations. The model learns a strong anatomical prior from ~84K training samples: the ConvNeXt encoder compresses the X-ray into a 1024-dim feature vector, and the implicit decoder learns to predict occupancy at any 3D query point conditioned on that vector. It's essentially learning "given this X-ray appearance, what's the most likely 3D bone shape?" When a second (lateral) view is available, we fuse both feature vectors, which resolves much of the depth ambiguity.

**Q: What is an implicit neural representation, and why use it over voxel regression?**
A: Instead of directly outputting a 64³ voxel grid, our decoder takes a 3D coordinate (x, y, z) as input and predicts the probability that point is inside bone. At inference we query a dense grid of points and threshold to get the voxel volume. The advantage is resolution-agnostic training — we can train on sparse point samples (2048 per example) rather than full 64³ grids, and at inference we can use MISE (Multi-resolution IsoSurface Extraction) to output at 256³ without retraining.

**Q: How do you generate training data from CT scans?**
A: We use Digitally Reconstructed Radiographs (DRRs). Given a 3D CT volume, we simulate X-ray projection by ray-casting through the volume and integrating attenuation — the same physics as a real X-ray machine. We generate 8 angle variations per subject (AP, lateral, and rotated views). The ground-truth 3D bone mask comes from the CT segmentation (CADS label 5 = bone). This gives us paired (X-ray → 3D bone) training data without needing real paired clinical datasets.

**Q: How does your preprocessing pipeline handle real X-rays vs synthetic DRRs?**
A: DRR images have known intensity distributions, but real X-rays vary wildly in contrast, exposure, and equipment. Our `normalize_to_drr` preprocessing applies percentile-based normalization (clipping at 1st and 99th percentiles) plus gamma correction to map real X-ray intensities into the DRR-like range the model was trained on. Users can toggle this off for synthetic inputs via the UI checkbox.

**Q: What metric do you use to evaluate reconstruction quality?**
A: Dice coefficient (volumetric overlap between predicted and ground-truth bone voxels). It ranges from 0 (no overlap) to 1 (perfect match). Our current best validation Dice is ~0.64, which captures overall bone shape well but misses fine detail. We also visually inspect reconstructions via saved GLB meshes at validation time.

**Q: Why ConvNeXt over a standard ResNet or Vision Transformer?**
A: ConvNeXt modernizes the ConvNet design with transformer-inspired tricks (larger kernels, LayerNorm, GELU, inverted bottleneck) while keeping the efficiency and inductive biases of convolutions. For medical images — which are single-channel, grayscale, and texture-heavy — ConvNets tend to be more data-efficient than ViTs. ConvNeXt gives us transformer-level performance without needing massive pretraining datasets.

**Q: How do you go from the voxel grid to a 3D mesh?**
A: Marching cubes algorithm. The implicit decoder outputs occupancy probabilities on a 64³ grid. We threshold at 0.5 to get a binary volume, then run marching cubes to extract the isosurface as a triangle mesh. We export as GLB (binary glTF) for web viewing. With MISE, we can adaptively refine to 256³ by only subdividing regions near the surface boundary.

**Q: What are the main failure modes?**
A: (1) Anatomy not well-represented in training data — e.g., extremities like hands/feet are rarer in CADS. (2) Overlapping structures in the X-ray projection can confuse the model. (3) The model can hallucinate symmetric structures when given only a single AP view. (4) Fine structures like thin fracture lines or small bone spurs are below the 64³ voxel resolution.

**Q: How does biplanar fusion help compared to single-view?**
A: The AP view gives good left-right and up-down spatial information but collapses front-back depth. The lateral view provides the missing depth axis. We concatenate both 1024-dim feature vectors into a 2048-dim vector and project back to 1024-dim. In practice, biplanar input significantly reduces depth ambiguity — ribs, vertebral bodies, and pelvic structures are much better resolved.

**Q: What's the inference time?**
A: On a GPU, single-view inference takes ~1-2 seconds (feature extraction + querying 64³ = 262K points + marching cubes). MISE at 256³ takes longer (~10-15s) due to iterative refinement. The web app streams the GLB result back to the browser for interactive 3D viewing via model-viewer.

**Q: How does this compare to CT in terms of clinical utility?**
A: CT gives ~0.5mm isotropic resolution with full soft tissue contrast. Our output is a coarse bone-only mesh (~1-2mm effective resolution at 64³). It's not a CT replacement for diagnosis, but it could enable: (1) surgical planning where only X-rays are available, (2) screening for gross skeletal abnormalities, (3) reducing unnecessary CT referrals, and (4) 3D visualization for patient education. The key advantage is 99% less radiation (0.1 mSv vs 10 mSv) and dramatically lower cost.

**Q: What would you need to do to make this clinically deployable?**
A: (1) Train on real paired X-ray + CT datasets (not just synthetic DRRs), (2) validate on diverse patient populations and imaging equipment, (3) increase resolution (128³ or 256³ training), (4) add uncertainty quantification so clinicians know when to trust the output, (5) regulatory approval (FDA 510(k) or CE marking), and (6) integration with PACS/DICOM clinical workflows.

**Q: Why did you build your own model instead of using an existing 3D reconstruction model?**
A: General 3D reconstruction models (like Hunyuan3D, TripoSR) are trained on natural images and objects — they don't understand medical anatomy. Medical X-rays are grayscale projections with overlapping structures, fundamentally different from photos. We needed a domain-specific architecture trained on anatomical data to learn the relationship between X-ray appearance and 3D bone structure.
