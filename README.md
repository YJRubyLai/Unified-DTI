# Unified-DTI: Multimodal Contrastive Learning for Drug-Target Interaction Prediction

Ying-Ju Lai, Tianyuzhou Liang, Po-Yuan Chen, Yu-Che Tsai, George C. Tseng, Yufei Huang, Yu-Chiao Chiu

*Bioinformatics*, 2025 — Proceedings of ECCB2026

---

Accurate prediction of drug-target interactions (DTIs) is fundamental to drug discovery and mechanistic understanding. Most existing methods rely on molecular structure and protein sequence while overlooking cellular phenotypes that reflect downstream biological effects. We propose a two-stage contrastive learning framework that integrates drug structures, protein sequences, and **Cell Painting** morphological profiles into a unified embedding space for DTI prediction.

Cell Painting is a high-throughput multiplexed fluorescence imaging assay that captures systems-level morphological responses to chemical and genetic perturbations. The key biological intuition is that drugs which inhibit a target and CRISPR knockouts of that same target both disrupt the same downstream cellular pathways — producing similar morphological signatures — even though their molecular representations are completely different modalities.

Stage 1 trains two independent contrastive models: one aligning drug SMILES (ChemBERTa-2) with protein sequences (ESM-2), and another aligning compound Cell Painting profiles with CRISPR knockdown profiles. Stage 2 then bridges these two modality-specific spaces via multi-positive contrastive learning, producing a single unified 512-dim embedding that jointly encodes molecular binding and cellular phenotype.



![Model Workflow](model_workflow.png)

## Data

Training uses 282,236 DTI pairs curated from the [MOTIVE dataset](https://github.com/carpenter-singh-lab/2024_Arevalo_NeurIPS_MotiVE), retaining only direct binding interactions. Cell Painting profiles for drugs and CRISPR knockouts are sourced from the [JUMP Cell Painting Gallery (cpg0016)](https://github.com/jump-cellpainting/datasets).

Large embedding files (`.pt`) and trained model checkpoints are stored on **OneDrive** due to file size — download link below.

> **OneDrive data download**: *(link)*

After downloading, place files under the corresponding `embedding/` folders as described in the repository structure below.

## Installation

```bash
git clone https://github.com/YJRubyLai/Unified-DTI.git
cd Unified-DTI
pip install -r requirements.txt
```

## Run

Training proceeds in three steps. Each step reads from and writes to paths set in the stage's `config.py` — update these to match your local paths before running.

**Step 1 — Structure-based contrastive model**

Trains on pre-computed ChemBERTa-2 SMILES embeddings (384-dim) and ESM-2 protein embeddings (1280-dim), projecting both to a shared 512-dim space via InfoNCE loss.

```bash
cd Stage1.1_structure/model
python train.py
```

**Step 2 — Image-based contrastive model**

Trains on compound Cell Painting profiles (737-dim) and CRISPR knockdown profiles (599-dim), projecting both to 512-dim via InfoNCE loss.

```bash
cd Stage1.2_image/model
python train.py
```

**Step 3 — Unified multimodal alignment**

First, generate projected 512-dim embeddings from the two trained Stage 1 models:

```bash
python stage2/embedding/generate_embeddings.py
```

Then train the Stage 2 model, which aligns structure and image embeddings using multi-positive InfoNCE (drug-target pairs sharing the same ID are positives):

```bash
cd stage2/model
python train.py
```

## Quick Test with Toy Data

To verify the pipeline without downloading the full embeddings, generate small synthetic data:

```bash
python create_toy_data.py
```

This creates a `toy_data/` directory with matching `.pt` files and CSVs. Update the paths in each `config.py` to point there.

## Repository Structure

```
Unified-DTI/
├── create_toy_data.py
├── requirements.txt
├── Stage1.1_structure/
│   ├── data/
│   │   ├── train_pairs.csv          # columns: SMILES, Protein
│   │   └── test_pairs.csv
│   ├── embedding/                   # download from OneDrive
│   │   ├── SMILE_embedding.pt       # {SMILES -> tensor[384]}
│   │   └── protein_embedding.pt     # {protein_seq -> tensor[1280]}
│   └── model/
│       ├── config.py
│       ├── dataset.py
│       ├── model.py
│       ├── loss.py
│       └── train.py
├── Stage1.2_image/
│   ├── data/
│   │   ├── train_pairs.csv          # columns: SMILES, Protein
│   │   └── test_pairs.csv
│   ├── embedding/                   # download from OneDrive
│   │   ├── All_compound_image.pt    # {SMILES -> tensor[737]}
│   │   └── All_CRISPR_image.pt      # {protein_seq -> tensor[599]}
│   └── model/
│       ├── config.py
│       ├── dataset.py
│       ├── model.py
│       ├── loss.py
│       └── train.py
└── stage2/
    ├── data/
    │   ├── train_pairs.csv          # columns: structure, image, id
    │   └── test_pairs.csv
    ├── embedding/
    │   ├── generate_embeddings.py
    │   └── ...                      # projected .pt files, download from OneDrive
    └── model/
        ├── config.py
        ├── dataset.py
        ├── model.py
        ├── loss.py
        └── train.py
```

## Citation

```bibtex
@article{lai2025unified,
  title   = {Multimodal contrastive learning for integrating molecular representations
             and cellular phenotypes in drug-target interaction prediction},
  author  = {Lai, Ying-Ju and Liang, Tianyuzhou and Chen, Po-Yuan and Tsai, Yu-Che
             and Tseng, George C. and Huang, Yufei and Chiu, Yu-Chiao},
  journal = {Bioinformatics},
  year    = {2025},
  doi     = {10.1093/bioinformatics/xxxxx}
}
```
