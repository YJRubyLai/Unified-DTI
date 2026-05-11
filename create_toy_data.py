"""
Create toy (synthetic) data for testing the pipeline without the full embeddings.
Generates small .pt embedding files and matching CSV files.

Usage:
    python create_toy_data.py

Output structure:
    toy_data/
        Stage1.1_structure/
            data/  train_pairs.csv, test_pairs.csv
            embedding/  SMILE_embedding.pt, protein_embedding.pt
        Stage1.2_image/
            data/  train_pairs.csv, test_pairs.csv
            embedding/  All_compound_image.pt, All_CRISPR_image.pt
        stage2/
            data/  train_pairs.csv, test_pairs.csv
            embedding/  contrastive_new_*.pt (all 4 files)
"""

import os
import torch
import pandas as pd
import numpy as np

# Dimensions matching the real data
SMILE_DIM = 384        # ChemBERTa embedding dim
PROTEIN_DIM = 1280     # ESM-2 650M embedding dim
COMPOUND_IMG_DIM = 737 # Compound image embedding dim
CRISPR_IMG_DIM = 599   # CRISPR image embedding dim
PROJ_DIM = 512         # Stage 1 projected embedding dim

# Number of toy samples
N_DRUGS = 10
N_PROTEINS = 8
N_TRAIN_PAIRS = 40
N_TEST_PAIRS = 16

torch.manual_seed(42)

# --- Toy SMILES and protein sequences ---
TOY_SMILES = [
    'CC(=O)Oc1ccccc1C(=O)O',
    'c1ccc(cc1)O',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',
    'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
    'CC(=O)NC1=CC=C(C=C1)O',
    'OC1=CC=CC=C1C(=O)O',
    'CN(C)CCC(c1ccccc1)c1ccccn1',
    'BrC1=CCCN(Cc2ccccn2)C1',
]

TOY_PROTEINS = [
    'MKTIIALSYIFCLVFA',
    'MVLSPADKTNVKAAW',
    'MTEYKLVVVGAGGVGKSALTIQLIQNHFV',
    'MASMTGGQQMGRDPNS',
    'MGSSHHHHHHSSGLVPRGSHMASMTG',
    'MKTAYIAKQRQISFVKSHFSRQ',
    'MEEPQSDPSVEPPLSQETFSDLWKLLP',
    'MGNCCCRRGAARSSSGTPQRDRDQERTV',
]


def make_embed_dict(keys, dim):
    """Create a dict of random unit-normalized embeddings."""
    return {k: torch.nn.functional.normalize(torch.randn(dim), dim=0) for k in keys}


def make_pairs_csv(smiles_list, protein_list, n_pairs, path):
    """Create a CSV with random drug-target pairs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_pairs):
        smile = smiles_list[rng.integers(len(smiles_list))]
        protein = protein_list[rng.integers(len(protein_list))]
        rows.append({'SMILES': smile, 'Protein': protein})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path} ({n_pairs} rows)")


def make_stage2_pairs_csv(smiles_list, protein_list, n_pairs, path):
    """Create a Stage 2 CSV with structure, image, id columns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_pairs):
        smile = smiles_list[rng.integers(len(smiles_list))]
        protein = protein_list[rng.integers(len(protein_list))]
        pair_id = f"{i // 4}_GENE{rng.integers(5)}"
        # Each pair appears as 4 combinations: (drug_struct, drug_img), (drug_struct, prot_img), etc.
        for struct_key, img_key in [
            (smile, smile),
            (smile, protein),
            (protein, smile),
            (protein, protein),
        ]:
            rows.append({'structure': struct_key, 'image': img_key, 'id': pair_id})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path} ({len(rows)} rows)")


def main():
    BASE = 'toy_data'
    print("Creating toy data...")

    # ===== Stage 1.1 Structure =====
    print("\n[Stage 1.1 Structure]")
    smile_embeds = make_embed_dict(TOY_SMILES, SMILE_DIM)
    protein_embeds = make_embed_dict(TOY_PROTEINS, PROTEIN_DIM)

    emb_dir = os.path.join(BASE, 'Stage1.1_structure', 'embedding')
    os.makedirs(emb_dir, exist_ok=True)
    torch.save(smile_embeds, os.path.join(emb_dir, 'SMILE_embedding.pt'))
    torch.save(protein_embeds, os.path.join(emb_dir, 'protein_embedding.pt'))
    print(f"  Saved SMILE_embedding.pt ({len(smile_embeds)} entries, dim={SMILE_DIM})")
    print(f"  Saved protein_embedding.pt ({len(protein_embeds)} entries, dim={PROTEIN_DIM})")

    make_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TRAIN_PAIRS,
                   os.path.join(BASE, 'Stage1.1_structure', 'data', 'train_pairs.csv'))
    make_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TEST_PAIRS,
                   os.path.join(BASE, 'Stage1.1_structure', 'data', 'test_pairs.csv'))

    # ===== Stage 1.2 Image =====
    print("\n[Stage 1.2 Image]")
    compound_img_embeds = make_embed_dict(TOY_SMILES, COMPOUND_IMG_DIM)
    crispr_img_embeds = make_embed_dict(TOY_PROTEINS, CRISPR_IMG_DIM)

    emb_dir = os.path.join(BASE, 'Stage1.2_image', 'embedding')
    os.makedirs(emb_dir, exist_ok=True)
    torch.save(compound_img_embeds, os.path.join(emb_dir, 'All_compound_image.pt'))
    torch.save(crispr_img_embeds, os.path.join(emb_dir, 'All_CRISPR_image.pt'))
    print(f"  Saved All_compound_image.pt ({len(compound_img_embeds)} entries, dim={COMPOUND_IMG_DIM})")
    print(f"  Saved All_CRISPR_image.pt ({len(crispr_img_embeds)} entries, dim={CRISPR_IMG_DIM})")

    make_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TRAIN_PAIRS,
                   os.path.join(BASE, 'Stage1.2_image', 'data', 'train_pairs.csv'))
    make_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TEST_PAIRS,
                   os.path.join(BASE, 'Stage1.2_image', 'data', 'test_pairs.csv'))

    # ===== Stage 2 Unified =====
    print("\n[Stage 2 Unified]")
    new_smile_embs = make_embed_dict(TOY_SMILES, PROJ_DIM)
    new_protein_embs = make_embed_dict(TOY_PROTEINS, PROJ_DIM)
    new_img_drug_embs = make_embed_dict(TOY_SMILES, PROJ_DIM)
    new_img_protein_embs = make_embed_dict(TOY_PROTEINS, PROJ_DIM)

    emb_dir = os.path.join(BASE, 'stage2', 'embedding')
    os.makedirs(emb_dir, exist_ok=True)
    torch.save(new_smile_embs, os.path.join(emb_dir, 'contrastive_new_smile_embeddings.pt'))
    torch.save(new_protein_embs, os.path.join(emb_dir, 'contrastive_new_protein_embeddings.pt'))
    torch.save(new_img_drug_embs, os.path.join(emb_dir, 'contrastive_new_img_drug_embeddings.pt'))
    torch.save(new_img_protein_embs, os.path.join(emb_dir, 'contrastive_new_img_protein_embeddings.pt'))
    print(f"  Saved 4 projected embedding files ({PROJ_DIM}-dim each)")

    make_stage2_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TRAIN_PAIRS // 4,
                          os.path.join(BASE, 'stage2', 'data', 'train_pairs.csv'))
    make_stage2_pairs_csv(TOY_SMILES, TOY_PROTEINS, N_TEST_PAIRS // 4,
                          os.path.join(BASE, 'stage2', 'data', 'test_pairs.csv'))

    print("\nToy data created successfully in ./toy_data/")
    print("To run Stage 1.1 with toy data, update config.py paths to point to toy_data/...")


if __name__ == '__main__':
    main()
