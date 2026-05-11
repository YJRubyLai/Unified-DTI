"""
Generate projected embeddings from Stage 1 trained models.

This script takes the raw pre-computed embeddings (SMILE, protein, compound image,
CRISPR image) and projects them through the trained Stage 1 models to produce
unified 512-dim embeddings for Stage 2 training.

Usage:
    python generate_embeddings.py
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os

# Add Stage 1 model paths to allow importing their architectures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Stage1.1_structure/model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Stage1.2_image/model'))

# Import Stage 1.1 (Structure) model
import importlib
struct_model_module = importlib.import_module('model')  # Stage1.1 model.py
StructContrastiveModel = struct_model_module.ContrastiveModel

# Re-import Stage 1.2 (Image) model — has different architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Stage1.2_image/model'))
img_model_module = importlib.import_module('model')
ImgContrastiveModel = img_model_module.ContrastiveModel


STAGE1_STRUCTURE_CKPT = 'Stage1.1_structure/model/checkpoints/best_model.pt'
STAGE1_IMAGE_CKPT = 'Stage1.2_image/model/checkpoints/best_model.pt'

SMILE_EMBED_PATH = 'Stage1.1_structure/embedding/SMILE_embedding.pt'
PROTEIN_EMBED_PATH = 'Stage1.1_structure/embedding/protein_embedding.pt'
COMPOUND_IMAGE_EMBED_PATH = 'Stage1.2_image/embedding/All_compound_image.pt'
CRISPR_IMAGE_EMBED_PATH = 'Stage1.2_image/embedding/All_CRISPR_image.pt'

OUTPUT_DIR = 'stage2/embedding'


def project_embeddings(model, projector_fn, embed_dict, device, batch_size=512):
    """Project a dict of embeddings through a model projector."""
    keys = list(embed_dict.keys())
    projected = {}

    for i in tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i:i + batch_size]
        batch_tensor = torch.stack([embed_dict[k] for k in batch_keys]).to(device)
        with torch.no_grad():
            proj = projector_fn(batch_tensor)
        for k, v in zip(batch_keys, proj.cpu()):
            projected[k] = v

    return projected


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Stage 1.1 (Structure) model ---
    print("\nLoading Stage 1.1 (Structure) model...")
    struct_model = StructContrastiveModel(drug_dim=384, target_dim=1280, hidden_dim=1024, output_dim=512)
    struct_model.load_state_dict(torch.load(STAGE1_STRUCTURE_CKPT, map_location='cpu', weights_only=True))
    struct_model.to(device).eval()

    smile_dict = torch.load(SMILE_EMBED_PATH, weights_only=True)
    protein_dict = torch.load(PROTEIN_EMBED_PATH, weights_only=True)

    print(f"  Projecting {len(smile_dict)} SMILES embeddings...")
    new_smile_embs = project_embeddings(struct_model, struct_model.drug_projector, smile_dict, device)

    print(f"  Projecting {len(protein_dict)} protein embeddings...")
    new_protein_embs = project_embeddings(struct_model, struct_model.target_projector, protein_dict, device)

    torch.save(new_smile_embs, os.path.join(OUTPUT_DIR, 'contrastive_new_smile_embeddings.pt'))
    torch.save(new_protein_embs, os.path.join(OUTPUT_DIR, 'contrastive_new_protein_embeddings.pt'))
    print("  Saved structure projected embeddings.")

    # --- Load Stage 1.2 (Image) model ---
    print("\nLoading Stage 1.2 (Image) model...")
    img_model = ImgContrastiveModel(drug_dim=737, target_dim=599, hidden_dim=1024, output_dim=512)
    img_model.load_state_dict(torch.load(STAGE1_IMAGE_CKPT, map_location='cpu', weights_only=True))
    img_model.to(device).eval()

    compound_img_dict = torch.load(COMPOUND_IMAGE_EMBED_PATH, weights_only=True)
    crispr_img_dict = torch.load(CRISPR_IMAGE_EMBED_PATH, weights_only=True)

    print(f"  Projecting {len(compound_img_dict)} compound image embeddings...")
    new_img_drug_embs = project_embeddings(img_model, img_model.drug_projector, compound_img_dict, device)

    print(f"  Projecting {len(crispr_img_dict)} CRISPR image embeddings...")
    new_img_protein_embs = project_embeddings(img_model, img_model.target_projector, crispr_img_dict, device)

    torch.save(new_img_drug_embs, os.path.join(OUTPUT_DIR, 'contrastive_new_img_drug_embeddings.pt'))
    torch.save(new_img_protein_embs, os.path.join(OUTPUT_DIR, 'contrastive_new_img_protein_embeddings.pt'))
    print("  Saved image projected embeddings.")

    print("\nAll embeddings generated successfully.")
    print(f"  SMILE:          {len(new_smile_embs)} entries, dim={list(new_smile_embs.values())[0].shape}")
    print(f"  Protein:        {len(new_protein_embs)} entries, dim={list(new_protein_embs.values())[0].shape}")
    print(f"  Compound image: {len(new_img_drug_embs)} entries, dim={list(new_img_drug_embs.values())[0].shape}")
    print(f"  CRISPR image:   {len(new_img_protein_embs)} entries, dim={list(new_img_protein_embs.values())[0].shape}")


if __name__ == '__main__':
    main()
