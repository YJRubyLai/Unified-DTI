import torch
from torch.utils.data import Dataset
import pandas as pd


class StructureImageDataset(Dataset):
    """Dataset for structure-image pairs (unified multimodal DTI space).

    CSV must have columns: structure, image, id
    Embedding files are dicts: {key_string -> torch.Tensor (512-dim)}

    Both drug and protein embeddings are already projected to 512-dim by Stage 1 models.
    """

    def __init__(self, csv_path, drug_struct_path, drug_img_path, prot_struct_path, prot_img_path):
        self.data = pd.read_csv(csv_path)

        self.drug_structure = torch.load(drug_struct_path, weights_only=True)
        self.drug_image = torch.load(drug_img_path, weights_only=True)
        self.protein_structure = torch.load(prot_struct_path, weights_only=True)
        self.protein_image = torch.load(prot_img_path, weights_only=True)

        # Build stable string → int mapping for pair IDs (for multi-positive loss)
        unique_ids = sorted(self.data['id'].unique())
        self.id_to_int = {x: i for i, x in enumerate(unique_ids)}
        print(f"Encoded {len(unique_ids)} unique pair IDs.")

    def __len__(self):
        return len(self.data)

    def _get_embedding(self, key, column_type):
        if column_type == 'structure':
            if key in self.drug_structure:
                return self.drug_structure[key]
            if key in self.protein_structure:
                return self.protein_structure[key]
        elif column_type == 'image':
            if key in self.drug_image:
                return self.drug_image[key]
            if key in self.protein_image:
                return self.protein_image[key]
        raise KeyError(f"Key '{key[:30]}...' not found in any {column_type} embedding dict.")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'structure': self._get_embedding(row['structure'], 'structure'),
            'image': self._get_embedding(row['image'], 'image'),
            'structure_key': row['structure'],
            'image_key': row['image'],
            'id': self.id_to_int[row['id']],
        }
