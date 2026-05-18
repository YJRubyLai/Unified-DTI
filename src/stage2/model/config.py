import torch

CONFIG = {
    # Data paths (relative to repo root)
    'train_csv': 'stage2/data/train_pairs.csv',
    'test_csv': 'stage2/data/test_pairs.csv',
    'drug_struct_path': 'stage2/embedding/contrastive_new_smile_embeddings.pt',
    'drug_img_path': 'stage2/embedding/contrastive_new_img_drug_embeddings.pt',
    'prot_struct_path': 'stage2/embedding/contrastive_new_protein_embeddings.pt',
    'prot_img_path': 'stage2/embedding/contrastive_new_img_protein_embeddings.pt',

    # Model hyperparameters
    'input_dim': 512,   # projected embeddings from Stage 1 are all 512-dim
    'hidden_dim': 1024,
    'output_dim': 512,
    'dropout': 0.1,

    # Training hyperparameters
    'batch_size': 256,
    'learning_rate': 1e-5,
    'num_epochs': 30,
    'temperature': 0.10,
    'weight_decay': 1e-3,

    # Output
    'save_dir': 'stage2/model/checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
