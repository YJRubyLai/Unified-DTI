import torch

CONFIG = {
    # Data paths (relative to repo root)
    'train_csv': 'Stage1.1_structure/data/train_pairs.csv',
    'test_csv': 'Stage1.1_structure/data/test_pairs.csv',
    'drug_embeds_path': 'Stage1.1_structure/embedding/SMILE_embedding.pt',
    'target_embeds_path': 'Stage1.1_structure/embedding/protein_embedding.pt',

    # Model hyperparameters
    'hidden_dim': 1024,
    'output_dim': 512,

    # Training hyperparameters
    'batch_size': 256,
    'learning_rate': 1e-4,
    'num_epochs': 30,
    'temperature': 0.10,
    'weight_decay': 1e-5,

    # Output
    'save_dir': 'Stage1.1_structure/model/checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
