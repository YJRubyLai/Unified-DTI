import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import StructureImageDataset
from model import ContrastiveModel
from loss import MultiPositiveInfoNCELoss, compute_alignment_metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics_acc = {'pos_sim_mean': 0, 'neg_sim_mean': 0, 'gap': 0, 'alignment': 0}

    for batch in tqdm(dataloader, desc="Training"):
        struct_embeds = batch['structure'].to(device)
        img_embeds = batch['image'].to(device)
        labels = torch.tensor(batch['id'], dtype=torch.long).to(device)

        struct_proj, img_proj = model(struct_embeds, img_embeds)
        loss = criterion(struct_proj, img_proj, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            m = compute_alignment_metrics(struct_proj, img_proj, labels)
            for k in metrics_acc:
                metrics_acc[k] += m[k]

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in metrics_acc.items()}


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_acc = {'pos_sim_mean': 0, 'neg_sim_mean': 0, 'gap': 0, 'alignment': 0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            struct_embeds = batch['structure'].to(device)
            img_embeds = batch['image'].to(device)
            labels = torch.tensor(batch['id'], dtype=torch.long).to(device)

            struct_proj, img_proj = model(struct_embeds, img_embeds)
            loss = criterion(struct_proj, img_proj, labels)

            total_loss += loss.item()
            m = compute_alignment_metrics(struct_proj, img_proj, labels)
            for k in metrics_acc:
                metrics_acc[k] += m[k]

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in metrics_acc.items()}


def main():
    cfg = CONFIG
    device = cfg['device']
    print(f"Using device: {device}")

    os.makedirs(cfg['save_dir'], exist_ok=True)

    train_dataset = StructureImageDataset(
        cfg['train_csv'], cfg['drug_struct_path'], cfg['drug_img_path'],
        cfg['prot_struct_path'], cfg['prot_img_path'],
    )
    test_dataset = StructureImageDataset(
        cfg['test_csv'], cfg['drug_struct_path'], cfg['drug_img_path'],
        cfg['prot_struct_path'], cfg['prot_img_path'],
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    model = ContrastiveModel(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        output_dim=cfg['output_dim'],
        dropout=cfg['dropout'],
    ).to(device)

    criterion = MultiPositiveInfoNCELoss(temperature=cfg['temperature'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

    best_val_loss = float('inf')
    history = []

    for epoch in range(cfg['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_pos': train_metrics['pos_sim_mean'], 'train_neg': train_metrics['neg_sim_mean'],
            'train_gap': train_metrics['gap'],
            'val_pos': val_metrics['pos_sim_mean'], 'val_neg': val_metrics['neg_sim_mean'],
            'val_gap': val_metrics['gap'],
        }
        history.append(record)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train → pos: {train_metrics['pos_sim_mean']:.4f}, neg: {train_metrics['neg_sim_mean']:.4f}, gap: {train_metrics['gap']:.4f}")
        print(f"Val   → pos: {val_metrics['pos_sim_mean']:.4f}, neg: {val_metrics['neg_sim_mean']:.4f}, gap: {val_metrics['gap']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'best_model.pt'))
            print(f"  => New best model saved (val_loss={val_loss:.4f})")

        with open(os.path.join(cfg['save_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
