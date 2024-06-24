import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

def train_model():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss:.4f} \tLR: {curr_lr}')
        val_loss = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss:.4f}\n')
        scheduler.step(val_loss)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
            })

        if True:
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': model.gnn.state_dict(),
                'bert_state_dict': model.bert.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')

    return 'Training completed'

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

train_data_loader, val_data_loader = load_data(config)
config['sch']['steps'] = len(train_data_loader)

model = create_model(config)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

GNN_state_dict = torch.load(f'./checkpoints/individual_pretrained/balanced_data/GNN/{config["task"]}/model.pt')
model.gnn.load_state_dict(GNN_state_dict['gnn_state_dict'])

BERT_state_dict = torch.load(f'./checkpoints/individual_pretrained/balanced_data/BERT/{config["task"]}/model.pt')
model.bert.protbert.load_state_dict(BERT_state_dict['bert_state_dict'])

save_dir = './checkpoints/temp'
shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
shutil.copy('./model/network.py', f'{save_dir}/network.py')
if not config['debug']:
    run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='PeptideFold', name=run_name)

    save_dir = f'./checkpoints/CLIP_balanced_data/{config["task"]}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('./model/network.py', f'{save_dir}/network.py')

train_model()
wandb.finish()
