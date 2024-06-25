import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import yaml
from data.dataloader import load_data
from matplotlib.colors import LinearSegmentedColormap
from transformers import BertModel, BertConfig, logging
import torch_geometric as PyG
import argparse

# python tSNE_individual_embeddings.py --train_from_scratch to train from scratch
# python tSNE_individual_embeddings.py for inference from pre_trained models

logging.set_verbosity_error()

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        self.GConv = PyG.nn.SAGEConv(input_dim, 128)
        self.FC = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, hidden_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv(x, edge_index)
        x = self.FC(x)
        return PyG.nn.global_max_pool(x, data.batch)

class PeptideBERT(torch.nn.Module):
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()
        self.protbert = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd',
            config=bert_config,
            ignore_mismatched_sizes=True
        )

    def forward(self, inputs, attention_mask):
        output = self.protbert(inputs, attention_mask=attention_mask)
        return output.pooler_output

def train_gnn_model(train_loader, gnn_model, config):
    gnn_model.train()
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=config['optim']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
    #for epoch in range(50): # If shorter training is preferred
        for batch in tqdm(train_loader):
            batch = batch.to(config['device'])
            optimizer.zero_grad()
            output = gnn_model(batch)
            labels = batch.label.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return gnn_model

def train_bert_model(train_loader, bert_model, config):
    bert_model.train()
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=config['optim']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
    #for epoch in range(10): # If shorter training is preferred
        for batch in tqdm(train_loader):
            batch = batch.to(config['device'])
            optimizer.zero_grad()
            output = bert_model(batch.seq, batch.attn_mask)
            labels = batch.label.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return bert_model


def get_gnn_embeddings(dataloader, gnn_model):
    gnn_model.eval()
    gnn_embeddings = []
    labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            gnn_embs = gnn_model(batch)
            gnn_embeddings.append(gnn_embs.cpu().numpy())
            labels.extend(batch.label.cpu().numpy())

    return np.vstack(gnn_embeddings), np.array(labels)

def get_bert_embeddings(dataloader, bert_model):
    bert_model.eval()
    bert_embeddings = []
    labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            bert_embs = bert_model(batch.seq, batch.attn_mask)
            bert_embeddings.append(bert_embs.cpu().numpy())
            labels.extend(batch.label.cpu().numpy())

    return np.vstack(bert_embeddings), np.array(labels)


def evaluate_accuracy(dataloader, model, is_gnn=True):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            if is_gnn:
                outputs = model(batch)
            else:
                outputs = model(batch.seq, batch.attn_mask)
            _, predicted = torch.max(outputs, 1)
            labels = batch.label.long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def plot_embeddings(embeddings, labels, title, image_path, use_pca=True, plot_3d=False):
    if use_pca:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=3 if plot_3d else 2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    colors = [(0, 0.3, 0.8), (1, 0, 0)]
    n_bins = 100  
    cmap_name = 'custom_bwr'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    if plot_3d:
        fig = plt.figure(figsize=[5, 4], dpi=600)
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left = -.3)
        scatter = ax.scatter3D(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels, cmap=custom_cmap, s=10, alpha=0.7)
        ax.set_zlabel('Dimension 3', fontsize=12)
    else:
        width = 1.5
        fig, ax = plt.subplots(figsize=[5, 3], dpi=600)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(width)
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap=custom_cmap, s=10, alpha=0.7)

    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, title + (" - 3D.png" if plot_3d else " - 2D.png")))
    plt.close()

def save_model(model, model_name, checkpoint_path):
    torch.save(model.state_dict(), os.path.join(checkpoint_path, model_name + '.pth'))

def load_model(model, model_name, checkpoint_path):
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, model_name + '.pth')))
    return model

def main(gnn_model, bert_model, task, use_pca=True, train_from_scratch=True):
    image_path = os.path.join('plots/tSNE/', task)
    checkpoint_path = os.path.join('checkpoints/trained_from_scratch', task)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    train_data_loader, val_data_loader = load_data(config)

    if train_from_scratch:
        print("Training GNN model from scratch...")
        gnn_instance = train_gnn_model(train_data_loader, gnn_model, config)
        save_model(gnn_instance, "gnn_model", checkpoint_path)
        
        print("Training PeptideBERT model from scratch...")
        bert_instance = train_bert_model(train_data_loader, bert_model, config)
        save_model(bert_instance, "bert_model", checkpoint_path)
    else:
        print("Loading pre-trained GNN model...")
        gnn_instance = load_model(gnn_model, "gnn_model", checkpoint_path)
        
        print("Loading pre-trained PeptideBERT model...")
        bert_instance = load_model(bert_model, "bert_model", checkpoint_path)

    gnn_embeddings, labels = get_gnn_embeddings(val_data_loader, gnn_instance)
    bert_embeddings, _ = get_bert_embeddings(val_data_loader, bert_instance)

    plot_embeddings(gnn_embeddings, labels, "GNN Embeddings", image_path, use_pca, plot_3d=False)
    plot_embeddings(bert_embeddings, labels, "PeptideBERT Embeddings", image_path, use_pca, plot_3d=False)

    plot_embeddings(gnn_embeddings, labels, "GNN Embeddings", image_path, use_pca, plot_3d=True)
    plot_embeddings(bert_embeddings, labels, "PeptideBERT Embeddings", image_path, use_pca, plot_3d=True)

    gnn_accuracy = evaluate_accuracy(val_data_loader, gnn_instance, is_gnn=True)
    bert_accuracy = evaluate_accuracy(val_data_loader, bert_instance, is_gnn=False)
    
    print(f'GNN Model Accuracy: {gnn_accuracy * 100:.2f}%')
    print(f'PeptideBERT Model Accuracy: {bert_accuracy * 100:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Load models for task")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train models from scratch")

    args = parser.parse_args()

    torch.manual_seed(0)

    checkpoint = 'CLIP_balanced_data/nf'
    config = yaml.load(open(f'./checkpoints/{checkpoint}/config.yaml', 'r'), Loader=yaml.FullLoader)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    task = config['task']

    gnn_model = GNN(input_dim=config['network']['GNN']['input_dim'], hidden_dim=config['network']['GNN']['hidden_dim']).to(device)

    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['BERT']['hidden_size'],
        num_hidden_layers=config['network']['BERT']['hidden_layers'],
        num_attention_heads=config['network']['BERT']['attn_heads'],
        hidden_dropout_prob=config['network']['BERT']['dropout']
    )
    bert_model = PeptideBERT(bert_config).to(device)

    main(gnn_model, bert_model, task, train_from_scratch=args.train_from_scratch)
