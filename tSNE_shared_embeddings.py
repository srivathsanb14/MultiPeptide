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
from model.network import create_model

checkpoint = 'CLIP_balanced_data/hemo'
split_path = 'full/'
split = split_path[:-1]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0) # Plots might be seed-dependent

def get_embeddings(dataloader):
    model.eval()
    embeddings = []
    labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            output = model(batch)[0]
            embeddings.append(output.cpu().numpy())
            labels.extend(batch.label.cpu().numpy())

    return np.vstack(embeddings), np.array(labels)


def plot_embeddings(embeddings, labels, title, image_path, use_pca=True, plot_3d=False):
    if use_pca:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=3 if plot_3d else 2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    colors = [(0, 0.3, 0.8), (1, 0, 0)]  # Lighter blue and red
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


def main(task, use_pca=True):
    image_path = os.path.join('plots/tSNE/', task)
    os.makedirs(image_path, exist_ok=True)

    train_data_loader, val_data_loader = load_data(config)

    embeddings, labels = get_embeddings(val_data_loader)
    plot_embeddings(embeddings, labels, "Shared test embeddings", image_path, use_pca, plot_3d=False)
    plot_embeddings(embeddings, labels, "Shared test embeddings", image_path, use_pca, plot_3d=True)


if __name__ == "__main__":

    torch.manual_seed(0)

    config = yaml.load(open(f'./checkpoints/{checkpoint}/config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device

    task = config['task']

    model = create_model(config, get_embeddings=True)
    model.load_state_dict(torch.load(f'./checkpoints/{checkpoint}/model.pt')['bert_state_dict'], strict=False)

    main(task)
