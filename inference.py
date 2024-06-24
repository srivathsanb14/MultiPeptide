import torch
import yaml
import numpy as np
import random
from tqdm import tqdm
from data.dataloader import load_data
from model.network import create_model
from sklearn.metrics import balanced_accuracy_score as bas


def get_embeddings(dataloader, model, device):
    gnn_embeddings, bert_embeddings = [], []
    labels = []

    for batch in tqdm(dataloader, leave=False):
        batch = batch.to(device)
        output = model(batch)

        bert_embeddings.append(output[0].detach().cpu().numpy())
        gnn_embeddings.append(output[1].detach().cpu().numpy())
        labels.extend(batch.label.cpu().numpy())

    return np.vstack(bert_embeddings).T, np.vstack(gnn_embeddings).T, labels


def evaluate(val_data_loader, model, bert_embeddings, gnn_embeddings, labels, device):
    bert_labels, gnn_labels = [], []
    ground_truth = []
    num_correctb, num_correctg = 0, 0

    with torch.inference_mode():

        for batch in tqdm(val_data_loader, leave=False):
            batch = batch.to(device)
            output = model(batch)

            bert_output = output[0].detach().cpu().numpy()
            gnn_output = output[1].detach().cpu().numpy()

            bert_pred = np.argmax(
                bert_output @ bert_embeddings,
                axis=1
            )
            gnn_pred = np.argmax(
                gnn_output @ gnn_embeddings,
                axis=1
            )

            bert_label = list(map(labels.__getitem__, bert_pred))
            gnn_label = list(map(labels.__getitem__, gnn_pred))

            num_correctb += np.sum(
                bert_label == batch.label.cpu().numpy()
            )
            num_correctg += np.sum(
                gnn_label == batch.label.cpu().numpy()
            )

            bert_labels.extend(bert_label)
            gnn_labels.extend(gnn_label)
            ground_truth.extend(batch.label.cpu().numpy())

    bert_accuracy = num_correctb / len(val_data_loader.dataset)
    gnn_accuracy = num_correctg / len(val_data_loader.dataset)

    return bert_accuracy


def main(train_data_loader, val_data_loader, model, device, task):

    bert_embeddings, gnn_embeddings, labels = get_embeddings(train_data_loader, model, device)
    accuracy = evaluate(val_data_loader, model, bert_embeddings, gnn_embeddings, labels, device)
    
    return accuracy

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device
    task = config["task"]

    train_data_loader, val_data_loader = load_data(config)

    torch.manual_seed(0)
    
    model = create_model(config)
    model.eval()

    model_state_dict = torch.load(f'./checkpoints/CLIP_balanced_data/{config["task"]}/model.pt', map_location=device)
    model.bert.load_state_dict(model_state_dict, strict=False)

    accuracy = main(train_data_loader, val_data_loader, model, device, task)
    print(f'Accuracy: {accuracy}')
        
        