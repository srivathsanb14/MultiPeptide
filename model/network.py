import torch
import torch_geometric as PyG
from transformers import BertModel, BertConfig, logging
import matplotlib.pyplot as plt

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


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(ProjectionHead, self).__init__()

        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class PretrainNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, bert_config, projection_dim, dropout):
        super(PretrainNetwork, self).__init__()

        self.gnn = GNN(input_dim, hidden_dim)
        self.bert = PeptideBERT(bert_config)
        self.graph_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
        self.text_projection = ProjectionHead(bert_config.hidden_size, projection_dim, dropout)

    def forward(self, data):
        gnn_features = self.gnn(data)
        bert_features = self.bert(data.seq, data.attn_mask)

        gnn_embs = self.graph_projection(gnn_features)
        bert_embs = self.text_projection(bert_features)

        gnn_embs = gnn_embs / torch.linalg.norm(gnn_embs, dim=1, keepdim=True)
        bert_embs = bert_embs / torch.linalg.norm(bert_embs, dim=1, keepdim=True)

        return bert_embs, gnn_embs


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, bert_embs, gnn_embs, labels):
        gnn_sim = gnn_embs @ gnn_embs.T
        bert_sim = bert_embs @ bert_embs.T
        targets = self.softmax((gnn_sim + bert_sim) / (2 * self.temperature))
        logits = (bert_embs @ gnn_embs.T) / self.temperature

        gnn_loss = self.cross_entropy(logits.T, targets.T)
        bert_loss = self.cross_entropy(logits, targets)
        loss = (gnn_loss + bert_loss) / 2

        return loss.mean()

    def cross_entropy(self, logits, targets):
        log_probs = self.logsoftmax(logits)
        return (-targets * log_probs).sum(1)


def create_model(config, get_embeddings=False):
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['BERT']['hidden_size'],
        num_hidden_layers=config['network']['BERT']['hidden_layers'],
        num_attention_heads=config['network']['BERT']['attn_heads'],
        hidden_dropout_prob=config['network']['BERT']['dropout']
    )
    
    model = PretrainNetwork(
        input_dim=config['network']['GNN']['input_dim'],
        hidden_dim=config['network']['GNN']['hidden_dim'],
        bert_config=bert_config,
        projection_dim=config['network']['proj_dim'],
        dropout=config['network']['drp']
    ).to(config['device'])

    return model


def cri_opt_sch(config, model):
    criterion = CLIPLoss(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )

    return criterion, optimizer, scheduler
