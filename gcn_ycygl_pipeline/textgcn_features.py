from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .prepare_dataset import Sample, simple_tokenize


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[rowsum == 0] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def sparse_to_torch(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def build_text_graph(
    samples: list[Sample],
    min_df: int = 1,
    max_features: int | None = None,
    use_pmi: bool = True,
    window_size: int = 20,
    pmi_threshold: float = 0.0,
) -> tuple[torch.Tensor, int, list[str]]:
    texts = [sample.text for sample in samples]
    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        norm=None,
        use_idf=True,
        smooth_idf=False,
        sublinear_tf=False,
    )
    tfidf = vectorizer.fit_transform(texts).astype(np.float32)
    vocab = list(vectorizer.get_feature_names_out())

    doc_count, vocab_count = tfidf.shape
    total_nodes = doc_count + vocab_count
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    coo = tfidf.tocoo()
    rows.extend(coo.row.tolist())
    cols.extend((coo.col + doc_count).tolist())
    vals.extend(coo.data.astype(float).tolist())
    rows.extend((coo.col + doc_count).tolist())
    cols.extend(coo.row.tolist())
    vals.extend(coo.data.astype(float).tolist())

    if use_pmi:
        vocab_index = {word: i for i, word in enumerate(vocab)}
        windows: list[list[int]] = []
        for text in texts:
            ids = [vocab_index[token] for token in simple_tokenize(text) if token in vocab_index]
            if not ids:
                continue
            if len(ids) <= window_size:
                windows.append(ids)
            else:
                for start in range(len(ids) - window_size + 1):
                    windows.append(ids[start : start + window_size])

        word_window_freq: Counter[int] = Counter()
        pair_count: Counter[tuple[int, int]] = Counter()
        for window in windows:
            unique = sorted(set(window))
            word_window_freq.update(unique)
            for i, left in enumerate(unique):
                for right in unique[i + 1 :]:
                    pair_count[(left, right)] += 1

        total_windows = max(len(windows), 1)
        for (left, right), count in pair_count.items():
            p_i_j = count / total_windows
            p_i = word_window_freq[left] / total_windows
            p_j = word_window_freq[right] / total_windows
            pmi = math.log(p_i_j / (p_i * p_j))
            if pmi > pmi_threshold:
                left_node = doc_count + left
                right_node = doc_count + right
                rows.extend([left_node, right_node])
                cols.extend([right_node, left_node])
                vals.extend([pmi, pmi])

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float32)
    return sparse_to_torch(normalize_adj(adj)), doc_count, vocab


class TextGCN(nn.Module):
    def __init__(self, node_count: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.empty(node_count, hidden_dim))
        self.classifier_weight = nn.Parameter(torch.empty(hidden_dim, num_classes))
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.classifier_weight)

    def hidden(self, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, self.node_embeddings)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.hidden(adj)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        support = x @ self.classifier_weight + self.classifier_bias
        return torch.sparse.mm(adj, support)


def split_indices(samples: list[Sample], split: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for idx, sample in enumerate(samples):
        name = split[sample.data_id]
        if name == "train":
            train.append(idx)
        elif name == "valid":
            valid.append(idx)
        else:
            test.append(idx)
    return torch.tensor(train), torch.tensor(valid), torch.tensor(test)


@torch.no_grad()
def evaluate(model: TextGCN, adj: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    if indices.numel() == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}
    model.eval()
    logits = model(adj)[indices]
    gold = labels[indices]
    loss = nn.functional.cross_entropy(logits, gold).item()
    pred = logits.argmax(dim=1).cpu().numpy()
    y_true = gold.cpu().numpy()
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }


def train_textgcn_and_save(
    samples: list[Sample],
    split: dict[str, str],
    out_dir: Path,
    hidden_dim: int = 200,
    epochs: int = 200,
    lr: float = 0.02,
    dropout: float = 0.226,
    patience: int = 10,
    weight_decay: float = 0.0,
    seed: int = 100,
    device: str | None = None,
    use_pmi: bool = True,
    max_features: int | None = None,
    wandb_run=None,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    adj, doc_count, _ = build_text_graph(samples, use_pmi=use_pmi, max_features=max_features)
    adj = adj.to(device_obj)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long, device=device_obj)
    train_idx, valid_idx, test_idx = split_indices(samples, split)
    train_idx = train_idx.to(device_obj)
    valid_idx = valid_idx.to(device_obj)
    test_idx = test_idx.to(device_obj)

    class_count = max(sample.label for sample in samples) + 1
    model = TextGCN(adj.shape[0], hidden_dim, class_count, dropout).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_valid_loss = float("inf")
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(adj)[train_idx]
        loss = nn.functional.cross_entropy(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

        valid_metrics = evaluate(model, adj, labels, valid_idx if valid_idx.numel() else test_idx)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "textgcn_epoch": epoch,
                    "textgcn_train_loss": loss.item(),
                    "textgcn_valid_loss": valid_metrics["loss"],
                    "textgcn_valid_accuracy": valid_metrics["accuracy"],
                    "textgcn_valid_f1": valid_metrics["f1"],
                }
            )

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, adj, labels, test_idx)
    if wandb_run is not None:
        wandb_run.log(
            {
                "textgcn_test_accuracy": test_metrics["accuracy"],
                "textgcn_test_f1": test_metrics["f1"],
                "textgcn_test_loss": test_metrics["loss"],
            }
        )

    feature_dir = out_dir / "gcn_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        doc_features = model.hidden(adj)[:doc_count].detach().cpu()
    for idx, sample in enumerate(samples):
        torch.save(doc_features[idx].clone().float(), feature_dir / f"{sample.data_id}.pt")
    torch.save(model.state_dict(), out_dir / "textgcn_model.pt")
    return feature_dir
