import copy
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from recommend.cf_model import Model


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.metrics_train: List[Dict[str, float]] = []
        self.metrics_valid: List[Dict[str, float]] = []
        self.past_models: List[Model] = []
        self.optimizer = optimizer

    def evaluate(
        self,
        loader: DataLoader,
        tqdm_desc: str,
        tqdm_leave: bool,
        plot_hist: bool = True,
    ) -> Dict[str, float]:

        metrics = {"mse": 0.0, "mae_stars": 0.0}
        preds = []

        with torch.no_grad():  # type: ignore
            for movie_idx, user_idx, rating in tqdm(
                loader, desc=tqdm_desc, leave=tqdm_leave, position=1
            ):
                pred = self.model(movie_idx, user_idx)
                preds.append(pred.clone().detach().cpu().flatten().numpy())
                metrics["mse"] += F.mse_loss(pred, rating).cpu().item()
                metrics["mae_stars"] += F.l1_loss(pred * 5, rating * 5).cpu().item()
        if plot_hist:
            pd.Series(np.concatenate(preds)).hist()
            plt.show()
        return {name: value / len(loader) for name, value in metrics.items()}

    def train(
        self,
        loader_train: DataLoader,
        loader_train_mini: DataLoader,
        loader_valid: DataLoader,
        patience: int,
        max_epochs: int,
        evaluate_every_n_steps: int,
        early_stopping_metric: str,
        early_stopping_threshold: float,
    ) -> None:
        step = 0
        device = list(self.model.parameters())[0].device
        for epoch in range(max_epochs):
            for movie_idx, user_idx, rating in tqdm(
                loader_train, desc=f"epoch {epoch}", position=0
            ):
                pred = self.model(movie_idx, user_idx)
                F.mse_loss(pred, rating).backward()  # type: ignore
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % evaluate_every_n_steps == 0:
                    self.metrics_train.append(
                        self.evaluate(
                            loader_train_mini, "train evaluation", tqdm_leave=False, plot_hist=False
                        )
                    )
                    self.metrics_valid.append(
                        self.evaluate(
                            loader_valid, "validation evaluation", tqdm_leave=False, plot_hist=True
                        )
                    )
                    print("train metrics:     ", self.metrics_train[-1])
                    print("validation metrics:", self.metrics_valid[-1])
                    self.past_models.append(copy.deepcopy(self.model).cpu())
                    if len(self.metrics_train) > patience:
                        self.past_models.pop(0)
                        last_n = pd.DataFrame(self.metrics_train[-patience:])[early_stopping_metric]
                        if last_n.max() - last_n.min() < early_stopping_threshold:
                            device = list(self.model.parameters())[0].device
                            best_idx = last_n.argmin()
                            self.model = self.past_models[best_idx].to(device)
                step += 1


class RatingDataset(Dataset):
    def __init__(self, df_ratings: pd.DataFrame, movie2idx: pd.Series, user2idx: pd.Series):
        self.df_ratings = df_ratings
        self.movie2idx = movie2idx
        self.user2idx = user2idx

    def __len__(self) -> int:
        return len(self.df_ratings)

    def __getitem__(self, idx: int) -> Tuple[int, int, float]:
        row = self.df_ratings.iloc[idx]
        idx_movie = self.movie2idx[row.movie_id]
        idx_user = self.user2idx[row.username]
        rating = (row.stars / 5.0).astype(np.float32)
        return idx_movie, idx_user, rating
