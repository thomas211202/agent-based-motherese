import datetime
import json
import os
import uuid
import random

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import spearmanr, entropy
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import defaultdict

from torch.nn import DataParallel
from torch.nn.functional import one_hot

from scipy.spatial import distance

from egg.core import Callback, Interaction, language_analysis

from .logger import logger



class CommunicationMetricsCallback(Callback):
    def __init__(self, opts, log_dir="./src/data/logs/", device='cpu'):
        logger.debug(f"CommunicationMetricsCallback init")

        self.log_dir = log_dir

        self.opts = opts

        self.control_group = False

        self.uuid = uuid.uuid4()

        self.device = device

        self.round = 0

        self.df = pd.DataFrame(
            columns=[
                'run_uuid',
                'round',
                'epoch',
                'epochs_per_round',
                'accuracy',
                'mean_accuracy',
                'ub_accuracy',
                'lb_accuracy',
                'std_accuracy',
                'var_accuracy',
                'var_bern_acuracy',
                'topographic_similarity',
                'positional_disentanglement',
                'bag_of_symbols_disentanglement',
                'vocab_usage',
                'message_entropy',
                'message_length',
                'mean_message_length',
                'ub_message_length',
                'lb_message_length',
                'std_message_length',
                'var_message_length',
                'var_bern_message_length',
                'control_group',
                'params',
                'timestamp'
            ]
        )

        path = os.path.join(self.log_dir, str(self.opts.run_uuid), "raw")
        self.metrics_directory = os.path.dirname(path)
        os.makedirs(self.metrics_directory, exist_ok=True)

        self.df.to_csv(f"{self.metrics_directory}/metrics_data.csv", index=False)

    def set_control_group(self, control_group=True):
        self.control_group = control_group
        df = self.df
        df = df.loc[df['round'] == 0]
        df['control_group'] = control_group
        df.to_csv(f"{self.metrics_directory}/metrics_data.csv", mode='a', header=False, index=False)
        self.df = df

    def set_uuid(self, uuid):
        self.uuid = uuid

    def memory_efficient_pairwise_manhattan(self, images, batch_size = 500):
        N = images.shape[0]
        distances = torch.zeros((N, N), device=images.device)

        abs_sums = images.abs().sum(dim=(1, 2, 3))

        flat_images = images.reshape(N, -1)

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            chunk_i = flat_images[i:end_i]
            abs_sum_i = abs_sums[i:end_i]

            for j in range(i, N, batch_size):
                end_j = min(j + batch_size, N)
                chunk_j = flat_images[j:end_j]
                abs_sum_j = abs_sums[j:end_j]

                with torch.no_grad():
                    mins = torch.minimum(
                        chunk_i.unsqueeze(1),
                        chunk_j.unsqueeze(0)
                    ).sum(dim=-1)

                    chunk_distances = (
                        abs_sum_i.unsqueeze(1) +
                        abs_sum_j.unsqueeze(0) -
                        2 * mins
                    )

                    distances[i:end_i, j:end_j] = chunk_distances
                    if i != j:
                        distances[j:end_j, i:end_i] = chunk_distances.T

        return distances

    def calculate_topographic_similarity(
        self,
        messages: torch.Tensor,
        meanings: torch.Tensor,
        sample: float = 0.1
    ) -> float:

        meanings = meanings.to(self.device)
        messages = messages.to(self.device)

        N = messages.size(0)
        sample_size = int(N * sample)
        sampled_indices = random.sample(range(N), sample_size)

        sampled_messages = messages[sampled_indices]
        sampled_meanings = meanings[sampled_indices]

        sampled_messages = sampled_messages.view(sampled_messages.size(0), -1)

        distance_meanings = self.memory_efficient_pairwise_manhattan(sampled_meanings)
        distance_messages = torch.cdist(sampled_messages, sampled_messages, p=2)

        topsim = spearmanr(distance_messages.cpu().numpy(), distance_meanings.cpu().numpy(), axis=None).correlation

        return topsim

    def setup_plots(self):
        logger.debug("Setting up plots")
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Communication Metrics')
        self.metrics = [
            ("mean_accuracy", "Mean Accuracy", self.axes[0, 0]),
            ("topographic_similarity", "Topographic Similarity", self.axes[0, 1]),
            ("mean_message_length", "Mean Message Length", self.axes[1, 0]),
            ("positional_disentanglement", "Positional Disentanglement", self.axes[1, 1]),
        ]
        # Set up the axes once
        for metric, title, ax in self.metrics:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)

    def plot_metrics(self, epoch):
        """Plot all tracked metrics over time."""
        logger.debug(f"Plotting metrics for epoch {epoch}")
        directory = f"{self.log_dir}/{str(self.uuid)}/plots"
        os.makedirs(directory, exist_ok=True)

        df = self.df
        logger.debug(f" df shape: {df.shape}")
        df = df[df['run_uuid'] == self.opts.run_uuid]
        logger.debug(f" df shape: {df.shape}")

        if not hasattr(self, 'fig'):
            self.setup_plots()

        for metric, _, ax in self.metrics:
            if metric in df.columns:
                sns.lineplot(data=df, x="epoch", y=metric, ax=ax)

                for round in range(0, self.round):
                    ax.axvline(x=self.opts.n_epochs*round, color='r', linestyle='--', label=f"round {round}")
                ax.legend()

        plt.tight_layout()
        plot_filename = (
            f"metrics_plot_control_group_{self.uuid}.png"
            if self.control_group
            else f"metrics_plot_{self.uuid}.png"
        )
        plt.savefig(os.path.join(directory, plot_filename))


    def plot(self):
        df = self.df
        df["full_epoch"] = df["round"] * self.opts.n_epochs + df["epoch"]
        df = df.drop(['round'], axis=1)
        df = df.drop(['epoch'], axis=1)
        df = df[df["run_uuid"] == self.opts.run_uuid]

        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Communication Metrics')

        self.metrics = [
            ("mean_accuracy", "Mean Accuracy", ax[0, 0]),
            ("topographic_similarity", "Topographic Similarity", ax[0, 1]),
            ("mean_message_length", "Mean Message Length", ax[1, 0]),
            ("positional_disentanglement", "Positional Disentanglement", ax[1, 1]),
        ]



        for metric, title, ax in self.metrics:
            if metric in df.columns:
                sns.lineplot(data=df, x="full_epoch", y=metric, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
                for round in range(0, self.round):
                    ax.axvline(x=self.opts.n_epochs*round + self.opts.n_epochs+0.5, color='r', linestyle='--', label=f"round {round}")
                # ax.legend()
        plt.tight_layout()
        plot_filename = (
            f"metrics_plot_control_group_{self.uuid}.png"
            if self.control_group
            else f"metrics_final_plot_{self.uuid}.png"
        )
        directory = f"{self.log_dir}/{str(self.uuid)}/plots"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, plot_filename))
        plt.close(fig)


    def plot_combined(self, df=None):
        if df is None:
            logger.info(f"Reading metrics data from {self.metrics_directory}/metrics_data.csv")
            df = pd.read_csv(f"{self.metrics_directory}/metrics_data.csv")

        df["full_epoch"] = np.where(
            df["control_group"] == False,
            df["round"] * self.opts.n_epochs + df["epoch"].astype(int),
            df["epoch"] + np.where(df["round"] >= 1, 10, 0)
        )

        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Communication Metrics')

        self.metrics = [
            ("accuracy", "Mean Accuracy", ax[0, 0]),
            ("topographic_similarity", "Topographic Similarity", ax[0, 1]),
            ("message_length", "Mean Message Length", ax[1, 0]),
            ("positional_disentanglement", "Positional Disentanglement", ax[1, 1]),
        ]

        for metric, title, ax in self.metrics:
            if metric in df.columns:
                if metric == "accuracy" or metric == "message_length":
                    logger.info(f"Plotting {metric} with error bars")
                    sns.lineplot(data=df, x="full_epoch", y=f"mean_{metric}", ax=ax, hue="control_group")

                else:
                    logger.info(f"Plotting {metric}")
                    sns.lineplot(data=df, x="full_epoch", y=metric, ax=ax, hue="control_group")
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
                for round in range(0, self.opts.rounds):
                    ax.axvline(x=self.opts.n_epochs*round + self.opts.n_epochs+0.5, color='r', linestyle='--', label=f"round {round}")
        plt.tight_layout()
        plot_filename = f"combined_plot_{self.opts.run_uuid}.png"
        directory = f"{self.log_dir}/{str(self.uuid)}/plots"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, plot_filename))
        plt.close(fig)


    def on_validation_end(self, loss: float, interaction: Interaction, epoch: int):

        logger.info(f"Calculating metrics for epoch {epoch} round {self.round}")
        messages = interaction.message
        meanings = interaction.sender_input

        if messages is None or meanings is None:
            logger.warning(f"Epoch {epoch}: Missing messages or meanings in interaction")
            return

        topsim = self.calculate_topographic_similarity(messages, meanings)

        # not implemented
        posdis = 0
        bosdis = 0
        entropy = 0

        acc = interaction.aux.get('acc', None)
        if acc is not None:
            acc = acc.detach().numpy() if isinstance(acc, torch.Tensor) else acc

        message_length = interaction.message_length
        if message_length is not None:
            message_length = message_length.detach().numpy() if isinstance(message_length, torch.Tensor) else message_length

        row = pd.DataFrame(
            {
                'run_uuid': [self.opts.run_uuid],
                'round': [self.round],
                'epoch': [epoch],
                'epochs_per_round': [self.opts.n_epochs],
                'accuracy': [acc],
                'mean_accuracy': [np.mean(acc)],
                'ub_accuracy': [np.mean(acc) + 1.96 * np.std(acc)],
                'lb_accuracy': [np.mean(acc) - 1.96 * np.std(acc)],
                'std_accuracy': [np.std(acc)],
                'var_accuracy': [np.var(acc)],
                'var_bern_acuracy': [(np.mean(acc) * len(acc))],
                'topographic_similarity': [topsim],
                'positional_disentanglement': [posdis],
                'bag_of_symbols_disentanglement': [bosdis],
                'vocab_usage': [0],
                'message_entropy': [entropy],
                'message_length': [message_length],
                'mean_message_length': [np.mean(message_length)],
                'ub_message_length': [np.mean(message_length) + 1.96 * np.std(message_length)],
                'lb_message_length': [np.mean(message_length) - 1.96 * np.std(message_length)],
                'std_message_length': [np.std(message_length)],
                'var_message_length': [np.var(message_length)],
                'var_bern_message_length': [(np.mean(message_length) * len(message_length))],
                'control_group': [self.control_group],
                'params': [self.opts],
                'timestamp': [datetime.datetime.now()]
            }
        )

        if self.df.empty:
            logger.info(f"Creating new dataframe")
            self.df = pd.concat(
                [
                    None,
                    row
                ],
                ignore_index=True
            )
        else:
            logger.info(f"Appending to existing dataframe")
            self.df = pd.concat(
                [
                    self.df,
                    row
                ],
                ignore_index=True
            )

        self.df = pd.concat(
            [
                self.df if not self.df.empty else None,
                row
            ],
            ignore_index=True
        )

        row.to_csv(f"{self.metrics_directory}/metrics_data.csv", mode='a', header=False, index=False)

        results = {
            "Run_uuid": self.opts.run_uuid,
            "Epoch": epoch,
            "Mean accuracy": np.mean(acc),
            "Std accuracy": np.std(acc),
            "Topsim": topsim,
            "Posdis": posdis,
            "Bosdis": bosdis,
            "Message Entropy": entropy,
            "Mean Message Length": np.mean(message_length),
        }

        logger.table(results, title="Results")

        if epoch == self.opts.n_epochs or (self.control_group and epoch % self.opts.n_epochs == 0):
            self.round += 1

        if (self.round == self.opts.rounds and epoch == self.opts.n_epochs) or (
                self.control_group and epoch == (self.opts.n_epochs - 1) * self.opts.rounds
        ):
            logger.info(f"Metrics calculation complete for {self.opts.run_uuid}")
            self.plot()
            if self.control_group:
                self.plot_combined()


