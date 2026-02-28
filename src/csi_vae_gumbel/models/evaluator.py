from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

from csi_vae_gumbel.util import split_test_window


def _plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    plt.figure(figsize=(10, 8))
    # Normalize by row (True Labels) to see percentages
    matrix_perc = matrix.astype("float") / (matrix.sum(axis=1)[:, np.newaxis] + 1e-12)

    sns.heatmap(
        matrix_perc,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.show()
    plt.close()


def _plot_latent_tsne(latent_array: np.ndarray, label_array: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    """Plot t-SNE visualization of the VAE categorical latent space.

    Arguments:
        latent_array: Numpy array of shape (n_samples, latent_dim) containing the latent vectors.
        label_array: Numpy array of shape (n_samples,) containing the class labels.
        class_names: List of class names corresponding to the labels.
        out_dir: Output directory for saving the plot.

    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=42,
    )
    z_tsne = tsne.fit_transform(latent_array)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        z_tsne[:, 0],
        z_tsne[:, 1],
        c=label_array,
        cmap="tab10",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    # Create legend with class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Activities", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("t-SNE Visualization of VAE Categorical Latent Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/latent_tsne.png")
    plt.show()
    plt.close()


class Evaluator:
    """Evaluator for VAE training using a classifier."""

    def __init__(
        self,
        vae: torch.nn.Module,
        classifier: torch.nn.Module,
        dataloader: DataLoader,
        sample_window_size: int,
        overlap_size: int,
        classes: list[str],
        gpu_id: int,
        out_dir: Path | None = None,
    ) -> None:
        """Initialize the evaluator.

        Arguments:
            vae: The trained VAE model.
            classifier: The trained classifier model.
            dataloader: DataLoader for evaluation data.
            sample_window_size: Size of the window to sample from the test data.
            overlap_size: Overlap size between windows.
            classes: List of class names.
            gpu_id: GPU identifier for computation.
            out_dir: Optional output directory for saving confusion matrix and t-SNE plots.

        """
        self.__vae = vae.to(gpu_id)
        self.__classifier = classifier.to(gpu_id)
        self.__dataloader = dataloader
        self.__sample_window_size = sample_window_size
        self.__overlap_size = overlap_size
        self.__classes = classes
        self.__n_classes = len(classes)
        self.__gpu_id = gpu_id
        self.__out_dir = out_dir

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the VAE and classifier on the test dataset."""
        self.__classifier.eval()
        self.__vae.eval()

        accuracy_metric = MulticlassAccuracy(num_classes=self.__n_classes).to(self.__gpu_id)
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=self.__n_classes).to(self.__gpu_id)

        all_latents = []
        all_labels = []

        for x, y in self.__dataloader:
            original_batch_size = x.shape[0]

            x_r = split_test_window(x.to(self.__gpu_id), self.__sample_window_size, self.__overlap_size)

            _, z_hard, logits = self.__vae(x_r)

            # (B * n_windows, latent_dim, n_categories) → (B, latent_dim * n_categories * n_windows)
            n_windows = x_r.shape[0] // original_batch_size
            z_hard = z_hard.view(original_batch_size, n_windows, -1)
            z_hard = z_hard.reshape(original_batch_size, -1)
            logits = logits.view(original_batch_size, n_windows, -1)
            logits = logits.reshape(original_batch_size, -1)

            preds = torch.argmax(self.__classifier(z_hard), dim=1)
            accuracy_metric.update(preds, y.to(self.__gpu_id))
            confusion_matrix_metric.update(preds, y.to(self.__gpu_id))

            # Use the original latent vectors and labels for t-SNE visualization
            all_latents.append(logits)
            all_labels.append(y.to(self.__gpu_id))

        conf_matrix = confusion_matrix_metric.compute()
        accuracy = accuracy_metric.compute()
        latents_single = torch.cat(all_latents, dim=0)
        labels_single = torch.cat(all_labels, dim=0)
        latents = torch.zeros(
            latents_single.shape[0] * dist.get_world_size(),
            *latents_single.shape[1:],
            device=self.__gpu_id,
        )
        labels = torch.zeros(
            labels_single.shape[0] * dist.get_world_size(),
            device=self.__gpu_id,
            dtype=labels_single.dtype,
        )

        dist.all_reduce(conf_matrix, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)
        dist.all_gather_into_tensor(latents, latents_single)
        dist.all_gather_into_tensor(labels, labels_single)

        if self.__gpu_id == 0 and self.__out_dir is not None:
            _plot_confusion_matrix(conf_matrix.cpu().numpy(), self.__classes, self.__out_dir)
            _plot_latent_tsne(latents.cpu().numpy(), labels.cpu().numpy(), self.__classes, self.__out_dir)

        return accuracy.item()
