from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix


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
        test_window_ratio: int,
        classes: list[str],
        gpu_id: int,
        out_dir: Path | None = None,
    ) -> None:
        """Initialize the evaluator.

        Arguments:
            vae: The trained VAE model.
            classifier: The trained classifier model.
            dataloader: DataLoader for evaluation data.
            test_window_ratio: Ratio to combine multiple latent vectors for evaluation.
            classes: List of class names.
            out_dir: Output directory for saving results.
            gpu_id: GPU identifier for computation.
            out_dir: Output directory for saving plots.

        """
        self.__vae = vae.to(gpu_id)
        self.__classifier = classifier.to(gpu_id)
        self.__dataloader = dataloader
        self.__test_window_ratio = test_window_ratio
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
            batch_size = x.shape[0]
            window_size = x.shape[2] // self.__test_window_ratio

            x_r = x.view(batch_size * self.__test_window_ratio, x.shape[1], window_size, x.shape[3]).to(self.__gpu_id)

            _, z_hard, latents = self.__vae(x_r)

            # (B * test_window_ratio, latent_dim, n_categories) → (B, latent_dim * n_categories * test_window_ratio)
            z_hard = z_hard.view(batch_size, -1)
            latents = latents.view(batch_size, -1)

            preds = torch.argmax(self.__classifier(z_hard), dim=1)
            accuracy_metric.update(preds, y.to(self.__gpu_id))
            confusion_matrix_metric.update(preds, y.to(self.__gpu_id))

            # Use the original latent vectors and labels for t-SNE visualization
            all_latents.append(latents.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        conf_matrix = confusion_matrix_metric.compute().cpu().numpy()

        latent_array = np.concatenate(all_latents, axis=0)
        label_array = np.concatenate(all_labels, axis=0)

        if self.__out_dir is not None and self.__gpu_id == 0:
            _plot_confusion_matrix(conf_matrix, self.__classes, self.__out_dir)
            _plot_latent_tsne(latent_array, label_array, self.__classes, self.__out_dir)

        return accuracy_metric.compute().item()
