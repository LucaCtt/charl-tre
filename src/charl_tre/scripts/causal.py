import json
import logging
from pathlib import Path

import torch
from rich.logging import RichHandler

from charl_tre.causal.pipelines import PathSearchPipeline, RawDiscoveryPipeline
from charl_tre.settings import Settings

settings = Settings()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def causal() -> None:
    """Run the full causal discovery and path search pipelines, and log the results."""
    raw_result = RawDiscoveryPipeline(settings).run()
    raw_dir = Path(raw_result["output_dir"])
    logger.info("Raw graphs saved to: %s", raw_dir)

    path_result = PathSearchPipeline(settings, raw_dir).run()

    output_dir = Path(path_result["output_dir"])
    logger.info("Path search outputs saved to: %s", output_dir)

    metrics = json.loads((output_dir / "deterministic_classifier_metrics.json").read_text())
    summary = json.loads((output_dir / "graphs_summary.json").read_text())
    rules = json.loads((output_dir / "classification_rules.json").read_text())

    logger.info("n_latent_variables %s", summary["n_latent_variables"])
    logger.info("n_segments: %s", metrics["n_segments"])
    logger.info("accuracy: %s", metrics["accuracy"])
    logger.info("mean margin: %s", metrics["mean_prediction_margin"])

    for activity, activity_rules in rules["activities"].items():
        logger.info("\n[%s] %s rules", activity, len(activity_rules))
        for rule in activity_rules[:3]:
            logger.info(" - %s", rule["rule_text"])

    for activity, info in summary["activities"].items():
        top_path = info["top_paths"][0]["path"] if info["top_paths"] else []
        logger.info("[%s] edges=%s  top_path=%s", activity, info["n_edges"], " → ".join(top_path))


if __name__ == "__main__":
    causal()
