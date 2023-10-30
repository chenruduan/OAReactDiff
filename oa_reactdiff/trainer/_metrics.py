from typing import Dict, List
import numpy as np
import torch


def average_over_batch_metrics(batch_metrics: List[Dict], allowed: List = []):
    epoch_metrics = {}
    effective_batch = {}
    for ii, out in enumerate(batch_metrics):
        for k, v in out.items():
            if not (k in allowed or len(allowed) == 0):
                continue
            if ii == 0:
                epoch_metrics[k] = v
                effective_batch[k] = 1
            else:
                if not np.isnan(v):
                    epoch_metrics[k] += v
                    effective_batch[k] += 1
    for k in epoch_metrics:
        epoch_metrics[k] /= effective_batch[k]
    return epoch_metrics


def pretty_print(epoch, metric_dict, prefix="Train"):
    out = f"{prefix} epoch {epoch} "
    for k, v in metric_dict.items():
        out += f"{k} {v:.2f} "
    print(out)
