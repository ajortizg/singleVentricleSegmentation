import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance


class Dice(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> torch.Tensor:
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        return compute_dice(preds, targets).mean()


class Hausdorff(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> torch.Tensor:
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        return compute_hausdorff_distance(preds, targets).mean()
