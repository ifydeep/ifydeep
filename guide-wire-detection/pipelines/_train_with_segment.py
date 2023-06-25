import tqdm, copy
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from ._base import BaseTrainingPipeline
from metrics import dice_coeff
from typing import Tuple


class SegmentExclusiveTrainingPipeline(BaseTrainingPipeline):
    def __init__(self, **kwargs):
        super(SegmentExclusiveTrainingPipeline, self).__init__(**kwargs)

        # collect metrics in this dictionary
        self._train_metrics_dict = dict(
            bce_loss = [],
            dice_score = []
        )
        self._eval_metrics_dict = copy.deepcopy(self._train_metrics_dict)

    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        
        valid_modes = self._valid_modes()
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        _, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].plot(getattr(self, f"_{mode}_metrics_dict")["bce_loss"])
        axs[0].set_title(f"{mode} Binary Cross Entropy Loss")

        axs[1].plot(getattr(self, f"_{mode}_metrics_dict")["dice_score"])
        axs[1].set_title(f"{mode} Dice Score")

        plt.show()
        print("\n\n")


    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Tuple[float, float, float]:
        assert mode in self._valid_modes(), "Invalid Mode"
        getattr(self.model, mode)()
        bce_loss, dice_score = 0, 0
        
        for idx, (images, _, masks) in tqdm.tqdm(enumerate(dataloader)):
            images = images.to(self.device)       # shape: (N, n_channels, H, W)
            masks = masks.to(self.device)  # shape: (N, n_channels, H, W)    
                        
            pred_masks = self.model(images)
            batch_bce_loss = self.lossfunc(pred_masks, masks)
            
            if mode == "train":
                batch_bce_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            bce_loss += batch_bce_loss.item()
            dice_score += dice_coeff(pred_masks, masks)

        bce_loss /= (idx + 1)
        dice_score /= (idx + 1)

        verbosity_label = mode.title()
        if verbose:
            print((
                f"{verbosity_label} BCE Loss: {round(bce_loss, 4)}\
                  \t{verbosity_label} Dice Score: {round(dice_score, 4)}"
            ))
        getattr(self, f"_{mode}_metrics_dict")["bce_loss"].append(bce_loss)
        getattr(self, f"_{mode}_metrics_dict")["dice_score"].append(dice_score)

        return bce_loss, dice_score