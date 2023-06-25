import tqdm, copy
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from ._base import BaseTrainingPipeline
from metrics import dice_coeff
from typing import Tuple


class TrainingPipeline(BaseTrainingPipeline):
    def __init__(self, bbox_lossfunc, mask_lossfunc, **kwargs):
        super(TrainingPipeline, self).__init__(**kwargs)

        self.bbox_lossfunc = bbox_lossfunc
        self.mask_lossfunc = mask_lossfunc

        # collect metrics in this dictionary
        self._train_metrics_dict = dict(
            bbox_loss=[],
            confidence_loss=[],
            bce_loss = [],
            total_loss=[], 
            dice_score = [],
        )
        self._eval_metrics_dict = copy.deepcopy(self._train_metrics_dict)

    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        
        valid_modes = self._valid_modes()
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        _, axs = plt.subplots(1, 5, figsize=figsize)
        axs[0].plot(getattr(self, f"_{mode}_metrics_dict")["bbox_loss"])
        axs[0].set_title(f"{mode} Bounding Box Loss")

        axs[1].plot(getattr(self, f"_{mode}_metrics_dict")["confidence_loss"])
        axs[1].set_title(f"{mode} Confidence Loss")

        axs[2].plot(getattr(self, f"_{mode}_metrics_dict")["bce_loss"])
        axs[2].set_title(f"{mode} Binary Cross Entropy Loss")

        axs[3].plot(getattr(self, f"_{mode}_metrics_dict")["total_loss"])
        axs[3].set_title(f"{mode} Total Loss")

        axs[4].plot(getattr(self, f"_{mode}_metrics_dict")["dice_score"])
        axs[4].set_title(f"{mode} Dice Score")

        plt.show()
        print("\n\n")


    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Tuple[float, float, float]:
        assert mode in self._valid_modes(), "Invalid Mode"
        getattr(self.model, mode)()
        bbox_loss, confidence_loss, bce_loss, total_loss, dice_score = 0, 0, 0, 0, 0
        
        for idx, (images, bboxes, masks) in tqdm.tqdm(enumerate(dataloader)):
            images = images.to(self.device)       # shape: (N, n_channels, H, W)
            bboxes = bboxes.to(self.device)       # shape: (N, 5)    
            masks = masks.to(self.device)         # shape: (N, n_channels, H, W)    
                        
            pred_bboxes, pred_masks = self.model(images)
            batch_bbox_loss, batch_confidence_loss = self.bbox_lossfunc(pred_bboxes, bboxes)
            batch_bce_loss = self.mask_lossfunc(pred_masks, masks)

            batch_total_loss = batch_bbox_loss + batch_confidence_loss + batch_bce_loss
            if mode == "train":
                batch_total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            bbox_loss += batch_bbox_loss.item()
            confidence_loss += batch_confidence_loss.item()
            bce_loss += batch_bce_loss.item()
            total_loss += batch_total_loss.item()
            dice_score += dice_coeff(pred_masks, masks)

        bbox_loss /= (idx + 1)
        confidence_loss /= (idx + 1)
        bce_loss /= (idx + 1)
        total_loss /= (idx + 1)
        dice_score /= (idx + 1)

        verbosity_label = mode.title()
        if verbose:
            print((
                f"{verbosity_label} BBox Loss: {round(bbox_loss, 4)}\
                \t{verbosity_label} Confidence Loss: {round(confidence_loss, 4)}\
                \t{verbosity_label} BCE Loss: {round(bce_loss, 4)}\
                \n{verbosity_label} Dice Score: {round(dice_score, 4)}\
                \t{verbosity_label} Total Loss: {round(total_loss, 4)}"
            ))
        getattr(self, f"_{mode}_metrics_dict")["bbox_loss"].append(bbox_loss)
        getattr(self, f"_{mode}_metrics_dict")["confidence_loss"].append(confidence_loss)
        getattr(self, f"_{mode}_metrics_dict")["bce_loss"].append(bce_loss)
        getattr(self, f"_{mode}_metrics_dict")["total_loss"].append(total_loss)
        getattr(self, f"_{mode}_metrics_dict")["dice_score"].append(dice_score)

        return bbox_loss, confidence_loss, bce_loss, total_loss, dice_score