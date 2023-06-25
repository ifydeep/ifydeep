import os, torch, copy
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Iterable, Tuple, Optional

class BaseTrainingPipeline:
    def __init__(self, 
                *,
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                lossfunc: Optional[nn.Module]=None,
                device: str="cpu", 
                weight_init: bool=True,
                custom_weight_initializer: Any=None,
                dirname: str="./saved_model", 
                filename: str="model.pth.tar"):
        
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.dirname = dirname
        self.filename = filename
        
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

        # collect metrics in this dictionary
        self._train_metrics_dict = dict()
        self._eval_metrics_dict = copy.deepcopy(self._train_metrics_dict)
        

    def xavier_init_weights(self, m: nn.Module):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and (m.weight.requires_grad == True):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    

    def save_model(self):
        if not os.path.isdir(self.dirname): os.mkdir(self.dirname)
        state_dicts = {
            "network_params":self.model.state_dict(),
            "optimizer_params":self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, os.path.join(self.dirname, self.filename))
    

    def load_model(self):
        model_path = os.path.join(self.dirname, self.filename)
        if not os.path.exists(model_path):
            raise OSError(f"model is yet to be saved in path: {model_path}")
        saved_params = torch.load(model_path, map_location=self.device)
        return self.model.load_state_dict(saved_params["network_params"])
    

    def get_metric(self) -> Tuple[Dict[str, Iterable[float]], Dict[str, Iterable[float]]]:
        if not hasattr(self, "_train_metrics_dict") or not hasattr(self, "_eval_metrics_dict"):
            raise NotImplementedError(
                "This method is not implemented for pipeline objects without metrics"
            ) 
        return self._train_metrics_dict, self._eval_metrics_dict


    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        raise NotImplementedError("Method requires that the class is subclassed and implemented")
        

    def train(
            self, 
            dataloader: DataLoader, 
            verbose: bool=False) -> Tuple[float, float, float]:
        return self._feed(dataloader, "train", verbose)
    
    def evaluate(
            self, 
            dataloader: DataLoader, 
            verbose: bool=False) -> Tuple[float, float, float]:        
        with torch.no_grad():
            return self._feed(dataloader, "eval", verbose)

    def _feed(
            self, 
            dataloader: DataLoader, 
            mode: str, 
            verbose: bool=False) -> Tuple[float, float, float]:
        raise NotImplementedError("Method requires that the class is subclassed and implemented")
    

    def _valid_modes(self) -> Iterable[str]:
        return ["train", "eval"]