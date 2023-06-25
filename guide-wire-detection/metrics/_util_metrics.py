import torch

def dice_coeff(prediction: torch.Tensor, ground_truth: torch.Tensor, epsilon: float=1e-5):
    #input shape: N, C, H, W
    prediction = prediction.round()
    ground_truth = ground_truth
    intersection = torch.sum(torch.abs(ground_truth * prediction), dim=(2, 3))
    denominator = ground_truth.sum(dim=(2, 3)) + prediction.sum(dim=(2, 3))
    dice_coeff = ((2 * intersection + epsilon) / (denominator + epsilon)).mean(dim=(0, 1))
    return dice_coeff.cpu().item()