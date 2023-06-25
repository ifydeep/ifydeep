import torch
import torch.nn as nn
from typing import Tuple

class GWDetectionCriterion(nn.Module):
    def __init__(self, iou_threshold: float=0.5):
        super(GWDetectionCriterion, self).__init__()
        
        self.iou_threshold = iou_threshold
        self.confidence_criterion = nn.BCELoss()
        self.bbox_criterion = nn.HuberLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input shape: (N, num_boxes, confidence + len([x, y, w, h]))
        # target shape: (N, confidence + len([x, y, w, h]))
        # confidence score = tensor[..., 0]
        # x, y, w, h values = tensor[..., 1:]

        N, n_pred_boxes, B = input.shape

        bbox_loss = torch.tensor(0)
        confidence_loss = torch.tensor(0)

        for i in range(N):
            pred = input[i, ...]                                       # shape: (n_pred_boxes, 5)
            gt = target[i, ...]                                        # shape: (5, )
            gt = gt.unsqueeze(dim=0).tile(pred.shape[0], 1)            # shape: (n_pred_boxes, 5)
            pred_boxes = pred[:, 1:]                                   # shape: (n_pred_boxes, 4)
            gt_boxes = gt[:, 1:]                                       # shape: (n_pred_boxes, 4)
            pred_confidence = pred[:, 0]                               # shape: (n_pred_boxes, 1)
            gt_confidence = gt[:, 0]                                   # shape: (n_pred_boxes, 1)

            ious = GWDetectionCriterion.compute_iou(
                pred_boxes, gt_boxes
            )                                                          # shape: (n_pred_boxes, )
            invalid_indices = torch.nonzero(
                ious < self.iou_threshold,
                as_tuple=False
            ).squeeze()
            gt_confidence[invalid_indices] = 0                        # set confidence of all bboxes below threshold to 0.

            bbox_loss = bbox_loss + self.bbox_criterion(pred_boxes, gt_boxes)
            confidence_loss = confidence_loss + self.confidence_criterion(pred_confidence, gt_confidence)

        bbox_loss = bbox_loss / N
        confidence_loss = confidence_loss / N

        return bbox_loss, confidence_loss

    
    @staticmethod
    def compute_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        # pred_boxes shape: (len([x1, y1, w1, h1]), ) | (N, len([x1, y1, w1, h1]))
        # gt_boxes shape: (len([x2, y2, w2, h2]), ) | (N, len([x2, y2, w2, h2]))
        
        assert pred_boxes.ndim == gt_boxes.ndim <= 2, \
            f"\
                pred_boxes and gt_boxes must be of same shape, with dimensions \
                <= 2 got {pred_boxes.ndim}, {gt_boxes.ndim} \
            "
        
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.unsqueeze(dim=0)    # shape: (1, 4)
            gt_boxes = gt_boxes.unsqueeze(dim=0)        # shape: (1, 4)

        x1, y1, w1, h1 = pred_boxes.permute(1, 0)
        x2, y2, w2, h2 = gt_boxes.permute(1, 0)
        area1, area2 = (w1 * h1), (w2 * h2)

        x_lefts = torch.stack((x1, x2), dim=-1).max(dim=-1).values
        y_tops = torch.stack((y1, y2), dim=-1).max(dim=-1).values
        x_rights = torch.stack((x1+w1, x2+w2), dim=-1).min(dim=-1).values
        y_bottoms = torch.stack((y1+h1, y2+h2), dim=-1).min(dim=-1).values

        i_area = (x_rights-x_lefts) * (y_bottoms-y_tops)
        u_area = (area1 + area2) - i_area
        ious = i_area / u_area
        ious[x_rights < x_lefts] = 0
        ious[y_bottoms < y_tops] = 0
        return ious.squeeze()                           # shape: (N, ) | (1, )