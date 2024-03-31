import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = x / S - 0.5 * w
        y1 = y / S - 0.5 * h
        x2 = x / S + 0.5 * w
        y2 = y / S + 0.5 * h
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        return boxes

    def find_best_iou_boxes(self, box_pred_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        4) hint: use torch.diagnoal() on results of compute_iou
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        best_iou_list = []
        best_boxes_list = [] 
        
        for box_pred in box_pred_list:
            iou_list = []
            for i in range(box_pred.size(0)):
                iou = torch.diagnoal(self.compute_iou(box_pred[i, :4], box_target))
                iou_list.append(iou)

            iou_tensor = torch.stack(iou_list, dim=1)
            best_iou, best_idx = torch.max(iou_tensor, dim=1)

            best_boxes = torch.cat([self.xywh2xyxy(box_pred[best_idx, :4]), best_iou.view(-1, 1)], dim=1)

            best_iou_list.append(best_iou)
            best_boxes_list.append(best_boxes)

        best_ious = torch.stack(best_iou_list, dim=0)
        best_boxes = torch.stack(best_boxes_list, dim=0)
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        # 將 classes_pred 和 classes_target 展平為 (batch_size * S * S, 20)
        classes_pred_flat = classes_pred.view(-1, 20)
        classes_target_flat = classes_target.view(-1, 20)

        # 取出具有物體的位置的索引
        object_indices = has_object_map.view(-1).nonzero().squeeze()

        # 從預測和目標中選擇具有物體的位置
        classes_pred_selected = torch.index_select(classes_pred_flat, 0, object_indices)
        classes_target_selected = torch.index_select(classes_target_flat, 0, object_indices)

        # 計算交叉熵損失
        class_loss = F.cross_entropy(classes_pred_selected, classes_target_selected.argmax(dim=1))
        
        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here

        loss = 0.0

        for pred_boxes in pred_boxes_list:
            # 將 pred_boxes 展平為 (N * S * S, 5)
            pred_boxes_flat = pred_boxes.view(-1, 5)

            # 取出不包含物體的位置的索引
            no_object_indices = (1 - has_object_map.view(-1)).nonzero().squeeze()

            # 從預測中選擇不包含物體的位置
            pred_boxes_no_object = torch.index_select(pred_boxes_flat, 0, no_object_indices)

            # 計算不包含物體的位置的損失（置信度損失）
            loss += F.mse_loss(pred_boxes_no_object[:, 4], torch.zeros_like(pred_boxes_no_object[:, 4]))

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        # 計算包含物體的位置的損失（置信度損失）
        contain_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')

        return contain_loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        reg_loss = F.mse_loss(box_pred_response, box_target_response, reduction='sum')

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0
        inv_N = 1.0 / N
        # When you calculate the classification loss, no-object loss, regression loss, contain_object_loss
        # you need to multiply the loss with inv_N. e.g: inv_N * self.get_regression_loss(...)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)

        # compcute classification loss

        # compute no-object loss

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects

        # compute contain_object_loss

        # compute final loss

        # construct return loss_dict
        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': inv_N * reg_loss,
            'containing_obj_loss': inv_N * containing_obj_loss,
            'no_obj_loss': inv_N * no_obj_loss,
            'cls_loss': inv_N * cls_loss
        }
        return loss_dict
