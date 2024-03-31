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

    wh = rb - lt  # [N, M, 2]
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
        #top left
        x1 = x / self.S - 0.5 * w
        y1 = y / self.S - 0.5 * h
        #top right
        x2 = x / self.S + 0.5 * w
        y2 = y / self.S + 0.5 * h
        
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes

    def find_best_iou_boxes(self, box_pred, box_target):
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
        ###find the best iou of predict bbox and target bbox

        ### CODE ###
        # Your code here
        S = self.S
        B = self.B

        contains_object_response_mask = torch.cuda.ByteTensor(box_target.size()).zero_()#mask與target box相同大小的tensor，cell有最好的iou為1其他為0
        box_target_iou = torch.zeros(box_target.size()).cuda()#box_target_iou與target box相同大小的tensor

        for i in range(0, box_target.size()[0], B):#box_target.size()[0]==has_object_map(n,s,s,..)，該框框屬於物體的有幾個，box_target.size()[0]就有幾個
            # 屬於物體的每個部分，都要去看他自己的bbox（predict*2,target*1）
            box_p = box_pred[i:i+B, :4]#remove the confidence column, because it produce b box_pred thus it shift b units per iteration.
            box_p_center = Variable(torch.FloatTensor(box_p.size()))#create a tensor which its size is the same as box_p
            box_p_center = self.xywh2xyxy(box_p)#store the transformed box_p
            
            box_t = box_target[i,:4].view(-1,4)
            box_t_center = Variable(torch.FloatTensor(box_t.size()))
            box_t_center = self.xywh2xyxy(box_t)

            
            iou = compute_iou(box_p_center, box_t_center)#compute the iou
            max_iou, max_idx = iou.max(0)#figure out the biggest iou and its index
            max_idx = max_idx.data.cuda()
            
            
            contains_object_response_mask[i + max_idx] = 1 # assign 1 to the object position which bbox has the best iou.(標位置), like true/false(mask)
            
            
            box_target_iou[i + max_idx, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()# assign the best iou to the index(標同個位置上的iou)

        box_target_iou = Variable(box_target_iou).cuda()
        

        return box_target_iou, contains_object_response_mask



    # def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
    def get_class_prediction_loss(self, classes_pred, classes_target):
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
        class_loss = F.mse_loss(classes_pred, classes_target, reduction='sum')#simply calculate the mse loss between classes_pred & classes_target
        return class_loss

    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):#find out the cells which are not belong to the object 
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
        B = self.B
        C = 20
        T = B * 5 + C      # number of tensors

        no_object_prediction = pred_tensor[no_object_mask].view(-1, T)# find the cell which not belong to the object in pred_tensor
        no_object_target = target_tensor[no_object_mask].view(-1, T)# find the cell which not belong to the object in target_tensor
        

        no_object_prediction_mask = torch.cuda.ByteTensor(no_object_prediction.size()).zero_()#create a mask the same size with no_object_prediction,assign 0.
        # print(no_object_prediction_mask.shape) #(4XXX, 30)
        for i in range(B):
            no_object_prediction_mask[:, i*5+4] = 1 # assign 1 to confidence column

        no_object_prediction_mask = no_object_prediction_mask.bool()# because it will cause warning during training, thus transformed to bool.
        no_object_prediction_conf = no_object_prediction[no_object_prediction_mask]#find out where no_object_prediction_mask is in the prediction.
        no_object_target_conf = no_object_target[no_object_prediction_mask]#find out where no_object_prediction_mask is in the target.
        no_object_loss = F.mse_loss(no_object_prediction_conf, no_object_target_conf, reduction='sum')#calculate mse loss of these two.
        
        return no_object_loss

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
        contain_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')#simply calculate the mse loss between box_pred_conf & box_target_conf

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
        reg_loss = F.mse_loss(box_pred_response[:,:2], box_target_response[:,:2], reduction='sum') + \
                   F.mse_loss(torch.sqrt(box_pred_response[:,2:]), torch.sqrt(box_target_response[:,2:]), reduction='sum')#the regression formula,x,y,w^(1/2),h^(1/2)
        #yolo v1
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
        S = pred_tensor.size(1)
        B = self.B
        C = 20
        T = B*5 + C #30

        temp = has_object_map.int()#有超過threshold就記1,反之為0
        temp = temp.unsqueeze(dim=3)#(N,S,S)=>(N,S,S,1)
        target_box_conf = torch.cat([target_boxes, temp], dim=3) # N, S, S, 4+1
        target_box_conf = torch.cat([target_box_conf, target_box_conf], dim=3) # N, S, S, (4+1)*2=10, 湊出N,S,S,10，為了concate時可與target_cls合併出跟pred_tensor一樣維度
        target_tensor = torch.cat([target_box_conf, target_cls], dim=3) # N, S, S, 30, 湊出一個跟pred_tensor一樣維度的target_tensor
        # pred_boxes_list = pred_tensor[:, :, :, :B*5].reshape(B, N, S, S, 5)
        # pred_cls = pred_tensor[:, :, :, B*5:]
        contains_object_mask = target_tensor[:,:,:,4] > 0 #target tensor中有哪些有object, the same as has_object_map
        contains_object_mask = contains_object_mask.unsqueeze(-1).expand_as(target_tensor)#expand to the same size as target_tensor
        
        no_object_mask = target_tensor[:,:,:,4] == 0#target tensor中有哪些沒有object
        no_object_mask = no_object_mask.unsqueeze(-1).expand_as(target_tensor)#expand to the same size as target_tensor

        contains_object_pred = pred_tensor[contains_object_mask].view(-1, T)#在pred_tensor裡找出那些有物體的cell中，找出predict的值
        bounding_box_pred = contains_object_pred[:,:B*5].contiguous().view(-1,5)#承接上一行，對每個predict出的結果找出bbox的info.
        classes_pred = contains_object_pred[:,B*5:]#承接上上一行，對每個predict出的結果找出class的info.

        contains_object_target = target_tensor[contains_object_mask].view(-1, T)#在target_tensor裡找出那些有物體的cell中，找出target的值
        bounding_box_target = contains_object_target[:,:B*5].contiguous().view(-1,5)#承接上一行，對每個target的結果找出bbox的info.
        classes_target = contains_object_target[:,B*5:]#承接上上一行，對每個target的結果找出class的info.

        # 計算分類損失
        cls_loss = self.get_class_prediction_loss(classes_pred, classes_target)

        # 計算無物體損失
        # no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)
        no_obj_loss = self.get_no_object_loss(pred_tensor, target_tensor, no_object_mask)


        # 找到所有包含物體的單元中的2個（或 self.B）預測框中的最佳框和相應的IOU
        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(bounding_box_pred, bounding_box_target)

        contains_object_response_mask = contains_object_response_mask.bool()# to bool
        box_prediction_response = bounding_box_pred[contains_object_response_mask].view(-1, 5)#從bounding_box_pred找出相應的index=>找出best iou在哪
        box_target_response_iou = box_target_iou[contains_object_response_mask].view(-1, 5)#find the value of best iou
        box_target_response = bounding_box_target[contains_object_response_mask].view(-1, 5)#從bounding_box_target找出相應的index=>找出best iou在哪
       
        reg_loss = self.get_regression_loss(box_prediction_response[:,:4], box_target_response[:,:4])#remove confidence
 
        # 計算包含物體損失
        containing_obj_loss = self.get_contain_conf_loss(box_prediction_response[:,4], box_target_response_iou[:,4])#only take confidence to calculate conf_loss

        # 計算最終損失
        total_loss = cls_loss + self.l_noobj*no_obj_loss + self.l_coord*reg_loss + containing_obj_loss #total loss

        loss_dict = {
            'total_loss': inv_N * total_loss,
            'reg_loss': inv_N * reg_loss,
            'containing_obj_loss': inv_N * containing_obj_loss,
            'no_obj_loss': inv_N * no_obj_loss,
            'cls_loss': inv_N * cls_loss
        }
        return loss_dict
