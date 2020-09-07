# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from .Multimodal_Transformer_master.trans import transform
from .Multimodal_Transformer_master.src.train import *
from .Multimodal_Transformer_master.src.dataset import Multimodal_Datasets
from .Multimodal_Transformer_master.src import models
import numpy
prototypes_iter = 240000
length = 10
def update_roi_feature(roi_feature_all_level_class, x, targets):

        x_tar, levels, labels = x
        levels = levels.cpu().numpy().tolist()
        # print("==========================================")
        # print(labels)
        # print(levels)
        for i in range(len(labels)):
            ind = int(labels[i]-1)
            level = int(levels[i])
            feature = x_tar[i]
            if(len(roi_feature_all_level_class[level][ind]) == 0):
                roi_feature_all_level_class[level][ind].append(feature)
            else:
                flag = -2
                temp_len = len(roi_feature_all_level_class[level][ind])
                max_cos = 0
                for j in range(temp_len):
                    cos = torch.cosine_similarity(feature,roi_feature_all_level_class[level][ind][j],dim=0)
                    # print("label ind",i,"label",ind+1,"leng",temp_len,j,cos)
                    if(cos<0.6 ):
                        if(temp_len < length):
                            flag = -1
                        else:
                            if(cos > max_cos):
                                max_cos = cos
                                flag = j
                    else:
                        break
                    # print(ind,cos)
                if(flag == -1):
                    roi_feature_all_level_class[level][ind].append(feature)
                elif(flag >-1):
                    # for l in range(len(roi_feature_all_level_class[ind])):
                    #     print(roi_feature_all_level_class[ind][l][0:-250])
                    roi_feature_all_level_class[level][ind].pop(flag)
                    roi_feature_all_level_class[level][ind].append(feature)
                    # print("-----------------\n")
                    # for l in range(len(roi_feature_all_level_class[ind])):
                    #     print(roi_feature_all_level_class[ind][l][0:-250])
        # for i in range(4):
        #     for j in range(len(roi_feature_all_level_class[i])):
        #         print(len(roi_feature_all_level_class[i][j]),end=" ")
        #     print('\n')
        # import pdb
        # pdb.set_trace()
        return  roi_feature_all_level_class

def get_final_feature(roi_feature_all_level_class,iter):
        final_features = []
        for i in range(4):
            final_features.append(0)
        for level in range(len(roi_feature_all_level_class)):
            flag = 0
            for c in range(len(roi_feature_all_level_class[level])):
                for i in range(len(roi_feature_all_level_class[level][c])):
                    if(flag == 0 ):
                        final_features[level] = roi_feature_all_level_class[level][c][i].unsqueeze(0)
                        flag = 1
                    else:
                        flag += 1
                        final_features[level] = torch.cat((final_features[level],roi_feature_all_level_class[level][c][i].unsqueeze(0)),0)
                # print(level,c,len(roi_feature_all_level_class[level][c]))
            if(flag < 2):
                final_features[level] = torch.ones(5,7*7*256)
        if(iter%2500 == 0):
            with open("./feature_city5_foggy_size_101/"+str(iter)+".pkl",'wb') as f:
                pickle.dump(final_features, f, pickle.HIGHEST_PROTOCOL)
        for i in range(4):
            final_features[i] = final_features[i].cuda()
            # print("shape",final_features[i].shape)
        # print(final_feature.shape)
        return final_features

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        # self.hyp_params ,self.settings= transform(256)
        # self.multmodel = getattr(models, self.hyp_params.model+'Model')(self.hyp_params)
        # self.maxpool = torch.nn.MaxPool2d(7)
        # self.roi_feature_all_level_class = []
        # for i in range(4):
        #     self.roi_feature_all_level_class.append([])
        #     for j in range(cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES-1):
        #         self.roi_feature_all_level_class[i].append([])
        # self.iter =1

    def forward(self, i, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # print("features\n",len(features))
        # for i in range(5):
        #     print(features[i].shape)
        # if self.training:
        #     final_feature = get_final_feature(self.roi_feature_all_level_class,self.iter)
        #     self.iter += 1
        #     # print(self.w1,self.w2)
        # else:
        #     with open("./feature_city5_foggy_size_101/"+str(prototypes_iter)+".pkl","rb") as fo:
        #         final_feature=pickle.load(fo,encoding = 'bytes')
                
        # features = list(features)
        # for i in range(1,4):

        #     a = features[i].view(1, 256, -1)
        #     d_3 = features[i].shape[2]
        #     d_4 = features[i].shape[3]
        #     x_v = a.permute(0,2,1).contiguous()

        #     # print(final_feature.shape)
        #     n = final_feature[i].shape[0]
        #     k = final_feature[i].view(n,256,7,7).contiguous()
        #     # print("k",k)
        #     k = self.maxpool(k)
        #     k = k.view(n,256).contiguous()
        #     k = k.unsqueeze(0)
        #     # print(x_v.shape, x_l.shape)
        #     # print("in")
        #     # print(x_v.shape,k.shape)
        #     t_feature = self.multmodel(x_v,k)
        #     # print(t_feature)
        #     # ##c*c 1*192*256 1*256*192
        #     # q_1 = t_feature.permute(0,2,1)
        #     # k_1 = a.permute(0,2,1)
        #     # s = torch.bmm(q_1, k_1 )
        #     # s = torch.softmax(s,dim=-1)
        #     # # print(s_max,s/s_max)
        #     # a_feature = torch.bmm(s, a)
        #     # a_feature = a_feature.view(1,256,d_3,d_4).contiguous()

        #     # ## n*n
        #     # q = t_feature
        #     # k = a
        #     # # print("q,k",q.shape,k.shape)
        #     # attention = torch.bmm(q, k)
        #     # attention  = torch.softmax(attention,dim=-1)
        #     # p_feature = torch.bmm(attention, k.permute(0,2,1))
        #     # p_feature = p_feature.permute(0,2,1).view(1,256,d_3,d_4).contiguous()
        #     # import matplotlib.pyplot as plt 
        #     # import numpy  
        #     # # plt.matshow(t_feature[0][0].cpu().numpy())
        #     # plt.matshow(features[4][0][0].cpu().numpy())
        #     # print("1",features[4].shape,"\n2",t_feature.shape)
        #     # max1,_ = torch.max(features[4],1)
        #     # print(features[4][0][0].shape,max1.squeeze(0).shape)
        #     # plt.matshow(max1.squeeze(0).cpu().numpy())


        #     features[i] =  features[i] + t_feature.permute(0,2,1).view(1,256,d_3,d_4).contiguous()
            # print(i,features[i])
        # print("2",features[4][0][0].shape)
        # plt.matshow(features[4][0][0].cpu().numpy())
        # max2,_ = torch.max(features[4],1)
        # print(max2.shape)
        # plt.matshow(max2.squeeze(0).cpu().numpy())
        # plt.show()
        # import pdb
        # pdb.set_trace()
        


        
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses, x_tar = self.roi_heads(features, proposals, 1, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # self.roi_feature_all_level_class = update_roi_feature(self.roi_feature_all_level_class, x_tar, targets)
            return losses

        return result
