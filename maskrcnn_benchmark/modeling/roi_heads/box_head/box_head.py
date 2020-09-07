# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.detector.Multimodal_Transformer_master.trans import transform
from maskrcnn_benchmark.modeling.detector.Multimodal_Transformer_master.src.train import *
from maskrcnn_benchmark.modeling.detector.Multimodal_Transformer_master.src.dataset import Multimodal_Datasets
from maskrcnn_benchmark.modeling.detector.Multimodal_Transformer_master.src import models
length = 50
cos_thre = 0.6
feature_path = './feature_1024/'
prototypes_iter = 215000
def update_roi_feature(roi_feature_all_class, x):

        x_tar, labels = x
        # print("==========================================")
        # print(labels)
        # print(levels)
        for i in range(len(labels)):
            ind = int(labels[i]-1)
            feature = x_tar[i]
            if(len(roi_feature_all_class[ind]) == 0):
                roi_feature_all_class[ind].append(feature)
            else:
                flag = 0
                temp_len = len(roi_feature_all_class[ind])
                max_cos = 0
                min_cos = 2
                for j in range(temp_len):
                    cos = torch.cosine_similarity(feature,roi_feature_all_class[ind][j],dim=0)
                    # print("label ind",i,"label",ind+1,"leng",temp_len,j,cos)
                    if(cos > max_cos):
                        max_cos = cos
                        flag = j
                    if(cos < min_cos):
                        min_cos = cos
                    # print(ind,cos)
                if(temp_len < length and min_cos < cos_thre):
                    roi_feature_all_class[ind].append(feature)
                if(temp_len == length and min_cos < cos_thre):
                    # for l in range(len(roi_feature_all_class[ind])):
                    #     print(roi_feature_all_class[ind][l][0:-250])
                    # print(flag)
                    roi_feature_all_class[ind].pop(flag)
                    roi_feature_all_class[ind].append(feature)
                    # print("-----------------\n")
                    # for l in range(len(roi_feature_all_class[ind])):
                    #     print(roi_feature_all_class[ind][l][0:-250])
        # for i in range(4):
        #     for j in range(len(roi_feature_all_class[i])):
        #         print(len(roi_feature_all_class[i][j]),end=" ")
        #     print('\n')
        # import pdb
        # pdb.set_trace()
        return  roi_feature_all_class

def get_final_feature(roi_feature_all_class,iter):
        final_features = 0
        flag = 0
        for c in range(len(roi_feature_all_class)):
            for i in range(len(roi_feature_all_class[c])):
                if(flag == 0 ):
                    final_features = roi_feature_all_class[c][i].unsqueeze(0)
                    flag = 1
                else:
                    final_features = torch.cat((final_features,roi_feature_all_class[c][i].unsqueeze(0)),0)
            # print(level,c,len(roi_feature_all_class[c]))
        if(flag ==0):
            final_features = torch.ones(5,1024)
        if(iter%2500 == 0):
            with open(feature_path + str(iter)+".pkl",'wb') as f:
                pickle.dump(final_features, f, pickle.HIGHEST_PROTOCOL)
        
        final_features = final_features.cuda()
            # print("shape",final_features[i].shape)
        # print(final_feature.shape)
        return final_features
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.hyp_params ,self.settings= transform(1024)
        self.roi_multmodel = getattr(models, self.hyp_params.model+'Model')(self.hyp_params)

        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.roi_feature_all_class = []
        for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES-1):
            self.roi_feature_all_class.append([])
        self.iter =1

    def forward(self, features, proposals, prototypes, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x ,_ , _= self.feature_extractor(features, proposals)
        if self.training:
            with torch.no_grad():
                x_tar,_, levels = self.feature_extractor(features,targets)
                labels = targets[0].get_field("labels").cpu().numpy().tolist()
                self.roi_feature_all_class = update_roi_feature(self.roi_feature_all_class, (x_tar, labels))
                k = get_final_feature(self.roi_feature_all_class, self.iter)
                self.iter += 1
        else:
            with open(feature_path+str(prototypes_iter)+".pkl","rb") as fo:
                k = pickle.load(fo,encoding = 'bytes')
        q = x.unsqueeze(0)
        k = k.unsqueeze(0)
        # print(q.shape,k.shape)
        x_hance = self.roi_multmodel(q,k)
        # print(x.shape,x_hance.shape)
        x = torch.cat((x, x_hance.squeeze(0)), 1)
        # print(x.shape)
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, []
        with torch.no_grad():
            labels = targets[0].get_field("labels").cpu().numpy().tolist()
        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            (x_tar,levels,labels)
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
