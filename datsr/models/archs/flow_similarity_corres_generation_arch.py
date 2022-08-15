import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from datsr.models.archs.arch_util import tensor_shift
from datsr.models.archs.ref_map_util import feature_match_index
from datsr.models.archs.vgg_arch import VGGFeatureExtractor
import pdb
logger = logging.getLogger('base')


class FlowSimCorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19'):
        super(FlowSimCorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def forward(self, dense_features, img_ref_hr):
        batch_offset_relu3 = []
        batch_offset_relu2 = []
        batch_offset_relu1 = []
        flows_relu3 = []
        flows_relu2 = []
        flows_relu1 = []
        similarity_relu3 = []
        similarity_relu2 = []
        similarity_relu1 = []
        # pdb.set_trace()
        for ind in range(img_ref_hr.size(0)):  # [9, 3, 160, 160]
            feat_in = dense_features['dense_features1'][ind]    # [256, 40, 40]
            feat_ref = dense_features['dense_features2'][ind]   # [256, 40, 40]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(
                feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            _max_idx, _max_val = feature_match_index(   # [38, 38], [38, 38]
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)
            
            # similarity for relu3_1
            sim_relu3 = F.pad(_max_val, (1,1,1,1)).unsqueeze(0)
            similarity_relu3.append(sim_relu3)
            # offset map for relu3_1
            offset_relu3 = self.index_to_flow(_max_idx)   # [1, 40, 40, 2]
            flows_relu3.append(offset_relu3)
            # shift offset relu3
            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j))  # [1, 40, 40, 2]
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0)  # [9, 40, 40, 2]
            batch_offset_relu3.append(shifted_offset_relu3)
            
            # similarity for relu2_1
            # pdb.set_trace()
            sim_relu2 = torch.repeat_interleave(sim_relu3, 2, 1)  # [1, 80, 40, 2]
            sim_relu2 = torch.repeat_interleave(sim_relu2, 2, 2)  # [1, 80, 80, 2]
            similarity_relu2.append(sim_relu2)
            # offset map for relu2_1
            offset_relu2 = torch.repeat_interleave(offset_relu3, 2, 1)  # [1, 80, 40, 2]
            offset_relu2 = torch.repeat_interleave(offset_relu2, 2, 2)  # [1, 80, 80, 2]
            offset_relu2 *= 2
            flows_relu2.append(offset_relu2)
            # shift offset relu2
            shifted_offset_relu2 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu2, (i * 2, j * 2))
                    shifted_offset_relu2.append(flow_shift)
            shifted_offset_relu2 = torch.cat(shifted_offset_relu2, dim=0)  # [9, 80, 80, 2]
            batch_offset_relu2.append(shifted_offset_relu2)

            # similarity for relu1_1
            sim_relu1 = torch.repeat_interleave(sim_relu2, 2, 1)  # [1, 80, 40, 2]
            sim_relu1 = torch.repeat_interleave(sim_relu1, 2, 2)  # [1, 80, 80, 2]
            similarity_relu1.append(sim_relu1)
            # offset map for relu1_1
            offset_relu1 = torch.repeat_interleave(offset_relu3, 4, 1)
            offset_relu1 = torch.repeat_interleave(offset_relu1, 4, 2)
            offset_relu1 *= 4
            flows_relu1.append(offset_relu1)
            # shift offset relu1
            shifted_offset_relu1 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu1, (i * 4, j * 4))
                    shifted_offset_relu1.append(flow_shift)
            shifted_offset_relu1 = torch.cat(shifted_offset_relu1, dim=0)
            batch_offset_relu1.append(shifted_offset_relu1)

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu3 = torch.stack(batch_offset_relu3, dim=0)
        batch_offset_relu2 = torch.stack(batch_offset_relu2, dim=0)
        batch_offset_relu1 = torch.stack(batch_offset_relu1, dim=0)

        # flows
        pre_flow = {}
        pre_flow['relu3_1'] = torch.cat(flows_relu3, dim=0)
        pre_flow['relu2_1'] = torch.cat(flows_relu2, dim=0)
        pre_flow['relu1_1'] = torch.cat(flows_relu1, dim=0)

        pre_offset = {}
        pre_offset['relu1_1'] = batch_offset_relu1  # [9, 9, 160, 160, 2]
        pre_offset['relu2_1'] = batch_offset_relu2  # [9, 9, 80, 80, 2]
        pre_offset['relu3_1'] = batch_offset_relu3  # [9, 9, 40, 40, 2]

        # similarity
        pre_similarity = {}
        pre_similarity['relu3_1'] = torch.stack(similarity_relu3, dim=0)
        pre_similarity['relu2_1'] = torch.stack(similarity_relu2, dim=0)
        pre_similarity['relu1_1'] = torch.stack(similarity_relu1, dim=0)
        
        img_ref_feat = self.vgg(img_ref_hr)
        # 'relu1_1': [9, 64, 160, 160]
        # 'relu2_1': [9, 128, 80, 80]
        # 'relu3_1': [9, 256, 40, 40]

        return [pre_offset, pre_flow, pre_similarity], img_ref_feat
