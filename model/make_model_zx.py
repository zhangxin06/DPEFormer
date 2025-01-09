import torch
import torch.nn as nn
import copy
import random
import torchvision
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
    deit_small_patch16_224_TransReID, fbm_qkv
import torch.nn.functional as F
from model.perceiver import ffn
import numpy as np


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except BaseException:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def shuffle_unit_without_global(features, shift=5, group=4):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, shift:], features[:, 0: shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except BaseException:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def shuffle_unit_random(features, proposal_k, shift=1):
    m = random.randint(0, proposal_k - 1)
    batchsize = features.size(0)
    result_shuffle = []
    for j in range(batchsize):
        kk = []
        for i in range(proposal_k - m):
            kk.append(features[j][m + i].unsqueeze(0))
        for i in range(m):
            kk.append(features[j][i].unsqueeze(0))
        kk = torch.cat(kk, dim=0)
        result_shuffle.append(kk.unsqueeze(0))
    result = torch.cat(result_shuffle, dim=0)

    gap_4 = nn.AdaptiveAvgPool1d(4)
    part_feature = gap_4(result.permute(0, 2, 1))

    part_feature = part_feature.permute(0, 2, 1)
    part_feature = part_feature[:, torch.randperm(part_feature.size(1)), :]

    return part_feature


def Patch_Selection_Dynamic(x1_global, x1_local_all,proposal_k=24):
    B = x1_global.size(0)
    distance_with_global = torch.matmul(x1_global.unsqueeze(dim=1), x1_local_all.permute(0, 2, 1))
    distance_pairs = torch.matmul(x1_local_all, x1_local_all.permute(0, 2, 1))
    anchor = torch.argmax(distance_with_global, dim=-1)
    result = []
    tongji = []
    for b in range(B):
        archor_distance = distance_pairs[b, anchor[b], :]
        features = archor_distance.sort(dim=-1, descending=True)
        features = features.values
        shift = torch.cat([features[:, 1:], features[:, -1:]], dim=-1)
        diff = features-shift
        tongji.append(diff.argmax().cpu())
    mean_value = int(np.ceil(np.mean(tongji)))
    min_value = int(np.ceil(np.min(tongji)))

    if min_value<24:
        min_value=24

    gap_x = nn.AdaptiveAvgPool1d(min_value)

    for b in range(B):
        archor_distance = distance_pairs[b, anchor[b], :]
        proposal_k = int(tongji[b])

        if proposal_k<24:
            proposal_k=24

        top_k = archor_distance.topk(proposal_k)[1]

        if proposal_k != min_value:
            last = gap_x(x1_local_all[b, top_k, :].permute(0,2,1))
            last = last.permute(0,2,1)
            result.append(last)
        else:
            top_k = archor_distance.topk(proposal_k)[1]
            result.append(x1_local_all[b, top_k, :])

    result = torch.cat(result, dim=0)
    return result, min_value


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def feature_search(memory, x, B, k=5, label=None, training=False):
    x = F.normalize(x, dim=1)
    memory = F.normalize(memory, dim=1)
    distmat = torch.matmul(x, memory.t())

    index = torch.topk(distmat, k=k, dim=1)[1]
    candidates = []

    for i in range(B):
        if training:
            if label[i] in index[i]:
                candidates.append(
                    index[i, label[i] != index[i]].unsqueeze(dim=0))
            else:
                candidates.append(
                    index[i][0:k - 1].unsqueeze(dim=0))
        else:
            candidates.append(index[i][0:k - 1].unsqueeze(dim=0))

    candidates = torch.cat(candidates, dim=0)
    latents = memory[candidates]
    return latents


def cosine_similarity(x):
    x_norm = F.normalize(x, dim=-1)
    dist = torch.matmul(x_norm, x_norm.permute(0, 1, 3, 2))
    dist = dist.mean(dim=-1, keepdim=True)

    return torch.sum(x * dist, dim=2)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'els':
            self.in_planes = 2048
            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet50.layer4[0].conv2.stride = [1, 1]
            resnet50.layer4[0].downsample[0].stride = [1, 1]
            self.base = nn.Sequential(*list(resnet50.children())[:-2])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(
            self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    # label is unused if self.cos_layer == 'no'
    def forward(self, x, label=None, cam_label=None, view_label=None):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class nonLocal_zx(nn.Module):
    def __init__(self, in_dim):
        super(nonLocal_zx, self).__init__()
        self.conv_query = nn.Linear(in_dim, in_dim)
        self.conv_part = nn.Linear(in_dim, in_dim)
        self.conv_value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.param = nn.Parameter(torch.zeros(1))
        self.linear1 = nn.Linear(2*in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, part_feature, global_feature):
        f_part = self.conv_query(part_feature)
        # print(f_query.shape)

        f_global = self.conv_part(global_feature)
        # print(f_part.shape)
        f_value = self.conv_value(global_feature).unsqueeze(2)

        feature = torch.cat([f_part, f_global], dim=-1)
        feature = self.linear1(feature)
        feature = self.linear2(feature)


        energy = self.sigmoid(feature)

        similarity = energy.unsqueeze(1)


        f_value = torch.matmul(f_value, similarity) + global_feature.unsqueeze(2)

        final_feat = part_feature.unsqueeze(2) + torch.matmul(f_value, self.param.unsqueeze(0))

        return final_feat.squeeze(2)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.using_REM = True
        self.REM_zx = nn.ModuleList([nonLocal_zx(768) for _ in range(4)])
        self.attention_global = nn.ModuleList([fbm_qkv(768) for _ in range(4)])


        self.backbone = 'els'
        self.instances, self.parts, self.part_number = 9, 4, 3
        self.num_classes, self.ID_LOSS_TYPE = num_classes, cfg.MODEL.ID_LOSS_TYPE
        self.gap, self.gap_n, self.gap_3, self.gap_2 = nn.AdaptiveAvgPool1d(
            1), nn.AdaptiveAvgPool1d(self.parts), nn.AdaptiveAvgPool1d(3), nn.AdaptiveAvgPool1d(2)

        self.gap_4 = nn.AdaptiveAvgPool1d(4)

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num,
                                                        view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.in_planes = 768
        self.in_planes_ori = 768

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        b = []
        for i in range(4):
            b2 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            b.append(b2)
        self.b2 = nn.ModuleList(b)

        # for token classifier
        self.classifier1 = nn.Linear(
            self.in_planes * self.parts,
            self.num_classes,
            bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.classifier1_4 = nn.Linear(
            self.in_planes * 4,  # 3
            self.num_classes,
            bias=False)
        self.classifier1_4.apply(weights_init_classifier)

        self.classifier1_3 = nn.Linear(
            self.in_planes * 3,  # 3
            self.num_classes,
            bias=False)
        self.classifier1_3.apply(weights_init_classifier)

        self.classifier1_2 = nn.Linear(
            self.in_planes * 2,
            self.num_classes,
            bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.in_planes * 3,
                                     self.num_classes, bias=False)

        self.classifier3 = nn.Linear(
            self.in_planes, self.num_classes, bias=False)
        self.classifier3.apply(weights_init_classifier)

        self.bottleneck_global = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_global.bias.requires_grad_(False)
        self.bottleneck_global.apply(weights_init_kaiming)

        bn_list = []
        for _ in range(self.parts + 6):
            temp = nn.BatchNorm1d(self.in_planes)
            temp.bias.requires_grad_(False)
            temp.apply(weights_init_kaiming)
            bn_list.append(temp)
        self.bottleneck_local = nn.ModuleList(bn_list)

        # this is bn list for stage 2
        bn_list = []
        for _ in range(self.parts):
            temp = nn.BatchNorm1d(self.in_planes)
            temp.bias.requires_grad_(False)
            temp.apply(weights_init_kaiming)
            bn_list.append(temp)
        self.bottleneck_local_stage2 = nn.ModuleList(bn_list)

        # this is classifier_list
        fc_list = []
        for _ in range(self.parts + 6):
            temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
            temp.apply(weights_init_classifier)
            fc_list.append(temp)
        self.classifier_local = nn.ModuleList(fc_list)

        # for recons
        self.ffn2 = ffn(latent_dim=self.in_planes)
        self.ffn3 = ffn(latent_dim=self.in_planes)


    def feature_extraction(self, x, B):
        x = self.base(x)
        x_global_feat, x_local_feat = x[0], x[1]
        x_local_feat_all = x_local_feat.clone()
        x_local_feat = self.gap_n(x_local_feat.permute(0, 2, 1))
        return x_global_feat, x_local_feat, x_local_feat_all


    def forward(self, x, memory_only=False, recon=False, labels=None, memory_features=None,):
        B = x.size(0)


        if recon==True:
            x1 = self.feature_extraction(x, B=B)
            x1_global, x1, x1_local_all = x1[0], x1[1], x1[2]

            x1 = x1.permute(0, 2, 1)

            x1_list, x1_3_list = [], []
            if self.backbone == 'res50':
                for i in range(self.parts):
                    x1_list.append(self.reduction_list[i](x1[:, i, :]))
            else:
                for i in range(self.parts):
                    x1_list.append(self.bottleneck_local[i](x1[:, i, :]))

            x1_bn = torch.cat(x1_list, dim=1)
            result, proposal_k = Patch_Selection_Dynamic(x1_global, x1_local_all)
            gap_4 = nn.AdaptiveAvgPool1d(4)  # part_number=4
            part_feature1 = gap_4(result.permute(0, 2, 1))
            part_feature1 = part_feature1.permute(0, 2, 1)

            part1 = []
            if self.using_REM:
                for i in range(self.parts):
                    a = self.REM_zx[i](part_feature1[:, i, :], x1_global)
                    part1.append(a.unsqueeze(dim=1))
            part_feature1 = torch.cat(part1, dim=1)
            x1_new_list = []

            # part_number=4
            for i in range(4):  # 4
                x1_new_list.append(self.bottleneck_local[i + 4](part_feature1[:, i, :]))
            x1_new = torch.cat(x1_new_list, dim=1)

            return x1_new



        if memory_only is True:
            x = self.feature_extraction(
                x, B)[1]
            feat_list = []
            for i in range(self.parts):
                if self.backbone == 'res50':
                    feat_list.append(self.reduction_list[i](x[:, :, i]))
                else:
                    feat_list.append(self.bottleneck_local[i](x[:, :, i]))

            return torch.cat(feat_list, dim=-1)


        else:
            if self.training:
                x = x[:, 0: 6, :, :]

                x = torch.chunk(x, 2, dim=1)
                x1, x2 = x[0], x[1]
                x1, x2 = self.feature_extraction(
                    x1, B), self.feature_extraction(x2, B)

                x1_global, x1, x1_local_all = x1[0], x1[1], x1[2]
                x2_global, x2, x2_local_all = x2[0], x2[1], x2[2]

                result, proposal_k = Patch_Selection_Dynamic(x1_global,x1_local_all)

                gap_4 = nn.AdaptiveAvgPool1d(4)
                part_feature1 = gap_4(result.permute(0, 2, 1))
                part_feature1 = part_feature1.permute(0, 2, 1)

                part1 = []
                if self.using_REM:
                    for i in range(self.parts):
                        a = self.attention_global[i](part_feature1[:, i, :], x1_global)
                        part1.append(a.unsqueeze(dim=1))
                part_feature1 = torch.cat(part1, dim=1)

                result, proposal_k = Patch_Selection_Dynamic(x2_global, x2_local_all)
                part_feature2 = gap_4(result.permute(0, 2, 1))
                part_feature2 = part_feature2.permute(0, 2, 1)

                part2 = []
                if self.using_REM:
                    for i in range(self.parts):
                        a = self.attention_global[i](part_feature2[:, i, :], x2_global)
                        part2.append(a.unsqueeze(dim=1))
                part_feature2 = torch.cat(part2, dim=1)

                cls_global_1_1 = self.classifier3(x1_global)
                cls_global_2_1 = self.classifier3(x2_global)

                x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)

                x1_list, x1_part_list, x2_list, x2_part_list = [], [], [], []
                cls_1_part, cls_1_self_part, cls_2_part, cls_2_self_part = [], [], [], []
                if self.backbone == 'res50':
                    for i in range(self.parts):
                        x1_list.append(self.reduction_list[i](x1[:, i, :]))
                        x2_list.append(self.reduction_list[i](x2[:, i, :]))
                else:
                    for i in range(self.parts):  # 4
                        x1_list.append(self.bottleneck_local[i](x1[:, i, :]))
                        x2_list.append(self.bottleneck_local[i](x2[:, i, :]))

                    for j in range(4):  # part_number=4
                        x1_part_list.append(
                            self.bottleneck_local[j + 4](part_feature1[:, j, :]))
                        x2_part_list.append(
                            self.bottleneck_local[j + 4](part_feature2[:, j, :]))
                        cls_1_self_part.append(
                            self.classifier_local[j + 4](x1_part_list[j]))
                        cls_2_self_part.append(
                            self.classifier_local[j + 4](x2_part_list[j]))

                x1_bn = torch.cat(x1_list, dim=1)
                x2_bn = torch.cat(x2_list, dim=1)


                x1_new = torch.cat(x1_part_list, dim=1)
                x2_new = torch.cat(x2_part_list, dim=1)
                cls_1_0 = self.classifier1(x1_bn)
                cls_2_0 = self.classifier1(x2_bn)

                # part_number=4
                cls_3_0 = self.classifier1_4(x1_new)
                cls_4_0 = self.classifier1_4(x2_new)

                return [cls_1_0, cls_2_0, cls_3_0, cls_4_0] + [cls_global_1_1,
                                                               cls_global_2_1] + cls_1_self_part + cls_2_self_part, x1_bn, x2_bn, x1_new, x2_new

            else:
                x1 = self.feature_extraction(x, B=B)
                x1_global, x1, x1_local_all = x1[0], x1[1], x1[2]

                x1 = x1.permute(0, 2, 1)

                x1_list, x1_3_list = [], []
                if self.backbone == 'res50':
                    for i in range(self.parts):
                        x1_list.append(self.reduction_list[i](x1[:, i, :]))
                else:
                    for i in range(self.parts):
                        x1_list.append(self.bottleneck_local[i](x1[:, i, :]))


                x1_bn = torch.cat(x1_list, dim=1)

                result, proposal_k = Patch_Selection_Dynamic(x1_global, x1_local_all)
                gap_4 = nn.AdaptiveAvgPool1d(4)  # part_number=4
                part_feature1 = gap_4(result.permute(0, 2, 1))
                part_feature1 = part_feature1.permute(0, 2, 1)

                part1 = []
                if self.using_REM:
                    for i in range(self.parts):
                        a = self.attention_global[i](part_feature1[:, i, :], x1_global)
                        part1.append(a.unsqueeze(dim=1))
                part_feature1 = torch.cat(part1, dim=1)


                x1_new_list = []

                for i in range(4):  # 4
                    x1_new_list.append(self.bottleneck_local[i + 4](part_feature1[:, i, :]))
                x1_new = torch.cat(x1_new_list, dim=1)

                return x1_bn, x1_new

    def load_param(self, trained_path):

        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model_1(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        model = build_transformer(
            num_class,
            camera_num,
            view_num,
            cfg,
            __factory_T_type)
        print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
