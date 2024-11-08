#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange
from mmdet.core import reduce_mean
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
                             predict_instance_segmentation_and_trajectories

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(self, 
                 # General
                 receptive_field=3,
                 n_future=4,
                 spatial_extent=(50, 50),
                 ignore_index=255,

                 # BEV
                 grid_conf = None,

                 bev_size=(200, 200),
                 bev_emb_dim=256,
                 bev_proj_dim=64,
                 bev_proj_nlayers=1,

                 # Query
                 query_dim=256,
                 query_mlp_layers=3,
                 detach_query_pos=True,
                 temporal_mlp_layer=2,

                 # Transformer
                 transformer_decoder=None,

                 attn_mask_thresh=0.5,
                 # Loss
                 sample_ignore_mode='all_valid',
                 aux_loss_weight=1.,

                 loss_mask=None,
                 loss_dice=None,

                 # Cfgs
                 init_cfg=None,

                 # Eval
                 pan_eval=False,
                 test_seg_thresh:float=0.5,
                 test_with_track_score=False,
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index  = ignore_index

        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, grid_conf)
        
        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True)
        )

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.attn_mask_thresh = attn_mask_thresh
        
        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)

        # - temporal-mlps
        # query_out_dim = bev_proj_dim

        temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)
            
        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)
        
        # - upsampleAdds
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        # Query
        self.mode_fuser = nn.Sequential(
                nn.Linear(query_dim, bev_proj_dim),
                nn.LayerNorm(bev_proj_dim),
                nn.ReLU(inplace=True)
            )
        self.multi_query_fuser =  nn.Sequential(
                nn.Linear(query_dim * 3, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim * 2, bev_proj_dim),
            )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
        
        # Loss
        # For matching
        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.aux_loss_weight = aux_loss_weight

        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)

        self.pan_eval = pan_eval
        self.test_seg_thresh  = test_seg_thresh

        self.test_with_track_score = test_with_track_score
        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        """
        G^t通过MLP生成掩码cur_ins_emb_for_mask_attn
        通过爱因斯坦求和约定：cur_ins_emb_for_mask_attn @ F^t 得到attn_mask的分数
        attn_mask的分数过激活函数+和阈值比较得到二值的attn_mask：O^t_m
        对attn_mask进行插值上采样得到mask_pred

        Args:
            state (_type_): F^t
            ins_query (_type_): G^t

        Returns:
            注意力掩码O^t_m, 插值上采样的掩码mask_pred, 用于掩码注意力的时间嵌入
        """
        # state: b, c, h, w
        # ins_query: b, q, c
        ins_embed = self.temporal_mlp_for_mask(
            ins_query 
        )
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()
        
        # if a mask is all True(all background), then set it all False.
        attn_mask[torch.where(
            attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred,
            self.bev_size,
            mode='bilinear',
            align_corners=False
        )  # Supervised by gt

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):
        """
        x: BEV
        ins_query: G^t = MLP_t(Q_A, Q_A_pos, pool(Q_ctx)

        在其中
        F: state
        G: ins_query
        base_state: F^0; base_ins_query: G^0
        last_state: F^t-1; last_ins_query: G^t-1
        cur_state: F^t; cur_ins_query: G^t
        """
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])
        #* state: F^t
        #* F^0初始化：BEV采样->投影->下采样 base_state: [b, d, h/4, w/4] -> F^0
        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        #* ins_query: G^t
        #* 初始化G^0
        base_ins_query = ins_query
        # 记录F^t-1和G^t-1
        last_state = base_state           #* F^t-1
        last_ins_query = base_ins_query   #* G^t-1
        #* 存储未来状态F^t、预测掩码M^t(O^t_m上采样)、时间查询G^t和用于掩码注意力的时间嵌入O^t: 预测n_future_blocks步
        future_states = []                #* {F^t}
        mask_preds = []                   #* {M^t}        
        temporal_query = []               #* {G^t}
        temporal_embed_for_mask_attn = [] #* {O^t_m}
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1

        #* 逐步预测未来状态
        for i in range(self.n_future_blocks):
            #* F^t: 对F^t-1进行下采样
            cur_state = self.downscale_convs[i](last_state)  # /4 -> /8

            # Attention
            #* G^t: G^t-1通过MLP
            cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
            temporal_query.append(cur_ins_query)

            #* 获取O^t_m
            #* 最终有两个掩码: 
            #* attn_mask:用于注意力计算的掩码O^t_m
            #* mask_pred:attn_mask的插值上采样, 用于最终预测结果的掩码
            #* 还有一个输出: cur_ins_emb_for_mask_attn, G^t过MLP得到
            # G^t通过MLP生成掩码cur_ins_emb_for_mask_attn
            # 通过爱因斯坦求和约定：cur_ins_emb_for_mask_attn @ F^t 得到attn_mask的分数
            # attn_mask的分数过激活函数+和阈值比较得到二值的attn_mask：O^t_m
            # 对attn_mask进行插值上采样得到mask_pred: M^t
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
            attn_masks = [None, attn_mask] 

            mask_preds.append(mask_pred)  # /1
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            # 
            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')
            
            # 过n层transformer 只取decoder
            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                #* decoder
                #* D^t(F^t) = MHCA(MHSA(F^t), G^t, attn_mask = O^t_m)
                cur_state = trans_layer(
                    query=cur_state,  # [h'*w', b, c]
                    key=cur_ins_query,  # [nq, b, c]
                    value=cur_ins_query,  # [nq, b, c]
                    query_pos=None,  
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )  # out size: [h'*w', b, c]

            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
            
            # Upscale to /4
            #* transformer输出D^t(F^t)通过上采样和残差连接得到F^t
            cur_state = self.upsample_adds[i](cur_state, last_state) # /8 -> /4

            # Out
            #* 在未来状态中存储F^t，并将F^t作为下一步的F^t-1
            future_states.append(cur_state)  # [b, d, h/4, w/4]
            last_state = cur_state

        #* 整合{F^t}, {G^t}, {M^t}, {O^t_m}
        future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
        temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
        mask_preds = torch.stack(mask_preds, dim=2)  # [b, q, t, h, w]
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)  # [b, t, q, d]

        #* 将F^t通过卷积decoder上采样得到像素级的F_dec^t
        future_states = self.dense_decoder(future_states)  # /4 -> /1

        #* U^t = MLP(M^t)
        ins_occ_query = self.query_to_occ_feat(ins_query)    # [b, t, q, query_out_dim]
        
        #* 得到最终的像素级Occ占用
        # O^t_A = U^t @ F_dec^t
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
        
        #* 输出像素级Occ(O^t_A)和预测掩码
        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        ins_query = outs_dict.get('traj_query', None)       # [n_dec, b, nq, n_modes, dim]
        track_query = outs_dict['track_query']              # [b, nq, d]
        track_query_pos = outs_dict['track_query_pos']      # [b, nq, d]

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        ins_query = ins_query[-1]
        ins_query = self.mode_fuser(ins_query).max(2)[0]
        ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))
        
        return ins_query

    # With matched queries [a small part of all queries] and matched_gt results
    def forward_train(
                    self,
                    bev_feat,
                    outs_dict,
                    gt_inds_list=None,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                ):
        # 生成地面实况标签和相关输入: 地面实况分割, 地面实况实例, 地面实况图像有效性
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)
        #* 从motionformer输出中提取所有匹配的地面实况 ID
        all_matched_gt_ids = outs_dict['all_matched_idxes']  # list of tensor, length bs
        
        #* 将motion的Q_ctx 和 Q_A 过MLP生成G^0##########################
        # G^0 = MLP_t(Q_A, Q_A_pos, pool(Q_ctx)
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        #* Occ前向传播
        #* 输入： BEV，Q_ctx和Q过MLP的到的G^0
        #* 输出： 像素级预测掩码(注意力掩码插值上采样)，像素级Occ(O^t_A)
        mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)
        
        # Get pred and gt
        #* 获取预测和gt，计算loss#############################
        ins_seg_targets_batch  = gt_instance # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch
        # img_valid flag, for filtering out invalid samples in sequence when calculating loss
        #* 计算loss时过滤掉无效的样本
        img_is_valid = gt_img_is_valid  # [1, 7]
        assert img_is_valid.size(1) == self.receptive_field + self.n_future,  \
                f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
        frame_valid_mask = img_is_valid.bool()      # 所有帧数的有效性掩码
        # 过去和未来的有效性掩码
        past_valid_mask  = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]  # [1, 5]  including current frame

        # only supervise when all 3 past frames are valid
        # 检查过去的3帧是否都有效
        past_valid = past_valid_mask.all(dim=1)
        future_frame_mask[~past_valid] = False
        
        # 计算batch中的loss，初始化为dict
        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()       
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

        # 遍历batch中的每个样本
        bs = ins_query.size(0)
        assert bs == 1
        for ind in range(bs):
            # Each gt_bboxes contains 3 frames, we only use the last one
            # 获取地面实况索引
            cur_gt_inds   = gt_inds_list[ind][-1]

            cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt]
            
            # Re-order gt according to matched_gt_inds
            cur_gt_inds   = cur_gt_inds[cur_matched_gt]
            
            # Deal matched_gt: -1, its actually background(unmatched)
            cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
            cur_gt_inds[cur_matched_gt == -2] = -2  

            # 未来帧掩码
            frame_mask = future_frame_mask[ind]  # [t]

            # Prediction
            # 获取预测和目标
            ins_seg_preds = ins_seg_preds_batch[ind]   # [q(n_gt for matched), t, h, w]
            ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
            mask_preds = mask_preds_batch[ind]
            
            # Assigned-gt
            # 分配地面实况目标
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                # -1 for unmatched query
                # If ins_seg_targets is all 255, ignore (directly append occ-and-flow gt to list)
                # 255 for special object --> change to -20 (same as in occ_label.py)
                # -2 for no_query situation
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2] :  # false positive query (unmatched)
                    ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
                else:
                    SPECIAL_INDEX = -20
                    if ins_id.item() == self.ignore_index:
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    ins_tgt = (ins_seg_targets == ins_id).long()  # [t, h, w], 0 or 1
                
                ins_seg_targets_ordered.append(ins_tgt)
            # 分配后的地面实况目标叠起来
            ins_seg_targets_ordered = torch.stack(ins_seg_targets_ordered, dim=0)  # [n_gt, t, h, w]
            
            # Sanity check
            # 检查预测和目标的size是否一致
            t, h, w = ins_seg_preds.shape[-3:]
            assert t == 1+self.n_future, f"{ins_seg_preds.size()}"
            assert ins_seg_preds.size() == ins_seg_targets_ordered.size(),   \
                            f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
            
            num_total_pos = ins_seg_preds.size(0)  # Check this line 

            # loss for a sample in batch
            #* 计算损失
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
            
            #* 分割的损失(实例级)
            cur_dice_loss = self.loss_dice(
                ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)
            #* 实例级掩码的损失(实例级)
            cur_mask_loss = self.loss_mask(
                ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )
            #* 辅助分割的损失(attn级) 
            cur_aux_dice_loss = self.loss_dice(
                mask_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask
            )
            #* 辅助掩码的损失(attn级)
            cur_aux_mask_loss = self.loss_mask(
                mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )

            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
            loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

        # batch的损失归一化
        loss_dict['loss_dice'] = loss_dice / bs
        loss_dict['loss_mask'] = loss_mask / bs
        loss_dict['loss_aux_dice'] = loss_aux_dice / bs
        loss_dict['loss_aux_mask'] = loss_aux_mask / bs

        #* 训练阶段只返回loss: Occ只参与数值优化，不参与planner的网络
        return loss_dict

    def forward_test(
                    self,
                    bev_feat,
                    outs_dict,
                    no_query=False,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                ):
        #* 从地面实况中提取未来n步的Occ标签
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)

        # 提取未来n步的地面实况
        out_dict = dict()
        out_dict['seg_gt']  = gt_segmentation[:, :1+self.n_future]  # [1, 5, 1, 200, 200]
        out_dict['ins_seg_gt'] = self.get_ins_seg_gt(gt_instance[:, :1+self.n_future])  # [1, 5, 200, 200]
        #* Q_A为空：返回全零结果： 相当于没有Occ预测
        if no_query:
            # output all zero results
            out_dict['seg_out'] = torch.zeros_like(out_dict['seg_gt']).long()  # [1, 5, 1, 200, 200]
            out_dict['ins_seg_out'] = torch.zeros_like(out_dict['ins_seg_gt']).long()  # [1, 5, 200, 200]
            return out_dict

        #* 将motion的Q_ctx 和 Q_A 过MLP生成G^0##########################
        # G^0 = MLP_t(Q_A, Q_A_pos, pool(Q_ctx)
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        
        #* Occ前向传播
        #* 输入： BEV，Q_ctx和Q过MLP的到的G^0
        #* 输出： 像素级预测掩码(注意力掩码插值上采样)，像素级Occ(O^t_A)
        #  测试时不需要掩码，只需要Occ
        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)

        out_dict['pred_ins_logits'] = pred_ins_logits       # 输出装填

        #* 取未来n步的Occ，过激活函数转化为概率
        pred_ins_logits = pred_ins_logits[:,:,:1+self.n_future]  # [b, q, t, h, w]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()  # [b, q, t, h, w]
        
        #* 测试时是否使用track score，默认为True
        #? 用track score对Occ进行加权: track_scores * Occ：track_scores为1或0, 只关注track置信度高的Occ
        if self.test_with_track_score:
            track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)  # [b, q]
            track_scores = track_scores[:, :, None, None, None]
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]

        out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid    # 输出装填
        
        # 取出Occ预测的概率的最大值作为分数
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        #* 阈值化得到Occ分割，大于阈值为1，小于阈值为0
        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)  # [b, t, 1, h, w]
        out_dict['seg_out'] = seg_out
        # 评估时是否进行实例分割，默认为True
        if self.pan_eval:
            # ins_pred
            pred_consistent_instance_seg =  \
                predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)  # bg is 0, fg starts with 1, consecutive
            
            out_dict['ins_seg_out'] = pred_consistent_instance_seg  # [1, 5, 200, 200]
        
        #* 输出：'seg_gt', 'ins_seg_gt': 地面实况分割和实例分割；'pred_ins_logits': Occ原始输出
        #* 'pred_ins_sigmoid': Occ预测的概率；'seg_out', 'ins_seg_out': 预测的Occ分割和实例分割
        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = gt_instance  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        if not self.training:
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]

        gt_segmentation = gt_segmentation[:, :self.n_future+1].long().unsqueeze(2)
        gt_instance = gt_instance[:, :self.n_future+1].long()
        gt_img_is_valid = gt_img_is_valid[:, :self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
