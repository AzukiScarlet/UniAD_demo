#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head

@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        seg_head=None,                # 语义分割的头: Mapformer              
        motion_head=None,             # Motionformer的头
        occ_head=None,                # Occ的头
        planning_head=None,           # Planning的头
        task_loss_weight=dict(  # 一般用默认
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        **kwargs,                    
    ):
        super(UniAD, self).__init__(**kwargs)      # 初始化cfg中的参数

        # cfg中若有，初始化各个任务的头
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)
        
        # 初始化各个任务的权重, 确保有track, map, motion, occ, planning
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
            {'track', 'occ', 'motion', 'map', 'planning'}

    # 检查是否有各个任务的头, @property装饰器，只读属性
    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None
    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None
    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        """
        在训练前，测试前向传播
        """
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """
        根据 return_loss=True 的设置，将调用 forward_train 或 forward_test。
        
        此设置将改变预期的输入。当 return_loss=True 时，即train阶段，img 和 img_metas 是单层嵌套的（即 torch.Tensor 和 list[dict]）；
        
        而当 return_loss=False 时，即测试阶段，img 和 img_metas 应该是双层嵌套的（即 list[torch.Tensor] 和 list[list[dict]]），外层列表表示测试时的增强。
        """
        # 训练是需要返回loss，测试不需要
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points')) # 装饰器，自动将输入转换为fp16，提高训练效率，减少显存占用。
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,

                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,
                      **kwargs,  # [1, 9]
                      ):
        """
        训练过程前向传播，such as tracking, segmentation, motion prediction, occupancy prediction, and planning.
        输入为(N, C, H, W)的图像张量，返回损失字典。

            Args:
                img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
                img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
                gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
                gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
                gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
                l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
                l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
                timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
                gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
                gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
                gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
                gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
                gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
                gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
                gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
                gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
                gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
                gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
                gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
                gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
                gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
                gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
                gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
                sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
                sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
                command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
                gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
                gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        #* 各个模块核心输出
        #* BEV： bev_embed: 从trackformer的encoder中提取
        #* track_former: outs_track
        #* map_former: outs_seg
        #* motion_former: outs_motion
        #* occ_former: outs_occ
        #* planning_former: outs_planning

        # loss初始化
        losses = dict()
        len_queue = img.size(1) # 输入图像队列长度
        #* 感知部分###################################################

        #* track_former的前向传播，返回losses和outs_track
        # outs_track是一个字典，["bev_embed", "bev_pos",
        #            "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
        #            "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        # 这里还要处理图片，还要BEV等等，因此写了一个专门的函数`forward_track_train`
        #* 本质是一整个Transformer的前向传播
        #* encoder输出BEV特征
        #* decoder输出Trackformer的Query
        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track') # 加权loss
        losses.update(losses_track)
        
        # 如果是tiny模型，需要上采样bev
        outs_track = self.upsample_bev_if_tiny(outs_track)
        # 从输出中获取BEV嵌入和位置
        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]


        #* map_former的前向传播，返回losses和outs_seg
        #* outs_seg是一个字典，包含了(outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, args_tuple, reference)
        #* 其中args_tuple是一个元组，包含了(memory, memory_mask, memory_pos, **query**, _, query_pos, hw_lvl)
        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')  # 加权loss
            losses.update(losses_seg)

        #* 预测部分######################################################################
        outs_motion = dict()   # 初始化motion的输出
        # Forward Motion Head
        if self.with_motion_head:
            #* 输入: BEV特征：B(bev_embed)，track的输出：outs_track(Q_a), Map的输出：uts_seg(Q_m)
            #* 输出: losses_motion, outs_motion, track_box 
            #* 其中outs_motion包含了motionformer的每一层decoder的Q_ctx，同时还将Q_ctx中自车的Q分离出来
            #? loss计算中使用了数值优化来平滑轨迹？？？？？？？？？？
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            #* 输入: BEV, 预测信息: outs_motion(Q_A, Q_ctx)
            #* 输出: Occ的losses
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                # 如果没有Q_A则直接用0填充(一般不用)
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                            bev_embed, 
                            outs_motion, 
                            gt_inds_list=gt_inds,
                            gt_segmentation=gt_segmentation,
                            gt_instance=gt_instance,
                            gt_img_is_valid=gt_occ_img_is_valid,
                        )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)
        

        # Forward Plan Head
        if self.with_planning_head:
            #* 输入: BEV, 预测信息: outs_motion(Q_A, Q_ctx), command: 规划指令，数据集里面有
            outs_planning = self.planning_head.forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt 只用于评估
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt 只用于评估
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function
        """
        # 检查输入img和img_metas是否为list
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        
        #* 处理img和img_metas
        img = [img] if img is None else img
        # 场景切换时，前一帧的BEV特征置0
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 视频测试模式下，不需要前一帧的BEV特征
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        #* 从canbus中获取两帧之间的位置和角度变化量
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # 处理第一帧变化量为0
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # 处理后续帧依次记录为和前一帧的差值
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        #* 取第一帧的数据作为输入
        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        #* BEV和track_former
        #* 输入：img, 上一帧的BEV特征
        #* 过Trackformer：和Train一样过encoder得到BEV(encoder参数被冻结，no_grad)
        #* 过decoder得到track的Query
        #  输出：["bev_embed", "bev_pos","track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
        #            "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        #* track_former输出的BEV和Q_A再过一个解码器获得真实的track结果
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model 
        # # 如果是tiny模型，需要上采样bev       
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        # 从输出中获取第一帧的BEV
        bev_embed = result_track[0]["bev_embed"]

        #* map_former
        #* 输入：BEV特征
        #* 输出：Map Former分割的bbox和类别信息: `pts_bbox`: 预测的bbox, `ret_iou`: IOU结果, `args_tuple`: 网络输出Q之类的信息
        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        #* motion_former
        #* 输入：BEV特征，track的输出，Map的输出
        #* 输出：Motionformer的预测轨迹, 网络输出相关(传到下游网络中的Qctx，Q_A等)
        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']
        
        #* occ_former 
        #* 输入：BEV特征, Motionformer的输出Qctx, Q_A，Occ_no_query: Q_A是否为空, 地面真实信息
        #* #* 输出：'seg_gt', 'ins_seg_gt': 地面实况分割和实例分割；'pred_ins_logits': Occ原始输出
        #* 'pred_ins_sigmoid': Occ预测的概率；'seg_out', 'ins_seg_out': 预测的Occ分割和实例分割
        outs_occ = dict()
        if self.with_occ_head:
            # 如果没有Q_A则直接用0填充输出，一般不用，用的是地面真实信息
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        #* planner
        #* 输入：BEV特征，Motionformer的输出Qctx, Q_A，Occ，gt的规划信息，规划指令
        #* 输出：规划最终轨迹
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        # 从输出中的第一帧pop掉一些不需要的信息

        #* trackformer部分：{"bev_embed", "bev_pos","track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
        #*            "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"}
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        #* mapformer部分：{'ret_iou', 'pts_bbox', 'args_tuple'}
        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        
        #* motionformer部分：traj_results：最终轨迹
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        
        #* Occformer部分：{'seg_gt', 'ins_seg_gt', 'pred_ins_logits','pred_ins_sigmoid', 'seg_out', 'ins_seg_out'}
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        #* planner部分：{'planning_gt', 'result_planning'}
        
        #* 将各个模块的输出整合到result中
        #* result为一个list，每个元素是一帧的输出的字典，用'token'字段标记
        #* 包括trackformer, mapformer, motionformer, occformer, planner的输出
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
