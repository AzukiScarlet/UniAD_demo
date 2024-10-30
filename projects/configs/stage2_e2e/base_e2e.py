_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

# Update-2023-06-12: 
# [Enhance] Update some freezing args of UniAD 
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]     #* 车辆类别 class_names 中的索引
group_id_list = [[0,1,2,3,4], [6,7], [8], [5,9]] # 车辆类别的分组：机动车、非机动车、行人、障碍物

# 输入模态：使用相机和外部信息
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

_dim_ = 256  # 基础维度
_pos_dim_ = _dim_ // 2  # 位置维度 
_ffn_dim_ = _dim_ * 2   # 前馈网络维度
_num_levels_ = 4        # 多头注意力的数量
bev_h_ = 200            
bev_w_ = 200
_feed_dim_ = _ffn_dim_  # 前馈网络输入维度
_dim_half_ = _pos_dim_  # 位置维度的一半
canvas_size = (bev_h_, bev_w_)  # 鸟瞰图尺寸


# NOTE: 您可以将队列长度从 5 更改为 3 来节省 GPU 内存，但可能会影响性能。
queue_length = 3  # each sequence contains `queue_length` frames.

###* 轨迹预测参数 ###
predict_steps = 12      #* 预测未来步数
predict_modes = 6       #* 预测6种模式
fut_steps = 4           # 未来步数
past_steps = 4          # 过去步数
use_nonlinear_optimizer = True         #* 训练时使用非线性优化器


##* Occ流参数	
occ_n_future = 4	     # 交互流的未来步数      
occ_n_future_plan = 6    # 规划的未来步数
occ_n_future_max = max([occ_n_future, occ_n_future_plan])	

###* 规划 ###
planning_steps = 6         # 规划步数
use_col_optim = True       #* 使用碰撞优化(牛顿法再优化)
# there exists multiple interpretations of the planning metric, where it differs between uniad and stp3/vad
# uniad: computed at a particular time (e.g., L2 distance between the predicted and ground truth future trajectory at time 3.0s)
# stp3: computed as the average up to a particular time (e.g., average L2 distance between the predicted and ground truth future trajectory up to 3.0s)
planning_evaluation_strategy = "uniad"  # uniad or stp3

###* Occ 范围和步长 ### 
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Other settings
train_gt_iou_threshold=0.3 # 训练时的IoU阈值

#* 模型设置################################################################
model = dict(
    type="UniAD",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,       # 使用网格掩码
    video_test_mode=True,     # 视频测试模式
    num_query=900,            # 查询数量
    num_classes=10,           # 类别数量, nuscenes中有10个类别
    vehicle_id_list=vehicle_id_list,   #* 车辆类别索引
    pc_range=point_cloud_range,
    
    # 图像主干网络配置
    img_backbone=dict(
        type="ResNet",         #* 残差网络
        depth=101,
        num_stages=4,          # 网络阶段数
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        # 可变性卷积网络
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),

    # 图像颈部网络配置
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),

    #* 冻结相关网络的配置，第二阶段BEV相关的encoder被冻结，参数不再更新
    freeze_img_backbone=True,
    freeze_img_neck=True,      #* 第二阶段冻结图像颈部网络
    freeze_bn=True,            #* 冻结BN层
    freeze_bev_encoder=True,   #****** 冻结BEV编码器
    
    # 得分阈值
    score_thresh=0.4,
    filter_score_thresh=0.35,

    # 查询交互模块
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    
    # encoder的记忆模块
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    #* 轨迹跟踪损失函数配置
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
    ),  # loss cfg for tracking
    
    #* 轨迹跟踪头配置：TrackFormer
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,    # 同步类别平均因子
        with_box_refine=True,        # 使用盒子细化
        as_two_stage=False,          #* 不进行两阶段检测
        past_steps=past_steps,
        fut_steps=fut_steps,

        #* 感知的transformer配置
        #* encoder生成BEV；decoder生成track特征
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,         # 预先旋转BEV
            use_shift=True,
            use_can_bus=True,             # 调用can_bus总线数据
            embed_dims=_dim_,

            #* encoder部分
            # 6层，每一层：TemporalSelfAttention-norm-SpatialCrossAttention-norm-前馈网络-norm
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    # 注意力配置
                    #* attn_cfgs中的attn_cfg[i]会被自动映射到operation_order中的各个attn操作中
                    # eg: operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
                    #    attn_cfgs[0] -> self_attn, attn_cfgs[1] -> cross_attn
                    # 最终执行为 TemporalSelfAttention -> norm -> SpatialCrossAttention -> norm -> FeedForward -> norm
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        # 空间交叉注意力, 空间维度的交叉注意力agent-agent，内部使用可变形注意力
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            # 内部使用可变形注意力
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    ##########* 操作顺序##########
                    # 每个网络不同配置的核心
                    # 自注意力-norm-交叉注意力-norm-前馈网络-norm
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            #* decoder部分
            # 6层，每一层：MultiheadAttention-norm-CustomMSDeformableAttention-norm-前馈网络-norm
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    # 注意力配置
                    attn_cfgs=[
                        # 多头注意力，8个头
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        # 可变形注意力
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    ##########* 操作顺序##########
                    # 自注意力-norm-交叉注意力-norm-前馈网络-norm
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        # 边界框编码器
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        #* 可学习的位置编码
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        # 分类损失，边界框损失，IoU损失
        # IoU：交并比损失用于评估两个边界框之间的重叠程度
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),

    #* 地图感知模块：MapFormer
    seg_head=dict(
        type='PansegformerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        num_query=300,              # 查询数量
        num_classes=4,              # 总类别数
        num_things_classes=3,       # 物体类别数
        num_stuff_classes=1,        # 非物体类别数
        in_channels=2048,           # 输入通道数
        sync_cls_avg_factor=True,
        as_two_stage=False,         #* 不进行两阶段检测
        with_box_refine=True,       # 使用边界框细化
        # 用于全景分割的可变形transformer
        transformer=dict(
            type='SegDeformableTransformer',
            # encoder: 6层，每一层：多尺度可变形注意力-norm-前馈网络-norm
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                         ),
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            # decoder: 6层：多头自注意力-norm-多尺度可变形注意力-norm-前馈网络-norm
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    # 注意力配置
                    attn_cfgs=[
                        # 多头自注意力，8个头
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        # 多尺度可变形注意力
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                ),
            ),
        ),
        # 位置编码
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        # 分类损失，边界框损失，IoU损失，掩模损失
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        # thing和stuff的transformer头
        thing_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0),
                ),
            # 采样器: 伪采样：仅保留接口不进行采样，直接使用所有样本
            sampler =dict(type='PseudoSampler'),
            sampler_with_mask =dict(type='PseudoSampler_segformer'),
        ),
    ),
    #*####################################################

    #* OccFormer
    occ_head=dict(
        type='OccHead',

        grid_conf=occflow_grid_conf,
        ignore_index=255,

        bev_proj_dim=256,
        bev_proj_nlayers=4,

        # Transformer
        attn_mask_thresh=0.3,
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=5,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,  # change to 512
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        # Query
        query_dim=256,
        query_mlp_layers=3,

        aux_loss_weight=1.,
        loss_mask=dict(
            type='FieryBinarySegmentationLoss',
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0,
            ignore_index=255,
        ),
        loss_dice=dict(
            type='DiceLossWithMasks',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            ignore_index=255,
            loss_weight=1.0),

        
        pan_eval=True,
        test_seg_thresh=0.1,
        test_with_track_score=True,
    ),
    motion_head=dict(
        type='MotionHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=10,
        predict_steps=predict_steps,
        predict_modes=predict_modes,
        embed_dims=_dim_,
        loss_traj=dict(type='TrajLoss', 
            use_variance=True, 
            cls_loss_weight=0.5, 	
            nll_loss_weight=0.5, 	
            loss_weight_minade=0., 	
            loss_weight_minfde=0.25),
        num_cls_fcs=3,
        pc_range=point_cloud_range,
        group_id_list=group_id_list,
        num_anchor=6,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        anchor_info_path='data/others/motion_anchor_infos_mode6.pkl',
        transformerlayers=dict(
            type='MotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',
                        num_steps=predict_steps,
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1),
                ],

                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        ),
    ),
    planning_head=dict(
        type='PlanningHeadSingleMode',
        embed_dims=256,
        planning_steps=planning_steps,
        loss_planning=dict(type='PlanningLoss'),
        loss_collision=[dict(type='CollisionLoss', delta=0.0, weight=2.5),
                        dict(type='CollisionLoss', delta=0.5, weight=1.0),
                        dict(type='CollisionLoss', delta=1.0, weight=0.25)],
        use_col_optim=use_col_optim,
        planning_eval=True,
        with_adapter=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)
dataset_type = "NuScenesE2EDataset"
data_root = "data/nuscenes/"
info_root = "data/infos/"
file_client_args = dict(backend="disk")
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test=info_root + f"nuscenes_infos_temporal_val.pkl"


train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,

        with_future_anns=True,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds 
        ins_inds_add_1=True,    # ins_inds start from 1
    ),

    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 

    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "gt_fut_traj",
            "gt_fut_traj_mask",
            "gt_past_traj",
            "gt_past_traj_mask",
            "gt_sdc_bbox",
            "gt_sdc_label",
            "gt_sdc_fut_traj",
            "gt_sdc_fut_traj_mask",
            "gt_lane_labels",
            "gt_lane_bboxes",
            "gt_lane_masks",
             # Occ gt
            "gt_segmentation",
            "gt_instance", 
            "gt_centerness", 
            "gt_offset", 
            "gt_flow",
            "gt_backward_flow",
            "gt_occ_has_invalid_frame",	
            "gt_occ_img_is_valid",	
            # gt future bbox for plan	
            "gt_future_boxes",	
            "gt_future_labels",	
            # planning	
            "sdc_planning",	
            "sdc_planning_mask",	
            "command",
        ],
    ),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=data_root),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,
         with_label_3d=False, 
         with_attr_label=False,

         with_future_anns=True,
         with_ins_inds_3d=False,
         ins_inds_add_1=True, # ins_inds start from 1
         ),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                       filter_invisible=False),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", keys=[
                                            "img",
                                            "timestamp",
                                            "l2g_r_mat",
                                            "l2g_t",
                                            "gt_lane_labels",
                                            "gt_lane_bboxes",
                                            "gt_lane_masks",
                                            "gt_segmentation",
                                            "gt_instance", 
                                            "gt_centerness", 
                                            "gt_offset", 
                                            "gt_flow",
                                            "gt_backward_flow",
                                            "gt_occ_has_invalid_frame",	
                                            "gt_occ_img_is_valid",	
                                            # planning	
                                            "sdc_planning",	
                                            "sdc_planning_mask",	
                                            "command",
                                        ]
            ),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,

        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
        
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'map', 'track','motion'],
        

        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'map', 'track','motion'],
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 20
evaluation = dict(
    interval=4,
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1)
# load_from = "ckpts/uniad_base_track_map.pth"
# 接着训练
load_from = "experiments/2024_10_26_UniAD_origin/stage1_track_map/base_track_map/latest.pth"

find_unused_parameters = True