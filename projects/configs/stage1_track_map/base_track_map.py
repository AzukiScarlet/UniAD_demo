_base_ = ["../_base_/datasets/nus-3d.py",
        "../_base_/default_runtime.py"]

# Update-2023-06-12: 
# [Enhance] Update some freezing args of UniAD 
# [Bugfix] Reproduce the from-scratch results of stage1
# 1. Remove loss_past_traj in stage1 training
# 2. Unfreeze neck and BN
# --> Reproduced tracking result: AMOTA 0.393

#* 训练前的相关参数设置

# 解冻“neck”层和 Batch Normalization 层，以便能够再现第一阶段从头开始训练的结果
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# 如果点云范围发生变化，模型也应相应地更改其点云范围
# [x_min, y_min, z_min, x_max, y_max, z_max]
#* grid_size 个 voxel 组成point cloud range
#* eg. grid_size = [512, 512, 1] 个 voxel_size = [0.2, 0.2, 8]
#* 构成point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]   # 定义体素的大小(分辨率)
patch_size = [102.4, 102.4]  # 定义图像的大小
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)  # 图像标准化配置 (均值， 标准差， 是否转换为RGB格式)
#* nuscens 场景中将agent分为10类
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

# 输入模态：仅使用相机和外部数据
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
queue_length = 1  # 每个序列包含 `queue_length` 帧。

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
planning_evaluation_strategy = "uniad"  # 采用uniad的评价策略

###* Occ 范围和步长 ### 
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Other settings
train_gt_iou_threshold=0.3    # 训练时的IoU阈值

#* 模型设置################################################################
model = dict(
    type="UniAD",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,       # 使用网格掩码
    video_test_mode=True,     # 视频测试模式
    num_query=900,            # 查询数量
    num_classes=10,           # 类别数量, nuscenes中有10个类别
    pc_range=point_cloud_range,

    # 图像主干网络配置
    img_backbone=dict(
        type="ResNet",   #* 残差网络
        depth=101,
        num_stages=4,    # 网络阶段数
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN2d", requires_grad=False),   # Batch Normalization配置
        norm_eval=True,
        style="caffe",            # caffe框架
        # 可变性卷积网络
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),      # 各阶段是否使用DCN
    ),
    # 图像颈部网络配置
    img_neck=dict(
        type="FPN", # 特征金字塔网络
        in_channels=[512, 1024, 2048], # 输入通道数, 与img_backbone的out_indices对应
        out_channels=_dim_, # 输出通道数
        start_level=0,
        add_extra_convs="on_output",  # 在输出上添加额外的卷积
        num_outs=4,       # 输出数量
        relu_before_extra_convs=True,  # 在额外的卷积之前使用ReLU
    ),

    # 冻结相关网络的配置
    freeze_img_backbone=True, # 冻结图像主干网络
    freeze_img_neck=False,    # 不冻结图像颈部网络
    freeze_bn=False,          # 不冻结Batch Normalization
    
    # 得分阈值
    score_thresh=0.4,
    filter_score_thresh=0.35,
    
    # 查询交互模块
    qim_args=dict(    
        qim_type="QIMBase",  # QIM 
        merger_dropout=0,    # 合并器的dropout
        update_query_pos=True, # 更新查询位置
        fp_ratio=0.3,        
        random_drop=0.1,     
    ),  # hyper-param for query dropping mentioned in MOTR
    
    # encoder的记忆模块
    mem_args=dict(
        memory_bank_score_thresh=0.0,   
        memory_bank_len=4,
    ),

    #* 轨迹跟踪损失函数配置
    loss_cfg=dict(
        type="ClipMatcher",        
        num_classes=10,         # nuscenes中有10个类别
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],   # 类别权重
        assigner=dict(
            type="HungarianAssigner3DTrack",                  # 匈牙利分配器
            cls_cost=dict(type="FocalLossCost", weight=2.0),  # 分类损失
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),  # 回归成本
            pc_range=point_cloud_range,                       # 点云范围
        ),
        # 轨迹损失：使用focalloss, 二分类交叉熵损失
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        # 边界框损失：使用L1损失
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        # 过去轨迹损失，权重为0，不使用
        loss_past_traj_weight=0.0,
    ),  # loss cfg for tracking

    #* 轨迹跟踪头配置：TrackFormer
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,  # 同步类平均因子
        with_box_refine=True,      # 使用边界框细化
        as_two_stage=False,        # 不进行两阶段检测
        past_steps=past_steps,
        fut_steps=fut_steps,

        #* 感知的transformer配置
        #* encoder生成BEV；decoder生成track特征
        transformer=dict(
            type="PerceptionTransformer",  
            rotate_prev_bev=True,       # 预先旋转BEV
            use_shift=True,
            use_can_bus=True,           # 调用can_bus总线数据
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
                        # 时序自注意力, 时间维度的自注意力agent-self
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
                    feedforward_channels=_ffn_dim_,      # 前馈网络通道数
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
            type="NMSFreeCoder",    # 柠檬树：非极大值抑制自由编码器，这里不使用NMS
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
        with_box_refine=True,      # 使用边界框细化
        # 用于全景分割的可变形transformer
        transformer=dict(
            type='SegDeformableTransformer',
            # encoder: 6层，每一层：多尺度可变形注意力-norm-前馈网络-norm
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    # 多尺度可变形注意力
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
            # 分配器: 匈牙利分配器
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

    #* 模型训练和测试配置
    train_cfg=dict(
        # 点云处理参数
        pts=dict(
            # grid_size 个 voxel 组成 point cloud range
            grid_size=[512, 512, 1],   #2D网格大小 
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,       
            # 匈牙利分配器
            assigner=dict(
                type="HungarianAssigner3D",
                # 分类损失，回归损失，IoU损失
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

#* 数据设置################################################################
dataset_type = "NuScenesE2EDataset"
data_root = "data/nuscenes/"
info_root = "data/infos/"
file_client_args = dict(backend="disk")
# 训练集、验证集和测试集的注释信息
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl" 
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test=info_root + f"nuscenes_infos_temporal_val.pkl"

#* 训练和测试数据pipeline设置，模型的数据流，架构全在这里###############################################
#* 数据读取， 数据预处理， 创建模型， 评估模型结果，模型调仓
train_pipeline = [
    # 数据读取，从data_root中读取图像，转化为float32格式
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    # 图像增强
    dict(type="PhotoMetricDistortionMultiViewImage"),
    # 加载3D注释
    dict(
        type="LoadAnnotations3D_E2E",
        # 3D边界和标签
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,

        with_future_anns=True,  # occ_flow gt 用于Occ flow的地面真实值
        # 实例索引，从1开始
        with_ins_inds_3d=True,  # ins_inds  实例索引，从1开始
        ins_inds_add_1=True,    # ins_inds start from 1
    ),

    # 生成OccFlow标签(只生成车辆的OccFlow标签)
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 
    # 过滤无效的样本
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),  # 过滤范围外的样本
    dict(type="ObjectNameFilterTrack", classes=class_names),                   # 过滤不在类别中的样本
    # 图像标准化，填充，故事化为3D数据输出
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),                      
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    
    #* 从处理后的数据中收集指定的key，并将它们打包成一个字典
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",               # 真实3D边界框
            "gt_labels_3d",               # 真实3D标签
            "gt_inds",                    # 真实3D索引
            "img",                        # 图像
            "timestamp",                  # 时间戳
            "l2g_r_mat",                  # 从局部坐标系到全局坐标系的旋转矩阵
            "l2g_t",                      # 从局部坐标系到全局坐标系的平移矩阵
            "gt_fut_traj",                # 真实未来轨迹
            "gt_fut_traj_mask",           # 真实未来轨迹掩码
            "gt_past_traj",               # 真实过去轨迹
            "gt_past_traj_mask",          # 真实过去轨迹掩码
            "gt_sdc_bbox",                # 真实自车边界框
            "gt_sdc_label",               # 真实自车标签            
            "gt_sdc_fut_traj",            # 真实自车未来轨迹
            "gt_sdc_fut_traj_mask",       # 真实自车未来轨迹掩码
            "gt_lane_labels",             # 真实车道标签
            "gt_lane_bboxes",             # 真实车道边界框
            "gt_lane_masks",              # 真实车道掩码
            # Occ gt
            "gt_segmentation",            # 语义分割结果
            "gt_instance",                # 实例分割
            "gt_centerness",              # 中心度
            "gt_offset",                  # 偏移
            "gt_flow",                    # 正向流信息
            "gt_backward_flow",           # 反向流信息
            "gt_occ_has_invalid_frame",   # Occ是否有无效帧
            "gt_occ_img_is_valid",        # Occ图像是否有效
            # gt future bbox for plan	
            "gt_future_boxes",	          # 未来边界框
            "gt_future_labels",	          # 未来标签
            # planning	
            "sdc_planning",	              # 自车规划信息
            "sdc_planning_mask",	      # 自车规划掩码
            "command",                    # 控制命令
        ],
    ),
]
# 测试数据pipeline设置
test_pipeline = [
    # 数据读取 + 图像标准化 + 填充 + 故事化为3D数据输出
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=data_root),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # 测试时不需要3D边界框和标签，只需要未来注释
    dict(type='LoadAnnotations3D_E2E',  
        with_bbox_3d=False,
        with_label_3d=False, 
        with_attr_label=False,

        with_future_anns=True,
        with_ins_inds_3d=False,
        ins_inds_add_1=True, # ins_inds start from 1
        ),
    # 生成OccFlow标签(只生成车辆的OccFlow标签)
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),
    # 多尺度图像增强
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),  # 缩放尺寸
        pts_scale_ratio=1,      # 点云缩放比例
        flip=False,             # 不翻转
        transforms=[
            # 数据格式化以适合3D任务
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            # 从处理后的数据中收集指定的key，并将它们打包成一个字典
            # 测试时不再需要真实的3D边界框和标签，真实未来和过去轨迹等等
            #* 此处为第一阶段训练，因此Occ和planning的真实信息需要
            dict(
                type="CustomCollect3D", keys=[
                                            "img",                         # 图像
                                            "timestamp",                   # 时间戳
                                            "l2g_r_mat",                   # 从局部坐标系到全局坐标系的旋转矩阵
                                            "l2g_t",                       # 从局部坐标系到全局坐标系的平移矩阵
                                            "gt_lane_labels",              # 真实车道标签
                                            "gt_lane_bboxes",              # 真实车道边界框
                                            "gt_lane_masks",               # 真实车道掩码
                                            # Occ gt
                                            "gt_segmentation",             # 语义分割结果
                                            "gt_instance",                 # 实例分割
                                            "gt_centerness",               # 中心度
                                            "gt_offset",                   # 偏移
                                            "gt_flow",                     # 正向流信息
                                            "gt_backward_flow",            # 反向流信息
                                            "gt_occ_has_invalid_frame",    # Occ是否有无效帧
                                            "gt_occ_img_is_valid",         # Occ图像是否有效 
                                            # planning	
                                            "sdc_planning",	               # 自车规划信息
                                            "sdc_planning_mask",	       # 自车规划掩码
                                            "command",                     # 控制命令
                                        ]
            ),
        ],
    ),
]

#* 训练数据配置
data = dict(
    samples_per_gpu=1,            # 每个GPU的batch_size
    workers_per_gpu=8,            # 每个GPU分配数据加载的线程数
    # 训练数据配置:train_pipeline
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,      #* train_pipeline流程
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
        use_nonlinear_optimizer=use_nonlinear_optimizer,    #* 使用非线性优化器

        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,

        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),

    # 验证数据配置:test_pipeline
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,                #* test_pipeline流程
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,      #* 使用非线性优化器
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track', 'map'],         #* 评价模块：检测，跟踪，建图

        # Occ流参数
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    # 测试数据配置:test_pipeline，第一阶段对感知训练，无Occ和planning信息
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,              #* test_pipeline流程
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,      #* 使用非线性优化器
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'map', 'track'],     #* 评价模块：检测，建图，跟踪
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
# 优化器设置
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

# 
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) 
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
#* 训练6个epoch
total_epochs = 4
#* 每6次迭代，进入一次评估
evaluation = dict(
    interval=6,
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)
#* runner设置
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# 记录训练日志，Tensorboard日志
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# 检查点设置,每训练一个epoch保存一次
checkpoint_config = dict(interval=1)

# *预训练模型
# load_from = "ckpts/bevformer_r101_dcn_24ep.pth"
# 接着训练
load_from = "projects/work_dirs/stage1_track_map/base_track_map/latest.pth"

#* 运行配置
find_unused_parameters = True