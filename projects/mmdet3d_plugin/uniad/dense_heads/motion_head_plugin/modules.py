#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    trajectory_coordinate_transform
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MotionTransformerDecoder(BaseModule):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = IntentionInteraction()
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers)])
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction() for i in range(self.num_layers)])
        self.bev_interaction_layers = nn.ModuleList(
            [build_transformer_layer(transformerlayers) for i in range(self.num_layers)])

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*4, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_trajs=None,
                traj_reg_branches=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                **kwargs):
        """Forward function for `MotionTransformerDecoder`.
        Args:
            agent_query (B, A, D)
            map_query (B, M, D) 
            map_query_pos (B, G, D)
            static_intention_embed (B, A, P, D)
            offset_query_embed (B, A, P, D)
            global_intention_embed (B, A, P, D)
            learnable_intention_embed (B, A, P, D)
            det_query_pos (B, A, D)
        Returns:
            None
        """
        #* 中间状态和参考轨迹
        intermediate = []
        intermediate_reference_trajs = []

        # 扩展track_query和track_query_pos的维度，使其与agent_level_embedding的维度相同
        B, _, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # static intention embedding, which is imutable throughout all layers
        #* 静态意图嵌入： 将agent过MLP后与scene和learnable嵌入相加得到所有层通用的嵌入
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        # 参考轨迹
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()
        #* 初始化一个Q_ctx
        query_embed = torch.zeros_like(static_intention_embed)
        
        # 遍历每一层
        for lid in range(self.num_layers):
            # fuse static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding
            #* 动态嵌入：上一层的输出，使用 anchor_embedding 过MLP进行初始化
            #* Q_pos = MLP(PE(I^s)) + MLP(PE(I^a)) + MLP(PE(x_0)) + MLP(PE(X^{l - 1}))
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
            #* 融合静态和动态嵌入作为查询意图嵌入：静态意图和动态意过MLP -> Q_pos
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)
            
            #* 意图嵌入和查询嵌入融合： Q_pos和Q_ctx过MLP : 每一层的真正输入
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))
            
            #* 上面通过Q_pos 和 Q_ctx 过 MLP 得到了本层的输入Q

            #* Q_ctx和Q_A交互 -> Q_a
            # Q_a = MHCA(MHSA(Q_ctx), Q_A)
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)
            
            #* Q_ctx和Q_M交互 -> Q_m
            # Q_m = MHCA(MHSA(Q_ctx), Q_M)
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)
            
            # interaction between agents and bev, ie. interaction between agents and goals
            # implemented with deformable transformer
            #* Q和BEV和目标点交互 -> Q_g
            # 可变形注意力Q_g = DeformAttn(Q_ctx, x^{l-1}, B)
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs)
            
            # 融合Q_a, Q_m, Q_pos, 扩展后的原始Q_ctx -> Q_ctx
            # Q_ctx = MLP([Q_a, Q_m, Q_g, Q_A])
            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            if traj_reg_branches is not None:
                # 更新参考轨迹：用Q_ctx生成
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)
                
                # we predict speed of trajectory and use cumsum trick to get the trajectory
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

                # update embedding, which is used in the next layer
                # only update the embedding of the last step, i.e. the goal
                # *4个MLP (场景级锚点，agent级锚点，agent真实坐标， 预测目标) -> Q_pos
                # Q_pos = MLP(PE(I^s)) + MLP(PE(I^a)) + MLP(PE(x_0)) + MLP(PE(X^{l - 1}))
                ep_offset_embed = reference_trajs.detach()
                ep_ego_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=True, with_rotation_transform=False).squeeze(2).detach()
                ep_agent_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=False, with_rotation_transform=True).squeeze(2).detach()

                #* MLP(PE(I^a))和
                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range))) 
                #* MLP(PE(x_0))和MLP(PE(X^{l - 1}))
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
                #* MLP(PE(I^s))
                scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                    norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        #* 输出每一层的Q和参考轨迹
        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)


class TrackAgentInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        query: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B*A, -1, -1)
        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class MapInteraction(BaseModule):
    """
    Modeling the interaction between the agent and the map
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        x: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        mem = key.expand(B*A, -1, -1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class IntentionInteraction(BaseModule):
    """
    Modeling the interaction between anchors
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out
