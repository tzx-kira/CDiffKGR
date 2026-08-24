import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import math
from torch.utils.checkpoint import checkpoint
from modules.pcgrad import PCGrad


def log_to_file(message, filename="/home/homes/tzx/ssd/kgrec/KGIN/EditKG-main/EditKG-main/result/last-fm消融实验7-27(1).txt"):
    with open(filename, "a") as f:
        f.write(message + "\n")
    print(message)


# ==================== 改进后的双路三元组去噪器 (Enhanced DualTriDenoise) ====================

class RelationAwareProjection(nn.Module):
    """
    关系感知投影模块：为每个关系类型学习独立的投影矩阵。
    不同关系在不同语义空间中应使用不同的度量方式。
    参考: TransR (Lin et al., 2015) 和 RAAKGC (Yuan et al., 2025, AAAI) 的关系投影思想。
    使用对角矩阵以降低参数量，避免过拟合。
    """

    def __init__(self, dim, n_relations):
        super(RelationAwareProjection, self).__init__()
        self.dim = dim
        self.n_relations = n_relations
        # 为每个关系学习一个可学习的对角投影权重和偏置
        self.relation_proj = nn.Parameter(torch.randn(n_relations, dim) * 0.01)
        self.relation_bias = nn.Parameter(torch.zeros(n_relations, dim))

    def forward(self, entity_emb, relation_ids):
        """
        对实体嵌入进行关系特定的投影变换。
        Args:
            entity_emb: [batch_size, dim]
            relation_ids: [batch_size]
        Returns:
            projected_emb: [batch_size, dim]
        """
        proj_weights = self.relation_proj[relation_ids]  # [batch_size, dim]
        proj_bias = self.relation_bias[relation_ids]  # [batch_size, dim]
        projected = entity_emb * proj_weights + proj_bias
        return projected


class MultiHeadTripletAttention(nn.Module):
    """
    多头三元组注意力模块：使用多头注意力机制捕捉h, r, t之间的复杂交互。
    修正版本：将h和t视为两个token，r作为Query。
    参考: HGT (Hu et al., 2020) 的类型特定多头注意力 和
          SRGAN (Liu et al., 2025, Knowledge-Based Systems) 的语义关系感知图注意力网络思想。

    设计思路：使用关系r作为Query，头实体h和尾实体t作为Key和Value，
    计算"关系r与头尾实体的匹配程度"，捕捉三元组的语义一致性。
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadTripletAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 将h, r, t映射到Q, K, V空间
        self.q_proj = nn.Linear(dim, dim)   # 用于r
        self.k_proj = nn.Linear(dim, dim)   # 用于h和t
        self.v_proj = nn.Linear(dim, dim)   # 用于h和t
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, h, r, t):
        """
        Args:
            h, r, t: [batch_size, dim]
        Returns:
            attended_features: [batch_size, dim]
        """
        batch_size = h.size(0)
        device = h.device

        # 1. 线性投影
        # Q: 关系r
        q = self.q_proj(r)  # [B, D]
        # K, V: 将h和t拼接成序列长度为2的矩阵
        # 形状: [B, 2, D]
        kv = torch.stack([h, t], dim=1)   # [B, 2, D]
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # 2. 多头分割
        # Q: [B, H, 1, HD]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, HD]
        # K, V: [B, 2, H, HD] -> [B, H, 2, HD]
        k = k.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. 计算注意力分数
        # scores: [B, H, 1, 2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 4. 加权聚合
        # out: [B, H, 1, HD] -> [B, D]
        out = torch.matmul(attn, v)  # [B, H, 1, HD]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)
        out = self.out_proj(out)

        # 5. 残差连接 + LayerNorm
        out = self.layer_norm(out + r)   # 使用关系r作为残差
        return out


class AdaptiveFusionGate(nn.Module):
    """
    自适应融合门控：动态调整注意力和三元组真实性的权重。
    参考: KDGNN (Zhang et al., 2024, The Computer Journal) 的双通道平衡机制 和
          GraphFusionSBR (2025) 的选择性融合机制。

    设计思路：根据关系嵌入和当前分数动态生成融合权重，
    使模型能够根据关系类型自适应地选择更可靠的信号。
    """

    def __init__(self, dim):
        super(AdaptiveFusionGate, self).__init__()
        # 基于关系类型和当前特征动态生成融合权重
        # 输入: atten_score(1) + auth_score(1) + relation_emb(dim) = dim+2
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim + 2, dim),   # 修正：输入维度为 dim+2
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, atten_score, auth_score, relation_emb):
        """
        Args:
            atten_score: [batch_size] 注意力分数
            auth_score: [batch_size] 三元组真实性分数
            relation_emb: [batch_size, dim] 关系嵌入
        Returns:
            gate_weights: [batch_size, 2] 融合权重
            fused_score: [batch_size] 最终融合分数
        """
        batch_size = atten_score.size(0)
        # 拼接输入特征：两个分数 + 关系嵌入 (注意顺序)
        combined = torch.cat([
            atten_score.unsqueeze(-1),
            auth_score.unsqueeze(-1),
            relation_emb
        ], dim=-1)  # [batch_size, dim+2]

        gate_weights = self.gate_mlp(combined)  # [batch_size, 2]
        atten_w = gate_weights[:, 0]
        auth_w = gate_weights[:, 1]

        fused_score = atten_w * atten_score + auth_w * auth_score
        return gate_weights, fused_score


class EnhancedTripletDenoiser(nn.Module):
    """
    改进后的生成KG去噪器。

    核心改进:
    1. 关系感知投影：为每个关系学习独立的语义投影空间，使真实性度量更准确
    2. 多头三元组注意力：捕捉h, r, t之间的复杂交互模式，提取深层语义特征
    3. 深度注意力评分网络：多层非线性变换提取注意力特征
    4. 关系感知真实性评分：在关系特定空间中度量TransE距离
    5. 自适应融合门控：动态调整两个维度的权重，实现智能融合

    参考论文:
    - SRGAN (Liu et al., 2025, Knowledge-Based Systems): 语义关系感知图注意力
    - KDGNN (Zhang et al., 2024, The Computer Journal): 双通道去噪与平衡
    - RAAKGC (Yuan et al., 2025, AAAI): 关系感知锚点增强
    - HGT (Hu et al., 2020): 类型特定多头注意力
    """

    def __init__(self, dim, n_relations, num_heads=4, dropout=0.1):
        super(EnhancedTripletDenoiser, self).__init__()
        self.dim = dim
        self.n_relations = n_relations
        self.num_heads = num_heads

        # ===== 1. 关系感知投影模块 =====
        # 为头实体和尾实体分别学习关系特定的投影
        self.relation_proj_h = RelationAwareProjection(dim, n_relations)
        self.relation_proj_t = RelationAwareProjection(dim, n_relations)

        # ===== 2. 多头三元组注意力模块 =====
        # 捕捉h, r, t之间的语义交互
        self.triplet_attention = MultiHeadTripletAttention(dim, num_heads=num_heads, dropout=dropout)

        # ===== 3. 深度注意力评分网络 =====
        # 输入: h_proj, r, t_proj, attended_features 拼接 -> dim * 4
        # 满足约束: MLP层数>=3, 隐藏层维度<=512
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # ===== 4. 关系感知真实性评分 =====
        # 在投影后的空间中计算TransE距离，使用可学习的缩放因子
        self.auth_scale = nn.Parameter(torch.ones(1))

        # ===== 5. 自适应融合门控 =====
        self.fusion_gate = AdaptiveFusionGate(dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h, r, t, relation_ids=None):
        """
        Args:
            h: [batch_size, dim] 头实体嵌入
            r: [batch_size, dim] 关系嵌入
            t: [batch_size, dim] 尾实体嵌入
            relation_ids: [batch_size] 关系类型ID (用于关系感知投影)
        Returns:
            final_weight: [batch_size] 最终去噪权重
            atten_score: [batch_size] 注意力分数
            auth_score: [batch_size] 三元组真实性分数
            gate_weights: [batch_size, 2] 门控权重 (用于日志记录)
        """
        batch_size = h.size(0)

        # ===== 步骤1: 关系感知投影 =====
        # 将头尾实体投影到关系特定的语义空间
        if relation_ids is not None:
            h_proj = self.relation_proj_h(h, relation_ids)
            t_proj = self.relation_proj_t(t, relation_ids)
        else:
            h_proj = h
            t_proj = t

        # ===== 步骤2: 多头三元组注意力 =====
        # 捕捉h, r, t之间的语义交互
        attended = self.triplet_attention(h_proj, r, t_proj)  # [batch_size, dim]

        # ===== 步骤3: 注意力评分 =====
        # 拼接投影后的实体、关系、注意力特征
        cat_emb = torch.cat([h_proj, r, t_proj, attended], dim=-1)  # [batch_size, dim * 4]
        atten_logits = self.attention_mlp(cat_emb).squeeze(-1)  # [batch_size]
        atten_score = torch.sigmoid(atten_logits)

        # ===== 步骤4: 关系感知真实性评分 =====
        # 在投影空间中计算TransE距离: ||h_proj + r - t_proj||
        dist = torch.norm(h_proj + r - t_proj, p=2, dim=-1)
        # 使用可学习的缩放因子，使模型自适应调整真实性敏感度
        auth_score = torch.exp(-self.auth_scale * dist)

        # ===== 步骤5: 自适应融合 =====
        # 动态调整attention score和authenticity score的权重
        gate_weights, final_weight = self.fusion_gate(atten_score, auth_score, r)

        return final_weight, atten_score, auth_score, gate_weights


class Aggregator(nn.Module):
    """用户-物品交互聚合器"""

    def __init__(self, n_users, n_items, n_entity, n_relation, gamma, max_iter):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = gamma
        self.max_iter = int(max_iter)
        self.dim = 128
        self.activation = nn.LeakyReLU()

    def forward(self, user_emb, item_emb, interact_mat):
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()
        user_item_mat = torch.sparse.FloatTensor(
            torch.cat([mat_row, mat_col]).view(2, -1), mat_val,
            size=[self.n_users, self.n_items])
        item_user_mat = torch.sparse.FloatTensor(
            torch.cat([mat_col, mat_row]).view(2, -1), mat_val,
            size=[self.n_items, self.n_users])
        user_agg_cf = torch.sparse.mm(user_item_mat, item_emb)
        item_agg_cf = torch.sparse.mm(item_user_mat, user_emb)
        return user_agg_cf, item_agg_cf


# ==================== 图卷积模块====================
class GraphConv(nn.Module):
    def __init__(self, channel, n_hops, n_users,
                 n_items, n_entities, n_relations, interact_mat, gamma, max_iter,
                 device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.channel = channel
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device

        # 关系嵌入 (原KG + 生成KG)
        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))
        self.relation_weight = nn.Parameter(relation_weight)

        n_relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))
        self.n_relation_weight = nn.Parameter(n_relation_weight)

        # 实例化改进后的去噪器
        self.denoiser = EnhancedTripletDenoiser(
            channel, n_relations, num_heads=4, dropout=0.1
        ).to(self.device)
        self.generated_weight_stats = []  # 记录每一轮前向传播的去噪信息

        for i in range(n_hops):
            self.convs.append(
                Aggregator(n_users=n_users, n_items=n_items, n_entity=n_entities,
                           n_relation=n_relations, gamma=gamma, max_iter=max_iter).to(self.device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _update_knowledge(self, two_hpo_kg):
        self.n_edge_index = two_hpo_kg[:, [0, -1]].transpose(1, 0)
        self.n_edge_type = two_hpo_kg[:, 1]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out

    def KG_forward(self, entity_emb, edge_index, edge_type, relation_weight, is_generated=False):

        if edge_index is None or edge_type is None or edge_index.size(1) == 0:
            return entity_emb, torch.zeros(entity_emb.size(0), device=entity_emb.device)

        head, tail = edge_index
        h_emb = entity_emb[head]
        t_emb = entity_emb[tail]
        r_emb = relation_weight[edge_type]

        if is_generated:
            # 定义用于checkpoint的函数
            def run_denoiser(h, r, t, rel_ids):
                return self.denoiser(h, r, t, rel_ids)

            # 使用梯度检查点 (Gradient Checkpointing) 节省大量显存，绝不切块(chunk)
            # 只有在需要计算梯度且输入要求梯度时，才触发checkpoint
            if torch.is_grad_enabled() and h_emb.requires_grad:
                weight, atten_score, auth_score, gate_weights = checkpoint(
                    run_denoiser, h_emb, r_emb, t_emb, edge_type,
                    use_reentrant=False
                )
            else:
                weight, atten_score, auth_score, gate_weights = run_denoiser(
                    h_emb, r_emb, t_emb, edge_type
                )

            # 保存去噪指标用于epoch结束时log_to_file统计
            self.generated_weight_stats.append({
                'weight': weight.detach(),
                'atten': atten_score.detach(),
                'auth': auth_score.detach(),
                'gate_atten': gate_weights[:, 0].detach(),
                'gate_auth': gate_weights[:, 1].detach()
            })

            # 融合去噪权重至消息传递 (权重与特征广播)
            msg = weight.unsqueeze(-1) * (t_emb + r_emb)
        else:
            # 消息传递：原图不执行去噪 (tail_emb + rel_emb)
            msg = t_emb + r_emb

        # 聚合：对每个头节点累加消息
        entity_agg = scatter_sum(msg, head, dim=0, dim_size=entity_emb.size(0))
        degree = scatter_sum(torch.ones_like(head, dtype=torch.float), head, dim=0, dim_size=entity_emb.size(0))

        entity_agg = entity_agg / (degree.unsqueeze(-1) + 1e-9)
        entity_agg = entity_agg + entity_emb  # 残差连接
        return entity_agg, degree

    def forward(self, all_embed, all_embed_cf, edge_index, edge_type,
                n_edge_index, n_edge_type, interact_mat,
                mess_dropout=True, node_dropout=False, gumbel=True, epoch=0):

        # 每次forward清空上一轮的日志统计缓存
        self.generated_weight_stats = []

        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()
        user_item_mat = torch.sparse.FloatTensor(
            torch.cat([mat_row, mat_col]).view(2, -1), mat_val,
            size=[self.n_users, self.n_items])
        item_user_mat = torch.sparse.FloatTensor(
            torch.cat([mat_col, mat_row]).view(2, -1), mat_val,
            size=[self.n_items, self.n_users])

        user_embeds = all_embed[:self.n_users]
        item_embed = all_embed[self.n_users:self.n_users + self.n_items][:, :self.channel]
        entity_emb = all_embed[self.n_users:][:, self.channel:self.channel * 2]
        n_entity_emb = all_embed[self.n_users:][:, self.channel * 2:]

        # ===== 原KG卷积 (不去噪) =====
        entity_emb_res = entity_emb[:self.n_items]
        for i in range(len(self.convs)):
            entity_emb, _ = self.KG_forward(entity_emb, edge_index, edge_type, self.relation_weight, is_generated=False)
            entity_emb_res = entity_emb_res + F.normalize(entity_emb[:self.n_items])
            torch.cuda.empty_cache()

        # ===== 生成KG卷积 (激活动态去噪器) =====
        n_entity_emb_res = n_entity_emb[:self.n_items]
        if n_edge_index is not None and n_edge_type is not None and n_edge_index.size(1) > 0:
            for i in range(len(self.convs)):
                n_entity_emb, _ = self.KG_forward(n_entity_emb, n_edge_index, n_edge_type, self.n_relation_weight,
                                                  is_generated=True)
                n_entity_emb_res = n_entity_emb_res + F.normalize(n_entity_emb[:self.n_items])
                torch.cuda.empty_cache()
        else:
            n_entity_emb_res = torch.zeros_like(entity_emb_res, device=self.device)

        # ===== 物品表示拼接 (dim * 3) =====
        item_embeds = torch.cat([item_embed, entity_emb_res, n_entity_emb_res], dim=-1)

        # ===== 用户-物品交互传播 =====
        user_embed_res = user_embeds
        item_embed_res = item_embeds
        for i in range(len(self.convs) - 1):
            user_embeds, item_embeds = self.convs[i](user_embeds, item_embeds, self.interact_mat)
            item_embed_res = item_embed_res + F.normalize(item_embeds)
            user_embed_res = user_embed_res + F.normalize(user_embeds)

        return user_embed_res, item_embed_res


# ==================== 推荐主模型 ====================
class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ui_sp_graph, item_rel_mask):
        super(Recommender, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']

        self.emb_size = args_config.dim

        self.context_hops = args_config.context_hops
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.item_rel_mask = torch.FloatTensor(item_rel_mask).to(self.device)
        self.ui_sp_graph = ui_sp_graph

        self.edge_index, self.edge_type = self._get_edges(graph)
        self.n_edge_index = None
        self.n_edge_type = None

        self.kg_triplets = None

        self.n_kg_triplets = None

        self.cet_loss = nn.CrossEntropyLoss(label_smoothing=0.85)
        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size * 3))
        self.all_embed = nn.Parameter(self.all_embed)
        self.all_embed_cf = None
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.ui_sp_graph).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         gamma=0.5, max_iter=2,
                         device=self.device,
                         node_dropout_rate=0.5,
                         mess_dropout_rate=0.1)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type_ = graph_tensor[:, -1]
        return index.t().long().to(self.device), type_.long().to(self.device)

    def update_generated_kg(self, generated_kg_tensor):
        if generated_kg_tensor is not None and len(generated_kg_tensor) > 0:
            self.gcn._update_knowledge(generated_kg_tensor)
            self.n_edge_index = self.gcn.n_edge_index
            self.n_edge_type = self.gcn.n_edge_type
            self.n_kg_triplets = generated_kg_tensor
        else:
            self.clear_generated_kg()

    def clear_generated_kg(self):
        self.n_edge_index = None
        self.n_edge_type = None
        self.gcn.n_edge_index = None
        self.gcn.n_edge_type = None
        self.n_kg_triplets = None

    def gcn_forword(self, user, pos_item, epoch=0):
        user_all_emb, item_all_emb = self.gcn(
            self.all_embed, self.all_embed_cf,
            self.edge_index, self.edge_type,
            self.n_edge_index, self.n_edge_type,
            self.interact_mat,
            mess_dropout=True, node_dropout=False, gumbel=False, epoch=epoch
        )
        user_emb = user_all_emb[user]
        score = torch.matmul(user_emb, item_all_emb.transpose(1, 0))
        rec_loss = self.cet_loss(score, pos_item)
        return rec_loss

    def forward(self, batch=None, mode="cf", epoch=0):
        if mode == "cf":
            user = batch['users']
            pos_item = batch['pos_items']
            return self.gcn_forword(user, pos_item, epoch)
        else:
            return torch.tensor(0.0, device=self.device)

    def log_denoise_stats(self, epoch):
        """记录并打印去噪器的统计信息，严格保留6位小数"""
        if not hasattr(self.gcn, 'generated_weight_stats') or len(self.gcn.generated_weight_stats) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 整合当前epoch所有批次的前向传播日志
        weights = torch.cat([s['weight'] for s in self.gcn.generated_weight_stats])
        atten_scores = torch.cat([s['atten'] for s in self.gcn.generated_weight_stats])
        auths = torch.cat([s['auth'] for s in self.gcn.generated_weight_stats])
        gate_attens = torch.cat([s['gate_atten'] for s in self.gcn.generated_weight_stats])
        gate_auths = torch.cat([s['gate_auth'] for s in self.gcn.generated_weight_stats])

        w_mean = weights.mean().item()
        w_max = weights.max().item()
        w_min = weights.min().item()
        atten_mean = atten_scores.mean().item()
        a_mean = auths.mean().item()
        g_atten_mean = gate_attens.mean().item()
        g_auth_mean = gate_auths.mean().item()

        log_msg = (f"[Denoise Info - Epoch {epoch}] "
                   f"Avg Weight: {w_mean:.6f} (Max: {w_max:.6f}, Min: {w_min:.6f}) | "
                   f"Avg Attention: {atten_mean:.6f} | Avg Authenticity: {a_mean:.6f} | "
                   f"Avg Gate Attention: {g_atten_mean:.6f} | Avg Gate Auth: {g_auth_mean:.6f}")
        log_to_file(log_msg)

        # 输出后清理缓存，准备下一次统计
        self.gcn.generated_weight_stats = []

        return w_mean, w_max, w_min, atten_mean, a_mean, g_atten_mean, g_auth_mean

    def compute_denoiser_regularization(self):
        """
        计算去噪器的正则化损失。
        包括:
        1. 关系投影矩阵的L2正则化 (防止过拟合)
        2. 真实性缩放因子的正则化
        """
        reg_loss = 0.0

        # 1. 关系投影矩阵L2正则化
        reg_loss += 1e-2 * torch.norm(self.gcn.denoiser.relation_proj_h.relation_proj, p=2)
        reg_loss += 1e-2 * torch.norm(self.gcn.denoiser.relation_proj_t.relation_proj, p=2)

        # 2. 真实性缩放因子的正则化
        reg_loss += 1e-1 * torch.norm(self.gcn.denoiser.auth_scale, p=2)

        return reg_loss

    def compute_relation_diversity_loss(self):
        """
        关系多样性损失：鼓励不同关系的投影矩阵保持多样性，避免坍缩到同一空间。
        参考: RAAKGC (Yuan et al., 2025, AAAI) 的关系感知锚点增强思想。

        原理：如果所有关系的投影矩阵都相似，则关系感知投影失去意义。
        通过惩罚关系间投影矩阵的高相似度，鼓励每个关系学习独特的语义空间。
        """
        # 获取所有关系的投影权重
        proj_h = self.gcn.denoiser.relation_proj_h.relation_proj  # [n_relations, dim]

        # 计算关系间投影矩阵的余弦相似度矩阵
        proj_h_norm = F.normalize(proj_h, p=2, dim=1)
        sim_matrix = torch.matmul(proj_h_norm, proj_h_norm.t())  # [n_relations, n_relations]

        # 对角线置0 (排除自身)
        mask = torch.eye(self.n_relations, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)

        # 鼓励非对角线元素尽可能小 (关系间保持差异)
        # 使用Frobenius范数惩罚高相似度
        diversity_loss = torch.mean(sim_matrix ** 2)

        return diversity_loss

    def generate(self, for_kgc=False, epoch=0):
        user_all_emb, item_all_emb = self.gcn(
            self.all_embed, self.all_embed_cf,
            self.edge_index, self.edge_type,
            self.n_edge_index, self.n_edge_type,
            self.interact_mat,
            mess_dropout=False, node_dropout=False, gumbel=False, epoch=epoch
        )

        # 由于 test(model...) 是基于 generate 进行评估，生成节点表示后立刻打印本 epoch 去噪日志
        self.log_denoise_stats(epoch)

        item_pred_emb = item_all_emb
        user_pred_emb = user_all_emb
        if for_kgc:
            placeholder = torch.zeros(self.n_items, device=self.device)
            return item_pred_emb, placeholder
        else:
            return item_pred_emb, user_pred_emb

    def rating(self, u_g_embeddings, i_g_embeddings, type="bpr"):
        if type == "bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()
        else:
            return (torch.cosine_similarity(
                u_g_embeddings[:, :self.emb_size].unsqueeze(1),
                i_g_embeddings[:, :self.emb_size].unsqueeze(0), dim=2).detach().cpu() +
                    torch.cosine_similarity(
                        u_g_embeddings[:, self.emb_size:].unsqueeze(1),
                        i_g_embeddings[:, self.emb_size:].unsqueeze(0), dim=2).detach().cpu())