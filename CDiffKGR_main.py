import os, sys, math, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from math import log
from tqdm import tqdm
from time import time
import multiprocessing
import scipy.sparse as sp
from collections import defaultdict
import datetime

from utils.parser import parse_args
from prettytable import PrettyTable
from utils.data_loader import load_data
from modules.EDKG import Recommender
from utils.evaluate import test
from utils.helper import early_stopping, _cal_npmi
from torch.utils.checkpoint import checkpoint
from modules.pcgrad import PCGrad

cores = multiprocessing.cpu_count()
n_users, n_items, n_entities, n_nodes, n_relations = 0, 0, 0, 0, 0


def log_to_file(message, filename="/home/homes/tzx/ssd/kgrec/KGIN/EditKG-main/EditKG-main/result/last-fm消融实验7-27(1).txt"):
    with open(filename, "a") as f:
        f.write(message + "\n")
    print(message)


def get_feed_data(train_entity_pairs, train_user_set):
    feed_dict = {}
    feed_dict['users'] = train_entity_pairs[:, 0]
    feed_dict['pos_items'] = train_entity_pairs[:, 1]
    return feed_dict


# ==================== TDM 模型相关 ====================
def get_original_relation_ids(original_relation_num, relation_id_offset, total_relation_num):
    """返回数据加载后原始KG关系在嵌入表中的真实索引。"""
    start = int(relation_id_offset)
    stop = start + int(original_relation_num)
    if start < 0 or original_relation_num <= 0:
        raise ValueError("relation_id_offset必须非负，original_relation_num必须为正数。")
    if stop > int(total_relation_num):
        raise ValueError(
            f"原始关系索引范围[{start}, {stop - 1}]超出关系嵌入表[0, {total_relation_num - 1}]。"
        )
    return list(range(start, stop))


def extract_item_triplets_for_tdm(triplets, n_items, original_relation_num,
                                  relation_id_offset):
    """提取物品-非物品实体三元组，并保留数据加载后的关系索引。"""
    relation_start = int(relation_id_offset)
    relation_stop = relation_start + int(original_relation_num)
    item_triplets = []
    for h, r, t in triplets:
        if not (relation_start <= int(r) < relation_stop):
            continue
        if h < n_items and t >= n_items:
            item_triplets.append([h, r, t])
        elif h >= n_items and t < n_items:
            item_triplets.append([t, r, h])
    if not item_triplets:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(item_triplets, dtype=np.int64)


def expand_generated_kg_with_original(generated_kg, original_triplets, n_items, device=None):
    if len(generated_kg) == 0:
        return generated_kg
    tail_set = set(generated_kg[:, 2])
    mask = np.isin(original_triplets[:, 0], list(tail_set))
    new_triplets_all = original_triplets[mask]
    non_item_mask = (new_triplets_all[:, 0] >= n_items) & (new_triplets_all[:, 2] >= n_items)
    new_triplets = new_triplets_all[non_item_mask]
    expanded = np.concatenate([generated_kg, new_triplets], axis=0)
    expanded = np.unique(expanded, axis=0)
    log_to_file(f"原始KG扩展: 生成尾实体数 {len(tail_set)}, "
                f"新增三元组 {len(new_triplets)}, 扩展后总数 {len(expanded)}")
    return expanded


class ConditionEncoder(nn.Module):
    """使用实数空间中的加法交互编码已知头实体与关系条件。"""

    def __init__(self, entity_dim, relation_dim, output_dim):
        super(ConditionEncoder, self).__init__()
        if entity_dim != relation_dim:
            raise ValueError("当前ConditionEncoder要求实体维度与关系维度一致。")
        self.linear = nn.Linear(entity_dim, output_dim)

    def forward(self, head_emb, rel_emb):
        return self.linear(head_emb + rel_emb)


class CTDenoiserBlock(nn.Module):
    """带条件缩放和残差连接的MLP式条件三元组去噪块。"""

    def __init__(self, hidden_dim, conditioning_dim, dropout=0.1):
        super(CTDenoiserBlock, self).__init__()
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_pointwise = nn.LayerNorm(hidden_dim)
        self.pointwise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, hidden_dim * 2),
        )

    def forward(self, x, conditioning):
        mlp_scale, pointwise_scale = self.scale_mlp(conditioning).chunk(2, dim=-1)
        x = x + (1.0 + mlp_scale) * self.mlp(self.norm_mlp(x))
        x = x + (1.0 + pointwise_scale) * self.pointwise(self.norm_pointwise(x))
        return x


class ConditionalTripleDenoiser(nn.Module):
    def __init__(self, triple_dim, time_emb_dim, condition_dim, hidden_dim, num_blocks=2):
        super(ConditionalTripleDenoiser, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.triple_dim = triple_dim
        self.condition_dim = condition_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.condition_encoder = ConditionEncoder(
            triple_dim // 3, triple_dim // 3, condition_dim
        )
        self.triple_embedding = nn.Linear(triple_dim, hidden_dim)
        conditioning_dim = time_emb_dim + condition_dim
        self.blocks = nn.ModuleList([
            CTDenoiserBlock(hidden_dim, conditioning_dim, dropout=0.1)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.noise_output = nn.Linear(hidden_dim, triple_dim)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _timestep_embedding(self, timesteps, device):
        half_dim = max(self.time_emb_dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=device)
            / half_dim
        )
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        time_emb = time_emb[:, :self.time_emb_dim]
        if time_emb.size(-1) < self.time_emb_dim:
            time_emb = F.pad(time_emb, (0, self.time_emb_dim - time_emb.size(-1)))
        return self.time_embedding(time_emb)

    def forward(self, noisy_triple, timesteps, head_emb, rel_emb):
        time_emb = self._timestep_embedding(timesteps, noisy_triple.device)
        condition_emb = self.condition_encoder(head_emb, rel_emb)
        conditioning = torch.cat([time_emb, condition_emb], dim=-1)
        x = self.triple_embedding(noisy_triple)
        for block in self.blocks:
            x = block(x, conditioning)
        return self.noise_output(self.output_norm(x))


class TDMGaussianDiffusion(nn.Module):
    def __init__(self, noise_scale=1.0, noise_min=0.0001, noise_max=0.02,
                 steps=1000, device=None):
        super(TDMGaussianDiffusion, self).__init__()
        self.noise_scale = float(noise_scale)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.steps = int(steps)
        self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
        self.calculate_for_diffusion(device or torch.device("cpu"))

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        if not (0.0 < start <= end < 1.0):
            raise ValueError("扩散beta范围必须满足0 < noise_scale*noise_min <= noise_scale*noise_max < 1。")
        return np.linspace(start, end, self.steps, dtype=np.float64)

    def calculate_for_diffusion(self, device=None):
        device = device or torch.device("cpu")
        self.betas = self.betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=self.alphas_cumprod.dtype, device=device),
            self.alphas_cumprod[:-1]
        ])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        clipped_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(clipped_variance)

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        if timesteps.device != arr.device:
            timesteps = timesteps.to(arr.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, head_emb, rel_emb):
        predicted_noise = model(x_t, t, head_emb, rel_emb)
        alpha_t = self._extract_into_tensor(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        beta_t = self._extract_into_tensor(self.betas, t, x_t.shape)
        model_mean = (
            x_t - beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12)) * predicted_noise
        ) / torch.sqrt(alpha_t)
        model_log_variance = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return model_mean, model_log_variance

    def apply_condition_constraint(self, x_t, head_emb, rel_emb, grad_step=10, eta=0.01):
        """约束头实体和关系切片，并用其平均梯度近似更新尾实体切片。"""
        with torch.no_grad():
            x_t_adjusted = x_t.clone()
            triple_element_dim = x_t.shape[-1] // 3
            condition = torch.cat([head_emb, rel_emb], dim=-1)
            for step in range(int(grad_step)):
                head_rel_part = x_t_adjusted[:, :2 * triple_element_dim]
                head_rel_gradient = 2.0 * (head_rel_part - condition)
                x_t_adjusted[:, :2 * triple_element_dim] -= eta * head_rel_gradient
                tail_gradient = head_rel_gradient.view(
                    head_rel_gradient.size(0), 2, triple_element_dim
                ).mean(dim=1)
                x_t_adjusted[:, 2 * triple_element_dim:] -= eta * tail_gradient
                if step % 3 == 0 and step > 0:
                    x_t_adjusted.clamp_(-5.0, 5.0)
        return x_t_adjusted

    def compute_denoised_triple(self, model, x_t, timesteps, head_emb, rel_emb):
        alpha_bar_t = self._extract_into_tensor(self.alphas_cumprod, timesteps, x_t.shape)
        predicted_noise = model(x_t, timesteps, head_emb, rel_emb)
        return (
            x_t / torch.sqrt(alpha_bar_t)
            - torch.sqrt(torch.clamp(1.0 / alpha_bar_t - 1.0, min=0.0)) * predicted_noise
        )

    def training_losses_tdm(self, model, x_start, timesteps, head_emb, rel_emb,
                            neg_samples=None, margin=10.0):
        batch_size = x_start.size(0)
        timesteps = timesteps.to(x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        denoised_positive = self.compute_denoised_triple(
            model, x_t, timesteps, head_emb, rel_emb
        )
        d_positive = torch.norm(x_start - denoised_positive, p=1, dim=-1)
        positive_loss = -F.logsigmoid(margin - d_positive)
        negative_loss = torch.zeros(batch_size, device=x_start.device)
        if neg_samples:
            for neg_triple in neg_samples:
                x_t_neg = self.q_sample(neg_triple, timesteps, noise)
                denoised_negative = self.compute_denoised_triple(
                    model, x_t_neg, timesteps, head_emb, rel_emb
                )
                d_negative = torch.norm(x_start - denoised_negative, p=1, dim=-1)
                negative_loss += -F.logsigmoid(d_negative - margin)
            negative_loss = negative_loss / len(neg_samples)
        return (positive_loss + negative_loss).mean()

    def p_sample(self, model, initial_noise, steps, head_emb, rel_emb,
                 apply_constraint=True, grad_step=10, eta=0.01):
        model.eval()
        sampling_steps = min(max(int(steps), 1), self.steps)
        x_t = initial_noise
        with torch.no_grad():
            for i in range(sampling_steps - 1, -1, -1):
                t = torch.full((x_t.shape[0],), i, device=x_t.device, dtype=torch.long)
                model_mean, model_log_variance = self.p_mean_variance(
                    model, x_t, t, head_emb, rel_emb
                )
                if i > 0:
                    x_t = model_mean + torch.exp(0.5 * model_log_variance) * torch.randn_like(x_t)
                else:
                    x_t = model_mean
                if apply_constraint and i > 0:
                    x_t = self.apply_condition_constraint(
                        x_t, head_emb, rel_emb, grad_step=grad_step, eta=eta
                    )
        return x_t


def train_tdm_model(model, triplets, n_items, n_entities, epochs, device,
                    original_relation_num, relation_id_offset, args):
    tdm_triplets = extract_item_triplets_for_tdm(
        triplets, n_items, original_relation_num, relation_id_offset
    )
    if len(tdm_triplets) == 0:
        log_to_file("没有找到可用于TDM训练的物品三元组")
        return None, None, None, None
    log_to_file(f"用于TDM训练的物品三元组数量: {len(tdm_triplets)}")
    with torch.no_grad():
        entity_embeds = model.all_embed[model.n_users:].detach()
        entity_embeds_single = entity_embeds[:, :args.dim]
        relation_embeds = model.gcn.relation_weight.detach()
    get_original_relation_ids(
        original_relation_num, relation_id_offset, relation_embeds.size(0)
    )
    triple_dim = 3 * args.dim
    tdm_model = ConditionalTripleDenoiser(
        triple_dim=triple_dim,
        time_emb_dim=args.dim,
        condition_dim=args.dim,
        hidden_dim=1024,
        num_blocks=2,
    ).to(device)
    diffusion_model = TDMGaussianDiffusion(
        noise_scale=1.0,
        noise_min=0.0001,
        noise_max=0.02,
        steps=args.tdm_steps,
        device=device,
    ).to(device)
    tdm_optimizer = torch.optim.Adam(tdm_model.parameters(), lr=0.0001)
    best_loss = float('inf')
    best_model_state = None
    batch_size = 4096
    num_batches = math.ceil(len(tdm_triplets) / batch_size)
    for epoch in range(epochs):
        tdm_model.train()
        total_loss = 0.0
        indices = np.arange(len(tdm_triplets))
        np.random.shuffle(indices)
        shuffled_triplets = tdm_triplets[indices]
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(shuffled_triplets))
            batch_triplets = shuffled_triplets[start_idx:end_idx]
            if len(batch_triplets) == 0:
                continue
            batch_triplets = torch.as_tensor(batch_triplets, dtype=torch.long, device=device)
            h_indices = batch_triplets[:, 0]
            r_indices = batch_triplets[:, 1]
            t_indices = batch_triplets[:, 2]
            head_emb = entity_embeds_single[h_indices]
            rel_emb = relation_embeds[r_indices]
            tail_emb = entity_embeds_single[t_indices]
            triple_embeddings = torch.cat([head_emb, rel_emb, tail_emb], dim=-1)

            neg_t_indices = torch.randint(
                low=n_items, high=n_entities, size=(len(batch_triplets),), device=device
            )
            collision = neg_t_indices.eq(t_indices)
            while collision.any():
                neg_t_indices[collision] = torch.randint(
                    low=n_items, high=n_entities, size=(int(collision.sum().item()),), device=device
                )
                collision = neg_t_indices.eq(t_indices)
            neg_tail_emb = entity_embeds_single[neg_t_indices]
            neg_triple_embeddings = torch.cat([head_emb, rel_emb, neg_tail_emb], dim=-1)

            timesteps = torch.randint(
                0, diffusion_model.steps, (len(batch_triplets),), device=device
            ).long()
            tdm_optimizer.zero_grad(set_to_none=True)
            diffusion_loss = diffusion_model.training_losses_tdm(
                tdm_model,
                triple_embeddings,
                timesteps,
                head_emb,
                rel_emb,
                neg_samples=[neg_triple_embeddings],
                margin=args.tdm_margin,
            )
            diffusion_loss.backward()
            tdm_optimizer.step()
            total_loss += diffusion_loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(tdm_model.state_dict())
        if epoch % 50 == 0 or epoch == epochs - 1:
            log_to_file(
                f"TDM Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}"
            )
    if best_model_state is not None:
        tdm_model.load_state_dict(best_model_state)
        log_to_file(f"TDM模型训练完成，最佳损失: {best_loss:.6f}")
    return tdm_model, diffusion_model, entity_embeds_single, relation_embeds


def generate_triplets_with_tdm(tdm_model, diffusion_model, entity_embeds,
                               relation_embeds, n_items, n_entities, device,
                               rebuild_k=40, max_total_triplets=20000, args=None):
    log_to_file("开始使用TDM模型生成三元组...")
    non_item_embeds = entity_embeds[n_items:]
    num_non_item = non_item_embeds.size(0)
    if num_non_item == 0:
        log_to_file("没有非物品实体可用于生成尾实体")
        return np.empty((0, 3), dtype=np.int64)

    relation_ids = get_original_relation_ids(
        args.original_relation_num,
        args.relation_id_offset,
        relation_embeds.size(0),
    )
    all_generated_triplets = []
    all_distances = []
    batch_size = 4096
    k_per_rel = max(int(args.k_per_rel), 1)
    for item_idx in tqdm(range(0, n_items, batch_size), desc="生成三元组"):
        end_idx = min(item_idx + batch_size, n_items)
        batch_items = torch.arange(item_idx, end_idx, device=device)
        batch_size_actual = len(batch_items)
        head_emb = entity_embeds[batch_items]
        for rel_idx in relation_ids:
            rel_emb = relation_embeds[rel_idx].unsqueeze(0).expand(batch_size_actual, -1)
            initial_noise = torch.randn(batch_size_actual, 3 * args.dim, device=device)
            generated_triples = diffusion_model.p_sample(
                tdm_model,
                initial_noise,
                args.tdm_steps,
                head_emb,
                rel_emb,
                apply_constraint=True,
                grad_step=10,
                eta=0.01,
            )
            tail_embeds = generated_triples[:, 2 * args.dim:]
            sub_batch_size = min(4096, num_non_item)
            min_dists_list, min_indices_list = [], []
            for i in range(0, num_non_item, sub_batch_size):
                end_i = min(i + sub_batch_size, num_non_item)
                non_item_subset = non_item_embeds[i:end_i]
                dist_subset = torch.cdist(tail_embeds, non_item_subset, p=2)
                local_k = min(k_per_rel, non_item_subset.size(0))
                topk_dist, topk_local_idx = torch.topk(
                    dist_subset, k=local_k, dim=1, largest=False
                )
                min_dists_list.append(topk_dist)
                min_indices_list.append(topk_local_idx + i)
            all_min_dists = torch.cat(min_dists_list, dim=1)
            all_min_indices = torch.cat(min_indices_list, dim=1)
            final_k = min(k_per_rel, all_min_dists.size(1))
            global_min_dist, batch_idx = torch.topk(
                all_min_dists, k=final_k, dim=1, largest=False
            )
            global_min_indices = torch.gather(all_min_indices, 1, batch_idx)
            for j in range(batch_size_actual):
                item_id = int(batch_items[j].item())
                for kk in range(final_k):
                    tail_id = int(global_min_indices[j, kk].item()) + n_items
                    distance = float(global_min_dist[j, kk].item())
                    all_generated_triplets.append((item_id, rel_idx, tail_id, distance))
                    all_distances.append(distance)

    log_to_file(f"生成三元组总数: {len(all_generated_triplets)}")
    if not all_generated_triplets:
        return np.empty((0, 3), dtype=np.int64)

    all_distances_np = np.asarray(all_distances)
    if args.global_distance_threshold is None:
        threshold = np.percentile(all_distances_np, args.global_distance_percentile)
        threshold_desc = f"{args.global_distance_percentile:g}%分位数"
    else:
        threshold = float(args.global_distance_threshold)
        threshold_desc = "固定阈值"
    log_to_file(f"全局距离阈值（{threshold_desc}）: {threshold:.4f}")

    item_generated_triplets = defaultdict(list)
    for item_id, rel_idx, tail_id, dist in all_generated_triplets:
        if dist <= threshold:
            item_generated_triplets[item_id].append((dist, rel_idx, tail_id))

    h_list, r_list, t_list = [], [], []
    total_triplets = 0
    for item_id, generated_triplets in item_generated_triplets.items():
        generated_triplets.sort(key=lambda x: x[0])
        for dist, rel_idx, tail_id in generated_triplets[:rebuild_k]:
            h_list.append(item_id)
            r_list.append(rel_idx)
            t_list.append(tail_id)
            total_triplets += 1
            if total_triplets >= max_total_triplets:
                break
        if total_triplets >= max_total_triplets:
            break

    log_to_file(f"筛选后生成三元组数量: {total_triplets}")
    if total_triplets == 0:
        return np.empty((0, 3), dtype=np.int64)
    generated_triplets = np.column_stack([h_list, r_list, t_list]).astype(np.int64)
    log_to_file(f"生成三元组形状: {generated_triplets.shape}")
    return generated_triplets


# ==================== 主函数 ====================
if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # 加载数据
    train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, \
        all_sparse_graph, item_rel_mask, triplets, kg_dict = load_data(args)
    item_pmi_dict = _cal_npmi(user_dict['train_user_set'])
    pkl.dump(item_pmi_dict, open(args.dataset + "item_pair_pmi.pkl", "wb"))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']
    get_original_relation_ids(
        args.original_relation_num, args.relation_id_offset, n_relations
    )

    train_cf_pairs = torch.LongTensor(
        np.array([[cf[0], cf[1]] for cf in train_cf], np.int32)
    )

    # 推荐模型
    model = Recommender(n_params, args, graph, ui_sparse_graph, item_rel_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best = 0
    stopping_step = 0
    should_stop = False
    best_metric = {"recall": 0, "ndcg": 0, "precision": 0, "hit_ratio": 0}
    best_epoch = {"recall": 0, "ndcg": 0, "precision": 0, "hit_ratio": 0}

    iter_ = math.ceil(len(train_cf_pairs) / args.batch_size)
    tdm_model, diffusion_model = None, None
    tdm_entity_embeds, tdm_relation_embeds = None, None

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        current_epoch = epoch

        if epoch % 3 == 0 or epoch == 0:
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])
            all_feed_data['pos_index'] = torch.LongTensor(index)

        # 生成KG
        if args.use_tdm and epoch % args.kg_gen_freq == 0:
            log_to_file(f"Epoch {epoch}: Training TDM...")
            tdm_model, diffusion_model, tdm_entity_embeds, tdm_relation_embeds = train_tdm_model(
                model,
                triplets,
                n_items,
                n_entities,
                epochs=args.tdm_epochs,
                device=device,
                original_relation_num=args.original_relation_num,
                relation_id_offset=args.relation_id_offset,
                args=args,
            )

            if tdm_model is not None:
                log_to_file(f"Epoch {epoch}: Generating the generated KG...")
                generated_kg = generate_triplets_with_tdm(
                    tdm_model,
                    diffusion_model,
                    tdm_entity_embeds,
                    tdm_relation_embeds,
                    n_items,
                    n_entities,
                    device,
                    rebuild_k=args.tdm_rebuild_k,
                    max_total_triplets=args.tdm_max_triplets,
                    args=args,
                )

                if len(generated_kg) > 0:
                    expanded_generated_kg = expand_generated_kg_with_original(
                        generated_kg, triplets, n_items, device
                    )
                    log_to_file(f"Expanded generated KG size: {len(expanded_generated_kg)}")
                    generated_kg_tensor = torch.as_tensor(
                        expanded_generated_kg, dtype=torch.long, device=device
                    )
                    model.update_generated_kg(generated_kg_tensor)
                    log_to_file(
                        f"Updated generated KG with {len(expanded_generated_kg)} triplets."
                    )
                else:
                    log_to_file("Generated KG empty, clearing.")
                    model.clear_generated_kg()
            else:
                log_to_file("TDM training failed.")
                model.clear_generated_kg()

        # 训练推荐模型
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_denoiser_reg_loss = 0.0
        total_diversity_loss = 0.0
        train_s_t = time()

        for i in tqdm(range(iter_)):
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            batch = {
                'pos_index': all_feed_data['pos_index'][
                    i * args.batch_size:(i + 1) * args.batch_size
                ].to(device),
                'users': all_feed_data['users'][
                    i * args.batch_size:(i + 1) * args.batch_size
                ].to(device),
                'pos_items': all_feed_data['pos_items'][
                    i * args.batch_size:(i + 1) * args.batch_size
                ].to(device),
            }

            # 推荐损失（保持原始CrossEntropyLoss）
            rec_loss = model(batch, epoch=current_epoch)
            denoiser_reg_loss = model.compute_denoiser_regularization()
            diversity_loss = model.compute_relation_diversity_loss()
            total_batch_loss = (
                rec_loss
                + args.denoiser_reg_weight * denoiser_reg_loss
                + args.relation_diversity_weight * diversity_loss
            )

            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_rec_loss += rec_loss.item()
            total_denoiser_reg_loss += denoiser_reg_loss.item()
            total_diversity_loss += diversity_loss.item()

        train_e_t = time()
        nowtime = datetime.datetime.now()
        avg_total_loss = total_loss / max(iter_, 1)
        avg_rec_loss = total_rec_loss / max(iter_, 1)
        avg_denoiser_reg = total_denoiser_reg_loss / max(iter_, 1)
        avg_diversity = total_diversity_loss / max(iter_, 1)

        log_to_file(
            f"Time {nowtime}: epoch {epoch}, "
            f"total_loss {avg_total_loss:.6f}, "
            f"rec_loss {avg_rec_loss:.6f}, "
            f"denoiser_reg {avg_denoiser_reg:.6f}, "
            f"diversity_loss {avg_diversity:.6f}"
        )

        # 生成物品嵌入
        with torch.no_grad():
            item_embs, KG_mask = model.generate(for_kgc=True, epoch=current_epoch)
            item_embs = item_embs.detach().cpu().numpy()
            KG_mask = KG_mask.detach().cpu().numpy()

        # 测试
        model.eval()
        test_s_t = time()
        with torch.no_grad():
            ret = test(model, user_dict, n_params)
        test_e_t = time()

        for k in best_metric.keys():
            if ret[k][0] > best_metric[k]:
                best_metric[k] = ret[k][0]
                best_epoch[k] = epoch

        train_res = PrettyTable()
        train_res.field_names = [
            "Epoch", "training time", "testing time",
            "recall", "ndcg", "precision", "hit_ratio"
        ]
        train_res.add_row([
            epoch, train_e_t - train_s_t, test_e_t - test_s_t,
            ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']
        ])
        log_to_file(str(train_res))
        log_to_file(f"best_metric: {best_metric}")

        cur_best, stopping_step, should_stop = early_stopping(
            ret['recall'][0],
            cur_best,
            stopping_step,
            expected_order='acc',
            flag_step=args.flag_step,
        )

        if should_stop:
            break

        if args.save and ret['recall'][0] == cur_best:
            torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

    log_to_file('early stopping at %d, best recall@20: %.6f' % (epoch, cur_best))