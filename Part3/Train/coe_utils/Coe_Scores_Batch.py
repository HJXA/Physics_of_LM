import torch
import torch.nn.functional as F

class CoEScoreInfo_Train:
    """
    用于处理模型各层隐藏状态（hidden states），计算与隐藏状态演变相关的CoE（Combined Evolution）指标
    衡量模型内部表示在层间的变化幅度、角度及综合特征
    """
    def __init__(self, hidden_states):
        """
        初始化方法：存储各层隐藏状态
        
        参数:
            hidden_states: 模型各层的隐藏状态列表，形状为[层数, 隐藏状态维度]
        """
        self.hidden_states = hidden_states  # 各层隐藏状态

    
    def compute_CoE(self):
        """
        计算 CoE 角度相关指标 (Angle)
        返回形状均为 (B,) 的张量:
            z_ang: 全局首尾层夹角
            a_in: 第一层与第二层的夹角
            a_mid: 中间层的平均夹角
            a_out: 倒数第二层与最后一层的夹角
        """
        hs = self.hidden_states  # (B, L, D)
        B, L, D = hs.shape
        
        # ====================
        # 1. 计算 z_ang (全局首尾夹角)
        # ====================
        v_start = hs[:, 0, :]  # (B, D)
        v_end = hs[:, -1, :]   # (B, D)
        
        # dim=-1 表示在 D 维度上计算余弦相似度
        cos_global = F.cosine_similarity(v_start, v_end, dim=-1)  # (B,)
        cos_global = torch.clamp(cos_global, -1.0 + 1e-7, 1.0 - 1e-7)
        z_ang = torch.acos(cos_global)  # (B,) 对应原代码的 denominator_angle / z_ang

        # ====================
        # 2. 计算所有相邻层的逐层夹角
        # ====================
        vectors_a = hs[:, :-1, :]  # (B, L-1, D)
        vectors_b = hs[:, 1:, :]   # (B, L-1, D)
        
        cos_locals = F.cosine_similarity(vectors_a, vectors_b, dim=-1)  # (B, L-1)
        cos_locals = torch.clamp(cos_locals, -1.0 + 1e-7, 1.0 - 1e-7)
        angles_local = torch.acos(cos_locals)  # (B, L-1) ，包含所有相邻层夹角

        # ====================
        # 3. 提取 a_in, a_mid, a_out
        # ====================
        if L > 1:
            # angles_local[:, 0] 是第0层和第1层的夹角
            a_in = angles_local[:, 0]    # (B,)
            # angles_local[:, -1] 是倒数第2层和最后一层的夹角
            a_out = angles_local[:, -1]  # (B,)
        else:
            # 防御性编程：如果只有一层，夹角为0
            a_in = torch.zeros(B, device=hs.device)
            a_out = torch.zeros(B, device=hs.device)
            
        if L > 3:
            # 取出中间的所有夹角并求均值
            mids = angles_local[:, 1:-1]       # (B, L-3)
            a_mid = torch.mean(mids, dim=-1)   # (B,)
        else:
            # 如果层数 <= 3，说明没有中间层
            a_mid = torch.zeros(B, device=hs.device)

        z_ang_mean = z_ang.mean().detach().item()
        a_in_mean = a_in.mean().detach().item()
        a_mid_mean = a_mid.mean().detach().item()
        a_out_mean = a_out.mean().detach().item()

        return z_ang_mean, a_in_mean, a_mid_mean, a_out_mean

    def compute_CoE_Ang(self):
        """
        计算 CoE 角度 (Angle) 特征，支持 Batch 处理。
        衡量隐藏状态在层间的"角度变化幅度"特征归一化结果。
        """
        hs = self.hidden_states  # (B, L, D)

        # ====================
        # 1. 计算分母：全局夹角
        # ====================
        v_start = hs[:, 0, :]
        v_end = hs[:, -1, :]
        
        cos_global = F.cosine_similarity(v_start, v_end, dim=-1) # (B,)
        cos_global = torch.clamp(cos_global, -1.0 + 1e-7, 1.0 - 1e-7)
        denominator_angle = torch.acos(cos_global) + 1e-8 # (B,)

        # ====================
        # 2. 计算分子：逐层夹角
        # ====================
        vectors_a = hs[:, :-1, :] # (B, L-1, D)
        vectors_b = hs[:, 1:, :]  # (B, L-1, D)
        
        cos_locals = F.cosine_similarity(vectors_a, vectors_b, dim=-1) # (B, L-1)
        cos_locals = torch.clamp(cos_locals, -1.0 + 1e-7, 1.0 - 1e-7)
        angles_local = torch.acos(cos_locals) # (B, L-1)

        # 3. 归一化
        al_semdiff_norm = angles_local / denominator_angle.unsqueeze(-1) # (B, L-1)

        # 4. 统计
        al_semdiff_ave = torch.mean(al_semdiff_norm, dim=-1) # (B,)
        al_semdiff_var = torch.var(al_semdiff_norm, dim=-1) # (B,)
        angles_mean = torch.mean(angles_local, dim=-1) # (B,)

        # 返回均值或完整张量均可，这里返回标量均值，以方便直接记录
        return al_semdiff_norm.mean().item(), al_semdiff_ave.mean().item(), al_semdiff_var.mean().item(), denominator_angle.mean().item(), angles_mean.mean().item()







