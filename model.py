import torch
import math
import torch.nn as nn
import torch.autograd as autograd


class AMAdaptiveSelfAttention(nn.Module):
    """
    优化版：多头特征级自适应自注意力机制 (Multi-Head Feature-wise Attention)
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        # 确保特征维度可以被头数整除
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 将独立的线性层合并，提升计算效率
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 门控机制
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 【核心优化】：PINN 冷启动保护 (Cold-start Gating)
        # 将 bias 初始化为一个负数（如 -2.0）。
        # 这样在训练初期，sigmoid(-2.0) ≈ 0.11，门控处于微启状态。
        # 强迫网络在初期依赖 KAN 主干拟合基础物理规律，随着训练深入再自适应打开 Attention 捕捉微小突变。
        nn.init.constant_(self.gate_proj.bias, -2.0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        # h: [batch, hidden_dim]
        B, H = h.shape
        
        # 1. 投影并切分为多头：[B, 3, heads, head_dim]
        qkv = self.qkv_proj(h).view(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 每个维度都是 [B, heads, head_dim]

        # 2. 头内的特征级注意力 (Feature-wise attention WITHIN each head)
        # 这样每个 Head 可以专门负责一种物理特征子空间（如：头1管电热耦合，头2管空间传导）
        # q: [B, heads, head_dim, 1]
        # k: [B, heads, 1, head_dim]
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-2)  # 等价于 unsqueeze(2)
        
        # 计算特征间的协方差分数矩阵: [B, heads, head_dim, head_dim]
        scale = float(self.head_dim) ** 0.5
        attn_scores = torch.matmul(q, k) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 3. 聚合上下文
        # v: [B, heads, head_dim, 1]
        v = v.unsqueeze(-1)
        # context: [B, heads, head_dim, 1] -> squeeze -> [B, heads, head_dim]
        context = torch.matmul(attn_weights, v).squeeze(-1)
        
        # 重新组合为 [B, hidden_dim]
        context = context.view(B, H)

        # 4. 残差门控输出
        gate = torch.sigmoid(self.gate_proj(h))
        fused = h + gate * self.dropout(self.out_proj(context))
        
        return self.norm(fused)

class KANLayer(nn.Module):
    """
    优化版 KAN 层 (Physics-Enhanced KAN)
    修复了纯 RBF 容易造成的梯度消失和外推失效问题，加入了 Base 基础函数。
    """
    def __init__(self, in_dim, out_dim, num_centers=8, grid_margin=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_centers = num_centers
        
        # 优化 1: 引入基础变换分支 (Base Branch)
        # 使用 SiLU (Swish) 捕捉全局的连续非线性趋势，避免 RBF 在区间外衰减为0导致的外推灾难
        self.base_activation = nn.SiLU()
        self.base_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        # 优化 2: 扩展中心点网格 (Grid Extension)
        # 将网格向两侧延伸 grid_margin，增强对边界条件 (如表面散热、极端电压) 的感知能力
        grid_min, grid_max = -grid_margin, 1.0 + grid_margin
        self.centers = nn.Parameter(
            torch.linspace(grid_min, grid_max, num_centers).view(1, 1, num_centers)
        )
        
        # 优化 3: 科学的宽度初始化 (Prevent Multicollinearity)
        # 根据高斯中心点的间距 d 初始化宽度，使得 w ≈ 1 / (2 * d^2)
        # 这样能保证相邻的高斯基函数有适当的交叉，但又不会过度重叠
        d = (grid_max - grid_min) / (num_centers - 1)
        init_width = 1.0 / (2 * (d ** 2))
        # log(init_width) 才是我们需要的初始化值
        self.log_widths = nn.Parameter(
            torch.full((1, in_dim, num_centers), math.log(init_width))
        )
        
        # RBF 权重初始化
        self.rbf_weights = nn.Parameter(
            torch.randn(out_dim, in_dim, num_centers) * (1.0 / (in_dim * num_centers) ** 0.5)
        )

    def forward(self, x):
        # x: [batch, in_dim]
        
        # --- 分支 1: 全局基础映射 (Base Branch) ---
        # 确保网络在任何极端工况下都有稳定的梯度和预测能力
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        
        # --- 分支 2: 局部高斯精细拟合 (RBF Branch) ---
        x_ext = x.unsqueeze(-1) # [batch, in_dim, 1]
        widths = torch.exp(self.log_widths)  # 确保 widths > 0
        basis = torch.exp(-widths * (x_ext - self.centers)**2) # [batch, in_dim, centers]
        rbf_output = torch.einsum('bic, oic -> bo', basis, self.rbf_weights)
        
        # 最终输出为 全局趋势 + 局部精细非线性的组合
        return base_output + rbf_output

class MM_Q3D_KAN_PINN(nn.Module):
    def __init__(
        self,
        input_dim=7,
        hidden_dim=32,
        output_dim=1,
        use_am_attention=True,
        am_dropout=0.05,
    ):
        super().__init__()
        
        # 优化4: 物理特征变换 (Physics-Informed Feature Transformation)
        # forward 中增加两项增广特征: V^2 与 time-step envelope，因此 KAN1 输入维度 +2
        self.augmented_dim = input_dim + 2
        
        # KAN 主干网络
        self.kan1 = KANLayer(self.augmented_dim, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, output_dim)
        self.use_am_attention = bool(use_am_attention)
        self.am_attention = AMAdaptiveSelfAttention(hidden_dim, dropout=am_dropout)
        
        # --- 优化1 & 3: 物理参数动态辨识与单调性先验 ---
        # 内阻不再是常数，而是 R_in(T) = R_base * exp(-R_alpha * T)
        self.R_in_base = nn.Parameter(torch.tensor([0.05], requires_grad=True))
        self.R_in_alpha = nn.Parameter(torch.tensor([1.0], requires_grad=True)) # 温度影响系数
        
        self.h_conv = nn.Parameter(torch.tensor([15.0], requires_grad=True))
        # Temperature envelope time constant used in forward()
        self.tau = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        
        # 物理常量基准值 (25℃)
        self.rho_cp_base = 2600.0 * 850.0 
        self.k_cond_base = 3.0

    def forward(self, x):
        t = x[:, 0:1]
        V = x[:, 5:6]
        
        v_sq = V ** 2 
        t_step = torch.exp(-10.0 * t) 
        x_augmented = torch.cat([x, v_sq, t_step], dim=-1)
        
        h1 = torch.tanh(self.kan1(x_augmented))
        if self.use_am_attention:
            h1 = self.am_attention(h1)
            
        # 【修改点 1】: 移除 torch.sigmoid，直接输出！
        # 因为我们后续加了 envelope 包络线，直接用线性输出即可
        nn_out = self.kan2(h1) 
        
        envelope = 1.0 - torch.exp(-self.tau.abs() * t)
        T_pred = nn_out * envelope
        
        # 【修改点 2】: 为了防止数值越界，使用 clamp 而不是 sigmoid
        # 允许输出略微超出 [0, 1] 的范围（如 -0.1 到 1.2），提升对最高温的捕捉能力
        return torch.clamp(T_pred, -0.1, 1.2)

    def compute_pde_residual(self, x_pde):
        """计算准3D热传导PDE+边界残差 (包含动态热力学参数与物理惩罚)"""
        x_pde.requires_grad_(True)
        T = self.forward(x_pde)
        
        # 一阶偏导
        grads = autograd.grad(T.sum(), x_pde, create_graph=True, allow_unused=True)[0]
        if grads is None:
            return torch.tensor(0.0, device=x_pde.device, requires_grad=True), torch.tensor(0.0, device=x_pde.device)
            
        dT_dt = grads[:, 0:1]
        dT_dx = grads[:, 1:2]
        dT_dy = grads[:, 2:3]
        dT_dz = grads[:, 3:4]
        
        # 二阶偏导 (代码简写，与原版一致)
        d2T_dx2 = autograd.grad(dT_dx.sum(), x_pde, create_graph=True, allow_unused=True)[0][:, 1:2] if autograd.grad(dT_dx.sum(), x_pde, create_graph=True, allow_unused=True)[0] is not None else torch.zeros_like(dT_dx)
        d2T_dy2 = autograd.grad(dT_dy.sum(), x_pde, create_graph=True, allow_unused=True)[0][:, 2:3] if autograd.grad(dT_dy.sum(), x_pde, create_graph=True, allow_unused=True)[0] is not None else torch.zeros_like(dT_dy)
        d2T_dz2 = autograd.grad(dT_dz.sum(), x_pde, create_graph=True, allow_unused=True)[0][:, 3:4] if autograd.grad(dT_dz.sum(), x_pde, create_graph=True, allow_unused=True)[0] is not None else torch.zeros_like(dT_dz)
        
        z = x_pde[:, 3:4]
        P_gas = x_pde[:, 6:7]
        
        # --- 优化1: 引入随温度动态变化的物性参数 ---
        # 假设 T 归一化前对应大约 40K 的温升 (25℃ - 65℃)
        T_physical_rise = T * 40.0 
        # 热导率随温度升高而降低 (-0.5%/K)
        k_cond_dynamic = self.k_cond_base * (1.0 - 0.005 * T_physical_rise)
        # 比热容随温度升高而增加 (+0.2%/K)
        rho_cp_dynamic = self.rho_cp_base * (1.0 + 0.002 * T_physical_rise)
        
        # --- 优化3: 动态内阻与单调性惩罚 (Monotonicity Prior) ---
        # 内阻 R_in 随温度升高指数下降
        R_in_dynamic = self.R_in_base.abs() * torch.exp(-self.R_in_alpha * T)
        # 单调性惩罚：如果网络尝试将 alpha 变成负数（即内阻随温度升高），则给予严厉惩罚
        loss_mono = torch.relu(-self.R_in_alpha) * 10.0
        
        # 空间异质性生热模型
        spatial_q = 1.0 + 0.5 * z**2
        Q_gen = spatial_q * (0.1 * R_in_dynamic + 0.05 * P_gas.abs())
        
        # PDE 残差 (使用动态参数)
        thermal_diffusivity = k_cond_dynamic / rho_cp_dynamic
        residual = dT_dt - thermal_diffusivity * (d2T_dx2 + d2T_dy2 + d2T_dz2) - (Q_gen / rho_cp_dynamic)

        # 边界条件
        surface_mask = x_pde[:, 1:2] > 0.99
        if surface_mask.any():
            dT_dn_surface = dT_dx[surface_mask]
            T_surface = T[surface_mask]
            bc_residual = -k_cond_dynamic[surface_mask] * dT_dn_surface - self.h_conv.abs() * T_surface
            bc_loss = torch.mean(bc_residual**2)
        else:
            bc_loss = torch.tensor(0.0, device=x_pde.device)
        
        residual = torch.clamp(residual, -1e2, 1e2)
        pde_loss = torch.mean(residual**2)
        
        total_pde_loss = pde_loss + 0.1 * bc_loss + loss_mono.mean()
        return total_pde_loss, loss_mono.mean()