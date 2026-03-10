import torch
import torch.nn as nn
import torch.autograd as autograd


class AMAdaptiveSelfAttention(nn.Module):
    """Feature-wise self-attention with adaptive residual gating."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        # h: [batch, hidden_dim]
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Build feature-feature attention map: [B, H, H]
        scale = float(self.hidden_dim) ** 0.5
        attn_scores = torch.matmul(q.unsqueeze(-1), k.unsqueeze(1)) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v.unsqueeze(-1)).squeeze(-1)

        gate = torch.sigmoid(self.gate_proj(h))
        fused = h + gate * self.dropout(self.out_proj(context))
        return self.norm(fused)

class KANLayer(nn.Module):
    """简化版 KAN 层，使用 RBF 基函数"""
    def __init__(self, in_dim, out_dim, num_centers=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.linspace(0, 1, num_centers).view(1, 1, num_centers))
        # widths 用 log 参数化保证正数，避免数值不稳定
        self.log_widths = nn.Parameter(torch.zeros(1, in_dim, num_centers))
        # 权重初始化更稳定 (Xavier/He 初始化)
        self.weights = nn.Parameter(torch.randn(out_dim, in_dim, num_centers) * (1.0 / (in_dim * num_centers) ** 0.5))

    def forward(self, x):
        # x: [batch, in_dim]
        x_ext = x.unsqueeze(-1) # [batch, in_dim, 1]
        widths = torch.exp(self.log_widths)  # 确保 widths > 0
        basis = torch.exp(-widths * (x_ext - self.centers)**2) # [batch, in_dim, centers]
        output = torch.einsum('bic, oic -> bo', basis, self.weights)
        return output

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
        # KAN 主干网络 [t, x, y, z, I, V, P] -> T
        self.kan1 = KANLayer(input_dim, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, output_dim)
        self.use_am_attention = bool(use_am_attention)
        self.am_attention = AMAdaptiveSelfAttention(hidden_dim, dropout=am_dropout)
        
        # --- 物理参数动态辨识 (Learnable Parameters) ---
        self.R_in = nn.Parameter(torch.tensor([0.05], requires_grad=True)) 
        self.h_conv = nn.Parameter(torch.tensor([15.0], requires_grad=True))
        
        # 物理常量 (预设)
        self.rho_cp = 2600.0 * 850.0 
        self.k_cond = 3.0

    def forward(self, x):
        h1 = torch.tanh(self.kan1(x))
        if self.use_am_attention:
            h1 = self.am_attention(h1)
        # Target temperature is min-max normalized to [0, 1].
        return torch.sigmoid(self.kan2(h1))

    def compute_pde_residual(self, x_pde):
        """计算准3D热传导PDE+边界残差 (改进数值稳定性)"""
        x_pde.requires_grad_(True)
        T = self.forward(x_pde)
        
        # 一阶偏导
        grads = autograd.grad(T.sum(), x_pde, create_graph=True, allow_unused=True)[0]
        if grads is None:
            return torch.tensor(0.0, device=x_pde.device, requires_grad=True)
            
        dT_dt = grads[:, 0:1]
        dT_dx = grads[:, 1:2]
        dT_dy = grads[:, 2:3]
        dT_dz = grads[:, 3:4]
        
        # 二阶偏导 (对一阶导数继续求导)
        d2T_dx2 = autograd.grad(dT_dx.sum(), x_pde, create_graph=True, allow_unused=True)[0]
        if d2T_dx2 is not None:
            d2T_dx2 = d2T_dx2[:, 1:2]
        else:
            d2T_dx2 = torch.zeros_like(dT_dx)
            
        d2T_dy2 = autograd.grad(dT_dy.sum(), x_pde, create_graph=True, allow_unused=True)[0]
        if d2T_dy2 is not None:
            d2T_dy2 = d2T_dy2[:, 2:3]
        else:
            d2T_dy2 = torch.zeros_like(dT_dy)
            
        d2T_dz2 = autograd.grad(dT_dz.sum(), x_pde, create_graph=True, allow_unused=True)[0]
        if d2T_dz2 is not None:
            d2T_dz2 = d2T_dz2[:, 3:4]
        else:
            d2T_dz2 = torch.zeros_like(dT_dz)
        
        # 提取多模态输入 [t, x, y, z, I, V, P]
        z = x_pde[:, 3:4]
        P_gas = x_pde[:, 6:7]
        
        # 空间异质性生热模型 (假设极耳 z=1 处产热权重更高)
        spatial_q = 1.0 + 0.5 * z**2
        Q_gen = spatial_q * (0.1 * self.R_in.abs() + 0.05 * P_gas.abs())  # 避免负值
        
        # 将方程除以 rho*Cp，避免巨大常数造成损失量级失衡。
        thermal_diffusivity = self.k_cond / self.rho_cp
        residual = dT_dt - thermal_diffusivity * (d2T_dx2 + d2T_dy2 + d2T_dz2) - (Q_gen / self.rho_cp)

        # 简化边界条件: x=1 的表面满足牛顿冷却 (T_amb 归一化后近似 0)
        surface_mask = x_pde[:, 1:2] > 0.99
        if surface_mask.any():
            dT_dn_surface = dT_dx[surface_mask]
            T_surface = T[surface_mask]
            bc_residual = -self.k_cond * dT_dn_surface - self.h_conv.abs() * T_surface
            bc_loss = torch.mean(bc_residual**2)
        else:
            bc_loss = torch.tensor(0.0, device=x_pde.device)
        
        # 数值稳定性检查
        residual = torch.clamp(residual, -1e2, 1e2)
        pde_loss = torch.mean(residual**2)
        return pde_loss + 0.1 * bc_loss