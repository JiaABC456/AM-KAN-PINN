import torch

def evolutionary_pde_resampling(x_batch, errors, subset_size, mutation_scale=0.02):
    """
    进化策略：时空 PDE 配点繁育
    针对离散高误差奇点，通过选择和变异生成新的物理约束点
    """
    batch_size = x_batch.size(0)
    subset_size = min(subset_size, batch_size)
    
    # 1. Selection (精英选择): 找出当前批次中误差最大的点
    # 比如我们挑选前 10% 的最难样本作为“父代”
    elite_size = max(1, int(batch_size * 0.1))
    _, elite_indices = torch.topk(errors.squeeze(), elite_size)
    elites = x_batch[elite_indices]
    
    # 2. Reproduction (繁育): 复制精英种群以填满 subset_size
    num_repeats = (subset_size // elite_size) + 1
    population = elites.repeat(num_repeats, 1)[:subset_size]
    
    # 3. Mutation (变异): 在离散点周围的高维时空空间注入变异
    # 只对连续的时空坐标 (t, x, y, z) 即前 4 列施加变异，
    # 外部工况条件 (I, V, P) 作为环境变量保持不变。
    mutation_noise = torch.randn_like(population) * mutation_scale
    mask = torch.zeros_like(population)
    mask[:, 0:4] = 1.0  # 激活 [t, x, y, z] 的变异掩码
    
    evolved_pde_points = population + mutation_noise * mask
    
    # 确保变异后的坐标不越过物理边界 (假设 t, x, y, z 归一化在 [0, 1])
    evolved_pde_points[:, 0:4] = torch.clamp(evolved_pde_points[:, 0:4], 0.0, 1.0)
    
    return evolved_pde_points.detach().requires_grad_(True)

def compute_ntk_weights(model, loss_data, loss_pde):
    """基于梯度迹(Trace)的动态权重计算 (改进数值稳定性)"""
    # 包含所有需要梯度的参数，而不仅仅是输出层
    params = [p for p in model.parameters() if p.requires_grad]
    grad_data = torch.autograd.grad(loss_data, params, retain_graph=True, allow_unused=True)
    grad_pde = torch.autograd.grad(loss_pde, params, retain_graph=True, allow_unused=True)
    
    trace_data = sum(torch.sum(g**2) for g in grad_data if g is not None)
    trace_pde = sum(torch.sum(g**2) for g in grad_pde if g is not None)
    
    # 避免除零，添加更大的 epsilon
    trace_data = torch.clamp(trace_data, min=1e-10)
    trace_pde = torch.clamp(trace_pde, min=1e-10)
    
    w_data = ((trace_data + trace_pde) / (2 * trace_data + 1e-8)).detach()
    w_pde = ((trace_data + trace_pde) / (2 * trace_pde + 1e-8)).detach()
    
    # 权重归一化到 [0.1, 1.0] 范围内，避免极端值
    w_data = torch.clamp(w_data, 0.1, 1.0)
    w_pde = torch.clamp(w_pde, 0.1, 1.0)
    
    return w_data, w_pde

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_pde=True,
    log_every=200,
    epoch=None,
    pde_every_n_batches=10,
    pde_warmup_epochs=2,
    pde_subset_size=128,
    max_pde_weight=0.3,
    pde_ramp_epochs=10,
    use_adaptive_pde_weight=True,
    amp_enabled=True,
    scaler=None,
    ema_state=None, # 新增：用于保持 EMA 状态
):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    use_amp = amp_enabled and str(device).startswith("cuda")

    if epoch is not None and pde_ramp_epochs > 0:
        pde_schedule = min(1.0, max(0.0, epoch / float(pde_ramp_epochs)))
    else:
        pde_schedule = 1.0

    pde_active_steps = 0
    pde_exception_steps = 0
    
    # 初始化 EMA 状态
    if ema_state is None:
        ema_state = {'ratio': 1.0}

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        pde_enabled_this_batch = (
            use_pde
            and (epoch is None or epoch > pde_warmup_epochs)
            and (pde_every_n_batches <= 1 or batch_idx % pde_every_n_batches == 0)
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            y_pred = model(x_batch)
            loss_data = criterion(y_pred, y_batch)
            
            # 记录当前每个点的绝对误差，用于进化算法的选择压力
            with torch.no_grad():
                pointwise_errors = torch.abs(y_pred - y_batch)

            loss_mono_val = torch.tensor(0.0, device=device)
            # Default to zero PDE loss; overwritten when PDE branch succeeds.
            loss_pde = torch.tensor(0.0, device=device)

            if pde_enabled_this_batch:
                try:
                    # 🚀 【引入进化计算】替换掉原来的 torch.randperm 随机采样
                    if pde_subset_size and pde_subset_size > 0:
                        # 开启变异率为 0.02 (约 2% 的时空偏移)
                        x_pde = evolutionary_pde_resampling(x_batch, pointwise_errors, pde_subset_size, mutation_scale=0.02)
                    else:
                        x_pde = x_batch.clone().detach().requires_grad_(True)
                        
                    # 对变异繁育出的新配点计算物理残差
                    loss_pde_raw, loss_mono_val = model.compute_pde_residual(x_pde)
                    
                    loss_pde = loss_pde_raw
                    pde_active_steps += 1
                except Exception as e:
                    pde_exception_steps += 1
                    loss_pde = torch.tensor(0.0, device=device)

            # 优化2: 使用 EMA 平滑自适应 PDE 权重 (SoftAdapt 的轻量级实现)
            if pde_enabled_this_batch and loss_pde.item() > 0:
                base_w_p = max_pde_weight * pde_schedule
                if use_adaptive_pde_weight:
                    current_ratio = (loss_data.detach() + 1e-8) / (loss_pde.detach() + 1e-8)
                    # 0.9 EMA 动量，防止单批次噪点导致的权重剧烈震荡
                    ema_state['ratio'] = 0.9 * ema_state['ratio'] + 0.1 * current_ratio.item()
                    w_p = torch.clamp(torch.tensor(ema_state['ratio']**0.5, device=device) * base_w_p, 0.0, max_pde_weight)
                else:
                    w_p = torch.tensor(base_w_p, device=device)
                w_d = torch.tensor(1.0, device=device)
            else:
                w_d, w_p = torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

            loss = w_d * loss_data + w_p * loss_pde
        
        if torch.isnan(loss):
            continue
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item()

        if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches):
            avg_loss = total_loss / batch_idx
            epoch_text = f"{epoch:03d}" if isinstance(epoch, int) else "---"
            mono_str = f" | Mono: {loss_mono_val.item():.4f}" if loss_mono_val.item() > 0 else ""
            print(
                f"  [Epoch {epoch_text}] Batch {batch_idx}/{num_batches} "
                f"| Loss: {loss.item():.6f} | Data: {loss_data.item():.6f} "
                f"| PDE: {loss_pde.item():.6f}{mono_str} | Avg: {avg_loss:.6f}"
            )

    return total_loss / len(dataloader)
    
def validate(model, dataloader, criterion, device):
    """验证集评估"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)