import torch

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

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # 2. PDE Loss (物理一致性) - 低频启用以提升速度和稳定性
        pde_enabled_this_batch = (
            use_pde
            and (epoch is None or epoch > pde_warmup_epochs)
            and (pde_every_n_batches <= 1 or batch_idx % pde_every_n_batches == 0)
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # 1. Data Loss
            y_pred = model(x_batch)
            loss_data = criterion(y_pred, y_batch)

            # 2. PDE Loss (物理一致性) - 低频启用以提升速度和稳定性
            if pde_enabled_this_batch:
                try:
                    if pde_subset_size and pde_subset_size > 0 and pde_subset_size < x_batch.size(0):
                        pde_idx = torch.randperm(x_batch.size(0), device=x_batch.device)[:pde_subset_size]
                        x_pde = x_batch[pde_idx]
                    else:
                        x_pde = x_batch
                    loss_pde = model.compute_pde_residual(x_pde.clone().detach().requires_grad_(True))
                    pde_active_steps += 1
                except Exception:
                    pde_exception_steps += 1
                    loss_pde = torch.tensor(0.0, device=device)
            else:
                loss_pde = torch.tensor(0.0, device=device)

            # 3. 自适应物理损失权重 (更轻量，避免每步NTK高开销)
            if pde_enabled_this_batch and loss_pde.item() > 0:
                base_w_p = max_pde_weight * pde_schedule
                if use_adaptive_pde_weight:
                    # Use sqrt ratio to reduce abrupt swings and cap by max_pde_weight.
                    ratio = torch.sqrt((loss_data.detach() + 1e-8) / (loss_pde.detach() + 1e-8)).clamp(0.1, 3.0)
                    w_p = torch.clamp(ratio * base_w_p, 0.0, max_pde_weight)
                else:
                    w_p = torch.tensor(base_w_p, device=device)
                w_d = torch.tensor(1.0, device=device)
            else:
                w_d, w_p = torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

            # 4. 反向传播
            loss = w_d * loss_data + w_p * loss_pde
        
        # 检查 NaN
        if torch.isnan(loss):
            print(f"⚠️  Warning: NaN detected in loss! data_loss={loss_data.item()}, pde_loss={loss_pde.item()}, w_d={w_d.item()}, w_p={w_p.item()}")
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

        # Print sparse, meaningful batch logs to confirm training is progressing.
        if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches):
            avg_loss = total_loss / batch_idx
            epoch_text = f"{epoch:03d}" if isinstance(epoch, int) else "---"
            print(
                f"  [Epoch {epoch_text}] Batch {batch_idx}/{num_batches} "
                f"| Loss: {loss.item():.6f} | Data: {loss_data.item():.6f} "
                f"| PDE: {loss_pde.item():.6f} | Avg: {avg_loss:.6f}"
            )

    if pde_active_steps > 0 or pde_exception_steps > 0:
        print(
            f"  [Epoch {epoch if epoch is not None else '-'}] PDE active steps={pde_active_steps}, "
            f"exceptions={pde_exception_steps}, schedule={pde_schedule:.3f}"
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