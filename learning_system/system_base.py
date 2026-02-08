import os
import gc
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from inner_loop_optimizers import LSLRGradientDescentLearningRule
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def set_torch_seed(seed):
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class RegressorBase(nn.Module):
    def __init__(self, input_shape, args):

        super(RegressorBase, self).__init__()
        self.args = args
        self.batch_size = args.meta_batch_size
        self.current_epoch = 0
        self.input_shape = input_shape
        self.rng = set_torch_seed(seed=0)

        self.args.rng = self.rng

    def _get_device(self):
        dev = getattr(self.args, 'device', None)
        if dev is not None:
            return dev
        if hasattr(self, 'regressor'):
            try:
                model = self.regressor
                if hasattr(model, 'module'):
                    model = model.module
                p = next(model.parameters())
                return p.device
            except Exception:
                pass
        return torch.device('cpu')

    def _compute_loss(self, pred, y, split):
        # 用 split 选择支持集
        if split.dim() == 0:
            split = split.unsqueeze(0)

        sup_mask = split.bool()

        if sup_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device), pred

        pred_sup = pred[sup_mask]
        y_sup = y[sup_mask]

        # shape 对齐
        try:
            task_type = getattr(self.args, "task_type", "regression")
        except Exception:
            task_type = "regression"

        if pred_sup.dim() == 1:
            pred_sup = pred_sup.unsqueeze(1)
        if y_sup.dim() == 1:
            y_sup = y_sup.unsqueeze(1)

        valid_mask = ~torch.isnan(y_sup)

        if not valid_mask.any():
            # 没有有效标签时返回 0（可反向传播的标量）
            return torch.tensor(0.0, device=pred.device, requires_grad=True), pred

        if task_type == "classification":
            # flatten 后按 mask 选择有效元素
            pred_flat = pred_sup.view(-1)
            y_flat = y_sup.view(-1).float()
            vm = valid_mask.view(-1)

            pred_f = pred_flat[vm]
            y_f = y_flat[vm]

            if pred_f.numel() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True), pred

            # ensure tensors on same device
            device = pred_f.device
            y_f = y_f.to(device)
            pred_f = pred_f.to(device)

            # dynamic pos_weight: neg / pos  (safe)
            use_pos_weight = bool(getattr(self.args, "use_pos_weight", False))
            eps = 1e-6
            pos_weight_tensor = None
            if use_pos_weight:
                pos = float(torch.sum(y_f == 1.0).item())
                neg = float(torch.sum(y_f == 0.0).item())
                if pos > 0:
                    pw = neg / (pos + eps)
                    # BCEWithLogitsLoss expects a Tensor for pos_weight (same device)
                    pos_weight_tensor = torch.tensor(pw, dtype=torch.float32, device=device)
                else:
                    # no positive samples in this batch -> fallback to unweighted
                    pos_weight_tensor = None

            focal_gamma = float(getattr(self.args, "focal_gamma", 0.0))
            if focal_gamma is not None and focal_gamma > 0.0:
                # focal loss variant (binary)
                # per-sample BCE (no reduction), then focal weight
                bce_per = F.binary_cross_entropy_with_logits(pred_f, y_f, reduction='none')
                p = torch.sigmoid(pred_f)
                # p_t = p if y=1 else 1-p
                p_t = p * y_f + (1 - p) * (1 - y_f)
                focal_weight = (1.0 - p_t).pow(focal_gamma)
                # optional alpha balancing (not using by default)
                alpha = float(getattr(self.args, "focal_alpha", 1.0))
                # if you want class-dependent alpha, could apply here.
                loss = (focal_weight * bce_per * alpha).mean()
                return loss, pred
            else:
                # use BCEWithLogitsLoss, possibly with pos_weight
                if pos_weight_tensor is not None:
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='mean')
                    loss = loss_fn(pred_f, y_f)
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                    loss = loss_fn(pred_f, y_f)
                return loss, pred

        # ---- regression branch ----
        else:
            # criterion = nn.SmoothL1Loss(beta=1.0, reduction='mean')
            criterion = nn.MSELoss(reduction='mean')
            loss = criterion(pred_sup, y_sup)
            return loss, pred


    def post_init(self, args):
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(args=args,
                                                                    init_learning_rate=args.update_lr,
                                                                    total_num_inner_loop_steps=self.args.num_updates,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.regressor.named_parameters()))
        self.temp = 0.2
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape, value.requires_grad)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)

        lslr_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # 关键：识别 LSLR 参数（通常名字里带 learning_rates 或 alpha）
            if "learning_rates" in name or "inner_loop_optimizer" in name:
                lslr_params.append(param)
                # print(f"Protected from weight decay: {name}") # 调试用
            else:
                other_params.append(param)

        # 2. 定义优化器分组
        optimizer_grouped_parameters = [
            {'params': other_params, 'weight_decay': args.weight_decay},
            {'params': lslr_params, 'weight_decay': 0.0}  # 保护 LSLR 参数，不衰减！
        ]

        print(f" >>> [Debug] LSLR Params protected: {len(lslr_params)} tensors")
        print(f" >>> [Debug] Other Params decaying: {len(other_params)} tensors")

        if len(lslr_params) == 0:
            print("!!! WARNING: No LSLR params found! Check parameter names. !!!")

        # 3. 初始化优化器
        opt = optim.Adam(optimizer_grouped_parameters, lr=args.meta_lr, amsgrad=False)
        self.optimizer = opt
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,
                                                              T_max=self.args.metatrain_iterations,
                                                              eta_min=self.args.min_learning_rate)

    def cossim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def get_sim_matrix(self, a, b):
        a_bool = (a > 0.).float()
        b_bool = (b > 0.).float()
        and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
        or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
        sim = and_res / or_res
        return sim

    def robust_square_error(self, a, b, topk_idx):
        abs_diff = torch.abs(a - b)
        square_mask = (abs_diff <= 2).float()
        linear_mask = 1. - square_mask
        square_error = (a - b) ** 2
        linear_error = 1. * (abs_diff - 2) + 4
        loss = square_error * square_mask + linear_error * linear_mask

        loss_select = torch.gather(loss, 0, topk_idx)
        return torch.mean(loss_select)

    def get_per_step_loss_importance_vector(self):
        try:
            num_updates = int(self.args.num_updates)
        except Exception:
            num_updates = 1

            # 默认均等权重（numpy）
        loss_weights = np.ones(shape=(num_updates,), dtype=np.float32) * (1.0 / max(1, num_updates))

        # 计算衰减参数（防止除零）
        multi_step_epochs = getattr(self.args, "multi_step_loss_num_epochs", 1)
        if multi_step_epochs is None or multi_step_epochs <= 0:
            multi_step_epochs = 1
        decay_rate = 1.0 / max(1, num_updates) / multi_step_epochs
        min_value_for_non_final_losses = 0.03 / max(1, num_updates)

        # 调整非最终步骤的权重
        current_epoch = getattr(self, "current_epoch", 0)
        for i in range(len(loss_weights) - 1):
            curr_value = max(loss_weights[i] - (current_epoch * decay_rate),
                             min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        # 调整最终步骤的权重
        last_val = min(
            loss_weights[-1] + (current_epoch * (num_updates - 1) * decay_rate),
            1.0 - ((num_updates - 1) * min_value_for_non_final_losses)
        )
        loss_weights[-1] = last_val

        # 选择 device：优先用 model 的 device（如果有 _get_device），否则用 self.device 或 CPU/GPU 自动选择
        if hasattr(self, "_get_device"):
            device = self._get_device()
        else:
            device = getattr(self, "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将 numpy -> torch.tensor（注意小写 tensor，支持 dtype 和 device）
        loss_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32, device=device)

        return loss_weights_tensor

    def get_weighted_training_param(self):
        params = self.regressor.named_parameters()
        for name, param in params:
            if param.requires_grad:
                if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                    yield param

    def get_inner_loop_parameter_dict(self, params):
        param_dict = {}
        for name, param in params:
            if param.requires_grad and ("predict" in name or "ffi_model" in name):
                param_dict[name] = param
        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx, sup_number):
        # 1. 计算梯度
        names = list(names_weights_copy.keys())
        params = [names_weights_copy[k] for k in names]

        # 开启 create_graph 是 LSLR 学习的关键！
        grads = torch.autograd.grad(loss, params, create_graph=use_second_order, allow_unused=True)

        updated_weights = {}
        for name, grad, param in zip(names, grads, params):
            if grad is None:
                updated_weights[name] = param
                continue

            lslr_key = name.replace(".", "-")

            lr_tensor = None

            # 尝试从可学习字典中获取 alpha
            if hasattr(self, "inner_loop_optimizer") and \
                    hasattr(self.inner_loop_optimizer, "names_learning_rates_dict"):

                # 获取整个参数字典
                lr_dict = self.inner_loop_optimizer.names_learning_rates_dict

                if lslr_key in lr_dict:
                    # 【重要】我们要取的是由 current_step_idx 索引的那一个切片
                    # shape 可能是 (T, ) 或者 (T, *param_shape)
                    # 假设你初始化时 shape 是 (total_steps + 1)，这里取第 step 个标量
                    full_alpha = lr_dict[lslr_key]

                    # 确保 current_step_idx 不越界
                    step_idx = min(current_step_idx, len(full_alpha) - 1)
                    lr_tensor = full_alpha[step_idx]

                    # print(f"DEBUG: Using Learnable LR for {name}") # 调试时可打开

            # ======================================================

            # 如果没找到可学习的 LR，或者发生了 Key Mismatch
            if lr_tensor is None:
                # print(f"DEBUG: Fallback to fixed LR for {name}") # 调试时可打开
                # 使用默认的标量学习率（这会导致 LSLR 失效！）
                lr_tensor = self.args.update_lr

            # 执行更新
            # 注意：这里构成了计算图：new_p 依赖于 lr_tensor
            updated_weights[name] = param - lr_tensor * grad

        return updated_weights

    def get_across_task_loss_metrics(self, total_losses, loss_weights):
        if hasattr(self, "_model_device"):
            device = self._model_device()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vals = []
        for l in total_losses:
            if l is None:
                continue
            if isinstance(l, torch.Tensor):
                vals.append(l.to(device))
            else:
                try:
                    vals.append(torch.tensor(float(l), device=device))
                except Exception:
                    continue

        if len(vals) == 0:
            return {'loss': torch.tensor(0.0, device=device)}

        loss_vec = torch.stack(vals)
        return {'loss': loss_vec.mean()}

    def get_init_weight(self):
        init_weight = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
        init_weight = init_weight["layer_dict.linear.weights"]
        return init_weight

    def inner_loop(self, x_task, y_task, assay_idx, split, is_training_phase, epoch, num_steps):
        task_losses = []
        per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
        support_loss_each_step = []

        # support count
        try:
            sup_num = int(torch.sum(split).item()) if torch.is_tensor(split) else int(sum(split))
        except Exception:
            sup_num = 0

        # initial snapshot of params (dict of tensors)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
        use_second_order = bool(self.args.second_order and epoch > self.args.first_order_to_second_order_epoch)
        last_target_preds = None

        # device
        device = self._model_device() if hasattr(self, "_model_device") else (
            getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )

        # ensure y tensor on device, shape (n,)
        if not torch.is_tensor(y_task):
            y_tensor = torch.tensor(list(y_task) if hasattr(y_task, "__iter__") else [y_task],
                                    dtype=torch.float32, device=device).view(-1)
        else:
            y_tensor = y_task.to(device).float().view(-1)

        # support mask on device (bool)
        if torch.is_tensor(split):
            support_mask = split.to(device).bool().view(-1)
        else:
            support_mask = torch.tensor(list(split) if hasattr(split, "__iter__") else [split],
                                        dtype=torch.bool, device=device).view(-1)

        # per-task normalization from support (only for regression)
        eps = 1e-8
        task_type = getattr(self.args, "task_type", "regression")

        # 1. 获取当前 Support Set 的样本数量
        # 注意: support_mask 是 bool 类型，sum() 后得到的是 Tensor，需要转 item()
        current_sup_num = int(support_mask.sum().item())

        y_task_norm = y_tensor
        y_mean = torch.tensor(0.0, device=device)
        y_std = torch.tensor(1.0, device=device)

        # inner loop steps
        for num_step in range(num_steps):
            # compute support loss (labels are normalized)
            support_loss, support_preds = self.net_forward(
                x=x_task, y=y_task_norm, assay_idx=assay_idx, split=split,
                is_support=True, weights=names_weights_copy,
                backup_running_statistics=(num_step == 0), training=True, num_step=num_step
            )

            try:
                support_loss_val = float(support_loss.detach().cpu().item())
            except Exception:
                try:
                    support_loss_val = float(support_loss)
                except Exception:
                    support_loss_val = 0.0
            support_loss_each_step.append(support_loss_val)

            # 1. 决定是否更新 (基于 Support Loss 是否异常)
            do_update = True
            init_loss = support_loss_each_step[0] if len(support_loss_each_step) > 0 else None
            loss_ratio_threshold = getattr(self.args, "support_loss_skip_ratio", 10.0)

            if init_loss is not None and support_loss_val is not None:
                # 如果 Loss 突然暴涨，可能是梯度爆炸，跳过这一步更新
                if support_loss_val >= init_loss * loss_ratio_threshold:
                    do_update = False

            # 2. 执行更新 (Fix: 只要 do_update 为 True 就更新，不再依赖 debug flag)
            if do_update:
                # 调用我们之前修好的 apply_inner_loop_update
                # 这里是你看到 DEBUG 打印的地方！
                names_weights_copy_new = self.apply_inner_loop_update(
                    loss=support_loss,
                    names_weights_copy=names_weights_copy,
                    use_second_order=(use_second_order if is_training_phase else False),
                    current_step_idx=num_step,
                    sup_number=current_sup_num
                )
            else:
                names_weights_copy_new = names_weights_copy

            # 赋值给下一步
            names_weights_copy = names_weights_copy_new

            # compute target/query loss using adapted weights (for multi-step or last-step)
            if is_training_phase:
                is_multi_step_optimize = self.args.use_multi_step_loss_optimization and epoch < self.args.multi_step_loss_num_epochs
                is_last_step = num_step == (self.args.num_updates - 1)
                if is_multi_step_optimize or is_last_step:
                    target_loss, target_preds = self.net_forward(
                        x=x_task, y=y_task_norm, assay_idx=assay_idx, split=split,
                        is_support=False, weights=names_weights_copy,
                        backup_running_statistics=False, training=True, num_step=num_step
                    )
                    last_target_preds = target_preds
                    if is_multi_step_optimize:
                        task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                    elif is_last_step:
                        task_losses.append(target_loss)

            # ensure at least one differentiable loss available
            if len(task_losses) == 0:
                try:
                    device_local = self._get_device() if hasattr(self, "_get_device") else device
                    tensor_zero = torch.tensor(0.0, device=device_local, dtype=torch.float32, requires_grad=True)
                except Exception:
                    tensor_zero = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                task_losses.append(tensor_zero)

        # final fallback for last_target_preds
        if last_target_preds is None:
            last_target_preds = torch.zeros_like(y_tensor, device=device)

        if not torch.is_tensor(last_target_preds):
            last_target_preds = torch.tensor(last_target_preds, dtype=torch.float32, device=device)

        # un-normalize predictions (only for regression)
        if task_type == "regression":
            last_target_preds = last_target_preds.view(-1) * (y_std + eps) + y_mean
        else: # classification
            last_target_preds = last_target_preds.view(-1)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return names_weights_copy, support_loss_each_step, task_losses, last_target_preds

    def get_metric(self, y, y_pred, split):
        # 获取任务类型
        task_type = getattr(self.args, "task_type", "regression")

        # 区分 Support (训练) 和 Query (测试) 的索引
        # split: 1 为 Support, 0 为 Query
        if split.dtype == torch.bool:
            split = split.int()

        sup_idx = torch.nonzero(split)[:, 0]  # 支持集索引
        tgt_idx = torch.nonzero(1 - split)[:, 0]  # 查询集索引

        # 提取数据 (Tensor)
        y_true_all = y.detach().cpu().numpy()
        y_pred_all = y_pred.detach().cpu().numpy()

        # Support Set 数据
        y_sup_true = y_true_all[sup_idx.cpu().numpy()]
        y_sup_pred = y_pred_all[sup_idx.cpu().numpy()]

        # Query Set 数据
        y_query_true = y_true_all[tgt_idx.cpu().numpy()]
        y_query_pred = y_pred_all[tgt_idx.cpu().numpy()]

        # 如果没有 Query 数据，返回空
        if len(y_query_true) == 0:
            return {"auc": 0.0, "prc_auc": 0.0, "acc": 0.0, "f1": 0.0}

        # ================== 分类任务逻辑 ==================
        if task_type == "classification":
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, average_precision_score

            # 1. Logits -> Probabilities
            probs_query = 1.0 / (1.0 + np.exp(-np.clip(y_query_pred, -20, 20)))

            # 2. 计算 ROC-AUC
            auc = 0.5
            if len(np.unique(y_query_true)) > 1:
                try:
                    auc = roc_auc_score(y_query_true, probs_query)
                except:
                    auc = 0.5

            # 3. 【新增】计算 PRC-AUC (Average Precision)
            prc_auc = 0.0
            if len(np.unique(y_query_true)) > 1:
                try:
                    # Average Precision 就是 PRC 曲线下面积
                    prc_auc = average_precision_score(y_query_true, probs_query)
                except:
                    # 如果只有一种类别，PRC通常未定义或为0
                    prc_auc = 0.0
            else:
                # 极端情况：Query Set 全是负样本（在MUV中常见）
                # 此时 PRC-AUC 无法计算，或者可以视为 0
                prc_auc = 0.0

            # 4. 计算其他指标 (Acc, F1) - 这里简单用 0.5 阈值，或者保留你之前的动态阈值逻辑
            pred_bin_query = (probs_query >= 0.5).astype(int)
            acc = accuracy_score(y_query_true, pred_bin_query)
            f1 = f1_score(y_query_true, pred_bin_query, zero_division=0)

            return {
                "auc": float(auc),
                "prc_auc": float(prc_auc),
                "acc": float(acc),
                "f1": float(f1),
                "pred": list(probs_query),
                "true": list(y_query_true)
            }

            # # (1) 将 Logits 转为概率 (Sigmoid)
            # probs_all = 1.0 / (1.0 + np.exp(-np.clip(y_pred_all, -20, 20)))
            #
            # prob_sup = probs_all[sup_idx.cpu().numpy()]
            # prob_query = probs_all[tgt_idx.cpu().numpy()]
            #
            # # 动态寻找最佳阈值 (基于 Support Set)
            # best_thr = 0.5  # 默认值
            #
            # # 只有当 Support Set 中同时存在正负样本时，才能计算 F1 并寻找阈值
            # if len(np.unique(y_sup_true)) > 1:
            #     try:
            #         # 计算 Precision-Recall 曲线
            #         prec, rec, ths = precision_recall_curve(y_sup_true, prob_sup)
            #         # 计算 F1 分数序列
            #         with np.errstate(divide='ignore', invalid='ignore'):
            #             f1s = 2.0 * prec * rec / (prec + rec)
            #         f1s = np.nan_to_num(f1s)  # 处理除零导致的 NaN
            #
            #         # 找到最大 F1 对应的下标
            #         best_idx = np.argmax(f1s)
            #
            #         # 获取对应阈值 (ths长度比prec/rec少1，通常取best_idx即可，防止越界)
            #         if best_idx < len(ths):
            #             best_thr = ths[best_idx]
            #     except Exception as e:
            #         # 如果报错，回退到 0.5
            #         best_thr = 0.5
            #
            # # 使用找到的阈值 (best_thr) 对 Query Set 进行预测
            # pred_bin_query = (prob_query >= best_thr).astype(int)
            #
            # # 计算 Query Set 的指标
            # auc = 0.5
            # if len(np.unique(y_query_true)) > 1:
            #     try:
            #         auc = roc_auc_score(y_query_true, prob_query)
            #     except:
            #         auc = 0.5
            #
            # acc = accuracy_score(y_query_true, pred_bin_query)
            # f1 = f1_score(y_query_true, pred_bin_query, zero_division=0)
            #
            # return {"auc": float(auc), "acc": float(acc), "f1": float(f1),
            #         "pred": list(prob_query), "true": list(y_query_true), "threshold": float(best_thr)}

        # ================== 回归任务 ==================
        else:
            # 计算 Support Set 均值 (用于 R2os)
            if len(y_sup_true) > 0:
                y_train_mean = np.mean(y_sup_true)
            else:
                y_train_mean = np.mean(y_query_true)

            rmse = np.sqrt(np.mean((y_query_true - y_query_pred) ** 2))
            mae = np.mean(np.abs(y_query_true - y_query_pred))

            # Standard R2
            ss_res = ((y_query_true - y_query_pred) ** 2).sum()
            ss_tot = ((y_query_true - y_query_true.mean()) ** 2).sum() + 1e-8
            standard_r2 = 1 - (ss_res / ss_tot)

            # Pearson R2
            if len(y_query_true) > 1:
                corr = np.corrcoef(y_query_true, y_query_pred)[0, 1]
                pearson_r2 = corr ** 2 if not np.isnan(corr) else 0.0
            else:
                pearson_r2 = 0.0

            # R2os
            numerator = ((y_query_true - y_query_pred) ** 2).sum()
            denominator = ((y_query_true - y_train_mean) ** 2).sum() + 1e-8
            R2os = 1 - (numerator / denominator)

            return {
                "r2": float(standard_r2),
                "pearson_r2": float(pearson_r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "R2os": float(R2os),
                "y_train_mean": float(y_train_mean),
                "pred": list(y_query_pred),
                "true": list(y_query_true)
            }

    def forward(self, data_batch, epoch, num_steps, is_training_phase, **kwargs):
        raise NotImplementedError

    def net_forward(self, x, y, assay_idx, split, weights, is_support, backup_running_statistics, training, num_step):
        raise NotImplementedError

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def meta_update(self, loss):
        # === 针对 QM9：开启梯度累加 (Batch=4 -> 等效 16) ===
        if self.args.datasource == 'qm9':
            if not hasattr(self, '_acc_counter'):
                self._acc_counter = 0

            # 定义累加步数
            acc_steps = 4

            # 1. Loss 缩放 (防止梯度过大)
            loss_scalar = loss.sum() / acc_steps

            # 2. 累积梯度 (不清零)
            loss_scalar.backward()

            self._acc_counter += 1

            # 3. 凑够 4 步才更新一次
            if self._acc_counter % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()  # 更新完才清空

        # === 其他数据集：保持原样 ===
        else:
            self.optimizer.zero_grad()
            loss_scalar = loss.sum()
            loss_scalar.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds, _, _ = self.forward(data_batch=data_batch, epoch=epoch,
                                                           num_steps=self.args.num_updates,
                                                           is_training_phase=True)
        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        # self.optimizer.zero_grad()
        # self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        if self.training:
            self.eval()
        losses, per_task_target_preds, final_weights, per_task_metrics = self.forward(data_batch=data_batch,
                                                                                epoch=self.current_epoch,
                                                                                num_steps=self.args.test_num_updates,
                                                                                is_training_phase=False)
        return losses, per_task_target_preds, final_weights, per_task_metrics

    def save_model(self, model_save_dir, state):
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):

        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
