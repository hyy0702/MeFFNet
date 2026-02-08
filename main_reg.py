import argparse
import copy
import random

import numpy as np
import torch
import os
import gc
import re
import math
import json
import torch.nn as nn
import pandas as pd
import glob
from torch.backends import cudnn

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import dataset_constructor
from learning_system import system_selector
from torch_geometric.data import Batch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', default='chembl', type=str)
    parser.add_argument('--eval_metric', type=str, default='rmse')
    parser.add_argument('--task_type', type=str, default='regression', choices=('regression', 'classification'))

    parser.add_argument('--use_pos_weight', default=False, action='store_true', help='Use dynamic pos_weight in BCE loss for classification')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='If >0 use focal loss with this gamma (binary focal).')
    parser.add_argument('--task_filter', type=str, default='', help='Only train/test tasks containing this string. Empty means all tasks.')

    parser.add_argument('--preprocessed_dir', default=None, type=str,help = '如果指定，直接从这个目录 load .pt 文件，而不做 RDKit 预处理')
    parser.add_argument('--model_name', default='ffinet', type=str)
    parser.add_argument('--dim_w', default=2048, type=int, help='dimension of w')
    parser.add_argument('--hid_dim', default=2048, type=int, help='dimension of w')
    parser.add_argument('--num_stages', default=2, type=int, help='num stages')
    parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
    parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
    parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
    parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
    parser.add_argument('--second_order', default=1, type=int, help='second_order')
    parser.add_argument('--first_order_to_second_order_epoch', default=10, type=int, help='first_order_to_second_order_epoch')

    parser.add_argument('--transfer_lr', default=0.004, type=float,  help='transfer_lr')
    parser.add_argument('--test_sup_num', default=4, type=float)
    parser.add_argument('--test_repeat_num', default=10, type=int)

    parser.add_argument('--test_write_file', default="./test_result_debug/", type=str)
    parser.add_argument('--expert_test', default="", type=str)
    parser.add_argument('--similarity_cut', default=0.5, type=float)

    parser.add_argument('--train_seed', default=1111, type=int, help='train_seed')
    parser.add_argument('--val_seed', default=1111, type=int, help='val_seed')
    parser.add_argument('--test_seed', default=1111, type=int, help='test_seed')

    parser.add_argument('--metatrain_iterations', default=80, type=int,help='number of metatraining iterations.')
    parser.add_argument('--meta_batch_size', default=1, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--min_learning_rate', default=0.0001, type=float, help='min_learning_rate')
    parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.00015, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=1, type=int, help='num_updates in maml')
    parser.add_argument('--test_num_updates', default=1, type=int, help='num_updates in maml')
    parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
    parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')


    ## Logging, saving, and testing options
    parser.add_argument('--logdir', default='', type=str,help='directory for summaries and checkpoints.')
    parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

    parser.add_argument('--cross_test', default=False, action='store_true')
    parser.add_argument('--use_byhand_lr', default=False, action='store_true')
    parser.add_argument('--begin_lrloss_epoch', default=50, type=int)

    # Backbone
    parser.add_argument('--feature_per_layer', type=int, nargs='+', default=[66, 128, 256],help='FFiNet每层的特征维度列表')
    parser.add_argument('--num_heads', type=int, default=4,help='FFiNet多头注意力头数')
    parser.add_argument('--pred_hidden_dim', type=int, default=512,help='FFiNet预测层隐藏维度')
    parser.add_argument('--pred_dropout', type=float, default=0.5,help='FFiNet预测层Dropout概率')
    parser.add_argument('--pred_layers', type=int, default=2,help='FFiNet预测层层数')
    parser.add_argument('--activation', default=nn.PReLU(),help='激活函数，如 nn.ReLU()')
    parser.add_argument('--residual', type=bool, default=True,help='是否使用残差连接')
    parser.add_argument('--num_tasks', type=int, default=1,help='预测任务数')
    parser.add_argument('--bias', type=bool, default=True,help='是否使用偏置项')
    parser.add_argument('--dropout', type=float, default=0.1,help='全局Dropout概率')
    parser.add_argument('--in_channels', type=int, default=66, help='Input channels for the model')

    # ablation
    parser.add_argument('--no_1hop', default=False, action='store_true', help='Disable 1-hop spatial encoding')
    parser.add_argument('--no_2hop', default=False, action='store_true', help='Disable 2-hop spatial encoding')
    parser.add_argument('--no_3hop', default=False, action='store_true', help='Disable 3-hop spatial encoding')
    parser.add_argument('--no_axial', default=False, action='store_true', help='Disable Axial Attention, use simple sum instead.')

    return parser

def run_train_iter(model, *args, **kwargs):
    # support DataParallel and DistributedDataParallel wrappers
    if hasattr(model, 'module'):
        return model.module.run_train_iter(*args, **kwargs)
    else:
        return model.run_train_iter(*args, **kwargs)

def run_validation_iter(model, *args, **kwargs):
    if hasattr(model, 'module'):
        return model.module.run_validation_iter(*args, **kwargs)
    else:
        return model.run_validation_iter(*args, **kwargs)

def train(args, model, dataloader, local_rank=0, world_size=1):
    Print_Iter = 200

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    begin_epoch = 0
    if args.resume == 1:
        begin_epoch = args.test_epoch + 1
    # Only rank0 runs initial test to get baseline
    last_test_result = None
    try:
        rank0 = (local_rank == 0)
    except NameError:
        rank0 = True
    if rank0:
        _, last_test_result = test(args, begin_epoch, model, dataloader, is_test=False)

    beat_epoch = -1
    print_loss = 0.0
    print_step = 0
    for epoch in range(begin_epoch, args.metatrain_iterations):
        raw_train_batches = list(dataloader.get_train_batches())

        train_data_all = []
        if args.task_filter:
            # 只有当任务名包含 task_filter 字符串时，才保留
            train_data_all = [
                batch for batch in raw_train_batches
                if args.task_filter in str(batch[3][0])
            ]
            # 如果 epoch=0，打印一次日志确认一下
            if epoch == begin_epoch:
                print(f"👉 [Train Filter] Kept {len(train_data_all)} tasks containing '{args.task_filter}'")
        else:
            train_data_all = raw_train_batches

        if world_size > 1:
            train_data_all = [b for i,b in enumerate(train_data_all) if i % world_size == local_rank]
        for step, cur_data in enumerate(train_data_all):
            meta_batch_loss, _ = run_train_iter(model,cur_data, epoch)

            loss_val = meta_batch_loss['loss']
            if hasattr(loss_val, 'item'):
                print_loss += loss_val.item()
            elif isinstance(loss_val, (float, int)):
                print_loss += loss_val

            print_step += 1

            if print_step % Print_Iter == 0 or step == len(train_data_all) - 1:
                avg_loss = print_loss / print_step if print_step > 0 else 0.0
                print('epoch: {}, iter: {}, mse: {:.4f}'.format(epoch, step, avg_loss))
                print_loss = 0.0
                print_step = 0


        _, test_result = test(args, epoch, model, dataloader, is_test=False)
        # torch.save(model.state_dict(), '{0}/{2}/model_{1}'.format(args.logdir, epoch, exp_string))
        if last_test_result is None:
            last_test_result = test_result
            beat_epoch = epoch
            torch.save(model.state_dict(), '{0}/{2}/model_best'.format(args.logdir, epoch, exp_string))
        else:
            # 2. 根据任务类型判断好坏
            # 回归任务 (RMSE/MAE): 越小越好
            if args.task_type == 'regression':
                is_better = test_result < last_test_result
            # 分类任务 (AUC/ACC): 越大越好
            else:
                is_better = test_result > last_test_result

            if is_better:
                last_test_result = test_result
                beat_epoch = epoch
                torch.save(model.state_dict(), '{0}/{2}/model_best'.format(args.logdir, epoch, exp_string))
                print(f"New best model saved at epoch {epoch} with result {test_result:.4f}")
        print("beat valid epoch is:", beat_epoch)


def test(args, epoch, model, dataloader, is_test=True):
    cir_num = args.test_repeat_num
    # containers for regression
    r2_list, R2os_list, rmse_list, mae_list = [], [], [], []
    # containers for classification
    auc_list, prc_list, acc_list, prec_list, rec_list, f1_list = [], [], [], [], [], []

    res_dict = {}
    all_pred_data = []

    # 加载 metadata.json
    import json
    metadata_path = os.path.join(args.preprocessed_dir, 'metadata.json')
    expanded_map = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta_json = json.load(f)
                expanded_map = meta_json.get('expanded_map', {})
                # expanded_map 的结构是:
                # Key: "clintox_Scaffold_X_Task_Y"
                # Value: ["Scaffold_X_Task_Y", task_idx, "原始任务名(如FDA_APPROVED)"]
        except Exception as e:
            print(f"[Warning] Failed to load metadata: {e}")

    # 用于存储拆分后的指标： {'FDA_APPROVED': {'auc': [], 'f1': []}, ...}
    task_breakdown = {}

    for cir in range(cir_num):
        # 强制固定每一轮测试的随机种子
        seed_val = args.test_seed + cir
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        if is_test:
            raw_batches = list(dataloader.get_test_batches(repeat_cnt=cir))
        else:
            raw_batches = list(dataloader.get_val_batches(repeat_cnt=cir))

            # ================= 【修改点：任务过滤】 =================
        test_data_all = []
        if args.task_filter:
            test_data_all = [
                batch for batch in raw_batches
                if args.task_filter in str(batch[3][0])
            ]
            # 这里的打印在 test 时可能会刷屏，可以注释掉或者只打印一次
            # print(f"👉 [Test/Val Filter] Kept {len(test_data_all)} tasks")
        else:
            test_data_all = raw_batches
        # ======================================================

        for step, cur_data in enumerate(test_data_all):
            ligands_x = None
            assay_name = f"task_{step}"
            try:
                if isinstance(cur_data, (list, tuple)) and len(cur_data) > 0:
                    first = cur_data[0]
                    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
                        ligands_x = first[0]
                    else:
                        ligands_x = first
                if isinstance(cur_data, (list, tuple)) and len(cur_data) > 3:
                    maybe = cur_data[3]
                    if isinstance(maybe, (list, tuple)) and len(maybe) > 0:
                        assay_name = str(maybe[0])
                    else:
                        assay_name = str(maybe)
            except Exception:
                ligands_x = None

            try:
                supn = int(float(args.test_sup_num))
            except Exception:
                supn = None

            try:
                if ligands_x is not None and supn is not None and len(ligands_x) <= supn:
                    continue
            except Exception:
                pass

            # run validation
            losses, per_task_target_preds, final_weights, per_task_metrics = run_validation_iter(model, cur_data)

            if per_task_metrics is None or len(per_task_metrics) == 0:
                continue

            # 提取真实值和预测值
            for task_idx, m in enumerate(per_task_metrics):
                if 'pred' in m and 'true' in m:
                    preds = m['pred']
                    trues = m['true']
                    smiles_list = m.get('smiles', [""] * len(preds))

                    if len(smiles_list) != len(preds):
                        smiles_list = ["Len_Mismatch"] * len(preds)

                    for p, t, s in zip(preds, trues, smiles_list):
                        all_pred_data.append({
                            "Epoch": epoch,
                            "Repeat_Index": cir,
                            "Task_Step": step,
                            "Task_Name": assay_name,
                            "SMILES": s,
                            "True_Value": t,
                            "Pred_Value": p,
                            "Split": "Test" if is_test else "Valid"
                        })

            m = per_task_metrics[0]

            # 获取当前子任务的 ID
            current_assay_name = cur_data[3][0]

            real_task_name = "Unknown"

            if current_assay_name in expanded_map:
                real_task_name = expanded_map[current_assay_name][2]
            elif f"{args.datasource}_{current_assay_name}" in expanded_map:
                real_task_name = expanded_map[f"{args.datasource}_{current_assay_name}"][2]

            if args.task_type == 'classification' and real_task_name != "Unknown":
                if real_task_name not in task_breakdown:
                    task_breakdown[real_task_name] = {'auc': [], 'prc': [], 'acc': [], 'f1': []}

                if m.get("auc") is not None:
                    task_breakdown[real_task_name]['auc'].append(m["auc"])
                prc_val = m.get("prc_auc", m.get("prc", None))
                if prc_val is not None:
                    task_breakdown[real_task_name]['prc'].append(prc_val)
                val_acc = m.get("accuracy", m.get("acc", None))
                if val_acc is not None:
                    task_breakdown[real_task_name]['acc'].append(val_acc)
                if m.get("f1") is not None:
                    task_breakdown[real_task_name]['f1'].append(m["f1"])

            if args.task_type == 'regression':
                rmse = m.get("rmse", 0.0)
                mae = m.get("mae", 0.0)
                r2 = m.get("r2", 0.0)
                R2os = m.get("R2os", 0.0)
                if rmse == 0.0 and r2 == 0.0 and R2os == 0.0 and mae == 0.0:
                    continue

                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)
                R2os_list.append(R2os)

            else:# classification
                auc = m.get("auc", None)
                prc = m.get("prc_auc", 0.0)
                acc = m.get("accuracy", m.get("acc", None))
                prec = m.get("precision", None)
                rec = m.get("recall", None)
                f1_val = m.get("f1", None)
                if auc is None:
                    continue

                auc_list.append(auc)
                prc_list.append(prc)
                if acc is not None: acc_list.append(acc)
                if prec is not None: prec_list.append(prec)
                if rec is not None: rec_list.append(rec)
                if f1_val is not None: f1_list.append(f1_val)

            assay_name = cur_data[3][0]
            # 累积到字典里
            res_dict.setdefault(assay_name, []).append(m)


    if len(all_pred_data) > 0 and args.train == 0:
        save_name = f"predictions_{args.datasource}_epoch_{epoch}.csv"
        if is_test:
            save_name = "test_" + save_name
        else:
            save_name = "valid_" + save_name

        save_path = os.path.join(args.logdir, save_name)

        try:
            df = pd.DataFrame(all_pred_data)
            df.to_csv(save_path, index=False)
            print(f">>> Predictions saved to: {save_path}")
        except Exception as e:
            print(f"Error saving prediction csv: {e}")

    if args.task_type == 'regression':
        if len(rmse_list) == 0:
            print("No valid regression tasks found during test.")
            metric_val = 0.5
        else:
            rmse_i = np.mean(rmse_list)
            mae_i = np.mean(mae_list) if len(mae_list) > 0 else 0.0
            median_r2 = np.median(r2_list) if len(r2_list) > 0 else 0.0
            mean_r2 = np.mean(r2_list) if len(r2_list) > 0 else 0.0
            valid_cnt = len([x for x in r2_list if x > 0.3])
            print('epoch is: {}, mean rmse is: {:.4f}, mean mae is: {:.4f}'.format(epoch, rmse_i, mae_i))
            print('epoch is: {}, r2: mean is: {:.4f}, median is: {:.4f}, cnt>0.3 is: {:.4f}'.format(epoch, mean_r2,
                                                                                                    median_r2,
                                                                                                    valid_cnt))
            median_r2os = np.median(R2os_list) if len(R2os_list) > 0 else 0.0
            mean_r2os = np.mean(R2os_list) if len(R2os_list) > 0 else 0.0
            valid_cnt = len([x for x in R2os_list if x > 0.3])
            print('epoch is: {}, R2os: mean is: {:.4f}, median is: {:.4f}, cnt>0.3 is: {:.4f}'.format(epoch, mean_r2os,
                                                                                                      median_r2os,
                                                                                                      valid_cnt))

            metric_name = getattr(args, "eval_metric", "rmse")
            metric_val = mae_i if metric_name == "mae" else rmse_i
    else:
        if len(auc_list) == 0:
            print("No valid classification tasks with AUC found during test.")
            metric_val = 0.5
        else:
            mean_auc = float(np.mean(auc_list))
            mean_prc = float(np.mean(prc_list))
            mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else None
            mean_prec = float(np.mean(prec_list)) if len(prec_list) > 0 else None
            mean_rec = float(np.mean(rec_list)) if len(rec_list) > 0 else None
            mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else None

            print(
                f"epoch {epoch}: mean ROC-AUC: {mean_auc:.4f}, mean PRC-AUC: {mean_prc:.4f} " +
                (f"mean acc: {mean_acc:.4f}, " if mean_acc is not None else "mean acc: N/A, ") +
                (f"mean f1: {mean_f1:.4f}" if mean_f1 is not None else "mean f1: N/A")
            )

            if args.task_type == 'classification' and len(task_breakdown) > 0:
                print("\n" + "=" * 75)
                print(f" >>> Per-Task Performance Breakdown (Epoch {epoch}) <<<")
                print(f"{'Task Name':<30} | {'AUC':<8} | {'PRC':<8} | {'Acc':<8} | {'F1':<8} | {'N_Tasks':<5}")
                print("-" * 75)

                sorted_tasks = sorted(task_breakdown.keys())
                for t_name in sorted_tasks:
                    metrics = task_breakdown[t_name]
                    # 计算平均值
                    avg_auc = np.mean(metrics['auc']) if metrics['auc'] else 0.0
                    avg_prc = np.mean(metrics['prc']) if metrics['prc'] else 0.0
                    avg_acc = np.mean(metrics['acc']) if metrics['acc'] else 0.0
                    avg_f1 = np.mean(metrics['f1']) if metrics['f1'] else 0.0
                    count = len(metrics['auc'])

                    # 格式化输出 (如果是 0.0 且 count>0，说明真的差；如果 count=0，显示 N/A)
                    s_auc = f"{avg_auc:.4f}" if metrics['auc'] else "N/A"
                    s_prc = f"{avg_prc:.4f}" if metrics['prc'] else "N/A"
                    s_acc = f"{avg_acc:.4f}" if metrics['acc'] else "N/A"
                    s_f1 = f"{avg_f1:.4f}" if metrics['f1'] else "N/A"

                    print(f"{t_name:<30} | {s_auc:<8} | {s_prc:<8}  | {s_acc:<8} | {s_f1:<8} | {count:<5}")
                print("=" * 75 + "\n")

            if args.eval_metric in ['prc', 'multi-PRC-AUC', 'PRC-AUC']:
                metric_val = mean_prc
            else:
                metric_val = mean_auc

    return res_dict, metric_val


def prepare_assay_feat(args, model, dataloader):
    if os.path.exists(args.train_assay_feat_all):
        return
    train_data_all = dataloader.get_train_batches_weighted()
    init_weight = model.get_init_weight().detach().cpu().numpy().squeeze()
    train_weights_all = []
    train_assay_names_all = []
    loss_all = []
    for train_idx, cur_data in tqdm(enumerate(train_data_all)):
        train_assay_names_all.append(cur_data[3][0])
        loss, _, final_weights, _ = model.run_validation_iter(cur_data)
        loss_all.append(loss['loss'].detach().cpu().item())
        task_weight = final_weights[0]["layer_dict.linear.weights"].detach().cpu().numpy().squeeze()
        task_feat = task_weight - init_weight
        train_weights_all.append(task_feat)
        if (train_idx+1)%200 == 0:
            print(sum(loss_all) / len(loss_all))

    print(sum(loss_all)/len(loss_all))
    np.save(args.train_assay_feat_all, np.array(train_weights_all))
    json.dump(train_assay_names_all, open(args.train_assay_idxes, "w"))


def setup_distributed(local_rank, world_size):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

    if not dist.is_available():
        print("[setup_distributed] distributed not available on this build")
        return

    if dist.is_initialized():
        print("[setup_distributed] process group already initialized (skipping)")
        return

    print(f"[setup_distributed] init_process_group backend=nccl rank={local_rank} world_size={world_size}")
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)

def main_worker(local_rank, args):
    # local_rank: GPU id for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ngpus = torch.cuda.device_count()
    world_size = ngpus if ngpus > 0 else 1
    if world_size > 1:
        setup_distributed(local_rank, world_size)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    try:
        args.local_rank = int(local_rank)
        args.world_size = int(world_size)
    except Exception:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    pid = os.getpid()
    env_info = {
        'LOCAL_RANK': os.environ.get('LOCAL_RANK'),
        'RANK': os.environ.get('RANK'),
        'WORLD_SIZE': os.environ.get('WORLD_SIZE'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')
    }
    print(f"[PROC START] pid={pid} local_rank={local_rank} world_size={world_size} device={device} env={env_info}",
          flush=True)

    # create model and move to device
    model = system_selector(args)(args=args, input_shape=(2, args.dim_w)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataloader = dataset_constructor(args)

    # Only rank 0 (local_rank==0) performs logging, checkpointing and full testing
    rank0 = (local_rank == 0)

    if args.train == 1:
        if args.resume == 1 and rank0:
            model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
            if not os.path.exists(model_file):
                model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
            print("resume training from", model_file)
            try:
                model.load_state_dict(torch.load(model_file), strict=False)
            except Exception as e:
                try:
                    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
                except:
                    model.module.load_state_dict(torch.load(model_file), strict=False)
        # start training (each process handles a subset of tasks)
        train(args, model, dataloader, local_rank=local_rank, world_size=world_size)
    elif args.train == 0:
        # For evaluation mode, only run on rank0 to avoid duplication
        if rank0:
            args.meta_batch_size = 1
            model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
            if not os.path.exists(model_file):
                model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
            print(f">>> [TEST MODE] Loading model from: {model_file}")
            if not isinstance(args.test_sup_num, list):
                args.test_sup_num = [args.test_sup_num]
            test_sup_num_all = copy.deepcopy(args.test_sup_num)
            for test_sup_num in test_sup_num_all:
                args.test_sup_num = test_sup_num
                try:
                    model.load_state_dict(torch.load(model_file), strict=False)
                except Exception as e:
                    try:
                        model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
                    except:
                        model.module.load_state_dict(torch.load(model_file), strict=False)
                test(args, args.test_epoch, model, dataloader, is_test=False)


def main():
    global exp_string
    datasource = args.datasource
    exp_string = f'data_{datasource}.mbs_{args.meta_batch_size}.metalr_0.00015.innerlr_{args.update_lr}'
    ngpus = torch.cuda.device_count()
    # If launched with torchrun / torch.distributed.run, LOCAL_RANK and WORLD_SIZE will be set
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        setup_distributed(local_rank, world_size)
        main_worker(local_rank, args)
    elif ngpus > 1:
        # fallback: spawn processes manually (less recommended)
        mp.spawn(main_worker, nprocs=ngpus, args=(args,))
    else:
        # single GPU / CPU fallback
        main_worker(0, args)

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    if "transfer" in args.model_name.lower() or "protonet" in args.model_name.lower():
        args.per_step_bn_statistics = False
    print(args)

    try:
        args.test_sup_num = json.loads(args.test_sup_num)
    except:
        args.test_sup_num = float(args.test_sup_num)
        if args.test_sup_num > 1:
            args.test_sup_num = int(args.test_sup_num)

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    random.seed(1)
    np.random.seed(2)

    datasource = args.datasource
    exp_string = f'data_{datasource}.mbs_{args.meta_batch_size}.metalr_0.00015.innerlr_{args.update_lr}'
    main()
