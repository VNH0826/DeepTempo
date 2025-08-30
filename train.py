import argparse
import os
import random
import time
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from model import DeepTempo
from utils.data_loader import load_aig_dataset
from config import get_config


def get_logger(name: str, logfile: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepTempo Training')

    parser.add_argument('--dataset', type=str, default='aig_circuits')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--split_ratio', type=str, default='0.05-0.05-0.9')
    parser.add_argument('--task_type', type=str, default='prob', choices=['prob', 'tt'])
    parser.add_argument('--model', type=str, default='DeepTempo')
    parser.add_argument('--layer_num', type=int, default=9)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--temporal_kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'mse'])
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.device}'
                               if torch.cuda.is_available() and args.device >= 0
                               else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    return args


def zero_normalization(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        return x
    mean_x = torch.mean(x)
    std_x = torch.std(x) + 1e-8
    return (x - mean_x) / std_x


def compute_aig_losses(edge_index_s: torch.Tensor, output: torch.Tensor,
                       loss_type: str = 'mae') -> dict:
    losses = {}

    # NOT gate constraint
    not_edges = edge_index_s[edge_index_s[:, 2] == -1]
    if not_edges.size(0) > 0:
        input_out = output[not_edges[:, 0]]
        output_out = output[not_edges[:, 1]]
        edges_sum = input_out + output_out
        target = torch.ones_like(edges_sum)

        if loss_type == 'mae':
            losses['not'] = F.l1_loss(edges_sum, target)
        else:
            losses['not'] = F.mse_loss(edges_sum, target)
    else:
        losses['not'] = torch.tensor(0.0, device=output.device)

    # AND gate constraint
    and_edges = edge_index_s[edge_index_s[:, 2] == 1]
    if and_edges.size(0) > 0:
        sorted_indices = torch.argsort(and_edges[:, 1])
        and_edges_sorted = and_edges[sorted_indices]

        if and_edges_sorted.size(0) % 2 == 0:
            inputs = output[and_edges_sorted[:, 0]].view(-1, 2)
            outputs = output[and_edges_sorted[:, 1]].view(-1, 2)
            min_inputs = torch.min(inputs, dim=1)[0]

            if loss_type == 'mae':
                losses['and'] = F.l1_loss(outputs[:, 0], min_inputs)
            else:
                losses['and'] = F.mse_loss(outputs[:, 0], min_inputs)
        else:
            losses['and'] = torch.tensor(0.0, device=output.device)
    else:
        losses['and'] = torch.tensor(0.0, device=output.device)

    return losses


def train_epoch(model, optimizer, train_data, args, logger):
    model.train()
    total_prob_loss = 0
    total_tt_loss = 0
    num_batches = 0

    random.shuffle(train_data)

    for batch_idx, batch in enumerate(tqdm(train_data, desc='Training')):
        (data_dir, edge_index_s, pi_edges_signed_tensor, tt_pair_index_tensor,
         node_features_tensor, node_labels_tensor, tt_dis_tensor) = batch

        node_labels_tensor = node_labels_tensor.unsqueeze(1)

        out_emb, out = model(node_features_tensor, edge_index_s)

        if args.loss_type == 'mae':
            prob_loss = F.l1_loss(out, node_labels_tensor)
        else:
            prob_loss = F.mse_loss(out, node_labels_tensor)

        if tt_pair_index_tensor.size(1) > 0:
            node_a = out_emb[tt_pair_index_tensor[0]]
            node_b = out_emb[tt_pair_index_tensor[1]]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(tt_dis_tensor)

            if args.loss_type == 'mae':
                tt_loss = F.l1_loss(emb_dis_z, tt_dis_z)
            else:
                tt_loss = F.mse_loss(emb_dis_z, tt_dis_z)
        else:
            tt_loss = torch.tensor(0.0, device=args.device)

        if args.task_type == 'prob':
            loss = prob_loss
        elif args.task_type == 'tt':
            loss = tt_loss
        else:
            loss = prob_loss + tt_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (batch_idx + 1) % args.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_prob_loss += prob_loss.item()
        total_tt_loss += tt_loss.item()
        num_batches += 1

    optimizer.step()
    optimizer.zero_grad()

    return total_prob_loss / num_batches, total_tt_loss / num_batches


def evaluate(model, eval_data, args, logger):
    model.eval()
    results = {
        'prob': 0.0, 'tt': 0.0, 'not': 0.0, 'and': 0.0,
        'level': defaultdict(lambda: {'value': 0.0, 'cnt': 0.0})
    }

    with torch.no_grad():
        for batch in tqdm(eval_data, desc='Evaluating'):
            (data_dir, edge_index_s, pi_edges_signed_tensor, tt_pair_index_tensor,
             node_features_tensor, node_labels_tensor, tt_dis_tensor) = batch

            node_labels_tensor = node_labels_tensor.unsqueeze(1)

            out_emb, out = model(node_features_tensor, edge_index_s)

            if args.loss_type == 'mae':
                prob_loss = F.l1_loss(out, node_labels_tensor)
            else:
                prob_loss = F.mse_loss(out, node_labels_tensor)

            if tt_pair_index_tensor.size(1) > 0:
                node_a = out_emb[tt_pair_index_tensor[0]]
                node_b = out_emb[tt_pair_index_tensor[1]]
                emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
                emb_dis_z = zero_normalization(emb_dis)
                tt_dis_z = zero_normalization(tt_dis_tensor)

                if args.loss_type == 'mae':
                    tt_loss = F.l1_loss(emb_dis_z, tt_dis_z)
                else:
                    tt_loss = F.mse_loss(emb_dis_z, tt_dis_z)
            else:
                tt_loss = torch.tensor(0.0, device=args.device)

            constraint_losses = compute_aig_losses(edge_index_s, out, args.loss_type)

            results['prob'] += prob_loss.item()
            results['tt'] += tt_loss.item()
            results['not'] += constraint_losses['not'].item()
            results['and'] += constraint_losses['and'].item()

    num_samples = len(eval_data)
    for key in ['prob', 'tt', 'not', 'and']:
        results[key] /= num_samples

    return results


def train_model(args, train_data, valid_data, logger):
    logger.info('Starting DeepTempo training...')

    model = DeepTempo(
        args=args,
        node_num=0,
        device=args.device,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        layer_num=args.layer_num,
        temporal_kernel_size=args.temporal_kernel_size,
        dropout=args.dropout,
        lamb=5
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_loss = float('inf')
    patience = args.patience
    best_model_state = None
    total_time = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_prob_loss, train_tt_loss = train_epoch(model, optimizer, train_data, args, logger)
        valid_results = evaluate(model, valid_data, args, logger)

        epoch_time = time.time() - start_time
        total_time += epoch_time

        logger.info(f'Epoch {epoch:03d} | '
                    f'Train - Prob: {train_prob_loss:.4f}, TT: {train_tt_loss:.4f} | '
                    f'Valid - Prob: {valid_results["prob"]:.4f}, TT: {valid_results["tt"]:.4f}, '
                    f'NOT: {valid_results["not"]:.4f}, AND: {valid_results["and"]:.4f} | '
                    f'Time: {epoch_time:.2f}s')

        current_loss = valid_results[args.task_type]
        if current_loss < best_loss:
            best_loss = current_loss
            patience = args.patience
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience -= 1
            if patience <= 0:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    avg_time = total_time / epoch
    logger.info(f'Training completed. Average time per epoch: {avg_time:.4f}s')

    return model, avg_time


def main():
    args = parse_arguments()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(args.output_dir, 'logs',
                            f'{args.task_type}_{args.model}_{args.layer_num}_{timestamp}.log')
    logger = get_logger(__name__, log_file)

    logger.info(f'Arguments: {args}')

    logger.info('Loading dataset...')
    train_data, valid_data, test_data = load_aig_dataset(args)
    logger.info(f'Data loaded - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')

    model, avg_train_time = train_model(args, train_data, valid_data, logger)

    if args.save_model:
        model_path = os.path.join(args.output_dir, 'models',
                                  f'{args.task_type}_{args.model}_{args.layer_num}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
        }, model_path)
        logger.info(f'Model saved to {model_path}')

    logger.info('Evaluating on test set...')
    test_results = evaluate(model, test_data, args, logger)

    logger.info('=' * 50)
    logger.info('FINAL RESULTS')
    logger.info('=' * 50)
    logger.info(f'Test - Prob: {test_results["prob"]:.4f}, TT: {test_results["tt"]:.4f}')
    logger.info(f'Constraints - NOT: {test_results["not"]:.4f}, AND: {test_results["and"]:.4f}')
    logger.info(f'Average training time: {avg_train_time:.4f}s')
    logger.info('=' * 50)


if __name__ == '__main__':
    main()