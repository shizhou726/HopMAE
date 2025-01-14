import argparse
import copy
import numpy as np
import torch
import os

from model import HopMAE, LogisticRegression
from save import load_dataset
from utils import accuracy, create_optimizer, set_random_seed
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def node_classification_evaluation(args, model, data_dict, feat_origin, train_mask, val_mask, test_mask, label):
    model.eval()
    with torch.no_grad():
        x = None
        for i in range(len(feat_origin) // args.batch_size + 1): 
            temp_x = model.embed(feat_origin[i * args.batch_size : (i+1) * args.batch_size])
            if x is None:
                x = temp_x
            else:
                x = torch.cat((x, temp_x), dim=0)

    classifier = LogisticRegression(args).to(args.device)
    optimizer_f = create_optimizer(args.optim_type, classifier, args.lr, args.weight_decay)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    label = data_dict['label'].to(args.device)
    
    best_val_acc = 0
    best_val_epoch = 0
    
    for epoch in range(args.max_epoch):
        classifier.train()
        logits = classifier(x)
        
        loss = criterion(logits[train_mask], label[train_mask])
        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()
        
        with torch.no_grad():
            classifier.eval()
            logits = classifier(x)
            val_acc = accuracy(logits[val_mask], label[val_mask])
            val_loss = criterion(logits[val_mask], label[val_mask])
        
            if val_acc >= best_val_acc: 
                best_val_acc = val_acc
                best_val_epoch = epoch
                test_acc = accuracy(logits[test_mask], label[test_mask])
                test_loss = criterion(logits[test_mask], label[test_mask])
            
    if not args.mute:
        print(f"--- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    return test_acc, best_val_acc


def main(args):
    acc_list = []
    best_acc_list = []
    files = os.listdir('./dataset')
    for file in files:
        if file.startswith(args.dataset):
            file_path = os.path.join('./dataset', file)
    feat_origin = torch.load(file_path)[:, :args.hops+1, :].to(args.device)
    # -------------------------------------------
    # to reproduce the results of the random split
    cls_seed = 0
    np.random.seed(cls_seed)
    # -------------------------------------------
    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        data_dict = load_dataset(args.dataset)
        set_random_seed(seed)

        batch_dataloader = DataLoader(TensorDataset(feat_origin), batch_size=args.batch_size, shuffle=True)
        setattr(args, 'input_dim', feat_origin.size(-1))
        setattr(args, 'num_classes', data_dict['label'].unique().size(0))
        model = HopMAE(args).to(args.device)
        optimizer = create_optimizer(args.pre_optim_type, model, args.pre_lr, args.pre_weight_decay)
        
        f_test_acc = 0
        best_val_acc = 0
        best_val_epoch = 0
        best_test_acc = 0
        best_test_epoch = 0

        epoch_iter = tqdm(range(args.pre_max_epoch))
        for epoch in epoch_iter:
            model.train()
            for _, batch in enumerate(batch_dataloader):
                x = batch[0]
                re_loss, ma_loss = model(x, args)
                loss = re_loss + args.alpha * ma_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_iter.set_description(f"# Epoch {epoch + 1}: re_loss: {re_loss.item():.4f} ma_loss: {ma_loss.item():.4f}")
            
            if (epoch + 1) % args.tau == 0:
                train_mask = data_dict['train_mask']
                val_mask = data_dict['val_mask']
                test_mask = data_dict['test_mask']
                label = data_dict['label'].to(args.device)
                test_acc, val_acc = node_classification_evaluation(args, model, data_dict, feat_origin, train_mask, val_mask, test_mask, label)

                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    best_test_epoch = epoch + 1

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch + 1
                    f_test_acc = test_acc
                    
        print(f"final --- TestAcc: {f_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- Best TestAcc: {best_test_acc:.4f} in epoch {best_test_epoch} --- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hops', type=int, default=13)
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attention_dropout', type=float, default=0.45)
    parser.add_argument('--n_encoder_layers', type=int, default=1)
    parser.add_argument('--n_decoder_layers', type=int, default=2)
    parser.add_argument('--mask_rate', type=float, default=0.2)
    parser.add_argument('--remask_rate', type=float, default=0.25)

    # training parameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--mute", action="store_true", default=False)
    parser.add_argument("--activation", type=str, default="relu")

    parser.add_argument('--pool_type', type=str, default='mean')
    parser.add_argument('--pre_optim_type', type=str, default='adadelta')
    parser.add_argument('--pre_max_epoch', type=int, default=200)
    parser.add_argument('--pre_lr', type=float, default=3e-2)
    parser.add_argument('--pre_weight_decay', type=float, default=5e-4)

    parser.add_argument('--tau', type=int, default=10)
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
