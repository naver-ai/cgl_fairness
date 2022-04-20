"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import torch
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks
import torch.nn.functional as F
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed
from data_handler.dataset_factory import DatasetFactory
from arguments import get_args
import time
import pickle
import os
from torch.utils.data import DataLoader
args = get_args()
###################################################################################################################################################################
"""
Used only for training group classifer
"""
###################################################################################################################################################################


def get_weights(loader, cuda=True):
    num_groups = loader.dataset.num_groups
    data_counts = torch.zeros(num_groups)
    data_counts = data_counts.cuda() if cuda else data_counts

    for data in loader:
        inputs, _, groups, _, _ = data
        for g in range(num_groups):
            data_counts[g] += torch.sum((groups == g))

    weights = data_counts / data_counts.min()
    return weights, data_counts


def focal_loss(input_values, gamma=10):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


def measure_ood(ind_probs, ood_probs, tpr_thres=0.95):
    n_pos = len(ind_probs)
    n_neg = len(ood_probs)

    labels = np.append(
        np.ones_like(ind_probs) * 1,
        np.ones_like(ood_probs) * 2
    )
    preds = np.append(ind_probs, ood_probs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auroc = metrics.auc(fpr, tpr)

    thres_indx = np.where(tpr >= tpr_thres)[0][0]
    tnr_at_tpr95 = 1 - fpr[thres_indx]

    temp_idx = np.argmax(((1 - fpr) * n_neg + tpr * n_pos) / (n_pos + n_neg))
    detection_acc = np.max(((1 - fpr) * n_neg + tpr * n_pos) / (n_pos + n_neg))
    if n_neg != n_pos:
        print(f'warning: n_neg ({n_neg}) != n_pos ({n_pos}). It may shows weird detection acc')
        print(f'current threshold ({thresholds[temp_idx]}): FPR ({fpr[temp_idx]}), TPR ({tpr[temp_idx]})')

    return {
        'auroc': auroc,
        'tnr_at_tpr95': tnr_at_tpr95,
        'detection_acc': detection_acc,
        'opt_thres': thresholds[temp_idx]
    }


def predict_thres(probs, true_idxs, false_idxs, val_idxs):
    print(len(true_idxs), len(false_idxs))
    val_false_idxs = list(set(false_idxs).intersection(val_idxs))
    val_true_idxs = list(set(true_idxs).intersection(val_idxs))

    val_true_maxprob = probs[val_true_idxs].max(dim=1)[0]
    val_false_maxprob = probs[val_false_idxs].max(dim=1)[0]
    r = measure_ood(val_true_maxprob.numpy(), val_false_maxprob.numpy())
    return r['opt_thres']


def predict_group(model, args, trainloader, testloader):
    model.cuda('cuda:{}'.format(args.device))
    target_attr = None
    target_attr = args.target
    if args.dataset == 'adult':
        target_attr = 'sex'
    elif args.dataset == 'compas':
        target_attr = 'race'
    dataset = DatasetFactory.get_dataset(args.dataset, split='train', sv_ratio=1, version='',
                                         target_attr=target_attr, add_attr=args.add_attr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    val_idxs = []
    if args.version == 'groupclf_val':
        val_idxs = trainloader.dataset.val_idxs

    test_idxs = []
    filename = f'{args.seed}_{args.sv}'
    if args.dataset == 'celeba':
        if args.target != 'Attractive':
            filename += f'_{args.target}'
        if args.add_attr is not None:
            filename += f'_{args.add_attr}'
    filename += '.pkl'
    filename = os.path.join(dataset.root, 'annotated_idxs', filename)
    with open(filename, 'rb') as f:
        idxs_dict = pickle.load(f)
    for key in idxs_dict['non-annotated'].keys():
        test_idxs.extend(idxs_dict['non-annotated'][key])

    model.eval()
    preds_list = []
    groups_list = []
    labels_list = []
    probs_list = []
    true_idxs = []
    false_idxs = []

    uncertainty = False
    if 'dropout' in args.model:
        uncertainty = True
        enable_dropout(model)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _, groups, labels, (idxs, _) = data
            if args.cuda:
                inputs = inputs.cuda()
                groups = groups.cuda()
                labels = labels.cuda()
                idxs = idxs.cuda()

            if uncertainty:
                preds, probs = mc_dropout(model, inputs)

            else:
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, 1)

            true_mask = groups == preds
            false_mask = groups != preds

            probs_list.append(probs.cpu())

            preds_list.append(preds.cpu())
            groups_list.append(groups.cpu())
            labels_list.append(labels.cpu())

            true_idxs.extend(list(idxs[true_mask].cpu().numpy()))
            false_idxs.extend(list(idxs[false_mask].cpu().numpy()))

    preds = torch.cat(preds_list)
    groups = torch.cat(groups_list)
    probs = torch.cat(probs_list)
    labels = torch.cat(labels_list)

    # calculate the test acc
    test_acc = (preds == groups)[test_idxs].sum().item() / len(test_idxs)
    print(f'test acc : {test_acc}')
    if args.version == 'groupclf_val':
        val_acc = (preds == groups)[val_idxs].sum().item() / len(val_idxs)
        print(f'val acc : {val_acc}')

    results = {}
    results['pred'] = preds
    results['group'] = groups
    results['probs'] = probs
    results['label'] = labels
    results['true_idxs'] = true_idxs
    results['false_idxs'] = false_idxs
    return results, val_idxs


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_dropout(model, inputs):
    temp_nl = 2
    f_pass = 10
    out_prob = []
    out_prob_nl = []
    for _ in range(f_pass):
        outputs = model(inputs)
        out_prob.append(F.softmax(outputs, dim=1))  # for selecting positive pseudo-labels
        out_prob_nl.append(F.softmax(outputs/temp_nl, dim=1))  # for selecting negative pseudo-labels
    out_prob = torch.stack(out_prob)
    out_prob_nl = torch.stack(out_prob_nl)
    out_std = torch.std(out_prob, dim=0)
    probs = torch.mean(out_prob, dim=0)
    out_prob_nl = torch.mean(out_prob_nl, dim=0)
    max_value, preds = torch.max(probs, dim=1)
    max_std = out_std.gather(1, preds.view(-1, 1)).squeeze()
    n_uncertain = (max_std > 0.05).sum()
    print('# of uncertain samples : ', n_uncertain)
    preds[max_std > 0.05] = -1
    return preds, probs


def main():
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    log_dir = os.path.join(args.result_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(log_dir)
    if 'groupclf' not in args.version:
        raise ValueError
    ########################## get dataloader ################################
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset,
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers,
                                                        target_attr=args.target,
                                                        add_attr=args.add_attr,
                                                        labelwise=args.labelwise,
                                                        sv_ratio=args.sv,
                                                        version=args.version,
                                                        )
    num_groups, num_classes, train_loader, test_loader = tmp
    ########################## get model ##################################

    num_model_output = num_classes if args.modelpath is not None else num_groups
    model = networks.ModelFactory.get_model(args.model, num_model_output, args.img_size)

    if args.modelpath is not None:
        model.load_state_dict(torch.load(args.modelpath))
        model.fc = nn.Linear(in_features=model.fc.weight.shape[1], out_features=num_groups, bias=True)

    model.cuda('cuda:{}'.format(args.device))
    teacher = None
    criterion = None
    if args.dataset in ['compas', 'adult']:
        weights = None
        weights, data_counts = get_weights(train_loader, cuda=args.cuda)
        cls_num_list = []
        for i in range(num_groups):
            cls_num_list.append(data_counts[i].item())
        beta = 0.999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        criterion = FocalLoss(weight=per_cls_weights).cuda()
    ########################## get trainer ##################################
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif 'SGD' in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                  optimizer=optimizer, teacher=teacher)

    ####################### start training or evaluating ####################

    if args.mode == 'train':
        start_t = time.time()
        trainer_.train(train_loader, test_loader, args.epochs, criterion=criterion)

        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        trainer_.save_model(save_dir, log_name)
    else:
        print('Evaluation ----------------')
        model_name = os.path.join(save_dir, log_name+'.pt')
        model.load_state_dict(torch.load(model_name))
        results, val_idxs = predict_group(model, args, train_loader, test_loader)
        if args.version == 'groupclf_val':
            opt_thres = predict_thres(results['probs'], results['true_idxs'], results['false_idxs'], val_idxs)
            results['opt_thres'] = opt_thres
        save_path = os.path.join('../data', args.dataset, args.date)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(results, os.path.join(save_path, log_name+'.pt'))
        print(os.path.join(save_path, log_name+'.pt'))
        return

    if args.evalset == 'all':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, log_dir, log_name)
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, log_dir, log_name)

    elif args.evalset == 'train':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, log_dir, log_name)
    else:
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, log_dir, log_name)

    print('Done!')


if __name__ == '__main__':
    main()
