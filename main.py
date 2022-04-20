"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
import torch
import torch.optim as optim
import numpy as np
import networks
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed
from adamp import AdamP
from tensorboardX import SummaryWriter

from arguments import get_args
import time
import os
args = get_args()


def main():
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    result_dir = os.path.join(args.result_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(result_dir)
    writer = None
    if args.record:
        log_dir = os.path.join(args.log_dir, args.date, dataset, args.method)
        check_log_dir(log_dir)
        writer = SummaryWriter(log_dir + '/' + log_name)

    ########################## get dataloader ################################
    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    else:
        args.img_size = 224
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset,
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers,
                                                        target_attr=args.target,
                                                        add_attr=args.add_attr,
                                                        labelwise=args.labelwise,
                                                        sv_ratio=args.sv,
                                                        version=args.version,
                                                        args=args
                                                        )
    num_classes, num_groups, train_loader, test_loader = tmp

    ########################## get model ##################################
    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    elif 'cifar' in args.dataset:
        args.img_size = 32

    model = networks.ModelFactory.get_model(args.model, num_classes, args.img_size,
                                            pretrained=args.pretrained, num_groups=num_groups)

    model.cuda('cuda:{}'.format(args.device))

    if args.modelpath is not None:
        model.load_state_dict(torch.load(args.modelpath))

    teacher = None
    if (args.method == 'mfd' or args.teacher_path is not None) and args.mode != 'eval':
        teacher = networks.ModelFactory.get_model(args.teacher_type, train_loader.dataset.num_classes, args.img_size)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cuda:{}'.format(args.t_device))))
        teacher.cuda('cuda:{}'.format(args.t_device))

    ########################## get trainer ##################################
    if 'Adam' in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'AdamP' in args.optimizer:
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'SGD' in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                  optimizer=optimizer, teacher=teacher)

    ####################### start training or evaluating ####################

    if args.mode == 'train':
        start_t = time.time()
        trainer_.train(train_loader, test_loader, args.epochs, writer=writer)
        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        trainer_.save_model(save_dir, log_name)

    else:
        print('Evaluation ----------------')
        model_to_load = args.modelpath
        trainer_.model.load_state_dict(torch.load(model_to_load))
        print('Trained model loaded successfully')

    if args.evalset == 'all':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, result_dir, log_name)
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, result_dir, log_name)

    elif args.evalset == 'train':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, result_dir, log_name)
    else:
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, result_dir, log_name)
    if writer is not None:
        writer.close()
    print('Done!')


if __name__ == '__main__':
    main()
