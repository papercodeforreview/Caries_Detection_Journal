import os
import sys
import torch.nn as nn
import torch
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
import utils_loss as KD_loss
from datetime import datetime
from time import sleep
import torchvision
import logging
import timeit

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        device = projections.device
        batch_size = projections.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")
    parser.add_argument('--log', type=str, default='./logs/logs.txt', help='logging file')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--cosine_epoch", default=100, type=int, help="Epoch for updating CosineAnnealingLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--data_update_epochs", "-de", default=10, type=int,
                        help="number of epochs when to refresh the coreset")
    parser.add_argument("--selection_epochs", "-se", default=5, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=0, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="Adam",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.01, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

    # Quantization and KD
    parser.add_argument('--adaptive', type=str, default='linear', help='Adaptive strategy for alpha (weighting coefficient)')
    parser.add_argument('--bitwidth', type=int, default=None, help='bitwidth for weights and activations')
    parser.add_argument('--teacher', type=str, default=None, help='teacher model') #'resnet101'

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('logs'):
        os.mkdir('logs')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    start_exp = 0
    for exp in range(start_exp, args.num_exp):

        
        start_epoch = 0

        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_{fr}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=start_exp,
                                                                                         epc=args.epochs,
                                                                                         fr=args.fraction)

        logging.info('\n================== Exp %d ==================\n' % exp)
        logging.info("model: {}, selection:{}, fraction:{}, lr:{}\n".format(args.model, args.selection, args.fraction, args.lr))

        start_time = timeit.default_timer()
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test,dst_train_cl  = datasets.__dict__[args.dataset] \
            (args.data_path)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        torch.random.manual_seed(args.seed)
        
        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        for model in models:
            if len(models) > 1:
                logging.info("| Training on model %s" % model)
            if model == 'DNN' and args.dataset == "Teeth":
                logging.info("Using DNN Pretrained Model")
                network = nets.__dict__[model]().to(args.device)
            else:
                network = nets.__dict__[model](channel, num_classes, im_size).to(args.device)

            if args.device == "cpu":
                logging.info("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            # load the training set and test set
            train_loader, test_loader, if_weighted, subset, selection_args = load_subset(args, 0, dst_train, dst_test, mean, std, network)

            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
            criterion1 =SupervisedContrastiveLoss().to(args.device)
            
            # criterion= criterion0+criterion1
            # Optimizer
            #if model == 'QResNet18':
            if args.bitwidth != None:

                logging.info("Using Adam optimizer with different lr")
                all_parameters = network.parameters()
                weight_parameters = []
                alpha_parameters = []

                weight_parameters_id = list(map(id, weight_parameters))
                alpha_parameters_id = list(map(id, alpha_parameters))
                other_parameters1 = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
                other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1))

                if args.dataset == 'ImageNet':
                    optimizer = torch.optim.Adam(
                            [{'params' : alpha_parameters, 'lr': args.lr / 10},
                            {'params' : other_parameters, 'lr': args.lr},
                            {'params' : weight_parameters, 'weight_decay': args.weight_decay, 'lr': args.lr}],
                            betas=(0.9,0.999)
                    )
                else:
                    optimizer = torch.optim.SGD(
                                [{'params' : alpha_parameters, 'lr': args.lr / 10},
                                {'params' : other_parameters, 'lr': args.lr},
                                {'params' : weight_parameters, 'weight_decay': args.weight_decay, 'lr': args.lr}],
                                momentum=args.momentum,nesterov=args.nesterov
                    )

            elif args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            # LR scheduler
            if args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_epoch, eta_min=args.min_lr)
            elif args.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            elif args.scheduler == "LambdaLR":
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
            else:
                scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
            
            best_prec1 = 0.0

            # Save the checkpont with only the susbet.
            if args.save_path != "" and args.resume == "":
                save_checkpoint({"exp": exp,
                                 "subset": subset,
                                 "sel_args": selection_args},
                                os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
                                             + "_") + "unknown.ckpt"), 0, 0.)


            #train the backbone
            

            #train the classify
            for epoch in range(start_epoch, args.epochs):

                if epoch != 0 and epoch % args.data_update_epochs == 0 :
                    print("Generating New Coreset")
                    train_loader, test_loader, if_weighted, subset, selection_args = load_subset(args, epoch, dst_train, dst_test, mean, std, network)

                model_teacher=None
                rec = init_recorder()
                train(train_loader, network, criterion, criterion1,model_teacher, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                             "epoch": epoch + 1,
                                             "state_dict": network.state_dict(),
                                             "opt_dict": optimizer.state_dict(),
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset,
                                             "sel_args": selection_args},
                                            os.path.join(args.save_path, checkpoint_name + (
                                                "" if model == args.model else model + "_") + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

            # Prepare for the next checkpoint
            if args.save_path != "":
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
                                     "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
                                     ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
                except:
                    save_checkpoint({"exp": exp,
                                     "epoch": args.epochs,
                                     "state_dict": network.state_dict(),
                                     "opt_dict": optimizer.state_dict(),
                                     "best_acc1": best_prec1,
                                     "rec": rec,
                                     "subset": subset,
                                     "sel_args": selection_args},
                                    os.path.join(args.save_path, checkpoint_name +
                                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)

            logging.info('Best accuracy: {}'.format(best_prec1))
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            print(f" {execution_time:.6f} s")
            start_epoch = 0
            checkpoint = {}
            sleep(2)

def load_subset(args, current_epoch, dst_train, dst_test, mean, std, current_model):
    selection_args = dict(epochs=args.selection_epochs,
                            selection_method=args.uncertainty,
                            balance=args.balance,
                            greedy=args.submodular_greedy,
                            function=args.submodular
                            )
    if args.selection == 'ACS':
        method = methods.__dict__[args.selection](dst_train, args, current_epoch, current_model, args.fraction, args.seed, **selection_args)
    else:
        method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
    subset = method.select()
    logging.info("The length for the subset:{}".format(len(subset["indices"])))

    # Augmentation
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        dst_train.transform = transforms.Compose(
            [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(), dst_train.transform])
    elif args.dataset == "ImageNet":
        dst_train.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Handle weighted subset
    if_weighted = "weights" in subset.keys()
    if if_weighted:
        dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
    else:
        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet":
        train_loader = DataLoaderX(dst_subset, batch_size=args.train_batch, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader, if_weighted, subset, selection_args


if __name__ == '__main__':
    main()
