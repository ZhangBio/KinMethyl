# encoding=utf-8

import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from simtsv.dataloader import FeaData
from simtsv.dataloader import clear_linecache,line2input
from torch.utils.data import DataLoader
from simtsv.models import BiGRU,CpG_regression_layer,CpG_regression_layer_single,context_kinetics_predicter,CombinedModelv2,kinetics_layer
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU 用
    torch.cuda.manual_seed_all(seed)  # 多GPU用

    torch.backends.cudnn.deterministic = True  # 确保每次结果一致
    torch.backends.cudnn.benchmark = False  # 禁用优化以避免非确定性
def generate_pseudo_negative(seq_batch, regressor_model, noise_std=0.2):
    """
    输入：
        seq_batch: Tensor，形状为 (B, L, 4) 或 (B, L)，表示独热编码或索引编码的DNA序列
        regressor_model: 训练好的 signal 回归模型，输入 seq → 输出 predIPD, predPW
        noise_std: float，噪声标准差（默认 0.2）

    输出：
        pseudo_signal: Tensor，形状为 (B, L, 2)，为伪造的IPD/PW负样本特征
    """
    regressor_model.eval()
    with torch.no_grad():
        predIPD, predPW = regressor_model(seq_batch)  # 应输出 shape: (B, L, 1)

        def add_noise(x):
            return x + torch.randn_like(x) * noise_std

        predIPD_noisy = add_noise(predIPD)
        predPW_noisy = add_noise(predPW)

        pseudo_signal = torch.cat([predIPD_noisy, predPW_noisy], dim=-1)  # (B, L, 2)

    return pseudo_signal

use_cuda = torch.cuda.is_available()
''' 添加新的model类型，需要添加1.model 2.loss 3.train 4.validation'''
def FloatTensor(tensor, device=0):
    if use_cuda:
        return torch.tensor(tensor, dtype=torch.float, device='cuda:{}'.format(device))
    return torch.tensor(tensor, dtype=torch.float)
def FloatTensor_cpu(tensor):
    return torch.tensor(tensor, dtype=torch.float)
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
def LongTensor(tensor, device=0):
    if use_cuda:
        return torch.tensor(tensor, dtype=torch.long, device='cuda:{}'.format(device))
    return torch.tensor(tensor, dtype=torch.long)
from simtsv.logger import mylogger
LOGGER = mylogger(__name__)
device = "cuda" if use_cuda else "cpu"

def train(args):

    total_start = time.time()
    if args.tseed:
        set_seed(args.tseed)

    LOGGER.info("[main]train starts")
    if use_cuda:
        LOGGER.info("GPU is available!")
    else:
        LOGGER.info("GPU is not available!")

    # dnacontig=DNAReference(args.ref).getcontigs()
    LOGGER.info("reading data..")
    train_dataset = FeaData(args.train_file)
    valid_dataset = FeaData(args.valid_file)
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.dl_num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.dl_num_workers)


    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(r"" + args.model_type + "\..*b\d+_epoch\d+\.ckpt*")
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile) is not None:
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    if args.model_type == "pretrain":
        model=CpG_regression_layer_single()
    elif args.model_type == "combined":
        pretrained_model=CpG_regression_layer_single()
        pretrained_model.load_state_dict(torch.load(args.seq_model, map_location=torch.device('cpu')))
        context_kinetics_model=context_kinetics_predicter(regression_layer=pretrained_model)
        model = CombinedModelv2(pretained=context_kinetics_model,structure=args.construct)

    else:
        raise ValueError("--model_type not right!")
    model=model.to(device)
    if args.init_model is not None:
        LOGGER.info("loading pre-trained model: {}".format(args.init_model))
        para_dict = torch.load(args.init_model, map_location=torch.device('cpu'))
        # para_dict = torch.load(model_path, map_location=torch.device(device))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)

    model = model.to(device)
    if not args.balance:
        weight=torch.tensor([1.0, 1.0])
    else:
        LOGGER.info("calculating weights.....")
        import pandas as pd
        data = pd.read_csv(args.train_file, delimiter='\t', header=None)
        labels = data.iloc[:, -1]
        # 统计每个类别的样本数量
        class_counts = labels.value_counts()
        weight = (1.0 / class_counts)
        weight= torch.tensor(weight)
    weight=weight.float().to(device)
    # Loss and optimizer
    if args.model_type in {"kinetics","combined","f","IPD","PW","contrast","seq"}:

        if args.focal:
            from simtsv.models import FocalLoss
            if args.smooth:
                criterion=FocalLoss(gamma=2,smoothing=0.05)
            else:
                criterion=FocalLoss()
        else:
            if args.smooth:
                criterion =nn.CrossEntropyLoss(weight=weight,label_smoothing=0.05)
            else:
                criterion = nn.CrossEntropyLoss(weight=weight)
    elif args.model_type in {"pretrain"}:
        criterion=nn.MSELoss()
    else:
        raise ValueError('incorrect model name')
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("--optim_type is not right!")
    if args.lr_scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay,
                                      patience=args.lr_patience, verbose=True)
    else:
        raise ValueError("--lr_scheduler is not right!")

    # Train the model
    total_step = len(train_loader)
    LOGGER.info("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    curr_best_accuracy_loc = 0
    curr_best_accuracy_epoches = []
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        accuracies_per_epoch = []
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            if len(sfeatures) == 4:
                fkmer, fipdm, fpwm, labels = sfeatures
            else:  # ==7
                fkmer, fipdm, fpwm, labels, chr, tpl, strand = sfeatures
            if args.model_type in {"pretrain"}:
                X = line2input(sfeatures, 'seq')
                #确定输入输出
                target_PW =fpwm.float()
                target_IPD = fipdm.float()
                X = X.to(device)
                labels = labels.to(device)
                target_PW = target_PW.to(device)
                target_IPD = target_IPD.to(device)
                # 前向
                predIPD, predPW = model(X, labels)
                loss = criterion(target_IPD, predIPD) + criterion(target_PW, predPW)
            elif args.model_type=="combined":
                #确定输入输出
                seq = line2input(sfeatures, 'seq41')
                X=line2input(sfeatures,'fk')
                seq=seq.to(device)
                X = X.to(device)
                labels = labels.to(device)
                # 前向
                outputs, logits = model(seq,X)
                loss=criterion(outputs, labels)
            else:
                raise ValueError("--model_type is not right!")
            tlosses.append(loss.detach().item())
        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    if args.model_type in {"combined"}:
                        vlosses, vlabels_total, vpredicted_total,vprob = [], [], [],[]
                        for vi, vsfeatures in enumerate(valid_loader):
                                vfkmer,vfipdm, vfpwm, vlabels = vsfeatures
                                if args.model_type == "combined":
                                    # 确定输入输出
                                    seq = line2input(vsfeatures, 'seq41')
                                    X = line2input(vsfeatures, 'fk')

                                    seq = seq.to(device)
                                    X = X.to(device)
                                    vlabels = vlabels.to(device)
                                    # 前向
                                    voutputs, vlogits = model(seq, X)
                                else:
                                    raise  Exception
                                vloss = criterion(voutputs, vlabels)
                                _, vpredicted = torch.max(vlogits.data, 1)
                                vlogits = vlogits.cpu()
                                vlabels = vlabels.cpu()
                                vpredicted = vpredicted.cpu()
                                vlosses.append(vloss.item())
                                vprob += vlogits[:, 1].tolist()
                                vlabels_total += vlabels.tolist()
                                vpredicted_total += vpredicted.tolist()
                                # print(vlabels,vpredicted,vlogits.data)
                        v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)

                        v_meanloss = np.mean(vlosses)
                        v_auc=metrics.roc_auc_score(vlabels_total, vprob)
                        v_precision = metrics.precision_score(vlabels_total, vpredicted_total)
                        v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
                        accuracies_per_epoch.append(v_accuracy)
                        if v_accuracy > curr_best_accuracy_epoch:
                            curr_best_accuracy_epoch = v_accuracy
                            # if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type +
                                       '.b{}_epoch{}.ckpt'.format(args.seq_len, epoch + 1))
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                curr_best_accuracy_loc = epoch + 1
                                no_best_model = False


                        time_cost = time.time() - start
                        try:
                            last_lr = scheduler.get_last_lr()
                            LOGGER.info('Epoch [{}/{}], Step [{}/{}]; LR: {:.4e}; TrainLoss: {:.4f}; '
                                        'ValidLoss: {:.4f}, AUC: {:.4f},'
                                        'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                        'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                        .format(epoch + 1, args.max_epoch_num, i + 1, total_step, last_lr,
                                                np.mean(tlosses), v_meanloss, v_auc,v_accuracy, v_precision, v_recall,
                                                curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                        except Exception:
                            LOGGER.info('Epoch [{}/{}], Step [{}/{}]; TrainLoss: {:.4f}; '
                                        'ValidLoss: {:.4f}, AUC: {:.4f},'
                                        'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                        'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                        .format(epoch + 1, args.max_epoch_num, i + 1, total_step,
                                                np.mean(tlosses), v_meanloss, v_auc, v_accuracy, v_precision, v_recall,
                                                curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                    elif args.model_type in {"pretrain"}:
                        vlosses, vlabels_IPD, vpredicted_IPD,vlabels_PW, vpredicted_PW = [], [], [],[],[]
                        for vi, vsfeatures in enumerate(valid_loader):
                            vfkmer,vfipdm, vfpwm, vlabels = vsfeatures
                            X = line2input(vsfeatures, 'seq')
                            target_PW = vfpwm.float()
                            target_IPD = vfipdm.float()
                            X = X.to(device)
                            vlabels = vlabels.to(device)
                            target_PW = target_PW.to(device)
                            target_IPD = target_IPD.to(device)
                            # 前向
                            predIPD, predPW = model(X, vlabels)
                            vloss = criterion(target_IPD, predIPD) + criterion(target_PW, predPW)
                            target_PW,target_IPD = target_PW.cpu(),target_IPD.cpu()
                            predIPD, predPW=predIPD.cpu(), predPW.cpu()
                            vlosses.append(vloss.item())
                            vlabels_IPD += target_IPD.reshape(-1).tolist() #(B,21,1)-->(B*21)
                            vlabels_PW += target_PW.reshape(-1).tolist()
                            vpredicted_IPD += predIPD.reshape(-1).tolist()
                            vpredicted_PW+=predPW.reshape(-1).tolist()

                        v_IPD_r2 = metrics.r2_score(vlabels_IPD, vpredicted_IPD)
                        v_IPD_mse = metrics.mean_squared_error(vlabels_IPD, vpredicted_IPD)
                        v_PW_r2 = metrics.r2_score(vlabels_PW, vpredicted_PW)
                        v_PW_mse = metrics.mean_squared_error(vlabels_PW, vpredicted_PW)
                        v_meanloss = np.mean(vlosses)
                        v_accuracy=(v_PW_r2+v_IPD_r2)/2
                        accuracies_per_epoch.append(v_accuracy)
                        if v_accuracy > curr_best_accuracy_epoch:
                            curr_best_accuracy_epoch = v_accuracy
                            # if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type +
                                       '.b{}_epoch{}.ckpt'.format(args.seq_len, epoch + 1))
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                curr_best_accuracy_loc = epoch + 1
                                no_best_model = False

                        time_cost = time.time() - start
                        try:
                            last_lr = scheduler.get_last_lr()
                            LOGGER.info('Epoch [{}/{}], Step [{}/{}]; LR: {:.4e}; TrainLoss: {:.4f}; '
                                        'ValidLoss: {:.4f}, '
                                        'Average_r2: {:.4f}, IPDr2: {:.4f}, IPDmse: {:.4f}, '
                                        'PWr2: {:.4f}, PWmse: {:.4f}, '
                                        'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                        .format(epoch + 1, args.max_epoch_num, i + 1, total_step, last_lr,
                                                np.mean(tlosses), v_meanloss, v_accuracy, v_IPD_r2, v_IPD_mse,
                                                v_PW_r2,v_PW_mse,
                                                curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                        except Exception:
                            LOGGER.info('Epoch [{}/{}], Step [{}/{}]; TrainLoss: {:.4f}; '
                                        'ValidLoss: {:.4f}, '
                                        'Average_r2: {:.4f}, IPDr2: {:.4f}, IPDmse: {:.4f}, '
                                        'PWr2: {:.4f}, PWmse: {:.4f}, '
                                        'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                        .format(epoch + 1, args.max_epoch_num, i + 1, total_step,
                                                np.mean(tlosses), v_meanloss, v_accuracy,v_IPD_r2, v_IPD_mse,
                                                v_PW_r2,v_PW_mse,
                                                curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                    else:
                        raise ValueError("unknown model type")
                    tlosses = []
                    start = time.time()
                    # sys.stdout.flush()
                model.train()

        if args.lr_scheduler == "ReduceLROnPlateau":
            if args.lr_mode_strategy == "mean":
                reduce_metric = np.mean(accuracies_per_epoch)
            elif args.lr_mode_strategy == "last":
                reduce_metric = accuracies_per_epoch[-1]
            elif args.lr_mode_strategy == "max":
                reduce_metric = np.max(accuracies_per_epoch)
            else:
                raise ValueError("--lr_mode_strategy is not right!")
            scheduler.step(reduce_metric)
        else:
            scheduler.step()

        curr_best_accuracy_epoches.append(curr_best_accuracy_epoch)
        if no_best_model and epoch >= args.min_epoch_num - 1:
            LOGGER.info("early stop!")
            break
    endtime = time.time()
    clear_linecache()
    if args.dl_offsets:
        train_dataset.close()
        valid_dataset.close()
    LOGGER.info("[main]train costs {:.1f} seconds, "
                "best accuracy: {} (epoch {})".format(endtime - total_start,
                                                      curr_best_accuracy,
                                                      curr_best_accuracy_loc))

def main():
    parser = argparse.ArgumentParser("train a model")
    st_input = parser.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = parser.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=False,default=".")

    # st_output.add_argument('--ref', type=str, required=True)

    st_train = parser.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    st_train.add_argument('--model_type', type=str, default="pretrain",

                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "default: attbigru2s")
    st_train.add_argument('--construct', type=int, default=1,
                          required=False,
                          help="1: # IPDR,PWR,seq"
                               "2: # IPDR,PWR"
                               "3: # IPDD,PWD,IPDR,PWR,seq"
                               "4: # IPDD,PWD,IPDR,PWR"
                               "5: # IPDD,PWD"
                               "6: # rawIPD,rawPW,seq"
                               "7: #rawIPD,rawPW,predIPD,predPW,seq"
                               "8: # rawIPD,rawPW,predIPD,predPW"
                               "9: # IPDD,PWD,seq"

                          )
    st_train.add_argument('--seq_model', type=str, default=None,
                          required=False,
                          help="model_dict for pretrained model")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiRNN model param
    st_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    st_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")
    st_train.add_argument('--focal', action='store_true',
                          help="whether use focal loss")
    st_train.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    st_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size, default 256")

    st_training = parser.add_argument_group("TRAINING")
    # model training
    st_training.add_argument('--balance', action='store_true',help="whether use balanced weight")
    st_training.add_argument('--smooth', action='store_true',help="whether use label smoothing")
    st_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                  "'Ranger' or 'LookaheadAdam', default Adam")
    st_training.add_argument('--batch_size', type=int, default=512, required=False)
    st_training.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', required=False,
                             choices=["StepLR", "ReduceLROnPlateau"],
                             help="StepLR or ReduceLROnPlateau, default StepLR")
    st_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001")
    st_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    st_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    st_training.add_argument('--lr_patience', type=int, default=0, required=False,
                             help="effective in ReduceLROnPlateau. default 0")
    st_training.add_argument('--lr_mode_strategy', type=str, default="last", required=False,
                             choices=["last", "mean", "max"],
                             help="effective in ReduceLROnPlateau. last, mean, or max, default last")
    st_training.add_argument("--max_epoch_num", action="store", default=50, type=int,
                             required=False, help="max epoch num, default 50")
    st_training.add_argument("--min_epoch_num", action="store", default=10, type=int,
                             required=False, help="min epoch num, default 10")
    st_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_training.add_argument('--step_interval', type=int, default=500, required=False)
    st_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")
    st_training.add_argument('--dl_offsets', action="store_true", default=False, required=False,
                             help="use file offsets loader")

    st_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    st_training.add_argument('--tseed', type=int, default=None,
                             help='random seed for pytorch')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()