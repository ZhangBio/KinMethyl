import argparse


import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import average_precision_score


from simtsv.dataloader import FeaData
from simtsv.dataloader import line2input,code_dna2base
from torch.utils.data import DataLoader
from simtsv.models import BiGRU,CpG_regression_layer,context_kinetics_predicter,CombinedModelv2
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
from simtsv.logger import mylogger
LOGGER = mylogger(__name__)
# 设置 PyTorch 随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import  csv
# 设置 NumPy 随机种子
np.random.seed(seed)
def feature2str(features,probs): #用于预测tensor变为原先输入格式的tsv文件
    n=len(features)
    if n==4:
        vfkmer, vfipdm, vfpwm, vlabels=features
    else:
        vfkmer, vfipdm, vfpwm, vlabels, vchr, vtpl, vstrand=features
    batchsize=len(vfkmer)
    data=[]
    for i in range(batchsize):
        fkmer=vfkmer[i]
        fipdm=vfipdm[i]
        fpwm=vfpwm[i]
        prob=probs[i]
        labels=vlabels[i]
        fkmer = "".join([code_dna2base[i] for i in fkmer.numpy().astype(int)])
        fipdm=",".join([f"{x:.6f}" for x in fipdm.numpy()])
        fpwm = ",".join([f"{x:.6f}" for x in fpwm.numpy()])
        labels=str(labels.numpy())
        prob=f"{prob:.16f}"

        if n==4:
            data.append([fkmer,fipdm,fpwm,labels,prob])
        else:
            chr,tpl,strand=vchr[i], vtpl[i], vstrand[i]
            data.append([fkmer,fipdm,fpwm,labels,str(chr),str(tpl),str(strand),prob])
    return data



def test(args):
        LOGGER.info("[main]train starts")
        if use_cuda:
            LOGGER.info("GPU is available!")
        else:
            LOGGER.info("GPU is not available!")

        LOGGER.info("reading data..")
        test_dataset = FeaData(args.data_file)
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                    )
        if args.model_type == "combined":
            structure_tables={
            1:  " #IPDR,PWR,seq",
            2: "# IPDR,PWR",
            3: "# IPDD,PWD,IPDR,PWR,seq",
            4: "# IPDD,PWD,IPDR,PWR",
            5: "# IPDD,PWD",
            6: "# rawIPD,rawPW,seq",
            7: "#rawIPD,rawPW,predIPD,predPW,seq",
            8: "# rawIPD,rawPW,predIPD,predPW",
            9: "# IPDD,PWD,seq",
            10:"gated",
            11: "additive"
            }
            model=CombinedModelv2(pretained=context_kinetics_predicter(),structure=args.construct)
            LOGGER.info("test on structure:{}, features:{}".format(args.construct,structure_tables.get(args.construct,args.construct)))
            # feature_layer = feature_GRU_layer()
            # model = CombinedModel(pretained=feature_layer)
        else:
            raise Exception
        try:
            model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
        except:
            for k, v in model.state_dict().items():
                print(k, v.shape)
            ckpt_path = args.model_file  # 例如 "./2/combined.b21_epoch3.ckpt"
            print("-----")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            for k, v in checkpoint.items():
                print(k, v.shape)
            raise Exception
        model.eval()
        model=model.to(device)
        with torch.no_grad():
                vlosses, vlabels_total, vpredicted_total = [], [], []
                vprob=[]
                for vi, vsfeatures in enumerate(test_loader):
                    if len(vsfeatures)==4:
                        vfkmer,vfipdm, vfpwm, vlabels = vsfeatures
                    else: #==7
                        vfkmer, vfipdm, vfpwm, vlabels,vchr,vtpl,vstrand = vsfeatures

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
                        raise  NotImplementedError("test.py is only for combined model")
                    if use_cuda:
                        vlogits=vlogits.cpu()
                        vlabels = vlabels.cpu()
                        vpredicted = vpredicted.cpu()
                    probi=vlogits[:, 1].tolist()
                    vprob += probi
                    vlabels_total += vlabels.tolist()
                    vpredicted_total += vpredicted.tolist()
                    if args.output:
                        data=feature2str(vsfeatures,probi)
                        with open(args.output, mode='a', newline='') as file:
                            writer = csv.writer(file,delimiter='\t')
                            # 逐行写入数据
                            writer.writerows(data)

                v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
                if args.model_type not in {'contrast'}:
                    v_auc = metrics.roc_auc_score(vlabels_total, vprob)
                    v_precision = metrics.precision_score(vlabels_total, vpredicted_total)
                    v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
                    aupr = average_precision_score(vlabels_total, vprob)
                    plot_density(vprob)
                LOGGER.info(' AUC: {:.4f},'
                            'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, AUPR:{:.4f}'
                            .format(v_auc, v_accuracy, v_precision, v_recall,aupr))
def plot_density(prob,savename="probability_density.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 假设 prob 是你预测的概率值数组
    # 使用 seaborn 绘制概率密度估计（KDE）
    sns.kdeplot(prob, fill=True, color="blue")
    # 也可以使用 matplotlib 绘制直方图
    plt.hist(prob, bins=30, density=True, alpha=0.5, color='gray')
    # 添加标题和标签
    plt.title('Probability Density of Predicted Values')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')

    # 显示图形
    plt.savefig(savename, dpi=300)
def main():
    parser = argparse.ArgumentParser("train a model")
    st_input = parser.add_argument_group("INPUT")
    st_input.add_argument('--data_file', type=str, required=True)
    st_input.add_argument('--model_file', type=str, required=True)

    st_train = parser.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    st_train.add_argument('--model_type', type=str, default="f",
                          choices=["kinetics","combined","f","contrast",'seq'],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "default: f")
    st_train.add_argument('--construct', type=int, default=1,
                          required=False,
                          help="1: # IPDR,PWR,seq"
                               "2: # IPDR,PWR"
                               "3: # IPDD,PWD,IPDR,PWR,seq"
                               "4: # IPDD,PWD,IPDR,PWR"
                               "5: # IPDD,PWD"
                          )
    st_train.add_argument('--batch_size', type=int, default=128,
                          required=False,
                          )
    st_train.add_argument('--o', type=int, default=128,
                          required=False,
                          )
    st_input.add_argument('--output', default=None,type=str) #是否把预测的logit输出到文件中
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()