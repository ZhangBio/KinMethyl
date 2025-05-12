#encoding=utf-8
#用于处理tsv文件
import os
import  random
from collections import defaultdict
import csv
import argparse
import warnings
iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
def match(motif,seq):
    n=len(motif)
    for i in range(n):
        if not seq[i] in iupac_alphabets[motif[i]]:
            return False
    return True
def pile_by_seq(args):
    file, outputfile=args.file,args.op
    if outputfile is None:
        fname, fext = os.path.splitext(file)
        outputfile = fname + ".features.tsv"
    else:
        outputfile = os.path.abspath(outputfile)
    dict_pile = defaultdict(lambda: [0, [0]*21, [0]*21, []])  # sum of pass, sum of ipd
    with open(file) as f:
        f_csv = csv.reader(f,delimiter='\t') #文件类型是tsv
        total_length=0
        skip=0
        for each_row in f_csv:
            total_length+=1
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label = each_row
            if args.type=="seq":
                idx = fkmer
            elif args.type=="pos":
                if chrom_pos=="-1":
                    skip +=1
                    continue
                idx=chrom+chrom_pos+strand
            elif args.type=="name": #按名字
                raise NotImplementedError #只会更多
            else:
                raise NotImplementedError
            if fpass==0:
                warnings.warn("fpass zero, corrected as 1")
                fpass=1

            fipdm=[ x*int(fpass) for x in list(map(float,fipdm.split(',')))]
            fpwm=[ x*int(fpass) for x in list(map(float, fpwm.split(',')))]
            if len(fipdm)!=21 or len(fpwm)!=21:
                continue
            dict_pile[idx][0] += int(fpass)
            dict_pile[idx][1] =[dict_pile[idx][1][i] +fipdm[i] for i in range(21)]
            dict_pile[idx][2] =[dict_pile[idx][2][i] +fpwm[i] for i in range(21)]
            dict_pile[idx][3] =  list(each_row)
        wf = open(outputfile, 'w')
        for idx in dict_pile:
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label =dict_pile[idx][3]
            fpass=dict_pile[idx][0]
            fipdm=",".join(["{:.6f}".format(i/fpass) for i in dict_pile[idx][1]])
            fpwm=",".join(["{:.6f}".format(i/fpass)for i in dict_pile[idx][2]])
            one_features_str="\t".join([chrom, chrom_pos, strand, seq_name, loc, fkmer, str(fpass), fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap,label])
            wf.write(one_features_str + "\n")
        wf.flush()
        print("total length:{} ,dict length:{},skiped:{}".format(total_length,len(dict_pile),skip))

#将ccsmeth生成的多列数据转换为4列 [seq,ipd,pw,label] #和simtsv中的输入一致
def convert(args):
    file, outputfile=args.file,args.op
    if outputfile is None:
        fname, fext = os.path.splitext(file)
        outputfile = fname + ".features.tsv"
    else:
        outputfile = os.path.abspath(outputfile)

    wf = open(outputfile, 'w')
    with open(file) as f:
        f_csv = csv.reader(f,delimiter='\t') #文件类型是tsv
        total_length=0
        skip=0
        for each_row in f_csv:
            total_length+=1
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label = each_row
            one_features_str="\t".join([fkmer, fipdm,fpwm,label])
            wf.write(one_features_str + "\n")
        wf.flush()
        print("input:{} converted to {}".format(file,outputfile))


# 合并相同位置的动力学数据
def pile_by_pos(args):
    file, outputfile=args.file,args.op
    if outputfile is None:
        fname, fext = os.path.splitext(file)
        outputfile = fname + ".features.tsv"
    else:
        outputfile = os.path.abspath(outputfile)

    n=1
    dict_pile = defaultdict(lambda: [0, [0]*n, [0]*n, []])  # sum of pass, sum of ipd
    with open(file) as f:
        f_csv = csv.reader(f,delimiter='\t') #文件类型是tsv
        total_length=0
        skip=0
        for each_row in f_csv:
            total_length+=1
            if not args.add_fpass:
                chrom, chrom_pos, strand, fipdm,fpwm = each_row
                fpass = 1
                each_row=[chrom, chrom_pos, strand, fipdm, fpwm,fpass]
            else:
                chrom, chrom_pos, strand, fipdm, fpwm,fpass = each_row
            if chrom_pos=="-1":
                skip +=1
                continue
            idx=chrom+"*"+chrom_pos+"*"+strand

            fipdm=[ x*int(fpass) for x in list(map(float,fipdm.split(',')))]
            fpwm=[ x*int(fpass) for x in list(map(float, fpwm.split(',')))]

            dict_pile[idx][0] += int(fpass)
            dict_pile[idx][1] =[dict_pile[idx][1][i] +fipdm[i] for i in range(n)]
            dict_pile[idx][2] =[dict_pile[idx][2][i] +fpwm[i] for i in range(n)]
            # dict_pile[idx][3] =  list(each_row)
        wf = open(outputfile, 'w')
        for idx in dict_pile:
            # chrom, chrom_pos, strand, fipdm,fpwm,fpass=dict_pile[idx][3]
            chrom, chrom_pos, strand=idx.split('*')
            fpass=dict_pile[idx][0]
            fipdm=",".join(["{:.6f}".format(i/fpass) for i in dict_pile[idx][1]])
            fpwm=",".join(["{:.6f}".format(i/fpass)for i in dict_pile[idx][2]])
            one_features_str="\t".join([chrom, chrom_pos, strand, fipdm,fpwm,str(fpass)])
            wf.write(one_features_str + "\n")
        wf.flush()
        print("total length:{} ,dict length:{},skiped:{}".format(total_length,len(dict_pile),skip))
def pile_by_name(args):
    file, outputfile=args.file,args.op
    if outputfile is None:
        fname, fext = os.path.splitext(file)
        outputfile = fname + ".features.tsv"
    else:
        outputfile = os.path.abspath(outputfile)
    dict_pile = defaultdict(lambda: [0, [0]*21, [0]*21, []])  # sum of pass, sum of ipd
    with open(file) as f:
        f_csv = csv.reader(f,delimiter='\t') #文件类型是tsv
        total_length=0
        for each_row in f_csv:
            # print(len(each_row))
            # _,_,_,_,_, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
            #     rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
            #     label = each_row
            total_length+=1
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label = each_row
            idx = seq_name[:seq_name.rfind("/")]
            fipdm=list(map(float,fipdm.split(','))) * int(fpass)
            fpwm=list(map(float, fpwm.split(','))) * int(fpass)
            if len(fipdm)!=21 or len(fpwm)!=21:
                continue
            dict_pile[idx][0] += int(fpass)
            dict_pile[idx][1] =[dict_pile[idx][1][i] +fipdm[i] for i in range(21)]
            dict_pile[idx][2] =[dict_pile[idx][2][i] +fpwm[i] for i in range(21)]
            dict_pile[idx][3] =  list(each_row)
        wf = open(outputfile, 'w')
        for idx in dict_pile:
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label =dict_pile[idx][3]
            fpass=dict_pile[idx][0]
            fipdm=",".join(["{:.6f}".format(i/fpass) for i in dict_pile[idx][1]])
            fpwm=",".join(["{:.6f}".format(i/fpass)for i in dict_pile[idx][2]])
            one_features_str="\t".join([chrom, chrom_pos, strand, seq_name, loc, fkmer, str(fpass), fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap,label])
            wf.write(one_features_str + "\n")
        wf.flush()
        print("total length:{} ,dict length:{}".format(total_length,len(dict_pile)))
#将tsv 分成两部分
def split_tsv(args):
    file, ratio=args.file,args.ratio
    ratio=list(map(float,ratio.split(",")))
    # 打开输入文件以读取内容
    name=file[:-4] #除去.tsv
    with open(file, 'r') as input_file:
        lines = input_file.readlines()

    # 随机打乱行
    random.shuffle(lines)

    # 计算每部分的行数
    total_lines = len(lines)
    part1_lines = int(total_lines * ratio[0]/10)
    part2_lines = int(total_lines * ratio[1] / 10)

    # 分割文件
    part1 = lines[:part1_lines]
    part2 = lines[part1_lines:part1_lines + part2_lines]
    part3 = lines[part1_lines + part2_lines:]

    # 写入分割后的文件
    with open(f'{name}_train.tsv', 'w') as part1_file:
        part1_file.writelines(part1)

    with open(f'{name}_dev.tsv', 'w') as part2_file:
        part2_file.writelines(part2)

    with open(f'{name}_test.tsv', 'w') as part3_file:
        part3_file.writelines(part3)
def average_kinetics(args):
    file,outputname=args.file,args.op
    with open(file) as f:
        f_csv = csv.reader(f,delimiter='\t') #文件类型是tsv
        total_length=0
        result_ipd=[]
        result_pw=[]
        for each_row in f_csv:
            total_length+=1
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label = each_row
            fipdm=list(map(float,fipdm.split(',')))
            fpwm=list(map(float, fpwm.split(',')))
            if len(fipdm)!=21 or len(fpwm)!=21:
                continue
            result_ipd.append(fipdm[10])
            result_pw.append(fpwm[10])
        density_plot(result_ipd,outputname+"ipd")
        density_plot(result_pw, outputname + "pw")
        print(f"mean ipd:{sum(result_ipd)/len(result_ipd)}, mean pw:{sum(result_pw)/len(result_pw)} ")
def density_plot(data,save_name):
    import matplotlib.pyplot as plt
    import seaborn as sns


    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建密度图
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, color="blue", shade=True)

    # 设置标题和标签
    plt.title("Density Plot", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # 保存为图片
    plt.savefig(f"{save_name}.png", dpi=300)  # 保存为300 DPI的PNG文件

#根据npass筛选
def filter(args):
    file = args.file
    op = args.op
    wf = open(op, 'w')
    m = n = 0
    with open(file) as f:
        f_csv = csv.reader(f, delimiter='\t')  # 文件类型是tsv
        for each_row in f_csv:
            m += 1
            chrom, chrom_pos, strand,  fipdm, fpwm, fpass = each_row
            if int (fpass)>=args.thres:
                n+=1
                wf.write("\t".join(each_row)+ "\n")
        wf.flush()
        print("total length:{} ,selected length:{}".format(m, n))
def generate_random_contrasive(args):
    import numpy as np
    file = args.file
    op = args.op
    wf = open(op, 'w')
    with open(file) as f:
        f_csv = csv.reader(f, delimiter='\t')  # 文件类型是tsv
        for each_row in f_csv:
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                label = each_row
            length=len(fipdm.split(","))
            fipdm=",".join([ str(round(np.random.normal(loc=0, scale=1),6)) for _ in range(length)]) if fipdm!="." else"."
            fipdsd = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if fipdsd!="." else"."
            fpwm = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if fpwm!="." else"."
            fpwsd = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if fpwsd!="." else"."
            ripdm = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if ripdm!="." else"."
            ripdsd = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if ripdsd!="." else"."
            rpwm = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if rpwm!="." else"."
            rpwsd = ",".join([str(round(np.random.normal(loc=0, scale=1), 6)) for _ in range(length)]) if rpwsd!="." else"."
            label=str(1-int(label)) #同序列但另一类别
            random_row=[chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap,
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap,  label ]
            wf.write("\t".join(random_row) + "\n") #写入random行
            wf.write("\t".join(each_row) + "\n") #写入原始行
        wf.flush()

#从tsv中选择 或移除某个motif
def select_motif(args):
        file = args.file
        op = args.op
        motif_list = []
        ex_motif_list = []
        if args.motif:
            motif = args.motif.split("*")
            shift = list(map(int, args.shift.split("*")))
            motif_list = list(zip(motif, shift))
        if args.ex_motif:
            ex_motif = args.ex_motif.split("*")
            ex_shift = list(map(int, args.ex_shift.split("*")))
            ex_motif_list = list(zip(ex_motif, ex_shift))
        wf = open(op, 'w')
        m=n=0
        with open(file) as f:
            f_csv = csv.reader(f, delimiter='\t')  # 文件类型是tsv
            for each_row in f_csv:
                m+=1 #原始行数
                chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                    label = each_row
                motif_check = True if motif_list == [] else False  #默认包括，但有motif要求时，默认不包括
                center=(len(fkmer)-1)//2

                for motif, shift in motif_list:
                    target = fkmer[center - shift + 1:center - shift + 1 + len(motif)] #C,1--->[10:11]
                    if match( motif,target):
                        motif_check = True #属于任何一个motif即可
                        break
                if not motif_check:  # 未包含在需要的motif中
                    continue

                ex_motif_check = False  # 是否排除motif，默认不排除
                for ex_motif, ex_shift in ex_motif_list:
                    target = fkmer[10 - ex_shift + 1:10 - ex_shift + 1 + len(ex_motif)]
                    if match(ex_motif,target) :
                        ex_motif_check = True #属于任意一个motif排除
                        break
                if ex_motif_check:  # 包含在需要排除的motif中
                    continue
                n+=1 #总计行数

                label=label if args.label==None else args.label
                each_row=[chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap,
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap,  label ]
                wf.write("\t".join(each_row)+ "\n")
            wf.flush()
            print("total length:{} ,selected length:{}".format(m, n))
def pile_by_none(args):
    file, outputfile = args.file, args.op
    if outputfile is None:
        fname, fext = os.path.splitext(file)
        outputfile = fname + ".features.tsv"
    else:
        outputfile = os.path.abspath(outputfile)
    wf = open(outputfile, 'w')
    with open(file) as f:
        f_csv = csv.reader(f, delimiter='\t')  # 文件类型是tsv
        total_length = 0
        for each_row in f_csv:
            total_length += 1
            chrom, chrom_pos, strand, seq_name, loc, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
            rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
            label = each_row
            fipdm = list(map(float, fipdm.split(',')))
            fpwm = list(map(float, fpwm.split(',')))
            if len(fipdm) != 21 or len(fpwm) != 21:
                continue
            one_features_str = "\t".join([fkmer, str(fipdm[10]),str(fpwm[10]),str(fpass)])
            wf.write(one_features_str + "\n")
        wf.flush()

def main(args):
    if args.command == 'pile':
        # args.full_kinetics
        pile_by_seq(args)
    elif args.command == 'split':
        split_tsv(args)
    elif args.command == 'select':
        select_motif(args)
    elif args.command == 'gen':
        generate_random_contrasive(args)
    elif args.command == 'pp':
        pile_by_pos(args)
    elif args.command == 'pn':
        pile_by_none(args)
    elif args.command == 'filter':
       filter(args)
    elif args.command=="plot_average":
        average_kinetics(args)
    elif args.command=="convert":
        convert(args)
    else:
        raise ValueError(" undefined command")
if __name__ == '__main__': #读取并保存为npz文件
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers.required = True
    subparsers.dest = 'command'

    parser_a = subparsers.add_parser('pile')
    parser_a.add_argument('file', help='fname1')
    parser_a.add_argument('op', help='fname1')
    parser_a.add_argument('--type', choices=['seq','pos','name'])

    parser_b = subparsers.add_parser('split', help='split help')
    parser_b.add_argument('file', help='full_kinetics')
    parser_b.add_argument('ratio', help='Methylated positions/kinetics')

    parser_c = subparsers.add_parser('select')
    parser_c.add_argument('file')
    parser_c.add_argument('op')
    parser_c.add_argument('--motif', default=None)
    parser_c.add_argument('--shift', default=None)
    parser_c.add_argument('--ex_motif', default=None)
    parser_c.add_argument('--ex_shift', default=None)
    parser_c.add_argument('--label', default=None,type=str,choices=["0","1"])

    parser_d = subparsers.add_parser('filter', help='filter help')
    parser_d.add_argument('file')
    parser_d.add_argument('op')
    parser_d.add_argument('thres',type=int ,default=20)

    parser_e = subparsers.add_parser('gen', help='generate random kinetics lines with same seq')
    parser_e.add_argument('file')
    parser_e.add_argument('op')

    parser_f = subparsers.add_parser('pp', help='pile by pos, work with tsv file that only has 5 columns')
    parser_f.add_argument('file')
    parser_f.add_argument('op')
    parser_f.add_argument('--add_fpass',action="store_true")

    parser_g = subparsers.add_parser('plot_average')
    parser_g.add_argument('file')
    parser_g.add_argument('op')

    parser_h = subparsers.add_parser('convert') #将由ccsmeth产生的数据列数减少
    parser_h.add_argument('file')
    parser_h.add_argument('op')


    parser_i = subparsers.add_parser('pn') #并不pile只是生成4列的文件用于单位点回归
    parser_i.add_argument('file')
    parser_i.add_argument('op')
    args = parser.parse_args()
    main(args)