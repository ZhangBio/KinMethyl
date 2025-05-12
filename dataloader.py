import numpy as np
import torch
from torch.utils.data import Dataset
import linecache
import os
import re
def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()

def is_valid_dna(seq):
    return re.fullmatch(r'[AGTC]+', seq.upper()) is not None
base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                 'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
                 'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
                 'Z': 15}

code_dna2base={0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N',
               5: 'W', 6: 'S', 7: 'M', 8: 'K', 9: 'R',
               10: 'Y', 11: 'B', 12: 'V', 13: 'D', 14: 'H',
               15: 'Z'}

def parse_a_line(line):
    words = line.strip().split("\t")
    n=len(words)
    if n==4:
        fkmer=words[0]
        try:
            fkmer = np.array([base2code_dna[x] for x in fkmer])
        except:
            print(words)
            raise Exception
        fipdm = np.array([float(x) for x in words[1].split(",")], dtype=float)
        fpwm = np.array([float(x) for x in words[2].split(",")], dtype=float)
        label=int(words[3])
        return fkmer,fipdm, fpwm,label
    elif n==7: #n==7
        fkmer=words[0]
        chrom=words[-2]
        tpl=words[-1]
        strand=1 #暂时只支持一条链
        try:
            fkmer = np.array([base2code_dna[x] for x in fkmer])
        except:
            print(words)
            raise Exception
        fipdm = np.array([float(x) for x in words[1].split(",")], dtype=float)
        fpwm = np.array([float(x) for x in words[2].split(",")], dtype=float)
        label=int(words[3])
        return fkmer,fipdm, fpwm,label,chrom,tpl,strand
    else: #用于返回ccsmeth生成的文件
        # chrom, chrom_pos, strand, seq_name, loc,
        #     fkmer_seq, npass_fwd, fkmer_im, fkmer_isd, fkmer_pm, fkmer_psd,
        #     fkmer_qual, fkmer_map,
        #     rkmer_seq, npass_rev, rkmer_im, rkmer_isd, rkmer_pm, rkmer_psd,
        #     rkmer_qual, rkmer_map,
        #     methy_label
        fkmer = np.array([base2code_dna[x] for x in words[5]])
        fipdm = np.array([float(x) for x in words[7].split(",")], dtype=float)
        fpwm = np.array([float(x) for x in words[9].split(",")], dtype=float)

        label = int(words[21])

        return fkmer,fipdm, fpwm,label

def generate_offsets(filename):
    offsets = []
    with open(filename, "r") as rf:
        offsets.append(rf.tell())
        while rf.readline():
            offsets.append(rf.tell())
    return offsets

class FeaData(Dataset):
    def __init__(self, filename, transform=None):
        self._filename = os.path.abspath(filename)
        self._transform = transform
        self._valid_indices = []

        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if line.strip() == "":
                    continue
                if self._is_valid_line(line):
                    self._valid_indices.append(i + 1)  # linecache is 1-based

        self._total_data = len(self._valid_indices)

    def _is_valid_line(self, line):
        try:
            words = line.strip().split("\t")
            if len(words)<=7: #由自己的流程产出的
                seq=words[0]
            else: #由ccsmeh +select产出的
                seq=words[5]
            return is_valid_dna(seq)
        except:
            return False

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        line_number = self._valid_indices[idx]
        line = linecache.getline(self._filename, line_number)
        output = parse_a_line(line)
        if self._transform is not None:
            output = self._transform(output)
        return output
def line2input(feature,type):
    if len(feature)==4:
        fkmer, fipdm, fpwm,label=feature
    else: #=7
        fkmer, fipdm, fpwm, label,_,_ ,_= feature

    if fipdm.shape[1]==41: #使用ccsmeth seq_len设置为41时提取会产生41context信息,但实际只使用21长度
        fipdm=fipdm[:,10:31]
        fpwm=fpwm[:,10:31]
    if type=="fk":
        X=np.concatenate([fipdm[:,:, None],fpwm[:,:, None]],axis=-1) #(B,21,2)
    elif type=="seq":
        if fkmer.shape[1]==41:
            X=fkmer[:,10:31, None]
        else:
            X=fkmer[:,:, None] #(B,L,1)
    elif type=="seq41":
        X=fkmer[:,:, None] #(B,L,1)
    elif type=="IPD":
        X=fipdm[:,:, None] #(B,L,1)
    elif type=="PW":
        X=fpwm[:,:, None] #(B,L,1)
    elif type=="fw":
        seq=fkmer[:,:, None] #(B,L,1)
        X = np.concatenate((seq,fipdm[:,:, None], fpwm[:,:, None]), axis=-1) #(B,21,3)
    else:
        raise NotImplementedError
    return torch.as_tensor(X).float()

