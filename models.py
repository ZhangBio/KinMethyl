import torch
from torch import nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self,num_inputs=2, num_hiddens=512,num_layers=2,output_size=2,dropout=0.3, **kwargs):
        super(BiGRU, self).__init__(**kwargs)
        self.num_directions = 2
        self.num_inputs = num_inputs #特征维度：V of (B,L,V)
        self.rnn =  nn.GRU(input_size=num_inputs, hidden_size=num_hiddens, bidirectional=True, num_layers=num_layers,dropout=dropout,batch_first=True)
        self.num_hiddens = self.rnn.hidden_size #128
        self.relu=nn.ReLU()
        self.linear1 = nn.Linear(self.num_hiddens * 2*21, output_size)
        # self.linear2 = nn.Linear(21, 2)
        self.softmax = nn.Softmax(1)
    def forward(self, X):
        self.rnn.flatten_parameters()
        output, _ = self.rnn(X)  # [B,L,2H]#默认使用全0的隐状态
        output =output.reshape((output.shape[0],-1))
        output = self.linear1(output)
        return output,self.softmax(output)


class feature_GRU_layer(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=10, dropout=0.3,output_size=256, **kwargs):
        super(feature_GRU_layer, self).__init__(**kwargs)
        self.num_directions = 2
        self.num_inputs = num_inputs  # 特征维度：V of (B,L,V)

        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, dropout=dropout,batch_first=True)
        self.num_hiddens = self.rnn.hidden_size  # 128
        self.linear=nn.Linear(num_hiddens*2,output_size)
    def forward(self, X):
        # self.rnn.flatten_parameters()  # 使用多个GPU时,权重会被分别存储,将权重压成一块，减少内存使用
        X, _ = self.rnn(X)  # [B,L,2H]#默认使用全0的隐状态
        X=self.linear(X) #(B,L,256)
        return X
class LSTM_CNN(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=4, dropout=0.3, output_size=256, **kwargs):
        super(LSTM_CNN, self).__init__(**kwargs)
        self.num_directions = 2
        self.num_inputs = num_inputs  # 特征维度：V of (B,L,V)

        self.rnn = nn.LSTM(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.cnn=nn.Conv1d(in_channels=num_hiddens * 2, out_channels=output_size, kernel_size=7, stride=1, padding=3)
        #不改变数据的长度
    def forward(self, X):
        # self.rnn.flatten_parameters()  # 使用多个GPU时,权重会被分别存储,将权重压成一块，减少内存使用
        X, _ = self.rnn(X)  # [B,L,2H]#默认使用全0的隐状态
        X=X.permute(0, 2, 1) #(B,256,L）
        X = self.cnn(X)  # (B,256,L）
        X = X.permute(0, 2, 1)  #(B,L,256)
        return X
class regression_layer(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=4, dropout=0.3, output_size=256, **kwargs):

        super(regression_layer, self).__init__(**kwargs)
        self.feature_layer=LSTM_CNN(num_inputs, num_hiddens, num_layers, dropout,output_size)
        self.num_inputs=num_hiddens*2
        self.MIPD_linear = nn.Linear(output_size, 1)
        self.MPW_linear = nn.Linear(output_size, 1)
        self.WPW_linear = nn.Linear(output_size, 1)
        self.WIPD_linear = nn.Linear(output_size, 1)

    def forward(self, X,input_type):
        X=self.feature_layer(X) #(B,L,2H)
        input_type=input_type.reshape(-1,1,1) #(B,1,1)
        output1=self.MIPD_linear(X)*input_type+(1-input_type)*self.WIPD_linear(X) #(B,L,1)
        output2 = self.MPW_linear(X)*input_type+(1-input_type)*self.WPW_linear(X) #(B,L,1)
        return output1.squeeze(-1).float(),output2.squeeze(-1).float() #(B,L)

class CpG_regression_layer(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=10, output_size=256, dropout=0.3, **kwargs):

        super(CpG_regression_layer, self).__init__(**kwargs)
        self.feature_layer=feature_GRU_layer(num_inputs, num_hiddens, num_layers, dropout,output_size)
        self.num_inputs=num_hiddens*2
        self.MIPD_linear = nn.Linear(output_size, 1)
        self.MPW_linear = nn.Linear(output_size, 1)
        self.WPW_linear = nn.Linear(output_size, 1)
        self.WIPD_linear = nn.Linear(output_size, 1)

    def forward(self, X,input_type):
        X=self.feature_layer(X) #(B,L,2H)
        input_type=input_type.reshape(-1,1,1) #(B,1,1)
        output1=self.MIPD_linear(X)*input_type+(1-input_type)*self.WIPD_linear(X) #(B,L,1)
        output2 = self.MPW_linear(X)*input_type+(1-input_type)*self.WPW_linear(X) #(B,L,1)
        return output1.squeeze(-1).float(),output2.squeeze(-1).float() #(B,L)
class CpG_regression_layer_single(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=2, output_size=256, dropout=0.3, **kwargs):

        super(CpG_regression_layer_single, self).__init__(**kwargs)
        self.feature_layer = feature_GRU_layer(num_inputs, num_hiddens, num_layers, dropout, output_size)
        self.num_inputs = num_hiddens * 2
        self.MIPD_linear = nn.Linear(output_size * 21, 1)
        self.MPW_linear = nn.Linear(output_size * 21, 1)
        self.WPW_linear = nn.Linear(output_size * 21, 1)
        self.WIPD_linear = nn.Linear(output_size * 21, 1)

    def forward(self, X):  # 注意，这里是为了单位点回归设置的
        X = self.feature_layer(X)  # (B,L,2H)
        X = X.reshape(X.shape[0], -1)  # (B,21*output_size)
        output1 = self.MIPD_linear(X) # (B,1)
        output2 = self.MPW_linear(X) # (B,1)
        return output1.float(), output2.float()  # (B,1)
class CombinedModel(nn.Module):#使用cat
    def __init__(self,pretained:feature_GRU_layer, num_inputs=2, num_hiddens=128, num_layers=2, dropout=0.3, **kwargs):

        super(CombinedModel, self).__init__(**kwargs)
        self.feature_layer = pretained #(B,L,4)-->(B,L,H)
        for param in self.feature_layer.parameters(): #冻住seq linear
            param.requires_grad = False
        # if self.feature_layer.rnn.hidden_size!=num_hiddens:
        #     raise ValueError ("unmatched hidden size, pretrained:{}, combined:{}".format(self.feature_layer.rnn.hidden_size,num_hiddens))
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, dropout=dropout,batch_first=True)
        self.num_hiddens = self.rnn.hidden_size  # 128
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.num_hiddens * 4*21, 2)

        self.linear2 = nn.Linear(self.num_hiddens * 2*21, 2)
    def forward(self,seq,kinetics):
        seq=self.feature_layer(seq)#B,L,2H
        kinetics, _ = self.rnn(kinetics)  # [B,L,2H]#默认使用全0的隐状态
        combined=torch.cat([seq,kinetics],dim=-1) #B,L,4H
        combined = combined.reshape((combined.shape[0], -1))
        output = self.linear1(combined)  # [B,L,2]

        kinetics=kinetics.reshape((kinetics.shape[0], -1))
        output2=self.linear2(kinetics) #B,L,2

        output=output+output2
        self.softmax = nn.Softmax(1)
        return output,self.softmax(output) #(B,C)
class kinetics_layer(nn.Module):
    def __init__(self, num_inputs=1, num_hiddens=512, num_layers=2, output_size=256, dropout=0.3, **kwargs):

        super(kinetics_layer, self).__init__(**kwargs)
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.num_hiddens = self.rnn.hidden_size  # 128
        self.linear = nn.Linear(num_hiddens * 2, output_size)
        self.num_inputs = num_hiddens * 2
        self.WPW_linear = nn.Linear(output_size * 21, 1)
        self.WIPD_linear = nn.Linear(output_size * 21, 1)

    def forward(self, X):  # 注意，这里是为了单位点回归设置的 #in:(B,21,1)--->out:(B,1),(B,1)
        X, _ = self.rnn(X)  # [B,L,2H]#默认使用全0的隐状态
        X = self.linear(X)  # (B,L,256)
        X = X.reshape(X.shape[0], -1)  # (B,21*output_size)
        output1 = self.WIPD_linear(X) # (B,1)
        output2 = self.WPW_linear(X) # (B,1)
        return output1.float(), output2.float()  # (B,1)
class context_kinetics_predicter(nn.Module):
        def __init__(self, regression_layer=None):
            super(context_kinetics_predicter, self).__init__()
            if regression_layer==None:
                # self.single_site_predicter=kinetics_layer()
                self.single_site_predicter=CpG_regression_layer_single()
            else:
                self.single_site_predicter=regression_layer
        def forward(self, X):  # 将41bp 拆分为21个21bp分别单位点预测 #(B,41,1)--->(B*21,21,1)--->(B*21,1)-->(B,21)

            X  = X.unfold(1, 21, 1).reshape(-1,21,1)  # (B,41,1)--->(B*21,21,1)
            X = 3 - X  # !!! strand flips here part1 complement
            X=X.flip(1)# !!! strand flips here part2 reverse
            IPDs,PWs=self.single_site_predicter(X) #(B*21,1)
            return IPDs.reshape(-1,21,1), PWs.reshape(-1,21,1)  # (B,21,1)
class CombinedModelv2(nn.Module): #seq变为
    def __init__(self,pretained:context_kinetics_predicter, structure=1, num_hiddens=128, num_layers=2, dropout=0.3, **kwargs):

        super(CombinedModelv2, self).__init__(**kwargs)
        self.context_kinetics_predicter = pretained #(B,21,1)-->(B,1),(B,1)

        self.context_kinetics_predicter.eval()#禁用drop out #0312
        for param in self.context_kinetics_predicter.parameters(): #冻住seq linear
            param.requires_grad = False

        self.structure=structure
        if structure in {11}:
            self.num_inputs = 2  # [IPD, PW]
            self.rnn_raw = nn.GRU(2, num_hiddens, bidirectional=True, num_layers=num_layers,
                                  dropout=dropout, batch_first=True)
            self.rnn_pred = nn.GRU(2, num_hiddens, bidirectional=True, num_layers=num_layers,
                                   dropout=dropout, batch_first=True)
            self.seq_linear = nn.Linear(4, num_hiddens * 2)  # seq is one-hot with 4 dims
            self.linear= nn.Linear(num_hiddens * 2 * 21, 2)
            self.softmax = nn.Softmax(1)
            self.eps = 1e-10
            return
        if structure in {1,6,9,14,15}: # 1:IPDR,PWR,seq  6:rawIPD,rawPW,seq,14:IPD+predIPD+seq, 15:PW+predPW+seq
            self.num_inputs=3
        elif structure in {2,5,12,13}: # 2:IPDR,PWR 5:IPDD,PWD,12:IPD+seq 13:PW+seq
            self.num_inputs=2
        elif structure in {3,7}: # 3:IPDD,PWD,IPDR,PWR,seq 7:rawIPD,rawPW,predIPD,predPW,seq
            self.num_inputs=5
        elif structure in {4,8}: # 4:IPDD,PWD,IPDR,PWR 8：rawIPD,rawPW,predIPD,predPW
            self.num_inputs=4
        elif structure in {10}: #gated kinetics
            self.num_inputs = 8
            self.gate_layer = nn.Linear(2, 2)  #
        elif structure in {16,17,18,19}: #16:IPDR ,17:PWR, 18:IPD, 19,PW
            self.num_inputs = 1
        else:
            raise ValueError("unrecognized model structure {}".format(structure))
        self.structure=structure
        self.relu = nn.ReLU()
        self.rnn = nn.GRU(self.num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.num_hiddens = self.rnn.hidden_size  # 128
        self.linear1 = nn.Linear(self.num_hiddens * 2*21, 2)
        self.softmax = nn.Softmax(1)
        self.eps=1e-10
    def get_embedding(self, seq, kinetics):
        predip,predpw=self.context_kinetics_predicter(seq) # (B,21,1)
        predk=torch.cat([predip+self.eps,predpw+self.eps],dim=-1) #(B,21,2)
        seq21=seq[:, 10:31, :]
        ipd=kinetics[:,:,0:1]
        pw=kinetics[:,:,1:2]
        if self.structure==11:
            seq = seq[:, 10:31, :].long()  # (B,21,1)
            seq_onehot = F.one_hot(seq.squeeze(-1), num_classes=4).float()
            seq_out = self.seq_linear(seq_onehot)  # (B,21,H*2)
            raw_out, _ = self.rnn_raw(kinetics)  # (B,21,H*2)
            pred_out, _ = self.rnn_pred(predk)  # (B,21,H*2)
            output = raw_out + pred_out + seq_out  # (B,21,H*2)
            embedding = output.reshape(output.shape[0], -1)  # (B, 21*2H)
            return embedding
        if self.structure==1: #IPDR,PWR,seq
            seq = seq[:, 10:31, :]  # (B,41,1)--->(B,21,1)
            combined = torch.cat([seq, kinetics / predk], dim=-1) #
        elif self.structure==2: #IPDR,PWR
            combined= kinetics / predk
        elif self.structure==3: #IPDD,PWD,IPDR,PWR,seq
            seq = seq[:, 10:31, :]
            combined=torch.cat([kinetics-predk,kinetics/predk,seq],dim=-1)
        elif self.structure==4: # IPDD,PWD,IPDR,PWR
            combined = torch.cat([kinetics - predk, kinetics / predk], dim=-1)  # B,L,4
        elif self.structure == 5:  # IPDD,PWD
            combined =kinetics - predk
        elif self.structure == 6:  # rawIPD,rawPW,seq 3
            seq = seq[:, 10:31, :]
            combined =torch.cat([kinetics, seq], dim=-1)
        elif self.structure == 7: #rawIPD,rawPW,predIPD,predPW,seq#5
            seq = seq[:, 10:31, :]
            combined = torch.cat([kinetics,predk, seq], dim=-1)
        elif self.structure == 8:#rawIPD,rawPW,predIPD,predPW
            combined = torch.cat([kinetics, predk], dim=-1)
        elif self.structure == 9:  # IPDD,PWD,seq
            seq = seq[:, 10:31, :]
            combined = torch.cat([kinetics-predk, seq], dim=-1)
        elif self.structure == 10:  # rawIPD, rawPW, predIPD, predPW, seq
            seq = seq[:, 10:31, :]  # (B,21,4)
            # Gated fusion
            seq = seq.long()
            seq_onehot = F.one_hot(seq.squeeze(-1), num_classes=4).float()
            gate = torch.sigmoid(self.gate_layer(predk))  # (B,21,2)
            filtered_predk = gate * predk  # gated pred signal
            combined = torch.cat([kinetics, filtered_predk, seq_onehot], dim=-1)  # (B,21,D)
        elif self.structure == 12:  # IPD+seq
            combined = torch.cat([ipd, seq21], dim=-1)
        elif self.structure == 13:  # PW+seq
            combined = torch.cat([pw, seq21], dim=-1)
        elif self.structure == 14:  # IPD+predIPD+seq
            combined = torch.cat([ipd,predip, seq21], dim=-1)
        elif self.structure == 15:  # pw+predpw+seq
            combined = torch.cat([pw,predpw, seq21], dim=-1)
        elif self.structure == 16:  # IPDR
            combined = ipd/(predip+self.eps)
        elif self.structure == 17:  # PWR
            combined = pw/(predpw+self.eps)
        elif self.structure == 18:  #ipd
            combined = ipd
        elif self.structure == 19:  #pw
            combined = pw
        else:
            raise Exception
        # 注意：需与 forward 保持一致
        output,_=self.rnn(combined) #(B,21,128*2)
        output=output.reshape(output.shape[0],-1)#(B,21*2H)
        return output
    def forward(self,seq,kinetics):

        embed=self.get_embedding(seq,kinetics)
        # combined =kinetics/predk
        if self.structure not in {11}:
            output=self.linear1(embed) # (B,2)
        else:
            output=self.linear(embed)
        return output,self.softmax(output) #(B,C)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets): #(B,C) (B,)
        num_classes = inputs.size(1)
        confidence = 1.0 - self.smoothing
        low_confidence = self.smoothing / (num_classes - 1)

        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed_targets = one_hot * confidence + low_confidence

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1 - probs).pow(self.gamma)

        loss = -(smoothed_targets * focal_weight * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
