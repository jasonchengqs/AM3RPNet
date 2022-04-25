import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        # self.P = args.period
        self.m = args.m_var
        self.hidC = args.h_cnn_short
        self.hidL = args.h_cnn_long
        self.hidR = args.h_rnn
        self.head = args.attn_head
        self.Ck = args.cnn_kernel
        
        # self.skip = args.skip
        # self.pt = (self.P - self.Ck)/self.skip
        # self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR, batch_first=True, bidirectional=True)
        
        self.conv2 = nn.Conv2d(self.hidC, self.hidL, kernel_size = (self.Ck, 1), stride=2)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=self.hidL, num_heads=self.head, batch_first=True)
        self.GRU2 = nn.GRU(self.hidL, self.hidR, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p = args.dropout)
 
    def forward(self, x):
        try:
            batch_size = x.size(0)
        except:
            batch_size = x.batch_sizes
        
        #CNN-short-term
        c = x.view(batch_size, 1, -1, self.m) # [b,hc,t,m]
        c = F.relu(self.conv1(c)) # [b,hc,t,1]
        c = self.dropout(c)
        c = torch.squeeze(c, 3) # [b,hc,t]
        #RNN-short-term
        r = c.permute(0, 2, 1).contiguous() # [b,t,hc]
        _, r = self.GRU1(r) # [2,b,hr]
        r = r.permute(1, 0, 2).contiguous() # [b,2,hr]
        # r = self.dropout(torch.squeeze(r, 1)) # [b,hr]
        r = self.dropout(r)

        #CNN-long-term
        l = c.view(batch_size, self.hidC, -1, 1)
        l = F.relu(self.conv2(l)) # [b,hc,ta,1]
        l = self.dropout(l)
        l = torch.squeeze(l, 3) # [b,hc,ta]
        #Attn
        l = l.permute(0, 2, 1).contiguous()
        a, _ = self.attn1(l, l, l, attn_mask=None, need_weights=False) # [b,ta,htr]
        #RNN-long-term
        _, a = self.GRU2(a) # [2,b,htr]
        a = a.permute(1, 0, 2).contiguous() # [b,2,htr]
        a = self.dropout(a)
 
        res = torch.cat((r,a),1) # [b,2+2,htr]
        res = res.view(batch_size, -1)
        
        # #highway
        # if (self.hw > 0):
        #     z = x[:, -self.hw:, :]
        #     z = z.permute(0,2,1).contiguous().view(-1, self.hw)
        #     z = self.highway(z)
        #     z = z.view(-1,self.m)
        #     res = res + z
            
        # if (self.output):
        #     res = self.output(res)
        return res
    
        
        
        
