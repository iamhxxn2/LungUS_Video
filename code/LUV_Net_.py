import torch.nn as nn
import torch
from utils import pad
import math

class MedVidNet_multi_attn_conv3_____(nn.Module):
    def __init__(self, encoder, num_heads, num_out=4, pooling_method='attn_multilabel_conv', drop_rate=0.0, debug=False, attn_hidden_size=32):
        super(MedVidNet_multi_attn_conv3_____, self).__init__()
        
        self.num_features = encoder.num_features
        assert self.num_features % num_heads == 0, "The number of encoder features must be divisible by the number of attention heads."
        self.num_heads = num_heads
        self.subspace_size = self.num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.num_out = num_out
        self.drop_rate = drop_rate
        self.debug = debug
        self.pool = pooling_method
        self.attn_hidden_size = attn_hidden_size
        self.encoder = encoder

        # Shared attention query vector
        self.attn_query_vecs = nn.Parameter(torch.randn(self.num_out, self.num_heads, self.subspace_size))
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Linear layer for Attention features
        self.attn_fc = nn.Sequential(
            nn.Linear(self.num_features * self.num_out, 4096),  # 중간 차원
            nn.ReLU(),
            nn.Linear(4096, self.num_out)  # 최종 차원
        )

    def forward(self, x, num_frames):
        h = self.encoder(x)  # Shape: [L*batch_size, num_features]
        
        ###############################################################
        # multi_attention_pool을 사용해 레이블별 feature 리스트(h_vid_outs) 반환
        h_vid_outs, attn = self.multi_attention_pool(h, num_frames)  # [batch_size, num_out, num_features]
        
        # 레이블별 h_vid_outs를 concat하여 하나의 feature로 결합
        h_vid_concat = h_vid_outs.view(h_vid_outs.size(0), -1)  # Shape: [batch_size, num_out * num_features]
        
        # 최종 출력
        # Linear transformation of Attention features
        output = self.attn_fc(h_vid_concat)  # Reduce to [batch_size, num_features]
        
        return output, attn

    def multi_attention_pool(self, h, num_frames):
        
        h_vid_lst = []
        attn_lst = []
        for i in range(self.num_out):
            
            h_query = h.view(-1, self.num_heads, self.subspace_size)
            attn_query_vecs = self.attn_query_vecs[i].to(h_query.device)
            
            alpha = (h_query * attn_query_vecs).sum(axis=-1) / self._scale
            alpha = pad(alpha, num_frames)
            for ix, n in enumerate(num_frames):
                alpha[n:, ix] = -50  # Mask for attention

            attn = torch.softmax(alpha, axis=0)
            h_query_pad = pad(h_query, num_frames)
            
            # Weighted sum of the features
            h_vid = torch.sum(h_query_pad * attn[..., None], dim=0)
            h_vid_wide = h_vid.view(-1, self.num_features)
            
            h_vid_lst.append(h_vid_wide)
            attn_lst.append(attn)
            
        h_vid_common = torch.stack(h_vid_lst, dim=1)  # Stack to keep separate channels for each label
        return h_vid_common, attn_lst
