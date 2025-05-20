import torch.nn as nn
import torch
from utils import pad
import math

class LUV_Net(nn.Module):
    def __init__(self, encoder, num_heads, num_out=4, pooling_method='attn_multilabel_conv', drop_rate=0.0, debug=False, attn_hidden_size=32, kernel_width=5):
        super(LUV_Net, self).__init__()
        
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

        # attention query vector for each patterns
        self.attn_query_vecs = nn.Parameter(torch.randn(self.num_out, self.num_heads, self.subspace_size))

        # Conv1d layer를 한층만으로 구성
        self.conv1d = nn.Sequential(
            nn.Conv1d(self.num_features, self.num_features, kernel_size=kernel_width, padding=(kernel_width - 1) // 2, bias=False),
#             nn.Conv1d(self.num_features, self.num_features, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Linear layer for Attention features
        self.attn_fc = nn.Sequential(
            nn.Linear(self.num_features * self.num_out, 4096),  # 중간 차원
            nn.ReLU(),
            nn.Linear(4096, self.num_features)  # 최종 차원
        )

        self.fc_out_final = nn.Sequential(
            nn.Linear(self.num_features, 512),  # 중간 레이어
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout 추가
            nn.Linear(512, self.num_out)  # 최종 출력
        )
        
    def forward(self, x, num_frames):
        h = self.encoder(x)  # Shape: [L*N, num_features]
        
        ###############################################################
        # multi_attention_pool을 사용해 레이블별 feature 리스트(h_vid_outs) 반환
        h_vid_outs, attn = self.multi_attention_pool(h, num_frames)  # [batch_size, num_out, num_features]
        
        # 레이블별 h_vid_outs를 concat하여 하나의 feature로 결합
        h_vid_concat = h_vid_outs.view(h_vid_outs.size(0), -1)  # Shape: [batch_size, num_out * num_features]
        
        # Linear transformation of Attention features
        attn_features = self.attn_fc(h_vid_concat)  # Reduce to [batch_size, num_features]
        
        ###############################################################
        # conv feature 추출
        h_pad = pad(h, num_frames)  # Shape: [L, batch_size, num_features]
        
        batch_size = h_pad.size(1)
        
        h_concat = h_pad.permute(1, 2, 0)  # Shape: [batch_size, num_features, L]

        # 1D 컨볼루션 적용
        conv_h = self.conv1d(h_concat)  # Shape: (batch_size, num_features, L)
        
        # 30 프레임의 features(마지막 차원을)에 대해서 pooling 진행
        conv_h = self.gap(conv_h).squeeze(-1)  # Shape: (batch_size, num_features)
        
        ###############################################################
        # 최종 결합 및 출력
        combined_features = attn_features + conv_h  # Element-wise Addition
#         combined_features = torch.cat((attn_features, conv_h), dim=1)  # [batch_size, num_features * (self.num_out + 1)]

        # 최종 출력 레이어를 통과하여 각 레이블별 출력 생성
        output = self.fc_out_final(combined_features)  # [batch_size, num_out]
        
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
