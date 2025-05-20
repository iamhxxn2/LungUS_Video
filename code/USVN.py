import torch.nn as nn
import torch
from utils import pad
import math
   
class MedVidNet(nn.Module):
    def __init__(self, train_layer, encoder, num_heads, num_out=4, pooling_method='attn', drop_rate=0.0, debug=False, attn_hidden_size=32):
#     def __init__(self, encoder, num_heads, num_out=5, pooling_method='attn', drop_rate=0.0, debug=False, attn_hidden_size=32):
        super(MedVidNet, self).__init__()
        # imagenet pretrained model case
        self.num_features = encoder.num_features
        # mae pretrained model case
#         self.num_features = encoder.features.norm5.num_features
        assert self.num_features % num_heads == 0, "The number of encoder features must be divisble by the number of attention heads."
        self.num_heads = num_heads
        self.subspace_size = self.num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.num_out = num_out
        self.drop_rate = drop_rate
        self.debug=debug
        self.pool=pooling_method
        self.attn_hidden_size = attn_hidden_size
        self.train_layer = train_layer
        
        self.encoder = encoder
        
#         if train_layer == 'all':
#             # frame encoder
#             self.encoder = encoder
        if train_layer == 'pooling':
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        #module to compute network output from pooled projections
        self.fc_out = nn.Sequential(
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.num_features, self.num_out)
        )
        
        if self.pool == 'attn':
            self.pool_func = self.attention_pool
            self.attn_query_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        elif self.pool == 'tanh_attn':
            self.pool_func = self.tanh_attention_pool
            self.V = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size, self.subspace_size))
            self.w = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size))
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'attn', 'max', or 'avg'")
    
    def forward(self, x, num_frames):
        print(f"Input shape: {x.shape}, on device: {x.device}\n") if self.debug else None
        if self.train_layer == 'all':
            # frame representations
            h = self.encoder(x)
            
        elif self.train_layer == 'pooling':
            with torch.no_grad():
                h = self.encoder(x)
        
        print(f"h shape: {h.shape}, on device: {h.device}") if self.debug else None
        # expect [L*N, h_dim]: torch.Size([120, 2208])
        
        h_vid, attn = self.pool_func(h, num_frames)
        # expect [N, num heads, subspace_size], [L, N, num_heads]
        print(f"h_vid shape: {h_vid.shape}, on device: {h_vid.device}") if self.debug else None
        # compute the output
        output = self.fc_out(h_vid)
        print(f"output shape: {output.shape}, on device: {output.device}") if self.debug else None
        # expect [N, output size]
        
        return output, attn
    
    def attention_pool(self, h, num_frames):
        # attention logits
        h_query = h.view(-1, self.num_heads, self.subspace_size)
        print(f"h_query shape: {h_query.shape}, on device: {h_query.device}") if self.debug else None
        # expect [L*N, num_heads, subspace_size]: torch.Size([120, 32, 69])
        
        # multigpu 설정
        # 명시적으로 attn_query_vecs를 h_query가 있는 GPU로 이동
        attn_query_vecs = self.attn_query_vecs.to(h_query.device)
    
        print(f"query vector shape: {attn_query_vecs.shape}, on device: {attn_query_vecs.device}") if self.debug else None
        # expect [num_heads, subspace_size]: torch.Size([32, 69])
        
#         alpha = (h_query * self.attn_query_vecs).sum(axis=-1) / self._scale
        # multigpu 설정
        alpha = (h_query * attn_query_vecs).sum(axis=-1) / self._scale
        print(f"alpha shape: {alpha.shape}, on device: {alpha.device}") if self.debug else None
        # expect [L*N, num_heads]: torch.Size([120, 32])
        
        # normalized attention
        alpha = pad(alpha, num_frames) # torch.Size([30, 4, 32])
        print(f"alpha shape: {alpha.shape}, on device: {alpha.device}") if self.debug else None
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-50
        print(f"alpha shape: {alpha.shape}, on device: {alpha.device}") if self.debug else None
        attn = torch.softmax(alpha, axis=0)
        print(f"attn shape: {attn.shape}, on device: {attn.device}") if self.debug else None
        # expect [L, N, num_heads]: torch.Size([30, 4, 32])
        
        # pool within subspaces
        h_query_pad = pad(h_query, num_frames)
        print(f"h_query_pad shape: {h_query_pad.shape}, on device: {h_query_pad.device}") if self.debug else None
        # expect [L, N, num heads, subspace_size]: torch.Size([30, 4, 32, 69])
        h_vid = torch.sum(h_query_pad * attn[...,None], axis=0)
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, num heads, subspace_size]: torch.Size([4, 32, 69])
        
        h_vid_wide = h_vid.view(-1, self.num_features)
        print(f"h_vid_wide shape: {h_vid_wide.shape}, on device: {h_vid_wide.device}") if self.debug else None
        # expect [N, h_dim]: torch.Size([4, 2208])
        
        return h_vid_wide, attn
    
    def tanh_attention_pool(self, h, num_frames):
        # attention logits
        h_query = h.view(-1, self.num_heads, self.subspace_size)
        print("h_query shape:", h_query.shape) if self.debug else None
        # expect [L*N, num_heads, subspace_size]
        print("query vector shape:", self.attn_query_vecs.shape) if self.debug else None
        # expect [num_heads, subspace_size]
        alpha = torch.einsum('ijk,jlk->ijl', h_query, self.V).tanh()
        print("alpha shape:", alpha.shape) if self.debug else None
        # expect [L*N, num_heads, attn_hidden_size]
        lamb = torch.einsum('ijl,jl->ij', alpha, self.w)
        print("lambda shape:", lamb.shape) if self.debug else None
        # expect [L*N, num_heads]
        
        # normalized attention
        lamb = pad(lamb, num_frames)
        for ix, n in enumerate(num_frames):
            lamb[n:, ix]=-50
        attn = torch.softmax(lamb, axis=0)
        print("attn shape:", attn.shape) if self.debug else None
        # expect [L, N, num_heads]
        
        # pool within subspaces
        h_query_pad = pad(h_query, num_frames)
        print("h_query_pad shape:", h_query_pad.shape) if self.debug else None
        # expect [L, N, num heads, subspace_size]
        h_vid = torch.sum(h_query_pad * attn[...,None] / self._scale, axis=0)
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, num heads, subspace_size]
        
        h_vid_wide = h_vid.view(-1, self.num_features)
        print("h_vid_wide shape:", h_vid_wide.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid_wide, attn
    
    def max_pool(self, h, num_frames):
        h_pad = pad(h, num_frames)
        print("h_pad shape:", h_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take max
        h_vid = h_pad.max(0).values
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid, None
    
    def avg_pool(self, h, num_frames):
        h_pad = pad(h, num_frames)
        print("h_pad shape:", h_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take avg
        h_vid = h_pad.mean(0)
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid, None