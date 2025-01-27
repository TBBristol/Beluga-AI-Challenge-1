import torch as th
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    """
    Takes (B, n_stacks, 6) -> (B, n_stacks, 256).
    """
    def __init__(self, n_heads=4, embed_dim=256, in_dim=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.n_heads = n_heads
        
        self.to_embed = nn.Linear(in_dim, embed_dim, bias = False)  # 6 -> 256 This is Q,K,V 
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,  # input is (B, N, E)
            bias = False,
            
        )
        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)

        #Fully connected layer after attention
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim))
        
    def forward(self, x):
        # x: shape (B, n_stacks, 6)
        x_emb = self.to_embed(x)  # (B, n_stacks, 256)
        
        # Self-attention
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)  # This calls attn Forward method (B, n_stacks, 256)
       
        # Feed-forward
        out = self.fc(attn_out)  # (B, n_stacks, 256)
        out = self.norm1(x_emb + out)

        if th.isnan(attn_out).any():
            print("NaN in output")
     
        return out  
    

    #TODO query over whether the skip is implemented properly as paper says X not Xembed
