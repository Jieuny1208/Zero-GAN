import torch
import torch.nn as nn
import torch.nn.parallel

from timm.models.vision_transformer import PatchEmbed, Block


# patch 한개의 임베딩을 받아서 patch 한장 나옴 (1차원으로 나옴)
# class MAEDecoder(nn.Module):
#     def __init__(self, embed_dim, decoder_embed_dim, decoder_depth, 
#                  decoder_num_heads, patch_size, in_channels, 
#                  mlp_ratio, norm_layer=nn.LayerNorm):
#         super(MAEDecoder, self).__init__()

#         # Project latent vectors to the decoder's dimension
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
#         # Transformer blocks for the decoder
#         self.decoder_blocks = nn.ModuleList([
#             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(decoder_depth)
#         ])
        
#         # Normalization layer at the end of the decoder
#         self.decoder_norm = norm_layer(decoder_embed_dim)

#         # Prediction layer to project the decoder outputs to the original patch dimensions
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)

#     def forward(self, x):
#         # Embed latent vectors into the decoder dimension
#         x = self.decoder_embed(x)

#         # Process through each decoder block
#         for block in self.decoder_blocks:
#             x = block(x)
#         x = self.decoder_norm(x)

#         # Final layer to project back to the input size
#         x = self.decoder_pred(x)
#         return x


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, patch_size, in_channels, mlp_ratio, num_patch, norm_layer=nn.LayerNorm):
        super(MAEDecoder, self).__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patch, decoder_embed_dim))  # 위치 인코딩 추가 # 16에 num_patches,, 
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)

    def forward(self, x):
        x = self.decoder_embed(x)
        x += self.pos_embed  # 위치 인코딩을 추가
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x