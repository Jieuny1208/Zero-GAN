import torch
import torch.nn as nn
import torch.nn.parallel
from timm.models.vision_transformer import PatchEmbed, Block

###
class GANEncoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    def __init__(self, image_size, latent_vector_dim, in_channels, discriminator_feature_num, n_extra_layers=0, add_final_conv=True):
        super(GANEncoder, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        main = nn.Sequential()
        # input is in_channels x image_size x image_size
        main.add_module('initial-conv-{0}-{1}'.format(in_channels, discriminator_feature_num),
                        nn.Conv2d(in_channels, discriminator_feature_num, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(discriminator_feature_num),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = image_size / 2, discriminator_feature_num

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, latent_vector_dim, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)

        return output


# 입력 shape: (batch_size, num_patches, embed_dim)
class MAEEncoder(nn.Module):
    """Encoder for extracting latent vectors from masked images."""
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MAEEncoder, self).__init__()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # Final normalization layer
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # Process each transformer block
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

