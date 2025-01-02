
##
import torch
import torch.nn as nn
import torch.nn.parallel

from models.networks.encoder import GANEncoder, MAEEncoder
from models.networks.decoder import MAEDecoder

from utils.data_utils import patches_to_image
from utils.pos_embed import get_2d_sincos_pos_embed


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """
    def __init__(self, opt):
        super(NetD, self).__init__()
        self.opt = opt
        model = GANEncoder(
            image_size=self.opt['image_size'],
            latent_vector_dim=1,
            in_channels=self.opt['in_channels'],
            discriminator_feature_num=self.opt['discriminator_feature_num'],
            n_extra_layers=self.opt['n_extra_layers'],
            add_final_conv=self.opt['add_final_conv'],
        )
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(*layers[-1:], nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1)

        return classifier, features

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# 테스트 할때는 mask_ratio=0 으로
class ImageToInput(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=opt['image_size'], patch_size=opt['patch_size'], in_chans=opt['in_channels'], embed_dim=opt['embed_dim'])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, opt['embed_dim']))
        pos_embed = get_2d_sincos_pos_embed(opt['embed_dim'], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, opt['embed_dim']), requires_grad=False)
        self.pos_embed.data.copy_(torch.tensor(pos_embed).float().unsqueeze(0))

    def forward(self, imgs, mask_ratio=0.75):
        x = self.patch_embed(imgs)
        batch_size, num_patches, embed_dim = x.shape
        x = x + self.pos_embed[:, :num_patches, :]

        # 마스킹
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.zeros_like(x)
        x_masked.scatter_(1, ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim), torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim)))

        return x_masked, ids_restore

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.opt = opt
        self.encoder1 = MAEEncoder(
            embed_dim= self.opt['embed_dim'],
            depth= self.opt['encoder_depth'],
            num_heads= self.opt['encoder_num_heads'],
            mlp_ratio= self.opt['mlp_ratio'],
            norm_layer=nn.LayerNorm
        ) # 출력 shape : (batch_size, num_patches, embed_dim)
        
        self.decoder = MAEDecoder(
            embed_dim= self.opt['embed_dim'],
            decoder_embed_dim= self.opt['decoder_embed_dim'],
            decoder_depth= self.opt['decoder_depth'],
            decoder_num_heads= self.opt['decoder_num_heads'],
            patch_size= self.opt['patch_size'],
            in_channels= self.opt['in_channels'],
            mlp_ratio= self.opt['mlp_ratio'],
            num_patch=int((self.opt['image_size'] / self.opt['patch_size'])**2),
            norm_layer=nn.LayerNorm
            ) # 여기서 나온걸 data_utils로 이미지로 다시 만들고 원래 input이랑 비교하면 될듯?
        self.encoder2 = GANEncoder(
            image_size= self.opt['image_size'], 
            latent_vector_dim= self.opt['latent_vector_dim'],
            in_channels= self.opt['in_channels'],
            discriminator_feature_num= self.opt['discriminator_feature_num'],
            n_extra_layers= self.opt['n_extra_layers'], 
            add_final_conv= self.opt['add_final_conv'], 
        )

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        gen_img = patches_to_image(gen_img, self.opt['image_size'], self.opt['patch_size'], self.opt['in_channels']) 
        latent_o = self.encoder2(gen_img)
        return latent_i, gen_img, latent_o


class ZeroGanGenerator(nn.Module):
    def __init__(self, opt):
        super(ZeroGanGenerator,self).__init__()
        self.generator = NetG(opt)

    def forward(self, x):
        latent_i, gen_img, latent_o = self.generator(x)
        return latent_i, gen_img, latent_o


class Discriminator(nn.Module):
    """
    입력:
        image: (batch_size, in_channels, image_size, image_size)
    출력:
        classifier(이미지가 real일 확률): (batch_size, 1)
        features(latent vector): (batch_size, latent_vector_dim)
    """
    def __init__(self, opt):
        super(Discriminator,self).__init__()
        self.discriminator = NetD(opt)

    def forward(self, x):
        classifier, features = self.discriminator(x)
        return classifier, features