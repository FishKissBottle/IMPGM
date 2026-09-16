import torch
import torch.nn as nn
try:
    from .VAE_block import DownBlock, MidBlock, UpBlock
except ImportError:  # Support running VAE_model.py directly from VAE_Code.
    from VAE_block import DownBlock, MidBlock, UpBlock

class AutoEncoder(nn.Module):
    """Encode images into variational latents and reconstruct them."""
    def __init__(self, img_channel=4, down_channels=[32, 64, 128], mid_inout_channels=[128, 128],
                 num_down_layers=1, num_mid_layers=1, num_up_layers=1,
                 z_channel=8, norm_channels=8):
        super().__init__()
        self.down_channels = down_channels
        self.mid_inout_channels = mid_inout_channels
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        
        self.z_channel = z_channel
        self.norm_channels = norm_channels

        # Assertion to validate the channel information
        assert self.mid_inout_channels[0] == self.down_channels[-1]
        assert self.mid_inout_channels[-1] == self.down_channels[-1]


        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(img_channel, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(in_channels=self.down_channels[i],
                                                 out_channels=self.down_channels[i + 1],
                                                 num_layers=self.num_down_layers,
                                                 norm_channels=self.norm_channels))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_inout_channels) - 1):
            self.encoder_mids.append(MidBlock(in_channels=self.mid_inout_channels[i],
                                              out_channels=self.mid_inout_channels[i + 1],
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channel * 2, kernel_size=3, padding=1)

        ##################### Reparameterize ###################### 
        self.pre_quant_conv = nn.Conv2d(self.z_channel * 2, self.z_channel * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.z_channel, self.z_channel, kernel_size=1)
        self.reparameterize_output = nn.Conv2d(z_channel, self.down_channels[-1], kernel_size=3, padding=(1, 1))

        ##################### Decoder ######################

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_inout_channels))):
            self.decoder_mids.append(MidBlock(in_channels=self.mid_inout_channels[i],
                                              out_channels=self.mid_inout_channels[i - 1],
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(in_channels=self.down_channels[i],
                                               out_channels=self.down_channels[i - 1],
                                               num_layers=self.num_up_layers,
                                               norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], img_channel, kernel_size=3, padding=1)


    def encode(self, x):
        x = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            x = down(x)
        for mid in self.encoder_mids:
            x = mid(x)
        x = self.encoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.encoder_conv_out(x)
        return x
    

    def reparameterize(self, x):

        z = self.pre_quant_conv(x)
        mu, logvar = torch.chunk(z, 2, dim=1)
        std = torch.exp(0.5 * logvar)

        if self.training:
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu

        z = self.post_quant_conv(z)

        return z, mu, logvar


    def decode(self, z):
        out = self.reparameterize_output(z)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_layers:
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)

        return out


    def forward(self, x):
        pre_reparam_z = self.encode(x)
        post_reparam_z, mean, logvar = self.reparameterize(pre_reparam_z)
        out = self.decode(post_reparam_z)
        return out, mean, logvar

