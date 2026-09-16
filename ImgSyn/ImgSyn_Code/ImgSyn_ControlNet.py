import torch
import torch.nn as nn
import torch.nn.functional as F
from IMPGM_Config import *
apply_task_config(globals(), IMGSYN_CONTROLNET_CONFIG)
from Diffusion_Block import *
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
import os
from IMPGM_Utils import load_model_for_eval


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):
    """Encode image-synthesis conditions into ControlNet residual features."""
    def __init__(self, 
                 ch=MODEL_CH,
                 conditional_ch=MODEL_CONDITIONAL_CH,
                 ch_mult=MODEL_CH_MULT,
                 attn_resolutions=MODEL_ATTN_RESOLUTIONS,
                 dropout=MODEL_DROPOUT,
                 resamp_with_conv=MODEL_RESAMP_WITH_CONV,
                 in_channels=MODEL_IN_CHANNELS,
                 resolution=MODEL_RESOLUTION,
                 prompt_dict=None,
                 ):
        super().__init__()
        if prompt_dict is None:
            prompt_dict = PROMPT_DICT
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.prompt_embedding_attn_len = ch // 4
        self.prompt_dict = prompt_dict
        self.prompt_embd_attn = nn.Embedding(len(prompt_dict),  self.prompt_embedding_attn_len)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])
        self.conditional_ch = conditional_ch

        self.condition_conv_block = nn.Sequential(
            nn.Conv2d(self.conditional_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # Map the condition toward latent resolution without the VAE.
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, ch, 3, padding=1, stride=2),  # Match the main UNet width instead of hard-coding 96.
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, ch, 3, padding=1))
        )

        self.conv_in = torch.nn.Conv2d(in_channels * 2, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        self.down = nn.ModuleList()
        self.zero_convs = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_a = nn.ModuleList()
            block_b = nn.ModuleList()
            self_attn = nn.ModuleList()
            cross_attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            downblocks_num = 2 if curr_res >= 32 else 3
            for i_block in range(downblocks_num):
                block_a.append(ResnetBlock(in_channels=block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:
                    if i_block == downblocks_num - 1:
                        downblock_base_head_dim = 32
                        downblock_heads = min(max(block_in // downblock_base_head_dim, 1), 8)
                        if curr_res >= 32:
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim, window_size=8, shift_size=0))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim, window_size=8, shift_size=4))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        if curr_res < 32:
                            self_attn.append(SelfAttention(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim))
                            cross_attn.append(CrossAttention(query_dim=block_in, heads=downblock_heads, dim_head=downblock_base_head_dim,
                                                             context_dim=self.prompt_embedding_attn_len,
                                                             dropout=dropout))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))                  

                self.zero_convs.append(self.make_zero_conv(block_out))

            down = nn.Module()
            down.block_a = block_a
            down.block_b = block_b
            down.self_attn = self_attn
            down.cross_attn = cross_attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                self.zero_convs.append(self.make_zero_conv(block_out))
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        midblock_base_head_dim = 32
        midblock_heads = min(max(block_in // midblock_base_head_dim, 1), 8)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.mid.self_attn_1 = SelfAttention(in_channels=block_in, heads=midblock_heads, head_dim=midblock_base_head_dim)
        self.mid.cross_attn_1 = CrossAttention(query_dim=block_in, heads=midblock_heads, dim_head=midblock_base_head_dim,
                                               context_dim=self.prompt_embedding_attn_len,
                                               dropout=dropout)
        
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.mid.self_attn_2 = SelfAttention(in_channels=block_in, heads=midblock_heads, head_dim=midblock_base_head_dim)
        self.mid.cross_attn_2 = CrossAttention(query_dim=block_in, heads=midblock_heads, dim_head=midblock_base_head_dim,
                                               context_dim=self.prompt_embedding_attn_len,
                                               dropout=dropout)
        

        self.mid.block_3 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.mid.block_out = self.make_zero_conv(ch * ch_mult[-1])

        self.zeros_in = self.make_zero_conv(ch)


    def make_zero_conv(self, channels):
        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))


    def forward(self, x, timesteps, prompt_str, conditional_element):

        prompt_idx = torch.tensor(
            [self.prompt_dict[i] for i in prompt_str],
            device=x.device,
        )
        prompt_embed_res_attn = self.prompt_embd_attn(prompt_idx)

        # timestep embedding
        temb = get_timestep_embedding(timesteps, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        conv_conditions = self.condition_conv_block(conditional_element)
        if conv_conditions.shape[-2:] != x.shape[-2:]:
            conv_conditions = F.interpolate(
                conv_conditions,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # downsampling
        hs = [self.conv_in(x) + conv_conditions]
        out_list = [self.zeros_in(hs[-1])]

        for i_level in range(self.num_resolutions):
            downblocks_num = len(self.down[i_level].block_a)
            for i_block in range(downblocks_num):

                h = self.down[i_level].block_a[i_block](hs[-1], temb)

                if i_block == downblocks_num - 1:
                    for idx in range(len(self.down[i_level].self_attn)):
                        if len(self.down[i_level].self_attn) > 0:
                            h = self.down[i_level].self_attn[idx](h)
                        if len(self.down[i_level].cross_attn) > 0:
                            h = self.down[i_level].cross_attn[idx](h, prompt_embed_res_attn)
                        if len(self.down[i_level].block_b) > 0:
                            h = self.down[i_level].block_b[idx](h, temb)

                hs.append(h)
                out_list.append(self.zero_convs[sum(len(self.down[l].block_a) + 1 for l in range(i_level)) + i_block](h))

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(hs[-1])
                hs.append(h)
                out_list.append(self.zero_convs[sum(len(self.down[l].block_a) + 1 for l in range(i_level)) + downblocks_num](h))


        # middle
        h = self.mid.block_1(hs[-1], temb)
        h = self.mid.self_attn_1(h)
        h = self.mid.cross_attn_1(h, prompt_embed_res_attn)
        h = self.mid.block_2(h, temb)
        h = self.mid.self_attn_2(h)
        h = self.mid.cross_attn_2(h, prompt_embed_res_attn)
        h = self.mid.block_3(h, temb)

        out_list.append(self.mid.block_out(h))

        return out_list



class ControlNet_on_ImgSyn_Diffusion(nn.Module):
    """Combine the ImgSyn diffusion backbone with its ControlNet branch."""
    def __init__(self,
                 ImgSyn_Diffusion_model_savepath,
                 ch=MODEL_CH,
                 out_ch=MODEL_OUT_CH,
                 ch_mult=MODEL_CH_MULT,
                 attn_resolutions=MODEL_ATTN_RESOLUTIONS,
                 dropout=MODEL_DROPOUT,
                 resamp_with_conv=MODEL_RESAMP_WITH_CONV,
                 in_channels=MODEL_IN_CHANNELS,
                 resolution=MODEL_RESOLUTION,
                 conditional_ch=MODEL_CONDITIONAL_CH,
                 ControlNet_weight=MODEL_CONTROLNET_WEIGHT,
                 prompt_dict=None,
                 ):
        super().__init__()

        if prompt_dict is None:
            prompt_dict = PROMPT_DICT
        self.ControlNet_weight = ControlNet_weight

        # Load the EMA backbone so ControlNet training matches deployed inference.
        ImgSyn_Diffusion_model_exist = os.path.exists(ImgSyn_Diffusion_model_savepath)
        if not ImgSyn_Diffusion_model_exist:
            raise FileNotFoundError(
                f"ImgSyn diffusion checkpoint does not exist: {ImgSyn_Diffusion_model_savepath}"
            )
        ImgSyn_Diffusion_model = ImgSyn_Diffusion_UNet(ch=ch,
                                                       out_ch=out_ch,
                                                       ch_mult=ch_mult,
                                                       attn_resolutions=attn_resolutions,
                                                       dropout=dropout,
                                                       resamp_with_conv=resamp_with_conv,
                                                       in_channels=in_channels,
                                                       resolution=resolution,
                                                       prompt_dict=prompt_dict
                                                       ).to(DEVICE)
        ImgSyn_Diffusion_model, _ = load_model_for_eval(
            ImgSyn_Diffusion_model_savepath,
            ImgSyn_Diffusion_model,
            map_location=DEVICE,
        )
        ImgSyn_Diffusion_model = ImgSyn_Diffusion_model.eval()
        for ImgSyn_Diffusion_param in ImgSyn_Diffusion_model.parameters():
            ImgSyn_Diffusion_param.requires_grad = False

        self.ImgSyn_Diffusion_model = ImgSyn_Diffusion_model

        ControlNet_model = ControlNet(ch=ch,
                                      conditional_ch=conditional_ch,
                                      ch_mult=ch_mult,
                                      attn_resolutions=attn_resolutions,
                                      dropout=dropout,
                                      resamp_with_conv=resamp_with_conv,
                                      in_channels=in_channels,
                                      resolution=resolution,
                                      prompt_dict=prompt_dict
                                      )
        self.ControlNet_model = ControlNet_model

    def train(self, mode=True):
        """Train the control branch while keeping the frozen backbone in eval mode."""
        super().train(mode)
        self.ImgSyn_Diffusion_model.eval()
        return self


    def forward(self, x, timesteps, prompt_str, fg_imgs_e, conditional_element):

        x = torch.cat((x, fg_imgs_e), dim=1)

        prompt_idx = torch.tensor(
            [self.ImgSyn_Diffusion_model.prompt_dict[i] for i in prompt_str],
            device=x.device,
        )
        prompt_embed_res_attn = self.ImgSyn_Diffusion_model.prompt_embd_attn(prompt_idx)

        temb = get_timestep_embedding(timesteps, self.ImgSyn_Diffusion_model.ch)
        temb = self.ImgSyn_Diffusion_model.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.ImgSyn_Diffusion_model.temb.dense[1](temb)

        # downsampling
        hs = [self.ImgSyn_Diffusion_model.conv_in(x)]
        for i_level in range(self.ImgSyn_Diffusion_model.num_resolutions):
            downblocks_num = len(self.ImgSyn_Diffusion_model.down[i_level].block_a)
            for i_block in range(downblocks_num):

                h = self.ImgSyn_Diffusion_model.down[i_level].block_a[i_block](hs[-1], temb)

                if i_block == downblocks_num - 1:
                    for idx in range(len(self.ImgSyn_Diffusion_model.down[i_level].self_attn)):
                        if len(self.ImgSyn_Diffusion_model.down[i_level].self_attn) > 0:
                            h = self.ImgSyn_Diffusion_model.down[i_level].self_attn[idx](h)
                        if len(self.ImgSyn_Diffusion_model.down[i_level].cross_attn) > 0:
                            h = self.ImgSyn_Diffusion_model.down[i_level].cross_attn[idx](h, prompt_embed_res_attn)
                        if len(self.ImgSyn_Diffusion_model.down[i_level].block_b) > 0:
                            h = self.ImgSyn_Diffusion_model.down[i_level].block_b[idx](h, temb)

                hs.append(h)

            if i_level != self.ImgSyn_Diffusion_model.num_resolutions - 1:
                hs.append(self.ImgSyn_Diffusion_model.down[i_level].downsample(hs[-1]))

        # middle
        h = self.ImgSyn_Diffusion_model.mid.block_1(hs[-1], temb)
        h = self.ImgSyn_Diffusion_model.mid.self_attn_1(h)
        h = self.ImgSyn_Diffusion_model.mid.cross_attn_1(h, prompt_embed_res_attn)
        h = self.ImgSyn_Diffusion_model.mid.block_2(h, temb)
        h = self.ImgSyn_Diffusion_model.mid.self_attn_2(h)
        h = self.ImgSyn_Diffusion_model.mid.cross_attn_2(h, prompt_embed_res_attn)
        h = self.ImgSyn_Diffusion_model.mid.block_3(h, temb)

        control_connections = self.ControlNet_model(x, timesteps, prompt_str, conditional_element)
        h += self.ControlNet_weight * control_connections.pop()

        # upsampling
        for i_level in reversed(range(self.ImgSyn_Diffusion_model.num_resolutions)):
            upblocks_num = len(self.ImgSyn_Diffusion_model.up[i_level].block_a)
            for i_block in range(upblocks_num):

                h = self.ImgSyn_Diffusion_model.up[i_level].block_a[i_block](torch.cat([h, hs.pop() + self.ControlNet_weight * control_connections.pop()], dim=1), temb)

                if i_block == upblocks_num - 1:
                    for idx in range(len(self.ImgSyn_Diffusion_model.up[i_level].self_attn)):
                        if len(self.ImgSyn_Diffusion_model.up[i_level].self_attn) > 0:
                            h = self.ImgSyn_Diffusion_model.up[i_level].self_attn[idx](h)
                        if len(self.ImgSyn_Diffusion_model.up[i_level].cross_attn) > 0:
                            h = self.ImgSyn_Diffusion_model.up[i_level].cross_attn[idx](h, prompt_embed_res_attn)
                        if len(self.ImgSyn_Diffusion_model.up[i_level].block_b) > 0:
                            h = self.ImgSyn_Diffusion_model.up[i_level].block_b[idx](h, temb)

            if i_level != 0:
                h = self.ImgSyn_Diffusion_model.up[i_level].upsample(h)

        # end
        h = self.ImgSyn_Diffusion_model.norm_out(h)
        h = nonlinearity(h)
        h = self.ImgSyn_Diffusion_model.conv_out(h)

        return h
