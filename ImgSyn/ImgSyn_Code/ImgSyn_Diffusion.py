import torch
import torch.nn as nn
from IMPGM_Config import *
apply_task_config(globals(), IMGSYN_DIFFUSION_CONFIG)
from Diffusion_Block import *
    


class ImgSyn_Diffusion_UNet(nn.Module):
    """Predict image-synthesis diffusion velocity with a conditional UNet."""
    def __init__(self, 
                 ch=96, 
                 out_ch=8, 
                 ch_mult=(1, 2, 3, 4), 
                 attn_resolutions=[8, 16, 32, 64], 
                 dropout=0.0, 
                 resamp_with_conv=True, 
                 in_channels=8,
                 resolution=64, 
                 prompt_dict={'NoObj': 0, 'Water':1, 'NoCloud':2, 'FewCloud':3, 'LessCloud':4, 'MoreCloud':5, 'ManyCloud':6},
                 ):
        super().__init__()
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


        self.conv_in = torch.nn.Conv2d(in_channels * 2, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        self.down = nn.ModuleList()
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
                        elif curr_res < 32:
                            self_attn.append(SelfAttention(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim))
                            cross_attn.append(CrossAttention(query_dim=block_in, heads=downblock_heads, dim_head=downblock_base_head_dim,
                                                             context_dim=self.prompt_embedding_attn_len,
                                                             dropout=dropout))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        
            down = nn.Module()
            down.block_a = block_a
            down.block_b = block_b
            down.self_attn = self_attn
            down.cross_attn = cross_attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)


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

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_a = nn.ModuleList()
            block_b = nn.ModuleList()
            self_attn = nn.ModuleList()
            cross_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            upblocks_num = 3 if curr_res >= 32 else 4
            for i_block in range(upblocks_num):
                if i_block == upblocks_num - 1:
                    skip_in = ch*in_ch_mult[i_level]
                block_a.append(ResnetBlock(in_channels=block_in+skip_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
                
                block_in = block_out

                if curr_res in attn_resolutions:
                    if i_block == upblocks_num - 1:
                        upblock_base_head_dim = 32
                        upblock_heads = min(max(block_in // upblock_base_head_dim, 1), 8)
                        if curr_res >= 32:
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim, window_size=8, shift_size=0))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim, window_size=8, shift_size=4))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        elif curr_res < 32:
                            self_attn.append(SelfAttention(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim))
                            cross_attn.append(CrossAttention(query_dim=block_in, heads=upblock_heads, dim_head=upblock_base_head_dim,
                                                             context_dim=self.prompt_embedding_attn_len,
                                                             dropout=dropout))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))

            up = nn.Module()
            up.block_a = block_a
            up.block_b = block_b
            up.self_attn = self_attn
            up.cross_attn = cross_attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order
            
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, timesteps, prompt_str, fg_imgs_e):

        x = torch.cat((x, fg_imgs_e), dim=1)

        prompt_idx = torch.tensor([self.prompt_dict[i] for i in prompt_str], device=DEVICE)
        prompt_embed_res_attn = self.prompt_embd_attn(prompt_idx)

        temb = get_timestep_embedding(timesteps, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        hs = [self.conv_in(x)]
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
                
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.mid.block_1(hs[-1], temb)
        h = self.mid.self_attn_1(h)
        h = self.mid.cross_attn_1(h, prompt_embed_res_attn)
        h = self.mid.block_2(h, temb)
        h = self.mid.self_attn_2(h)
        h = self.mid.cross_attn_2(h, prompt_embed_res_attn)
        h = self.mid.block_3(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            upblocks_num = len(self.up[i_level].block_a)
            for i_block in range(upblocks_num):
                
                h = self.up[i_level].block_a[i_block](torch.cat([h, hs.pop()], dim=1), temb)

                if i_block == upblocks_num - 1:
                    for idx in range(len(self.up[i_level].self_attn)):
                        if len(self.up[i_level].self_attn) > 0:
                            h = self.up[i_level].self_attn[idx](h)
                        if len(self.up[i_level].cross_attn) > 0:
                            h = self.up[i_level].cross_attn[idx](h, prompt_embed_res_attn)
                        if len(self.up[i_level].block_b) > 0:
                            h = self.up[i_level].block_b[idx](h, temb)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

