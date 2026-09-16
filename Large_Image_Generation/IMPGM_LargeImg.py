from torch import nn
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import *
import numpy as np
import random
from FgGen.FgGen_Code.FgGen_Diffusion import FgGen_Diffusion_UNet
from FgGen.FgGen_Code.FgGen_ControlNet import ControlNet_on_FgGen_Diffusion
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet import ControlNet_on_ImgSyn_Diffusion
from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from IMPGM_Utils import build_diagonal_tile_schedule, descale_latent, extract_high_frequency, load_controlnet_model_for_eval, load_standard_vae, scale_latent, load_model_for_eval, save_rgb_datas, save_tif_datas, save_msk_datas, set_random_seed, tensor_dilate, prepare_rgb_vis_tensor
from Diffusion_Sampler import DDPMSampler, DDIMSampler


class Generate_LargeImg(nn.Module):
    """Generate large images by composing tiled IMPGM predictions."""
    def __init__(self, 
                 sampler_mode, 
                 VAE_model_savepath, 
                 FgGen_Diffusion_model_savepath, 
                 ImgSyn_Diffusion_model_savepath,
                 FgGen_ControlNet_model_savepath,
                 ImgSyn_ControlNet_model_savepath,
                 FgSeg_model_savepath, 
                 beta_t,
                 overlap_rate,
                 overlap_buffer_thx=4*6,
                 is_FgGenInpaint_Resample=True,
                 is_ImgSynInpaint_Resample=True,
                 batch_size=8,
                 FgGen_RGB_savepath=None,
                 FgGen_TIF_savepath=None,
                 ImgSyn_RGB_savepath=None,
                 ImgSyn_TIF_savepath=None,
                 Msk_savepath=None,
                 is_Fixed_Prompt=False,
                 use_fggen_controlnet=False,
                 use_imgsyn_controlnet=False,
                 ddim_steps=500,
                 ddim_eta=0.0,
                 is_FgGen_Use_OpenOperation=False,
                 OpenOperation_Kernel_Size=3,
                 is_FgGen_Set_MinThreshold=False,
                 FgGen_MinThreshold=-0.6,
                 conditional_element_proj=None,
                 conditional_element_geotrans=None,
                 ):
        super().__init__()
        ddim_steps = int(ddim_steps)
        ddim_eta = float(ddim_eta)
        if sampler_mode not in ('ddpm', 'ddim'):
            raise ValueError("sampler_mode must be either 'ddpm' or 'ddim'.")
        if ddim_steps <= 0 or ddim_steps > len(beta_t):
            raise ValueError(
                f"ddim_steps must be in [1, {len(beta_t)}], got {ddim_steps}."
            )
        if ddim_eta < 0.0:
            raise ValueError(f"ddim_eta must be non-negative, got {ddim_eta}.")

        if (conditional_element_proj is None) != (conditional_element_geotrans is None):
            raise ValueError(
                "conditional_element_proj and conditional_element_geotrans must be provided together."
            )
        if isinstance(conditional_element_proj, str):
            conditional_element_proj = [conditional_element_proj]
        if conditional_element_geotrans is not None:
            if len(conditional_element_geotrans) == 6 and all(
                np.isscalar(value) for value in conditional_element_geotrans
            ):
                conditional_element_geotrans = [list(conditional_element_geotrans)]
        if conditional_element_proj is not None and len(conditional_element_proj) != 1:
            raise ValueError("Large-image generation expects exactly one projection WKT.")
        if conditional_element_geotrans is not None:
            if len(conditional_element_geotrans) != 1 or len(conditional_element_geotrans[0]) != 6:
                raise ValueError("Large-image generation expects one six-element geotransform.")

        if not 0.0 < overlap_rate < 1.0:
            raise ValueError(f"overlap_rate must satisfy 0 < overlap_rate < 1, got {overlap_rate}.")
        overlap_thx_e = int((IMG_SIZE // SIDELENGTH_SCALE_FACTOR) * overlap_rate)
        overlap_buffer_thx_e = overlap_buffer_thx // SIDELENGTH_SCALE_FACTOR
        if overlap_buffer_thx_e <= 0 or overlap_buffer_thx_e > overlap_thx_e:
            raise ValueError(
                "overlap_buffer_thx must be positive and no larger than the effective overlap width, "
                f"got overlap_buffer_thx={overlap_buffer_thx}, "
                f"effective overlap={overlap_thx_e * SIDELENGTH_SCALE_FACTOR}."
            )

        for savepath in (
            FgGen_RGB_savepath,
            FgGen_TIF_savepath,
            ImgSyn_RGB_savepath,
            ImgSyn_TIF_savepath,
            Msk_savepath,
        ):
            if savepath is not None:
                os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)


        self.is_FgGen_Use_OpenOperation = is_FgGen_Use_OpenOperation
        self.OpenOperation_Kernel_Size = OpenOperation_Kernel_Size
        self.is_FgGen_Set_MinThreshold = is_FgGen_Set_MinThreshold
        self.FgGen_MinThreshold = FgGen_MinThreshold

        self.T = len(beta_t)
        self.use_fggen_controlnet = bool(use_fggen_controlnet)
        self.use_imgsyn_controlnet = bool(use_imgsyn_controlnet)
        self.sampler_mode = sampler_mode
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        set_random_seed(RANDOM_SEED, deterministic=False)

        # Load Latent_model
        if not os.path.exists(VAE_model_savepath):
            raise Exception('VAE_MODEL is not available.')
        Latent_model, latent_scaling_factor, _ = load_standard_vae(save_path=VAE_model_savepath, device=DEVICE, load_ema=True)
        _latent_reparameterize = Latent_model.reparameterize
        _latent_decode = Latent_model.decode

        def _scaled_reparameterize(x):
            z, mu, logvar = _latent_reparameterize(x)
            return scale_latent(z, latent_scaling_factor), scale_latent(mu, latent_scaling_factor), logvar

        def _scaled_decode(x):
            return _latent_decode(descale_latent(x, latent_scaling_factor))

        Latent_model.reparameterize = _scaled_reparameterize
        Latent_model.decode = _scaled_decode
                
        self.Latent_model = Latent_model
        self.latent_scaling_factor = latent_scaling_factor

        # Load FgSeg_Unet_model
        FgSeg_UNet_exist = os.path.exists(FgSeg_model_savepath)
        if not FgSeg_UNet_exist:
            raise Exception('FGSEG_UNET_MODEL does not exist')
        else:
            FgSeg_UNet_model = FgSeg_UNet(in_channels=INPUT_CHANNELS, out_channels=UNET_OUTPUT_CHANNELS).to(DEVICE)
            FgSeg_UNet_model, _ = load_model_for_eval(
                FgSeg_model_savepath,
                FgSeg_UNet_model,
                map_location=DEVICE,
            )
            FgSeg_UNet_model = FgSeg_UNet_model.eval()
            for FgSeg_UNet_param in FgSeg_UNet_model.parameters():
                FgSeg_UNet_param.requires_grad = False

        self.FgSeg_model = FgSeg_UNet_model

        # Load FgGen_Diffusion_model
        FgGen_Diffusion_exist = os.path.exists(FgGen_Diffusion_model_savepath)
        if not FgGen_Diffusion_exist:
            raise Exception("FGGEN_DIFFUSION_MODEL does not exist.")
        else:
            FgGen_Diffusion_model = FgGen_Diffusion_UNet(ch=FGGEN_DIFFUSION_CONFIG.MODEL_CH,
                                                        out_ch=FGGEN_DIFFUSION_CONFIG.MODEL_OUT_CH,
                                                        ch_mult=FGGEN_DIFFUSION_CONFIG.MODEL_CH_MULT,
                                                        attn_resolutions=FGGEN_DIFFUSION_CONFIG.MODEL_ATTN_RESOLUTIONS,
                                                        dropout=FGGEN_DIFFUSION_CONFIG.MODEL_DROPOUT,
                                                        resamp_with_conv=FGGEN_DIFFUSION_CONFIG.MODEL_RESAMP_WITH_CONV,
                                                        in_channels=FGGEN_DIFFUSION_CONFIG.MODEL_IN_CHANNELS,
                                                        resolution=FGGEN_DIFFUSION_CONFIG.MODEL_RESOLUTION,
                                                        prompt_dict=PROMPT_DICT,
                                                        ).to(DEVICE)
            FgGen_Diffusion_model, _ = load_model_for_eval(FgGen_Diffusion_model_savepath, FgGen_Diffusion_model, map_location=DEVICE)
            FgGen_Diffusion_model = FgGen_Diffusion_model.eval()
            for FgGen_Diffusion_param in FgGen_Diffusion_model.parameters():
                FgGen_Diffusion_param.requires_grad = False

        self.FgGen_Diffusion_model = FgGen_Diffusion_model

        # Load ImgSyn_Diffusion_model
        ImgSyn_Diffusion_exist = os.path.exists(ImgSyn_Diffusion_model_savepath)
        if not ImgSyn_Diffusion_exist:
            raise Exception('IMGSYN_DIFFUSION_MODEL does not exist')
        else:
            ImgSyn_Diffusion_model = ImgSyn_Diffusion_UNet(ch=IMGSYN_DIFFUSION_CONFIG.MODEL_CH,
                                                           out_ch=IMGSYN_DIFFUSION_CONFIG.MODEL_OUT_CH,
                                                           ch_mult=IMGSYN_DIFFUSION_CONFIG.MODEL_CH_MULT,
                                                           attn_resolutions=IMGSYN_DIFFUSION_CONFIG.MODEL_ATTN_RESOLUTIONS,
                                                           dropout=IMGSYN_DIFFUSION_CONFIG.MODEL_DROPOUT,
                                                           resamp_with_conv=IMGSYN_DIFFUSION_CONFIG.MODEL_RESAMP_WITH_CONV,
                                                           in_channels=IMGSYN_DIFFUSION_CONFIG.MODEL_IN_CHANNELS,
                                                           resolution=IMGSYN_DIFFUSION_CONFIG.MODEL_RESOLUTION,
                                                           prompt_dict=PROMPT_DICT
                                                           ).to(DEVICE)
            ImgSyn_Diffusion_model, _ = load_model_for_eval(ImgSyn_Diffusion_model_savepath, ImgSyn_Diffusion_model, map_location=DEVICE)
            ImgSyn_Diffusion_model = ImgSyn_Diffusion_model.eval()
            for ImgSyn_Diffusion_param in ImgSyn_Diffusion_model.parameters():
                ImgSyn_Diffusion_param.requires_grad = False        

        self.ImgSyn_Diffusion_model = ImgSyn_Diffusion_model

        FgGen_ControlNet_model = None
        ImgSyn_ControlNet_model = None
        if self.use_fggen_controlnet:
            # Load FgGen_ControlNet_model
            FgGen_ControlNet_exist = os.path.exists(FgGen_ControlNet_model_savepath)
            if not FgGen_ControlNet_exist:
                raise Exception("FGGEN_CONTROLNET_MODEL does not exist.")
            FgGen_ControlNet_model = ControlNet_on_FgGen_Diffusion(FgGen_Diffusion_model_savepath=FgGen_Diffusion_model_savepath,
                                                                   ch=FGGEN_CONTROLNET_CONFIG.MODEL_CH,
                                                                   out_ch=FGGEN_CONTROLNET_CONFIG.MODEL_OUT_CH,
                                                                   ch_mult=FGGEN_CONTROLNET_CONFIG.MODEL_CH_MULT,
                                                                   attn_resolutions=FGGEN_CONTROLNET_CONFIG.MODEL_ATTN_RESOLUTIONS,
                                                                   dropout=FGGEN_CONTROLNET_CONFIG.MODEL_DROPOUT,
                                                                   resamp_with_conv=FGGEN_CONTROLNET_CONFIG.MODEL_RESAMP_WITH_CONV,
                                                                   in_channels=FGGEN_CONTROLNET_CONFIG.MODEL_IN_CHANNELS,
                                                                   resolution=FGGEN_CONTROLNET_CONFIG.MODEL_RESOLUTION,
                                                                   conditional_ch=FGGEN_CONTROLNET_CONFIG.MODEL_CONDITIONAL_CH,
                                                                   ControlNet_weight=FGGEN_CONTROLNET_CONFIG.MODEL_CONTROLNET_WEIGHT,
                                                                   prompt_dict=PROMPT_DICT
                                                                   ).to(DEVICE)
            FgGen_ControlNet_model, _ = load_controlnet_model_for_eval(
                FgGen_ControlNet_model_savepath,
                FgGen_ControlNet_model,
                map_location=DEVICE,
            )
            FgGen_ControlNet_model = FgGen_ControlNet_model.eval()
            for FgGen_ControlNet_param in FgGen_ControlNet_model.parameters():
                FgGen_ControlNet_param.requires_grad = False        

        if self.use_imgsyn_controlnet:
            # Load ImgSyn_ControlNet_model
            ImgSyn_ControlNet_exist = os.path.exists(ImgSyn_ControlNet_model_savepath)
            if not ImgSyn_ControlNet_exist:
                raise Exception('IMGSYN_CONTROLNET_MODEL does not exist')
            ImgSyn_ControlNet_model = ControlNet_on_ImgSyn_Diffusion(ImgSyn_Diffusion_model_savepath=ImgSyn_Diffusion_model_savepath,
                                                                     ch=IMGSYN_CONTROLNET_CONFIG.MODEL_CH,
                                                                     out_ch=IMGSYN_CONTROLNET_CONFIG.MODEL_OUT_CH,
                                                                     ch_mult=IMGSYN_CONTROLNET_CONFIG.MODEL_CH_MULT,
                                                                     attn_resolutions=IMGSYN_CONTROLNET_CONFIG.MODEL_ATTN_RESOLUTIONS,
                                                                     dropout=IMGSYN_CONTROLNET_CONFIG.MODEL_DROPOUT,
                                                                     resamp_with_conv=IMGSYN_CONTROLNET_CONFIG.MODEL_RESAMP_WITH_CONV,
                                                                     in_channels=IMGSYN_CONTROLNET_CONFIG.MODEL_IN_CHANNELS,
                                                                     resolution=IMGSYN_CONTROLNET_CONFIG.MODEL_RESOLUTION,
                                                                     conditional_ch=IMGSYN_CONTROLNET_CONFIG.MODEL_CONDITIONAL_CH,
                                                                     ControlNet_weight=IMGSYN_CONTROLNET_CONFIG.MODEL_CONTROLNET_WEIGHT,
                                                                     prompt_dict=PROMPT_DICT
                                                                     ).to(DEVICE)
            ImgSyn_ControlNet_model, _ = load_controlnet_model_for_eval(
                ImgSyn_ControlNet_model_savepath,
                ImgSyn_ControlNet_model,
                map_location=DEVICE,
            )
            ImgSyn_ControlNet_model = ImgSyn_ControlNet_model.eval()
            for ImgSyn_ControlNet_param in ImgSyn_ControlNet_model.parameters():
                ImgSyn_ControlNet_param.requires_grad = False

        self.FgGen_ControlNet_model = FgGen_ControlNet_model
        self.ImgSyn_ControlNet_model = ImgSyn_ControlNet_model


        sampler_class = DDPMSampler if sampler_mode == 'ddpm' else DDIMSampler
        FgGen_Sampler = None
        if not self.use_fggen_controlnet:
            FgGen_Sampler = sampler_class(
                FgGen_Diffusion_model,
                beta_t,
                is_ImgSyn=False,
                is_with_ControlNet=False,
                is_Inference=False,
            ).to(DEVICE)
        ImgSyn_Sampler = sampler_class(
            ImgSyn_Diffusion_model,
            beta_t,
            is_ImgSyn=True,
            is_with_ControlNet=False,
            is_Inference=False,
        ).to(DEVICE)
        FgGen_CtrlNet_Sampler = None
        if self.use_fggen_controlnet:
            FgGen_CtrlNet_Sampler = sampler_class(
                FgGen_ControlNet_model,
                beta_t,
                is_ImgSyn=False,
                is_with_ControlNet=True,
                is_Inference=False,
            ).to(DEVICE)
        ImgSyn_CtrlNet_Sampler = None
        if self.use_imgsyn_controlnet:
            ImgSyn_CtrlNet_Sampler = sampler_class(
                ImgSyn_ControlNet_model,
                beta_t,
                is_ImgSyn=True,
                is_with_ControlNet=True,
                is_Inference=False,
            ).to(DEVICE)
        
        self.VAE_model = Latent_model
        self.FgGen_Sampler  = FgGen_Sampler
        self.ImgSyn_Sampler = ImgSyn_Sampler
        self.FgGen_CtrlNet_Sampler = FgGen_CtrlNet_Sampler
        self.ImgSyn_CtrlNet_Sampler = ImgSyn_CtrlNet_Sampler

        self.ImgSlice_h, self.ImgSlice_w = IMG_SIZE, IMG_SIZE
        self.ImgSlice_e_h, self.ImgSlice_e_w = IMG_SIZE // SIDELENGTH_SCALE_FACTOR, IMG_SIZE // SIDELENGTH_SCALE_FACTOR

        self.overlap_rate = overlap_rate
        self.overlap_buffer_thx_e = overlap_buffer_thx // SIDELENGTH_SCALE_FACTOR
        self.overlap_thx_e = int((IMG_SIZE // SIDELENGTH_SCALE_FACTOR) * overlap_rate)
        self.overlap_nobuffer_thx_e = self.overlap_thx_e - self.overlap_buffer_thx_e
        self.interval_e = int(IMG_SIZE // SIDELENGTH_SCALE_FACTOR - self.overlap_thx_e) 
        self.overlap_thx = self.overlap_thx_e * SIDELENGTH_SCALE_FACTOR
        self.interval = self.interval_e * SIDELENGTH_SCALE_FACTOR

        self.batch_size = batch_size
        self.is_FgGenInpaint_Resample = is_FgGenInpaint_Resample
        self.is_ImgSynInpaint_Resample = is_ImgSynInpaint_Resample

        self.is_Fixed_Prompt = is_Fixed_Prompt

        self.FgGen_RGB_savepath = FgGen_RGB_savepath
        self.FgGen_TIF_savepath = FgGen_TIF_savepath
        self.ImgSyn_RGB_savepath = ImgSyn_RGB_savepath
        self.ImgSyn_TIF_savepath = ImgSyn_TIF_savepath
        self.Msk_savepath = Msk_savepath

        self.conditional_element_proj = conditional_element_proj
        self.conditional_element_geotrans = conditional_element_geotrans

    def _sampling_kwargs(self):
        if self.sampler_mode == 'ddim':
            return {'steps': self.ddim_steps, 'eta': self.ddim_eta}
        return {}


    @torch.no_grad()
    def _get_diagonals_points(self, rows, cols, multiplier):
        return build_diagonal_tile_schedule(rows, cols, multiplier)


    @torch.no_grad()
    def _get_obj_prompt(self, cur_prompt_str, obj_prompt_str, noobj_prompt_str, LargeImg_e, row_idx, col_idx, detect_pixel_thx, 
                        NoObj_Change_to_Obj_probability=0.75, is_fixed_prompt=False):

        if is_fixed_prompt:
            return cur_prompt_str

        if row_idx == 0 and col_idx == 0:
            chosen_prompt = cur_prompt_str
        else:
            if row_idx == 0 and col_idx > 0:
                ImgSlice_e_upper = None
                ImgSlice_e_left  = LargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx - self.interval_e: col_idx + self.overlap_thx_e]
            elif row_idx > 0 and col_idx == 0:
                ImgSlice_e_upper = LargeImg_e[:, :, row_idx - self.interval_e: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                ImgSlice_e_left  = None
            else:
                ImgSlice_e_upper = LargeImg_e[:, :, row_idx - self.interval_e: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                ImgSlice_e_left  = LargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx - self.interval_e: col_idx + self.overlap_thx_e]

            if ImgSlice_e_upper is not None:
                ImgSlice_upper = self.VAE_model.decode(ImgSlice_e_upper)     # [1, 4, 256, 256]
                MskSlice_upper = self.FgSeg_model(ImgSlice_upper)
                MskSlice_upper = torch.sigmoid(MskSlice_upper)
                MskSlice_upper = (MskSlice_upper >= 0.5).float()
                
            if ImgSlice_e_left is not None: 
                ImgSlice_left = self.VAE_model.decode(ImgSlice_e_left)                                         # [1, 4, 256, 256]
                MskSlice_left = self.FgSeg_model(ImgSlice_left)
                MskSlice_left = torch.sigmoid(MskSlice_left)
                MskSlice_left = (MskSlice_left >= 0.5).float()
                
            if row_idx == 0 and col_idx > 0:
                left_overlap_area = MskSlice_left[:, :, :, (self.ImgSlice_w - detect_pixel_thx):]
                if torch.sum(left_overlap_area) >= 1:
                    chosen_prompt = obj_prompt_str
                else:
                    random_num = random.random()
                    if random_num > NoObj_Change_to_Obj_probability:
                        chosen_prompt = obj_prompt_str
                    else:
                        chosen_prompt = noobj_prompt_str
            
            elif row_idx > 0 and col_idx == 0:
                upper_overlap_area = MskSlice_upper[:, :, (self.ImgSlice_h - detect_pixel_thx):, :]
                if torch.sum(upper_overlap_area) >= 1:
                    chosen_prompt = obj_prompt_str
                else:
                    random_num = random.random()
                    if random_num > NoObj_Change_to_Obj_probability:
                        chosen_prompt = obj_prompt_str
                    else:
                        chosen_prompt = noobj_prompt_str
            
            else:
                upper_overlap_area = MskSlice_upper[:, :, (self.ImgSlice_h - detect_pixel_thx):, :]
                left_overlap_area  = MskSlice_left[:, :, :, (self.ImgSlice_w - detect_pixel_thx):]
                if torch.sum(upper_overlap_area) >= 1 or torch.sum(left_overlap_area) >= 1:
                    chosen_prompt = obj_prompt_str
                else:
                    random_num = random.random()
                    if random_num > NoObj_Change_to_Obj_probability:
                        chosen_prompt = obj_prompt_str
                    else:
                        chosen_prompt = noobj_prompt_str

        return chosen_prompt    
    

    @torch.no_grad()
    def _get_obj_prompt_forControlNet(self, obj_prompt_str, noobj_prompt_str, FgGen_condtional_element, row_idx, col_idx):

        FgSlice_Condele = FgGen_condtional_element[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
        
        if torch.sum(FgSlice_Condele) >= 1:
            chosen_prompt = obj_prompt_str
        else:
            chosen_prompt = noobj_prompt_str
        
        return chosen_prompt


    @torch.no_grad()
    def _compute_grad_of_crossarea(self, 
                                   row_idx, col_idx, 
                                   row_idx_max, col_idx_max, 
                                   overlap_thx, ImgSlice_h, ImgSlice_w,
                                   zeros_thx, grad_thx, ones_thx,
                                   is_tailgrad=True):
        
        GradStripSeq_0to1 = torch.cat((torch.zeros(zeros_thx, dtype=torch.float32),
                                       torch.linspace(0.00, 1.00, grad_thx, dtype=torch.float32),
                                       torch.ones(ones_thx, dtype=torch.float32),
                                       )).to(DEVICE)                                                                     # long strip
        Vertical_GradStrip_0to1 = GradStripSeq_0to1.expand(ImgSlice_h, overlap_thx).unsqueeze(0).unsqueeze(0).clone()    # 0 -> 1
        Vertical_GradStrip_1to0 = 1.0 - Vertical_GradStrip_0to1                                                          # 1 -> 0

        Horizontal_GradStrip_0to1 = torch.transpose(Vertical_GradStrip_0to1, dim0=2, dim1=3)   
        Horizontal_GradStrip_1to0 = torch.transpose(Vertical_GradStrip_1to0, dim0=2, dim1=3)   

        Vertical_GradStrip_0to1_Tail = torch.ones((1, 1, overlap_thx, overlap_thx), dtype=torch.float).to(DEVICE)
        Vertical_GradStrip_1to0_Tail = torch.ones((1, 1, overlap_thx, overlap_thx), dtype=torch.float).to(DEVICE)
        Horizontal_GradStrip_0to1_Tail = torch.ones((1, 1, overlap_thx, overlap_thx), dtype=torch.float).to(DEVICE)
        Horizontal_GradStrip_1to0_Tail = torch.ones((1, 1, overlap_thx, overlap_thx), dtype=torch.float).to(DEVICE)

        if ones_thx == 0:

            VGS0to1T_Vgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS0to1T_Vgrad = VGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS0to1T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS0to1T_Hgrad = VGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            Vertical_GradStrip_0to1_Tail *= (VGS0to1T_Vgrad * VGS0to1T_Hgrad)

            VGS1to0T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS1to0T_Vgrad = VGS1to0T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS1to0T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS1to0T_Hgrad = VGS1to0T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            Vertical_GradStrip_1to0_Tail *= (VGS1to0T_Vgrad * VGS1to0T_Hgrad)            

            HGS0to1T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            HGS0to1T_Vgrad = HGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            HGS0to1T_Hgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            HGS0to1T_Hgrad = HGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            Horizontal_GradStrip_0to1_Tail *= (HGS0to1T_Vgrad * HGS0to1T_Hgrad)
        
            HGS1to0T_Vgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            HGS1to0T_Vgrad = HGS1to0T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            HGS1to0T_Hgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            HGS1to0T_Hgrad = HGS1to0T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            Horizontal_GradStrip_1to0_Tail *= (HGS1to0T_Vgrad * HGS1to0T_Hgrad)

        elif ones_thx > 0:

            # The first one
            VGS0to1T_Vgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS0to1T_Vgrad = VGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS0to1T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS0to1T_Hgrad = VGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Vertical_GradStrip_0to1_Tail *= VGS0to1T_Vgrad
            Vertical_GradStrip_0to1_Tail[:, :, :, (overlap_thx - ones_thx):] = 1.0
            Vertical_GradStrip_0to1_Tail *= VGS0to1T_Hgrad

            # The second one
            VGS1to0T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS1to0T_Vgrad = VGS1to0T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS1to0T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS1to0T_Hgrad = VGS1to0T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Vertical_GradStrip_1to0_Tail *= (VGS1to0T_Vgrad * VGS1to0T_Hgrad)
            Vertical_GradStrip_1to0_Tail[:, :, :, (overlap_thx - ones_thx):] = 0.0

            # The third one
            HGS0to1T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            HGS0to1T_Vgrad = HGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            HGS0to1T_Hgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            HGS0to1T_Hgrad = HGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Horizontal_GradStrip_0to1_Tail *= HGS0to1T_Vgrad
            Horizontal_GradStrip_0to1_Tail[:, :, (overlap_thx - ones_thx):, :] = 1.0
            Horizontal_GradStrip_0to1_Tail *= HGS0to1T_Hgrad

            # The fourth one
            HGS1to0T_Vgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            HGS1to0T_Vgrad = HGS1to0T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            HGS1to0T_Hgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            HGS1to0T_Hgrad = HGS1to0T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Horizontal_GradStrip_1to0_Tail *= (HGS1to0T_Vgrad * HGS1to0T_Hgrad)
            Horizontal_GradStrip_1to0_Tail[:, :, (overlap_thx - ones_thx):, :] = 0.0


        Vertical_GradMap_0to1 = torch.ones((1, 1, ImgSlice_h, ImgSlice_w), dtype=torch.float, device=DEVICE)
        Vertical_GradMap_0to1[:, :, :, :overlap_thx] *= Vertical_GradStrip_0to1
        Vertical_GradMap_1to0 = 1.0 - Vertical_GradMap_0to1        

        Horizontal_GradMap_0to1 = torch.ones((1, 1, ImgSlice_h, ImgSlice_w), dtype=torch.float, device=DEVICE)
        Horizontal_GradMap_0to1[:, :, :overlap_thx, :] *= Horizontal_GradStrip_0to1
        Horizontal_GradMap_1to0 = 1.0 - Horizontal_GradMap_0to1

        LeftTop_GradMap_0to1 = torch.ones((1, 1, ImgSlice_h, ImgSlice_w), dtype=torch.float, device=DEVICE)
        LeftTop_GradMap_0to1[:, :, :, :overlap_thx] *= Vertical_GradStrip_0to1
        LeftTop_GradMap_0to1[:, :, :overlap_thx, :] *= Horizontal_GradStrip_0to1
        LeftTop_GradMap_1to0 = 1.0 - LeftTop_GradMap_0to1

        if is_tailgrad:

            if row_idx != 0 and col_idx != col_idx_max:
                Horizontal_GradMap_0to1[:, :, :overlap_thx, ImgSlice_w - overlap_thx:] = Horizontal_GradStrip_0to1_Tail
                Horizontal_GradMap_1to0[:, :, :overlap_thx, ImgSlice_w - overlap_thx:] = Horizontal_GradStrip_1to0_Tail
                LeftTop_GradMap_0to1[:, :, :overlap_thx, ImgSlice_w - overlap_thx:] = Horizontal_GradStrip_0to1_Tail
                LeftTop_GradMap_1to0[:, :, :overlap_thx, ImgSlice_w - overlap_thx:] = Horizontal_GradStrip_1to0_Tail
            
            if row_idx != row_idx_max and col_idx != 0:
                Vertical_GradMap_0to1[:, :, ImgSlice_h - overlap_thx:, :overlap_thx] = Vertical_GradStrip_0to1_Tail
                Vertical_GradMap_1to0[:, :, ImgSlice_h - overlap_thx:, :overlap_thx] = Vertical_GradStrip_1to0_Tail
                LeftTop_GradMap_0to1[:, :, ImgSlice_h - overlap_thx:, :overlap_thx] = Vertical_GradStrip_0to1_Tail
                LeftTop_GradMap_1to0[:, :, ImgSlice_h - overlap_thx:, :overlap_thx] = Vertical_GradStrip_1to0_Tail

        return Horizontal_GradMap_0to1, Horizontal_GradMap_1to0, Vertical_GradMap_0to1, Vertical_GradMap_1to0, LeftTop_GradMap_0to1, LeftTop_GradMap_1to0


    @torch.no_grad()
    def main(
        self,
        LargeImg_h,
        LargeImg_w,
        prompt_str,
        obj_prompt_str,
        noobj_prompt_str,
        detect_pixel_thx=32,
        FgGen_original_msk=None,
        FgGen_conditional_element=None,
        ImgSyn_conditional_element=None,
    ):

        if not isinstance(LargeImg_h, (int, np.integer)) or not isinstance(LargeImg_w, (int, np.integer)):
            raise TypeError("LargeImg_h and LargeImg_w must be integers.")
        if LargeImg_h < self.ImgSlice_h or LargeImg_w < self.ImgSlice_w:
            raise ValueError(
                f"LargeImg_h and LargeImg_w must both be at least IMG_SIZE={self.ImgSlice_h}, "
                f"got ({LargeImg_h}, {LargeImg_w})."
            )
        requested_size = (LargeImg_h, LargeImg_w)

        if self.use_fggen_controlnet or self.use_imgsyn_controlnet:
            if FgGen_original_msk is None:
                raise ValueError("FgGen_original_msk is required when ControlNet is enabled.")
            if self.use_fggen_controlnet and FgGen_conditional_element is None:
                raise ValueError("FgGen_conditional_element is required.")
            if self.use_imgsyn_controlnet and ImgSyn_conditional_element is None:
                raise ValueError("ImgSyn_conditional_element is required.")

            conditional_tensors = {
                "FgGen_original_msk": FgGen_original_msk,
            }
            if FgGen_conditional_element is not None:
                conditional_tensors["FgGen_conditional_element"] = FgGen_conditional_element
            if ImgSyn_conditional_element is not None:
                conditional_tensors["ImgSyn_conditional_element"] = ImgSyn_conditional_element
            for name, tensor in conditional_tensors.items():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    raise ValueError(f"{name} must be a four-dimensional [B, C, H, W] tensor.")
                if tensor.shape[0] != 1 or tuple(tensor.shape[-2:]) != requested_size:
                    raise ValueError(
                        f"{name} must have batch size 1 and spatial size {requested_size}, "
                        f"got {tuple(tensor.shape)}."
                    )
            if FgGen_original_msk.shape[1] != 1:
                raise ValueError("FgGen_original_msk must contain exactly one channel.")
            if FgGen_conditional_element is not None and FgGen_conditional_element.shape[1] != 1:
                raise ValueError("FgGen_conditional_element must contain exactly one channel.")
            if (
                ImgSyn_conditional_element is not None
                and ImgSyn_conditional_element.shape[1] != INPUT_CHANNELS
            ):
                raise ValueError(
                    f"ImgSyn_conditional_element must contain {INPUT_CHANNELS} channels, "
                    f"got {ImgSyn_conditional_element.shape[1]}."
                )
            
        LargeImg_e_h = int((LargeImg_h - (LargeImg_h - self.ImgSlice_h) % self.interval) // SIDELENGTH_SCALE_FACTOR)
        LargeImg_e_w = int((LargeImg_w - (LargeImg_w - self.ImgSlice_w) % self.interval) // SIDELENGTH_SCALE_FACTOR)

        LargeImg_h = LargeImg_e_h * SIDELENGTH_SCALE_FACTOR
        LargeImg_w = LargeImg_e_w * SIDELENGTH_SCALE_FACTOR
        if (LargeImg_h, LargeImg_w) != requested_size:
            print(
                "-- Notice -- Requested large-image size "
                f"{requested_size} was adjusted to ({LargeImg_h}, {LargeImg_w}) "
                "to match the sliding-window interval."
            )

        FgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        SynLargeImg_e = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)

        NoGrad_FgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        NoGrad_SynLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)

        rows_num = int(LargeImg_e_h // self.interval_e) if LargeImg_e_h % self.interval_e != 0 else int(LargeImg_e_h // self.interval_e) - 1
        cols_num = int(LargeImg_e_w // self.interval_e) if LargeImg_e_w % self.interval_e != 0 else int(LargeImg_e_w // self.interval_e) - 1

        diagonals_points_list = self._get_diagonals_points(rows_num, cols_num, self.interval_e)

        row_idx_max, col_idx_max = max(diagonals_points_list)[0]


        for diagonal_points in diagonals_points_list:
            
            # This is one batch
            FgSlice_e_0_list = []
            FgSlice_e_t_list = []
            SynSlice_e_0_list = []
            SynSlice_e_t_list = []
            BldMsk_list = []
            Prompt_list = []

            if self.use_fggen_controlnet or self.use_imgsyn_controlnet:
                FgSlice_orimsk_list = []
            if self.use_fggen_controlnet:
                FgSlice_condele_list = []
            if self.use_imgsyn_controlnet:
                SynSlice_condele_list = []    

            # Collect materials
            for row_idx, col_idx in diagonal_points:

                BldGradMap_e_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                                    row_idx_max, col_idx_max,
                                                                    self.overlap_thx_e, self.ImgSlice_e_h, self.ImgSlice_e_w, 
                                                                    zeros_thx=self.overlap_thx_e - self.overlap_buffer_thx_e, grad_thx=self.overlap_buffer_thx_e, ones_thx=0,
                                                                    is_tailgrad=False)
                H_BldGradMap_0to1, H_BldGradMap_1to0, V_BldGradMap_0to1, V_BldGradMap_1to0, LT_BldGradMap_0to1, LT_BldGradMap_1to0 = BldGradMap_e_list

                # Generate slice No.0
                if row_idx == 0 and col_idx == 0:
                    FgSlice_e_t  = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)

                    if not self.use_fggen_controlnet:
                        chosen_prompt = self._get_obj_prompt(prompt_str, obj_prompt_str, noobj_prompt_str, FgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_Prompt)
                    else:
                        chosen_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)
                    
                    FgSlice_e_t_list.append(FgSlice_e_t)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    Prompt_list.append(chosen_prompt)

                # Top edge
                elif row_idx == 0 and col_idx > 0:
                    
                    # Compute Slice_e_0
                    FgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    FgSlice_e_0[:, :, :, :self.overlap_thx_e] = FgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :, :self.overlap_thx_e] = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    
                    # Compute Slice_e_t
                    FgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # Compute BldMsk
                    BldMsk = V_BldGradMap_0to1
                   
                    # Get the prompt
                    if not self.use_fggen_controlnet:
                        chosen_prompt = self._get_obj_prompt(chosen_prompt, obj_prompt_str, noobj_prompt_str, FgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_Prompt)
                    else:
                        chosen_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)
                    
                    FgSlice_e_0_list.append(FgSlice_e_0)
                    FgSlice_e_t_list.append(FgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    Prompt_list.append(chosen_prompt)

                # Left edge
                elif row_idx > 0 and col_idx == 0:

                    # Compute Slice_e_0
                    FgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    FgSlice_e_0[:, :, :self.overlap_thx_e, :] = FgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :self.overlap_thx_e, :] = SynLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    
                    # Compute Slice_e_t
                    FgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # Compute BldMsk
                    BldMsk = H_BldGradMap_0to1

                    if not self.use_fggen_controlnet:
                        chosen_prompt = self._get_obj_prompt(chosen_prompt, obj_prompt_str, noobj_prompt_str, FgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_Prompt)
                    else:
                        chosen_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)

                    FgSlice_e_0_list.append(FgSlice_e_0)
                    FgSlice_e_t_list.append(FgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    Prompt_list.append(chosen_prompt)

                else:
                    
                    # Compute Slice_e_0
                    FgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    FgSlice_e_0[:, :, :self.overlap_thx_e, :] = FgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    FgSlice_e_0[:, :, :, :self.overlap_thx_e] = FgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :self.overlap_thx_e, :] = SynLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    SynSlice_e_0[:, :, :, :self.overlap_thx_e] = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                  
                    # Compute Slice_e_t
                    FgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # Compute msk
                    BldMsk = LT_BldGradMap_0to1
     
                    if not self.use_fggen_controlnet:
                        chosen_prompt = self._get_obj_prompt(chosen_prompt, obj_prompt_str, noobj_prompt_str, FgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_Prompt)
                    else:
                        chosen_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)

                    FgSlice_e_0_list.append(FgSlice_e_0)
                    FgSlice_e_t_list.append(FgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    Prompt_list.append(chosen_prompt)

                # FgGen uses the foreground mask; ImgSyn uses high-frequency features after object inpainting.
                if self.use_fggen_controlnet or self.use_imgsyn_controlnet:
                    FgSlice_orimsk = FgGen_original_msk[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    FgSlice_orimsk_list.append(FgSlice_orimsk)
                if self.use_fggen_controlnet:
                    FgSlice_condele = FgGen_conditional_element[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    FgSlice_condele_list.append(FgSlice_condele)
                if self.use_imgsyn_controlnet:
                    SynSlice_condele = ImgSyn_conditional_element[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    SynSlice_condele_list.append(SynSlice_condele)


            # Start generating and write into FgLargeImg_e and SynLargeImg_e
            if row_idx == 0 and col_idx == 0:

                print('Prompt: ', Prompt_list)
                sampling_kwargs = self._sampling_kwargs()

                if self.use_fggen_controlnet:
                    FgSlice_e_set = self.FgGen_CtrlNet_Sampler.forward(
                        FgSlice_e_t_list[0],
                        Prompt_list,
                        is_record_process=False,
                        conditional_element=FgSlice_condele_list[0],
                        **sampling_kwargs,
                    )
                else:
                    FgSlice_e_set = self.FgGen_Sampler.forward(
                        FgSlice_e_t_list[0],
                        Prompt_list,
                        is_record_process=False,
                        **sampling_kwargs,
                    )

                if not self.use_imgsyn_controlnet:
                    SynSlice_e_set = self.ImgSyn_Sampler.forward(
                        SynSlice_e_t_list[0],
                        Prompt_list,
                        is_record_process=False,
                        fg_imgs_e=FgSlice_e_set,
                        **sampling_kwargs,
                    )
                else:

                    # Background ControlNet generation
                    SynSlice_condele_set = SynSlice_condele_list[0]
                    SynSlice_condele_e_set = self.VAE_model.encode(SynSlice_condele_set)
                    SynSlice_condele_e_set, _, _ = self.VAE_model.reparameterize(SynSlice_condele_e_set)

                    # Inpaint the Obj part of SynSlice_condele_set first
                    zero_maps = torch.zeros((len(FgSlice_e_set), INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(DEVICE)
                    zero_maps_e = self.VAE_model.encode(zero_maps)
                    zero_maps_e, _, _ = self.VAE_model.reparameterize(zero_maps_e)

                    FgSlice_orimsk_set = tensor_dilate(FgSlice_orimsk_list[0]) 
                    FgSlice_orimsk_e_set = F.interpolate(FgSlice_orimsk_set, size=(IMG_SIZE // SIDELENGTH_SCALE_FACTOR, IMG_SIZE // SIDELENGTH_SCALE_FACTOR), mode='bilinear')
                    
                    SynSlice_e_t_set = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)

                    SynSlice_condele_e_set = self.ImgSyn_Sampler.inpaint(
                        SynSlice_condele_e_set,
                        SynSlice_e_t_set,
                        FgSlice_orimsk_e_set,
                        prompt_str=['NoObj' for _ in range(len(FgSlice_e_set))],
                        is_record_process=False,
                        is_resample=self.is_ImgSynInpaint_Resample,
                        fg_imgs_e=zero_maps_e,
                        **sampling_kwargs,
                    )
                    SynSlice_condele_set = self.VAE_model.decode(SynSlice_condele_e_set)

                    # Then extract the high-frequency information
                    SynSlice_condele_set = SynSlice_condele_set.squeeze(0)
                    SynSlice_Condele_HFreq_set = extract_high_frequency(
                        SynSlice_condele_set,
                        CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
                    )
                    SynSlice_Condele_HFreq_set = SynSlice_Condele_HFreq_set.unsqueeze(0)                 

                    # Extract the Obj mask
                    FgSlice_set = self.VAE_model.decode(FgSlice_e_set)
                    FgSlice_msk_set = self.FgSeg_model(FgSlice_set)
                    FgSlice_msk_set = torch.sigmoid(FgSlice_msk_set)
                    FgSlice_msk_set = (FgSlice_msk_set >= 0.5).float()     # [6, 1, 256, 256]  

                    SynSlice_Condele_HFreq_set = SynSlice_Condele_HFreq_set * (1.0 - FgSlice_msk_set)       
                    
                    SynSlice_e_set = self.ImgSyn_CtrlNet_Sampler.forward(
                        SynSlice_e_t_list[0],
                        Prompt_list,
                        is_record_process=False,
                        fg_imgs_e=FgSlice_e_set,
                        conditional_element=SynSlice_Condele_HFreq_set,
                        **sampling_kwargs,
                    )
            
            else:
                FgSlice_e_0_set = torch.cat(FgSlice_e_0_list)
                FgSlice_e_t_set = torch.cat(FgSlice_e_t_list)
                SynSlice_e_0_set = torch.cat(SynSlice_e_0_list)
                SynSlice_e_t_set = torch.cat(SynSlice_e_t_list)
                BldMsk_set = torch.cat(BldMsk_list)  
                             
                if self.use_fggen_controlnet or self.use_imgsyn_controlnet:
                    FgSlice_orimsk_set = torch.cat(FgSlice_orimsk_list)
                if self.use_fggen_controlnet:
                    FgSlice_condele_set = torch.cat(FgSlice_condele_list)
                if self.use_imgsyn_controlnet:
                    SynSlice_condele_set = torch.cat(SynSlice_condele_list)

                print('Prompt: ', Prompt_list)

                # Process in batches
                Slice_totalnum = len(FgSlice_e_0_set)
                Loop_num = int(np.ceil(Slice_totalnum / self.batch_size))

                FgSlice_e_set = []
                SynSlice_e_set = []
                sampling_kwargs = self._sampling_kwargs()
                for Loop_idx in range(Loop_num):
                    
                    FgSlice_e_0_batch = FgSlice_e_0_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    FgSlice_e_t_batch = FgSlice_e_t_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    SynSlice_e_0_batch = SynSlice_e_0_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    SynSlice_e_t_batch = SynSlice_e_t_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    BldMsk_batch = BldMsk_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    Prompt_batch = Prompt_list[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    
                    if self.use_fggen_controlnet:
                        FgSlice_condele_batch = FgSlice_condele_set[
                            Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size
                        ]
                        FgSlice_e_batch = self.FgGen_CtrlNet_Sampler.inpaint(
                            FgSlice_e_0_batch,
                            FgSlice_e_t_batch,
                            BldMsk_batch,
                            Prompt_batch,
                            is_record_process=False,
                            is_resample=self.is_FgGenInpaint_Resample,
                            conditional_element=FgSlice_condele_batch,
                            **sampling_kwargs,
                        )
                    else:
                        FgSlice_e_batch = self.FgGen_Sampler.inpaint(
                            FgSlice_e_0_batch,
                            FgSlice_e_t_batch,
                            BldMsk_batch,
                            Prompt_batch,
                            is_record_process=False,
                            is_resample=self.is_FgGenInpaint_Resample,
                            **sampling_kwargs,
                        )

                    if self.use_imgsyn_controlnet:
                        FgSlice_orimsk_batch = FgSlice_orimsk_set[
                            Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size
                        ]
                        SynSlice_condele_batch = SynSlice_condele_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                        
                        SynSlice_condele_e_batch = self.VAE_model.encode(SynSlice_condele_batch)
                        SynSlice_condele_e_batch, _, _ = self.VAE_model.reparameterize(SynSlice_condele_e_batch)

                        # Inpaint the Obj part of SynSlice first
                        zero_maps = torch.zeros((len(FgSlice_e_batch), INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(DEVICE)
                        zero_maps_e = self.VAE_model.encode(zero_maps)
                        zero_maps_e, _, _ = self.VAE_model.reparameterize(zero_maps_e)

                        FgSlice_orimsk_batch = tensor_dilate(FgSlice_orimsk_batch) 
                        FgSlice_orimsk_e_batch = F.interpolate(FgSlice_orimsk_batch, size=(IMG_SIZE // SIDELENGTH_SCALE_FACTOR, IMG_SIZE // SIDELENGTH_SCALE_FACTOR), mode='bilinear')

                        SynSlice_condele_e_batch = self.ImgSyn_Sampler.inpaint(
                            SynSlice_condele_e_batch,
                            SynSlice_e_t_batch,
                            FgSlice_orimsk_e_batch,
                            prompt_str=['NoObj' for _ in range(len(FgSlice_e_batch))],
                            is_record_process=False,
                            is_resample=self.is_ImgSynInpaint_Resample,
                            fg_imgs_e=zero_maps_e,
                            **sampling_kwargs,
                        )
                        SynSlice_condele_batch = self.VAE_model.decode(SynSlice_condele_e_batch)

                        # Then extract the high-frequency information
                        SynSlice_Condele_HFreq_list = []
                        for SynSlice_Condele in SynSlice_condele_batch:
                            SynSlice_Condele_HFreq = extract_high_frequency(
                                SynSlice_Condele,
                                CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
                            )
                            SynSlice_Condele_HFreq_list.append(SynSlice_Condele_HFreq.unsqueeze(0))
                        SynSlice_Condele_HFreq_batch = torch.cat(SynSlice_Condele_HFreq_list).to(DEVICE)        

                        # Extract the Obj mask
                        FgSlice_batch = self.VAE_model.decode(FgSlice_e_batch)
                        FgSlice_msk_batch = self.FgSeg_model(FgSlice_batch)
                        FgSlice_msk_batch = torch.sigmoid(FgSlice_msk_batch)
                        FgSlice_msk_batch = (FgSlice_msk_batch >= 0.5).float()     # [6, 1, 256, 256]  

                        SynSlice_Condele_HFreq_batch = SynSlice_Condele_HFreq_batch * (1.0 - FgSlice_msk_batch)               

                        # Then denoise
                        SynSlice_e_batch = self.ImgSyn_CtrlNet_Sampler.inpaint(
                            SynSlice_e_0_batch,
                            SynSlice_e_t_batch,
                            BldMsk_batch,
                            Prompt_batch,
                            is_record_process=False,
                            is_resample=self.is_ImgSynInpaint_Resample,
                            fg_imgs_e=FgSlice_e_batch,
                            conditional_element=SynSlice_Condele_HFreq_batch,
                            **sampling_kwargs,
                        )
                    else:
                        SynSlice_e_batch = self.ImgSyn_Sampler.inpaint(
                            SynSlice_e_0_batch,
                            SynSlice_e_t_batch,
                            BldMsk_batch,
                            Prompt_batch,
                            is_record_process=False,
                            is_resample=self.is_ImgSynInpaint_Resample,
                            fg_imgs_e=FgSlice_e_batch,
                            **sampling_kwargs,
                        )
                    
                    FgSlice_e_set.append(FgSlice_e_batch)
                    SynSlice_e_set.append(SynSlice_e_batch)
                
                FgSlice_e_set = torch.cat(FgSlice_e_set)
                SynSlice_e_set = torch.cat(SynSlice_e_set)

            # Write back
            for idx, (FgSlice_e, SynSlice_e) in enumerate(zip(FgSlice_e_set, SynSlice_e_set)):
                row_idx, col_idx = diagonal_points[idx]
                FgSlice_e = FgSlice_e.unsqueeze(0)
                SynSlice_e = SynSlice_e.unsqueeze(0)  
                                                      
                MosGradMap_e_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                                    row_idx_max, col_idx_max,
                                                                    self.overlap_buffer_thx_e, self.ImgSlice_e_h, self.ImgSlice_e_w, 
                                                                    zeros_thx=0, grad_thx=self.overlap_buffer_thx_e, ones_thx=0,
                                                                    is_tailgrad=True)
                H_MosGradMap_0to1, H_MosGradMap_1to0, V_MosGradMap_0to1, V_MosGradMap_1to0, LT_MosGradMap_0to1, LT_MosGradMap_1to0 = MosGradMap_e_list

                if row_idx == 0 and col_idx == 0:
                    FgLargeImg_e[:, :, :self.ImgSlice_e_h, :self.ImgSlice_e_w] = FgSlice_e
                    SynLargeImg_e[:, :, :self.ImgSlice_e_h, :self.ImgSlice_e_w] = SynSlice_e        

                # Top edge
                elif row_idx == 0 and col_idx > 0:

                    new_row_idx = 0
                    new_col_idx = col_idx + self.overlap_nobuffer_thx_e

                    # Reset the target area to 0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0

                    new_V_MosGradMap_0to1 = V_MosGradMap_0to1[:, :, :, :(self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)]
                    new_V_MosGradMap_1to0 = V_MosGradMap_1to0[:, :, :, :(self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)]

                    # Pair the cropped vertical weights with the matching right-side content.
                    Fg_CoverageArea = NoGrad_FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Fg_CoverageArea * new_V_MosGradMap_1to0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += FgSlice_e[:, :, :, self.overlap_nobuffer_thx_e:] * new_V_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Syn_CoverageArea * new_V_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += SynSlice_e[:, :, :, self.overlap_nobuffer_thx_e:] * new_V_MosGradMap_0to1


                # Left edge
                elif row_idx > 0 and col_idx == 0:

                    new_row_idx = row_idx + self.overlap_nobuffer_thx_e
                    new_col_idx = 0
                    
                    # Reset the target area to 0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] = 0.0
                    
                    # Modify H_MosGradMap_1to0 and H_MosGradMap_0to1
                    new_H_MosGradMap_0to1 = H_MosGradMap_0to1[:, :, :(self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), :]
                    new_H_MosGradMap_1to0 = H_MosGradMap_1to0[:, :, :(self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), :]

                    # Pair the cropped horizontal weights with the matching lower content.
                    Fg_CoverageArea = NoGrad_FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)].clone()
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += Fg_CoverageArea * new_H_MosGradMap_1to0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += FgSlice_e[:, :, self.overlap_nobuffer_thx_e:, :] * new_H_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)].clone()
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += Syn_CoverageArea * new_H_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += SynSlice_e[:, :, self.overlap_nobuffer_thx_e:, :] * new_H_MosGradMap_0to1


                elif row_idx > 0 and col_idx > 0:
                    
                    new_row_idx = row_idx + self.overlap_nobuffer_thx_e
                    new_col_idx = col_idx + self.overlap_nobuffer_thx_e

                    # Reset the target area to 0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] = 0.0

                    if row_idx != row_idx_max:
                        # The buffered and unbuffered widths sum to overlap_thx_e.
                        FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                        SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0   
                    else:
                        # Note: at row_idx_max, self.overlap_buffer_thx_e no longer needs to be subtracted
                        FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                        SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0                          

                    # Crop both corner weight maps by the unbuffered overlap.
                    new_LT_MosGradMap_0to1 = torch.ones((1, 1, self.ImgSlice_e_h - self.overlap_nobuffer_thx_e, self.ImgSlice_e_w - self.overlap_nobuffer_thx_e), dtype=torch.float).to(DEVICE)
                    new_LT_MosGradMap_0to1[:, :, :self.overlap_buffer_thx_e, :self.overlap_buffer_thx_e] = LT_MosGradMap_0to1[:, :, :self.overlap_buffer_thx_e, :self.overlap_buffer_thx_e]
                    new_LT_MosGradMap_0to1[:, :, self.overlap_buffer_thx_e:, :self.overlap_buffer_thx_e] = LT_MosGradMap_0to1[:, :, self.overlap_thx_e:, :self.overlap_buffer_thx_e]
                    new_LT_MosGradMap_0to1[:, :, :self.overlap_buffer_thx_e, self.overlap_buffer_thx_e:] = LT_MosGradMap_0to1[:, :, :self.overlap_buffer_thx_e, self.overlap_thx_e:]
                    new_LT_MosGradMap_0to1[:, :, self.overlap_buffer_thx_e:, self.overlap_buffer_thx_e:] = LT_MosGradMap_0to1[:, :, self.overlap_thx_e:, self.overlap_thx_e:]

                    new_LT_MosGradMap_1to0 = torch.ones((1, 1, self.ImgSlice_e_h - self.overlap_nobuffer_thx_e, self.ImgSlice_e_w - self.overlap_nobuffer_thx_e), dtype=torch.float).to(DEVICE)
                    new_LT_MosGradMap_1to0[:, :, :self.overlap_buffer_thx_e, :self.overlap_buffer_thx_e] = LT_MosGradMap_1to0[:, :, :self.overlap_buffer_thx_e, :self.overlap_buffer_thx_e]
                    new_LT_MosGradMap_1to0[:, :, self.overlap_buffer_thx_e:, :self.overlap_buffer_thx_e] = LT_MosGradMap_1to0[:, :, self.overlap_thx_e:, :self.overlap_buffer_thx_e]
                    new_LT_MosGradMap_1to0[:, :, :self.overlap_buffer_thx_e, self.overlap_buffer_thx_e:] = LT_MosGradMap_1to0[:, :, :self.overlap_buffer_thx_e, self.overlap_thx_e:]
                    new_LT_MosGradMap_1to0[:, :, self.overlap_buffer_thx_e:, self.overlap_buffer_thx_e:] = LT_MosGradMap_1to0[:, :, self.overlap_thx_e:, self.overlap_thx_e:]


                    # Pair the cropped corner weights with the matching bottom-right content.
                    Fg_CoverageArea = NoGrad_FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Fg_CoverageArea * new_LT_MosGradMap_1to0
                    FgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += FgSlice_e[:, :, self.overlap_nobuffer_thx_e:, self.overlap_nobuffer_thx_e:] * new_LT_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()              
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Syn_CoverageArea * new_LT_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += SynSlice_e[:, :, self.overlap_nobuffer_thx_e:, self.overlap_nobuffer_thx_e:] * new_LT_MosGradMap_0to1
                    

            # Update NoGrad_FgLargeImg_e and NoGrad_SynLargeImg_e
            NoGrad_FgLargeImg_e = FgLargeImg_e.clone()
            NoGrad_SynLargeImg_e = SynLargeImg_e.clone()
              

        # Decode and write into the large image
        FgLargeImg    = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        SynLargeImg   = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        MskLargeImg   = torch.zeros((1, UNET_OUTPUT_CHANNELS, LargeImg_h, LargeImg_w)  , dtype=torch.uint8).to(DEVICE)

        NoGrad_FgLargeImg  = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        NoGrad_SynLargeImg  = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)

        for diagonal_points in diagonals_points_list:
            for row_idx, col_idx in diagonal_points:

                GradMap_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                               row_idx_max, col_idx_max,
                                                               self.overlap_thx, self.ImgSlice_h, self.ImgSlice_w, 
                                                               zeros_thx=0, grad_thx=self.overlap_thx, ones_thx=0,
                                                               is_tailgrad=True)
                H_GradMap_0to1, H_GradMap_1to0, V_GradMap_0to1, V_GradMap_1to0, LT_GradMap_0to1, LT_GradMap_1to0 = GradMap_list

                FgSlice_e = FgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.ImgSlice_e_w]
                SynSlice_e = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.ImgSlice_e_w]
                FgSlice = self.VAE_model.decode(FgSlice_e)  
                SynSlice = self.VAE_model.decode(SynSlice_e)  

                MskSlice = self.FgSeg_model(FgSlice)
                MskSlice = torch.sigmoid(MskSlice)
                MskSlice = (MskSlice >= 0.5).float()

                if row_idx == 0 and col_idx == 0:
                    FgLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = FgSlice
                    SynLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = SynSlice
                    MskLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = MskSlice

                # Top edge
                elif row_idx == 0 and col_idx > 0:

                    # Reset the target area to 0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0

                    # Handle the rest
                    Fg_CoverageArea = NoGrad_FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Fg_CoverageArea * V_GradMap_1to0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += FgSlice * V_GradMap_0to1
                    
                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * V_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * V_GradMap_0to1
                    
                    V_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    V_BldMskMap_1to0[:, :, :, :self.overlap_thx] = 1.0
                    Msk_CoverageArea = MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * V_BldMskMap_1to0
                    Msk_NewlyGened = MskSlice * V_BldMskMap_1to0
                    MskSlice = torch.min(Msk_NewlyGened, Msk_CoverageArea) + MskSlice * (1.0 - V_BldMskMap_1to0)
                    MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = MskSlice
                
                # Left edge
                elif row_idx > 0 and col_idx == 0:

                    # Reset the target area to 0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    
                    # Handle the rest       
                    Fg_CoverageArea = NoGrad_FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Fg_CoverageArea * H_GradMap_1to0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += FgSlice * H_GradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * H_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * H_GradMap_0to1

                    H_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    H_BldMskMap_1to0[:, :, :self.overlap_thx, :] = 1.0
                    Msk_CoverageArea = MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * H_BldMskMap_1to0
                    Msk_NewlyGened = MskSlice * H_BldMskMap_1to0
                    MskSlice = torch.min(Msk_NewlyGened, Msk_CoverageArea) + MskSlice * (1.0 - H_BldMskMap_1to0)
                    MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = MskSlice

                elif row_idx > 0 and col_idx > 0:

                     # Reset the target area to 0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    
                    if row_idx != row_idx_max:
                        FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0   
                    else:
                        FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0                          
                                     
                    # Handle the rest
                    Fg_CoverageArea = NoGrad_FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Fg_CoverageArea * LT_GradMap_1to0
                    FgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += FgSlice * LT_GradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()                  
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * LT_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * LT_GradMap_0to1

                    LT_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    LT_BldMskMap_1to0[:, :, :, :self.overlap_thx] = 1.0
                    LT_BldMskMap_1to0[:, :, :self.overlap_thx, :] = 1.0
                    Msk_CoverageArea = MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * LT_BldMskMap_1to0
                    Msk_NewlyGened = MskSlice * LT_BldMskMap_1to0
                    MskSlice = torch.min(Msk_NewlyGened, Msk_CoverageArea) + MskSlice * (1.0 - LT_BldMskMap_1to0)
                    MskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = MskSlice

            # Update NoGrad_FgLargeImg and NoGrad_SynLargeImg
            NoGrad_FgLargeImg = FgLargeImg.clone()
            NoGrad_SynLargeImg = SynLargeImg.clone()
 

        if self.ImgSyn_RGB_savepath is not None:
            save_rgb_datas(prepare_rgb_vis_tensor(SynLargeImg).cpu(), nrow=3, 
                           savepath=self.ImgSyn_RGB_savepath, 
                           is_showminmax=False, is_makegrid=False)  

        if self.FgGen_RGB_savepath is not None:
            save_rgb_datas(prepare_rgb_vis_tensor(FgLargeImg).cpu(), nrow=3, 
                           savepath=self.FgGen_RGB_savepath, 
                           is_showminmax=False, is_makegrid=False)  
            
        saved_proj = self.conditional_element_proj
        saved_geotrans = self.conditional_element_geotrans

        if self.ImgSyn_TIF_savepath is not None:
            _, saved_proj, saved_geotrans = save_tif_datas(SynLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                                                           savepath=self.ImgSyn_TIF_savepath,
                                                           is_showminmax=False)

        if self.FgGen_TIF_savepath is not None:
            _, saved_proj, saved_geotrans = save_tif_datas(FgLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                                                           savepath=self.FgGen_TIF_savepath,
                                                           is_showminmax=False)


        if self.Msk_savepath is not None:
            save_msk_datas(MskLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                           savepath=self.Msk_savepath, 
                           is_showminmax=False, is_makegrid=False)      


        return FgLargeImg, SynLargeImg, MskLargeImg
