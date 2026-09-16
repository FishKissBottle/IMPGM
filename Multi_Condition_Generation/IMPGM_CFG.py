import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import *
import os
from FgGen.FgGen_Code.FgGen_Diffusion import FgGen_Diffusion_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
from FgGen.FgGen_Code.FgGen_ControlNet import ControlNet_on_FgGen_Diffusion
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet import ControlNet_on_ImgSyn_Diffusion
from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from IMPGM_Utils import descale_latent, load_controlnet_model_for_eval, load_standard_vae, scale_latent, load_model_for_eval, set_random_seed, tensor_dilate
import torch
import torch.nn.functional as F
from Diffusion_Sampler import DDPMSampler, DDIMSampler
from Multi_Condition_Generation.Diffusion_Sampler_CFG import (
    DDIMSampler_CFG,
    DDPMSampler_CFG,
)
import cv2


class CFG():
    """Run classifier-free guidance for the IMPGM generation pipeline."""
    def __init__(self, 
                 sampler_mode,
                 beta_t,
                 Latent_model_savepath,
                 FgGen_Diffusion_model_savepath,
                 FgGen_ControlNet_model_savepath,
                 ImgSyn_Diffusion_model_savepath,
                 ImgSyn_ControlNet_model_savepath,
                 FgSeg_UNet_model_savepath,
                 is_FgGenInpaint_Resample=True,
                 is_ImgSynInpaint_Resample=True,
                 is_with_ControlNet=False,
                 *,
                 lambda_cloud,
                 lambda_object,
                 ):
        super().__init__()

        self.FgGen_Diffusion_model_savepath = FgGen_Diffusion_model_savepath
        self.FgGen_ControlNet_model_savepath = FgGen_ControlNet_model_savepath
        self.ImgSyn_Diffusion_model_savepath = ImgSyn_Diffusion_model_savepath
        self.ImgSyn_ControlNet_model_savepath = ImgSyn_ControlNet_model_savepath
        self.FgSeg_UNet_model_savepath = FgSeg_UNet_model_savepath

        self.beta_t = beta_t
        self.lambda_cloud = lambda_cloud
        self.lambda_object = lambda_object

        self.is_FgGenInpaint_Resample = is_FgGenInpaint_Resample
        self.is_ImgSynInpaint_Resample = is_ImgSynInpaint_Resample
        self.is_with_ControlNet = is_with_ControlNet

        self.sampler_mode = sampler_mode
        set_random_seed(RANDOM_SEED, deterministic=False)

        # Load Latent_model
        if not os.path.exists(Latent_model_savepath):
            raise Exception('VAE_MODEL is not available.')
        Latent_model, latent_scaling_factor, _ = load_standard_vae(save_path=Latent_model_savepath, device=DEVICE, load_ema=True)
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
        FgSeg_UNet_exist = os.path.exists(self.FgSeg_UNet_model_savepath)
        if not FgSeg_UNet_exist:
            raise Exception('FGSEG_UNET_MODEL does not exist')
        else:
            FgSeg_UNet_model = FgSeg_UNet(in_channels=INPUT_CHANNELS, out_channels=UNET_OUTPUT_CHANNELS).to(DEVICE)
            FgSeg_UNet_model, _ = load_model_for_eval(
                self.FgSeg_UNet_model_savepath,
                FgSeg_UNet_model,
                map_location=DEVICE,
            )
            FgSeg_UNet_model = FgSeg_UNet_model.eval()
            for FgSeg_UNet_param in FgSeg_UNet_model.parameters():
                FgSeg_UNet_param.requires_grad = False

        self.FgSeg_model = FgSeg_UNet_model

        # Load FgGen_Diffusion_model
        FgGen_Diffusion_exist = os.path.exists(self.FgGen_Diffusion_model_savepath)
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
            FgGen_Diffusion_model, _ = load_model_for_eval(
                self.FgGen_Diffusion_model_savepath,
                FgGen_Diffusion_model,
                map_location=DEVICE,
            )
            FgGen_Diffusion_model = FgGen_Diffusion_model.eval()
            for FgGen_Diffusion_param in FgGen_Diffusion_model.parameters():
                FgGen_Diffusion_param.requires_grad = False

        self.FgGen_Diffusion_model = FgGen_Diffusion_model

        # Load ImgSyn_Diffusion_model
        ImgSyn_Diffusion_exist = os.path.exists(self.ImgSyn_Diffusion_model_savepath)
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
            ImgSyn_Diffusion_model, _ = load_model_for_eval(
                self.ImgSyn_Diffusion_model_savepath,
                ImgSyn_Diffusion_model,
                map_location=DEVICE,
            )
            ImgSyn_Diffusion_model = ImgSyn_Diffusion_model.eval()
            for ImgSyn_Diffusion_param in ImgSyn_Diffusion_model.parameters():
                ImgSyn_Diffusion_param.requires_grad = False        

        self.ImgSyn_Diffusion_model = ImgSyn_Diffusion_model

        self.FgGen_ControlNet_model = None
        self.ImgSyn_ControlNet_model = None

        if self.is_with_ControlNet:
            # Load FgGen_ControlNet_model only for conditional generation.
            FgGen_ControlNet_exist = os.path.exists(self.FgGen_ControlNet_model_savepath)
            if not FgGen_ControlNet_exist:
                raise Exception("FGGEN_CONTROLNET_MODEL does not exist.")
            FgGen_ControlNet_model = ControlNet_on_FgGen_Diffusion(FgGen_Diffusion_model_savepath=self.FgGen_Diffusion_model_savepath,
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
                self.FgGen_ControlNet_model_savepath,
                FgGen_ControlNet_model,
                map_location=DEVICE,
            )
            FgGen_ControlNet_model = FgGen_ControlNet_model.eval()
            for FgGen_ControlNet_param in FgGen_ControlNet_model.parameters():
                FgGen_ControlNet_param.requires_grad = False        

            self.FgGen_ControlNet_model = FgGen_ControlNet_model

            # Load ImgSyn_ControlNet_model only for conditional generation.
            ImgSyn_ControlNet_exist = os.path.exists(self.ImgSyn_ControlNet_model_savepath)
            if not ImgSyn_ControlNet_exist:
                raise Exception('IMGSYN_CONTROLNET_MODEL does not exist')
            ImgSyn_ControlNet_model = ControlNet_on_ImgSyn_Diffusion(ImgSyn_Diffusion_model_savepath=self.ImgSyn_Diffusion_model_savepath,
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
                self.ImgSyn_ControlNet_model_savepath,
                ImgSyn_ControlNet_model,
                map_location=DEVICE,
            )
            ImgSyn_ControlNet_model = ImgSyn_ControlNet_model.eval()
            for ImgSyn_ControlNet_param in ImgSyn_ControlNet_model.parameters():
                ImgSyn_ControlNet_param.requires_grad = False

            self.ImgSyn_ControlNet_model = ImgSyn_ControlNet_model


    @torch.no_grad()
    def _Filterout_Smaller_Values(self, fg_imgs, fg_imgs_e, min_threshold, is_dilate=False):

        fg_imgs_channel_mean = fg_imgs.mean(dim=1, keepdim=False)
        threshold_msks = fg_imgs_channel_mean > min_threshold
        threshold_msks = threshold_msks.unsqueeze(1)
        threshold_msks = torch.where(threshold_msks, 1.00, 0.00).to(DEVICE)

        if is_dilate and torch.any(threshold_msks == 1.0):
            threshold_msks = tensor_dilate(threshold_msks) 
        threshold_msks_e = F.interpolate(threshold_msks, (64, 64), mode='bilinear').to(DEVICE)
        
        zero_maps = torch.zeros_like(fg_imgs, dtype=torch.float, device=DEVICE)
        zero_maps_e = self.Latent_model.encode(zero_maps)
        zero_maps_e, _, _ = self.Latent_model.reparameterize(zero_maps_e)

        fg_imgs_e = fg_imgs_e * threshold_msks_e + zero_maps_e * (1.0 - threshold_msks_e)
        fg_imgs = self.Latent_model.decode(fg_imgs_e)

        return fg_imgs_e, fg_imgs 


    @torch.no_grad()
    def _Open_Operation(self, fg_imgs_e, fg_msks, kernel_size, is_dilate=False):
        
        fg_msks = fg_msks.squeeze(1).to('cpu').numpy()                                  # [6, 256, 256]  
        fg_msks = np.transpose(fg_msks, (1, 2, 0))                                      # [256, 256, 6] 
        fg_msks = cv2.morphologyEx(fg_msks, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))

        fg_msks = np.transpose(fg_msks, (2, 0, 1))                                      # [6, 64, 64] 
        fg_msks = torch.tensor(fg_msks).unsqueeze(1).to(fg_imgs_e.device)               # [6, 1, 64, 64]     

        if is_dilate and torch.any(fg_msks == 1.0):
            fg_msks = tensor_dilate(fg_msks) 
        fg_msks_e = F.interpolate(fg_msks, (64, 64), mode='bilinear')

        # Modify fg_imgs_e
        zero_maps = torch.zeros((fg_imgs_e.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float, device=fg_imgs_e.device)
        zero_maps_e = self.Latent_model.encode(zero_maps)
        zero_maps_e, _, _ = self.Latent_model.reparameterize(zero_maps_e)

        fg_imgs_e = fg_imgs_e * fg_msks_e + zero_maps_e * (1.0 - fg_msks_e)
        fg_imgs = self.Latent_model.decode(fg_imgs_e)

        return fg_imgs, fg_msks_e, fg_msks


    @torch.no_grad()
    def FgGen(self, prompt_str, FgGen_conditional_element=None, is_openoperation=False, is_setminthreshold=False, is_dilate=False, **kwargs):

        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        else:
            kernel_size = 3
        if 'min_threshold' in kwargs:
            min_threshold = kwargs['min_threshold']
        else:
            min_threshold = -0.6          

        fg_z = torch.randn((len(prompt_str), LATENT_HIDDENCHANNEL, IMG_SIZE//SIDELENGTH_SCALE_FACTOR, IMG_SIZE//SIDELENGTH_SCALE_FACTOR)).to(DEVICE) 

        if FgGen_conditional_element is None:
            # Diffusion generation
            if self.sampler_mode == 'ddpm':
                FgGen_sampler = DDPMSampler(self.FgGen_Diffusion_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=False).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                FgGen_sampler = DDIMSampler(self.FgGen_Diffusion_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=False).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")
            
            fg_imgs_e = FgGen_sampler(fg_z, prompt_str, is_record_process=False)
            fg_imgs_e = fg_imgs_e.detach()

        elif FgGen_conditional_element is not None:
            # ControlNet generation
            if self.sampler_mode == 'ddpm':
                FgGen_sampler = DDPMSampler(self.FgGen_ControlNet_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=True).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                FgGen_sampler = DDIMSampler(self.FgGen_ControlNet_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=True).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")

            fg_imgs_e = FgGen_sampler(fg_z, prompt_str, is_record_process=False, conditional_element=FgGen_conditional_element)
            fg_imgs_e = fg_imgs_e.detach()

        fg_imgs = self.Latent_model.decode(fg_imgs_e)

        if is_setminthreshold:
            fg_imgs_e, fg_imgs = self._Filterout_Smaller_Values(fg_imgs, fg_imgs_e, min_threshold, is_dilate)

        fg_msks = self.FgSeg_model(fg_imgs)
        fg_msks = torch.sigmoid(fg_msks)
        fg_msks = (fg_msks >= 0.5).float()     # [6, 1, 256, 256]  

        # Post-processing: open operation to remove small fragments, applied to both img and msk
        if is_openoperation:
            fg_imgs, fg_msks_e, fg_msks = self._Open_Operation(fg_imgs_e, fg_msks, kernel_size, is_dilate)

        # No post-processing
        else:
            if is_dilate and torch.any(fg_msks == 1.0):
                fg_msks = tensor_dilate(fg_msks) 
            fg_msks_e = F.interpolate(fg_msks, (64, 64), mode='bilinear')
        
        return fg_imgs_e, fg_imgs, fg_msks_e, fg_msks


    def _build_imgsyn_cfg_sampler(self, conditional_element):
        if self.sampler_mode == 'ddpm':
            sampler_class = DDPMSampler_CFG
        elif self.sampler_mode == 'ddim':
            sampler_class = DDIMSampler_CFG
        else:
            raise ValueError("sampler_mode must be either 'ddpm' or 'ddim'.")
        control_model = self.ImgSyn_ControlNet_model if conditional_element is not None else None
        return sampler_class(
            self.ImgSyn_Diffusion_model,
            control_model,
            self.beta_t,
            is_Inference=False,
            lambda_cloud=self.lambda_cloud,
            lambda_object=self.lambda_object,
        ).to(DEVICE)

    @torch.no_grad()
    def ImgSyn_CFG(self, prompt_str_dict, fg_imgs_e_dict, fg_msks_Cloud_e, ImgSyn_conditional_element=None):

        syn_z = torch.randn((len(prompt_str_dict['prompt_str_Obj']), LATENT_HIDDENCHANNEL, IMG_SIZE//SIDELENGTH_SCALE_FACTOR, IMG_SIZE//SIDELENGTH_SCALE_FACTOR)).to(DEVICE)

        ImgSyn_sampler = self._build_imgsyn_cfg_sampler(ImgSyn_conditional_element)
        syn_imgs_e = ImgSyn_sampler(
            syn_z, prompt_str_dict, fg_imgs_e_dict, fg_msks_Cloud_e,
            is_record_process=False,
            conditional_element=ImgSyn_conditional_element,
        ).detach()

        syn_imgs = self.Latent_model.decode(syn_imgs_e)

        return syn_imgs_e, syn_imgs


    @torch.no_grad()
    def InPaint_inFgGen(self, fg_0_imgs_e, fg_t_imgs_e, msks_e, prompt_str, FgGen_conditional_element=None, is_openoperation=False, is_setminthreshold=False, is_dilate=False, **kwargs):

        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        else:
            kernel_size = 3
        if 'min_threshold' in kwargs:
            min_threshold = kwargs['min_threshold']
        else:
            min_threshold = -0.6

        zero_maps = torch.zeros((msks_e.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(DEVICE)
        zero_maps_e = self.Latent_model.encode(zero_maps)
        zero_maps_e, _, _ = self.Latent_model.reparameterize(zero_maps_e)

        if FgGen_conditional_element is None:
            if self.sampler_mode == 'ddpm':
                FgGen_sampler = DDPMSampler(self.FgGen_Diffusion_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=False).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                FgGen_sampler = DDIMSampler(self.FgGen_Diffusion_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=False).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")
            
            inpaint_fg_imgs_e = FgGen_sampler.inpaint(fg_0_imgs_e, fg_t_imgs_e, msks_e, prompt_str, is_record_process=False, is_resample=self.is_FgGenInpaint_Resample)
            inpaint_fg_imgs_e = inpaint_fg_imgs_e.detach()

        elif FgGen_conditional_element is not None:
            if self.sampler_mode == 'ddpm':
                FgGen_sampler = DDPMSampler(self.FgGen_ControlNet_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=True).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                FgGen_sampler = DDIMSampler(self.FgGen_ControlNet_model, self.beta_t, is_ImgSyn=False, is_with_ControlNet=True).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")

            inpaint_fg_imgs_e = FgGen_sampler.inpaint(fg_0_imgs_e, fg_t_imgs_e, msks_e, prompt_str, is_record_process=False, is_resample=self.is_FgGenInpaint_Resample, conditional_element=FgGen_conditional_element)
            inpaint_fg_imgs_e = inpaint_fg_imgs_e.detach()

        inpaint_fg_imgs = self.Latent_model.decode(inpaint_fg_imgs_e)

        if is_setminthreshold:
            inpaint_fg_imgs_e, inpaint_fg_imgs = self._Filterout_Smaller_Values(inpaint_fg_imgs, inpaint_fg_imgs_e, min_threshold)

        inpaint_fg_msks = self.FgSeg_model(inpaint_fg_imgs)
        inpaint_fg_msks = torch.sigmoid(inpaint_fg_msks)
        inpaint_fg_msks = (inpaint_fg_msks >= 0.5).float()     # [6, 1, 256, 256]   

        if is_openoperation:
            inpaint_fg_imgs, inpaint_fg_msks_e, inpaint_fg_msks = self._Open_Operation(inpaint_fg_imgs_e, inpaint_fg_msks, kernel_size)
        
        if is_dilate and torch.any(inpaint_fg_msks == 1.0):
            inpaint_fg_msks = tensor_dilate(inpaint_fg_msks) 
        inpaint_fg_msks_e = F.interpolate(inpaint_fg_msks, (64, 64), mode='bilinear')

        return inpaint_fg_imgs_e, inpaint_fg_imgs, inpaint_fg_msks_e, inpaint_fg_msks


    @torch.no_grad()
    def InPaint_inImgSyn(self, syn_0_imgs_e, syn_t_imgs_e, msks_e, prompt_str, ImgSyn_conditional_element=None):

        zero_maps = torch.zeros((msks_e.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(DEVICE)
        zero_maps_e = self.Latent_model.encode(zero_maps)
        zero_maps_e, _, _ = self.Latent_model.reparameterize(zero_maps_e)

        if ImgSyn_conditional_element is None:
            if self.sampler_mode == 'ddpm':
                ImgSyn_sampler = DDPMSampler(self.ImgSyn_Diffusion_model, self.beta_t, is_ImgSyn=True, is_with_ControlNet=False).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                ImgSyn_sampler = DDIMSampler(self.ImgSyn_Diffusion_model, self.beta_t, is_ImgSyn=True, is_with_ControlNet=False).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")

            inpaint_syn_imgs_e = ImgSyn_sampler.inpaint(syn_0_imgs_e, syn_t_imgs_e, msks_e, prompt_str, is_record_process=False, is_resample=self.is_ImgSynInpaint_Resample, fg_imgs_e=zero_maps_e)
            inpaint_syn_imgs_e = inpaint_syn_imgs_e.detach()
        else:
            if self.sampler_mode == 'ddpm':
                ImgSyn_sampler = DDPMSampler(self.ImgSyn_ControlNet_model, self.beta_t, is_ImgSyn=True, is_with_ControlNet=True).to(DEVICE)
            elif self.sampler_mode == 'ddim':
                ImgSyn_sampler = DDIMSampler(self.ImgSyn_ControlNet_model, self.beta_t, is_ImgSyn=True, is_with_ControlNet=True).to(DEVICE)
            else:
                raise Exception("sampler_mode must be either 'ddpm' or 'ddim'.")

            inpaint_syn_imgs_e = ImgSyn_sampler.inpaint(syn_0_imgs_e, syn_t_imgs_e, msks_e, prompt_str, is_record_process=False, is_resample=self.is_ImgSynInpaint_Resample, fg_imgs_e=zero_maps_e, conditional_element=ImgSyn_conditional_element)
            inpaint_syn_imgs_e = inpaint_syn_imgs_e.detach()

        inpaint_syn_imgs = self.Latent_model.decode(inpaint_syn_imgs_e)

        return inpaint_syn_imgs_e, inpaint_syn_imgs
    

    @torch.no_grad()
    def InPaint_inImgSyn_CFG(self, syn_0_imgs_e, syn_t_imgs_e, msks_e, prompt_str_dict, fg_imgs_e_dict, fg_msks_Cloud_e, ImgSyn_conditional_element=None):

        zero_maps = torch.zeros((msks_e.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(DEVICE)
        zero_maps_e = self.Latent_model.encode(zero_maps)
        zero_maps_e, _, _ = self.Latent_model.reparameterize(zero_maps_e)

        ImgSyn_CFG_sampler = self._build_imgsyn_cfg_sampler(ImgSyn_conditional_element)
        inpaint_syn_imgs_e = ImgSyn_CFG_sampler.inpaint(
            syn_0_imgs_e, syn_t_imgs_e, msks_e, prompt_str_dict, fg_imgs_e_dict, fg_msks_Cloud_e,
            is_record_process=False,
            is_resample=self.is_ImgSynInpaint_Resample,
            conditional_element=ImgSyn_conditional_element,
        ).detach()

        inpaint_syn_imgs = self.Latent_model.decode(inpaint_syn_imgs_e)

        return inpaint_syn_imgs_e, inpaint_syn_imgs
