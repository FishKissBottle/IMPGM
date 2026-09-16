from torch import nn
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
from VAE.VAE_Code.VAE_model import AutoEncoder
from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from IMPGM_Utils import build_diagonal_tile_schedule, extract_high_frequency, load_model, save_rgb_datas, save_tif_datas, save_msk_datas, tensor_dilate, prepare_rgb_vis_tensor
from IMPGM_Scheduler import build_beta_schedule_from_config
from torchvision.transforms.functional import gaussian_blur
from Multi_Condition_Generation.IMPGM_CFG import CFG
from PIL import Image
from IMPGM_TifReader import Tif_Read_and_Write


class Generate_LargeImg_CFG(nn.Module):
    """Generate large tiled images with the base classifier-free guidance workflow."""
    def __init__(self, 
                 sampler_mode, 
                 beta_t,
                 overlap_rate,
                 overlap_buffer_thx=4*6,
                 is_with_ControlNet=False,
                 is_FgGenInpaint_Resample=True,
                 is_ImgSynInpaint_Resample=True,
                 batch_size=8,
                 ObjFgGen_RGB_savepath=None,
                 ObjFgGen_TIF_savepath=None,
                 CloudFgGen_RGB_savepath=None,
                 CloudFgGen_TIF_savepath=None,
                 ImgSyn_RGB_savepath=None,
                 ImgSyn_TIF_savepath=None,
                 ObjMsk_savepath=None,
                 CloudMsk_savepath=None,
                 *,
                 lambda_cloud,
                 lambda_object,
                 is_Fixed_ObjPrompt=False,
                 is_Fixed_CloudPrompt=False,
                 is_ObjFgGen_Use_OpenOperation=False,
                 ObjOpenOperation_Kernel_Size=3,
                 is_CloudFgGen_Use_OpenOperation=False,
                 CloudOpenOperation_Kernel_Size=3,
                 is_CloudFgGen_Set_MinThreshold=False,
                 CloudFgGen_MinThreshold=-0.6,
                 conditional_element_proj=None,
                 conditional_element_geotrans=None,
                 ):
        super().__init__()

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
            ObjFgGen_RGB_savepath,
            ObjFgGen_TIF_savepath,
            CloudFgGen_RGB_savepath,
            CloudFgGen_TIF_savepath,
            ImgSyn_RGB_savepath,
            ImgSyn_TIF_savepath,
            ObjMsk_savepath,
            CloudMsk_savepath,
        ):
            if savepath is not None:
                os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)


        self.T = len(beta_t)
        self.is_with_ControlNet = is_with_ControlNet

        self.is_ObjFgGen_Use_OpenOperation = is_ObjFgGen_Use_OpenOperation
        self.ObjOpenOperation_Kernel_Size = ObjOpenOperation_Kernel_Size  
        self.is_CloudFgGen_Use_OpenOperation = is_CloudFgGen_Use_OpenOperation
        self.CloudOpenOperation_Kernel_Size = CloudOpenOperation_Kernel_Size
        self.is_CloudFgGen_Set_MinThreshold = is_CloudFgGen_Set_MinThreshold
        self.CloudFgGen_MinThreshold = CloudFgGen_MinThreshold

        CFG_module = CFG(sampler_mode,
                         beta_t,
                         Latent_model_savepath=VAE_MODEL_SAVEPATH,
                         FgGen_Diffusion_model_savepath=FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
                         FgGen_ControlNet_model_savepath=FGGEN_CONTROLNET_CONFIG.MODEL_SAVEPATH,
                         ImgSyn_Diffusion_model_savepath=IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
                         ImgSyn_ControlNet_model_savepath=IMGSYN_CONTROLNET_CONFIG.MODEL_SAVEPATH,
                         FgSeg_UNet_model_savepath=FGSEG_UNET_MODEL_SAVEPATH,
                         is_FgGenInpaint_Resample=is_FgGenInpaint_Resample,
                         is_ImgSynInpaint_Resample=is_ImgSynInpaint_Resample,
                         is_with_ControlNet=is_with_ControlNet,
                         lambda_cloud=lambda_cloud,
                         lambda_object=lambda_object,
                         )
        
        self.CFG_module  = CFG_module

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

        self.is_Fixed_ObjPrompt = is_Fixed_ObjPrompt
        self.is_Fixed_CloudPrompt = is_Fixed_CloudPrompt

        self.ObjFgGen_RGB_savepath = ObjFgGen_RGB_savepath
        self.ObjFgGen_TIF_savepath = ObjFgGen_TIF_savepath
        self.CloudFgGen_RGB_savepath = CloudFgGen_RGB_savepath
        self.CloudFgGen_TIF_savepath = CloudFgGen_TIF_savepath
        self.ImgSyn_RGB_savepath = ImgSyn_RGB_savepath
        self.ImgSyn_TIF_savepath = ImgSyn_TIF_savepath
        self.ObjMsk_savepath = ObjMsk_savepath
        self.CloudMsk_savepath = CloudMsk_savepath

        self.conditional_element_proj = conditional_element_proj
        self.conditional_element_geotrans = conditional_element_geotrans

    @torch.no_grad()
    def _get_diagonals_points(self, rows, cols, multiplier):
        return build_diagonal_tile_schedule(rows, cols, multiplier)


    @torch.no_grad()
    def _get_obj_prompt(self, cur_prompt_str, obj_prompt_str, noobj_prompt_str, LargeImg_e, row_idx, col_idx, detect_pixel_thx, 
                        NoObj_Change_to_Obj_probability=0.70, is_fixed_prompt=False):

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
                ImgSlice_upper = self.CFG_module.Latent_model.decode(ImgSlice_e_upper)     # [1, 4, 256, 256]
                MskSlice_upper = self.CFG_module.FgSeg_model(ImgSlice_upper)
                MskSlice_upper = F.sigmoid(MskSlice_upper)
                MskSlice_upper = (MskSlice_upper >= 0.5).float()
                
            if ImgSlice_e_left is not None: 
                ImgSlice_left = self.CFG_module.Latent_model.decode(ImgSlice_e_left)                                         # [1, 4, 256, 256]
                MskSlice_left = self.CFG_module.FgSeg_model(ImgSlice_left)
                MskSlice_left = F.sigmoid(MskSlice_left)
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
    def _get_cloud_prompt(self, cur_prompt_str, cloud_prompt_str_list, nocloud_prompt_str, LargeImg_e, row_idx, col_idx, detect_pixel_thx, 
                           NoCloud_Change_to_Cloud_probability=0.70, CloudCoverLevel_change_probability_list=[0.40, 0.40, 0.10, 0.10], is_fixed_prompt=False):

        if is_fixed_prompt:
            return cur_prompt_str

        # determine the order by probability
        if cur_prompt_str == nocloud_prompt_str:
            probrank_prompt_str_list = [prompt_str for prompt_str in cloud_prompt_str_list[:len(CloudCoverLevel_change_probability_list)]]
        else:
            cur_prompt_str_idx = cloud_prompt_str_list.index(cur_prompt_str)
            if cur_prompt_str_idx == 0:
                probrank_prompt_str_list = cloud_prompt_str_list
            elif cur_prompt_str_idx == len(cloud_prompt_str_list) - 1:
                probrank_prompt_str_list = cloud_prompt_str_list[::-1]
            else:
                probrank_prompt_str_list = [cloud_prompt_str_list[cur_prompt_str_idx - 1], cloud_prompt_str_list[cur_prompt_str_idx + 1]]
                shuffled_cloud_prompt_str_list = random.sample(cloud_prompt_str_list, len(cloud_prompt_str_list))      # shuffle the list
                for prompt_str in shuffled_cloud_prompt_str_list:
                    if prompt_str not in probrank_prompt_str_list:
                        probrank_prompt_str_list.append(prompt_str)


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
                ImgSlice_upper = self.CFG_module.Latent_model.decode(ImgSlice_e_upper)     # [1, 4, 256, 256]
                MskSlice_upper = self.CFG_module.FgSeg_model(ImgSlice_upper)
                MskSlice_upper = F.sigmoid(MskSlice_upper)
                MskSlice_upper = (MskSlice_upper >= 0.5).float() 
                
            if ImgSlice_e_left is not None: 
                ImgSlice_left = self.CFG_module.Latent_model.decode(ImgSlice_e_left)                                         # [1, 4, 256, 256]
                MskSlice_left = self.CFG_module.FgSeg_model(ImgSlice_left)
                MskSlice_left = F.sigmoid(MskSlice_left)
                MskSlice_left = (MskSlice_left >= 0.5).float() 
                
            if row_idx == 0 and col_idx > 0:
                left_overlap_area = MskSlice_left[:, :, :, self.ImgSlice_w - detect_pixel_thx:]
                if torch.sum(left_overlap_area) >= 1:
                    # cloud detected in the overlap area, pick from cloud_prompt_str_list
                    random_num1 = random.random()
                    prob_sum = 0.0
                    for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                        prob_sum += prob_val
                        if random_num1 <= prob_sum:
                            chosen_prompt = probrank_prompt_str_list[prob_idx]
                            break
                else:
                    random_num2 = random.random()
                    if random_num2 > NoCloud_Change_to_Cloud_probability:
                        chosen_prompt = None
                        random_num3 = random.random()
                        prob_sum = 0.0
                        for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                            prob_sum += prob_val
                            if random_num3 <= prob_sum:
                                chosen_prompt = probrank_prompt_str_list[prob_idx]
                                break                        
                    else:
                        chosen_prompt = nocloud_prompt_str

                            
            elif row_idx > 0 and col_idx == 0:
                upper_overlap_area = MskSlice_upper[:, :, self.ImgSlice_h - detect_pixel_thx:, :]
                if torch.sum(upper_overlap_area) >= 1:
                    random_num1 = random.random()
                    prob_sum = 0.0
                    for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                        prob_sum += prob_val
                        if random_num1 <= prob_sum:
                            chosen_prompt = probrank_prompt_str_list[prob_idx]
                            break

                else:
                    random_num2 = random.random()
                    if random_num2 > NoCloud_Change_to_Cloud_probability:
                        chosen_prompt = None
                        random_num3 = random.random()
                        prob_sum = 0.0
                        for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                            prob_sum += prob_val
                            if random_num3 <= prob_sum:
                                chosen_prompt = probrank_prompt_str_list[prob_idx]
                                break  
                    else:
                        chosen_prompt = nocloud_prompt_str

            else:
                upper_overlap_area = MskSlice_upper[:, :, self.ImgSlice_h - detect_pixel_thx:, :]
                left_overlap_area  = MskSlice_left[:, :, :, self.ImgSlice_w - detect_pixel_thx:]
                if torch.sum(upper_overlap_area) >= 1 or torch.sum(left_overlap_area) >= 1:
                    random_num1 = random.random()
                    prob_sum = 0.0
                    for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                        prob_sum += prob_val
                        if random_num1 <= prob_sum:
                            chosen_prompt = probrank_prompt_str_list[prob_idx]
                            break
                else:
                    random_num2 = random.random()
                    if random_num2 > NoCloud_Change_to_Cloud_probability:
                        chosen_prompt = None
                        random_num3 = random.random()
                        prob_sum = 0.0
                        for prob_idx, prob_val in enumerate(CloudCoverLevel_change_probability_list):
                            prob_sum += prob_val
                            if random_num3 <= prob_sum:
                                chosen_prompt = probrank_prompt_str_list[prob_idx]
                                break 
                    else:
                        chosen_prompt = nocloud_prompt_str

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
        Vertical_GradStrip_0to1 = GradStripSeq_0to1.expand(ImgSlice_h, overlap_thx).unsqueeze(0).unsqueeze(0).clone()    # 0 -> 1 gradient
        Vertical_GradStrip_1to0 = 1.0 - Vertical_GradStrip_0to1                                                          # 1 -> 0 gradient

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

            # the first one
            VGS0to1T_Vgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS0to1T_Vgrad = VGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS0to1T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS0to1T_Hgrad = VGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Vertical_GradStrip_0to1_Tail *= VGS0to1T_Vgrad
            Vertical_GradStrip_0to1_Tail[:, :, :, (overlap_thx - ones_thx):] = 1.0
            Vertical_GradStrip_0to1_Tail *= VGS0to1T_Hgrad

            # the second one
            VGS1to0T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            VGS1to0T_Vgrad = VGS1to0T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            VGS1to0T_Hgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            VGS1to0T_Hgrad = VGS1to0T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Vertical_GradStrip_1to0_Tail *= (VGS1to0T_Vgrad * VGS1to0T_Hgrad)
            Vertical_GradStrip_1to0_Tail[:, :, :, (overlap_thx - ones_thx):] = 0.0

            # the third one
            HGS0to1T_Vgrad = torch.linspace(0.00, 1.00, overlap_thx, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            HGS0to1T_Vgrad = HGS0to1T_Vgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()
            HGS0to1T_Hgrad = torch.linspace(1.00, 0.00, overlap_thx, dtype=torch.float32).to(DEVICE)
            HGS0to1T_Hgrad = HGS0to1T_Hgrad.expand(overlap_thx, overlap_thx).unsqueeze(0).unsqueeze(0).clone()

            Horizontal_GradStrip_0to1_Tail *= HGS0to1T_Vgrad
            Horizontal_GradStrip_0to1_Tail[:, :, (overlap_thx - ones_thx):, :] = 1.0
            Horizontal_GradStrip_0to1_Tail *= HGS0to1T_Hgrad

            # the fourth one
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
                Vertical_GradMap_0to1[:, :, ImgSlice_w - overlap_thx:, :overlap_thx] = Vertical_GradStrip_0to1_Tail
                Vertical_GradMap_1to0[:, :, ImgSlice_w - overlap_thx:, :overlap_thx] = Vertical_GradStrip_1to0_Tail
                LeftTop_GradMap_0to1[:, :, ImgSlice_w - overlap_thx:, :overlap_thx] = Vertical_GradStrip_0to1_Tail
                LeftTop_GradMap_1to0[:, :, ImgSlice_w - overlap_thx:, :overlap_thx] = Vertical_GradStrip_1to0_Tail

        return Horizontal_GradMap_0to1, Horizontal_GradMap_1to0, Vertical_GradMap_0to1, Vertical_GradMap_1to0, LeftTop_GradMap_0to1, LeftTop_GradMap_1to0


    @torch.no_grad()
    def main(self, LargeImg_h, LargeImg_w, first_obj_prompt_str, obj_prompt_str, noobj_prompt_str, first_cloud_prompt_str, 
             cloud_prompt_str_list, nocloud_prompt_str, detect_pixel_thx=32, *,
             FgGen_original_msk=None, FgGen_conditional_element=None,
             ImgSyn_conditional_element=None):

        if not isinstance(LargeImg_h, (int, np.integer)) or not isinstance(LargeImg_w, (int, np.integer)):
            raise TypeError("LargeImg_h and LargeImg_w must be integers.")
        if LargeImg_h < self.ImgSlice_h or LargeImg_w < self.ImgSlice_w:
            raise ValueError(
                f"LargeImg_h and LargeImg_w must both be at least IMG_SIZE={self.ImgSlice_h}, "
                f"got ({LargeImg_h}, {LargeImg_w})."
            )
        requested_size = (LargeImg_h, LargeImg_w)

        if self.is_with_ControlNet:
            conditional_tensors = {
                "FgGen_original_msk": FgGen_original_msk,
                "FgGen_conditional_element": FgGen_conditional_element,
                "ImgSyn_conditional_element": ImgSyn_conditional_element,
            }
            for name, tensor in conditional_tensors.items():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    raise ValueError(f"{name} must be a four-dimensional [B, C, H, W] tensor.")
                if tensor.shape[0] != 1 or tuple(tensor.shape[-2:]) != requested_size:
                    raise ValueError(
                        f"{name} must have batch size 1 and spatial size {requested_size}, "
                        f"got {tuple(tensor.shape)}."
                    )
            if FgGen_original_msk.shape[1] != 1 or FgGen_conditional_element.shape[1] != 1:
                raise ValueError("FgGen masks and conditions must each contain exactly one channel.")
            if ImgSyn_conditional_element.shape[1] != INPUT_CHANNELS:
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

        ObjFgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        CloudFgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        SynLargeImg_e = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)

        NoGrad_ObjFgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        NoGrad_CloudFgLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)
        NoGrad_SynLargeImg_e  = torch.zeros((1, LATENT_HIDDENCHANNEL, LargeImg_e_h, LargeImg_e_w), dtype=torch.float32).to(DEVICE)

        rows_num = int(LargeImg_e_h // self.interval_e) if LargeImg_e_h % self.interval_e != 0 else int(LargeImg_e_h // self.interval_e) - 1
        cols_num = int(LargeImg_e_w // self.interval_e) if LargeImg_e_w % self.interval_e != 0 else int(LargeImg_e_w // self.interval_e) - 1

        diagonals_points_list = self._get_diagonals_points(rows_num, cols_num, self.interval_e)
        row_idx_max, col_idx_max = max(diagonals_points_list)[0]


        for idx, diagonal_points in enumerate(diagonals_points_list):
            
            # this is one batch
            ObjFgSlice_e_0_list = []
            ObjFgSlice_e_t_list = []
            CloudFgSlice_e_0_list = []
            CloudFgSlice_e_t_list = []
            SynSlice_e_0_list = []
            SynSlice_e_t_list = []
            BldMsk_list = []
            ObjPrompt_list = []
            CloudPrompt_list = []

            if self.is_with_ControlNet:
                ObjFgSlice_orimsk_list = []
                ObjFgSlice_condele_list = []
                SynSlice_condele_list = []  

            for row_idx, col_idx in diagonal_points:

                BldGradMap_e_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                                    row_idx_max, col_idx_max,
                                                                    self.overlap_thx_e, self.ImgSlice_e_h, self.ImgSlice_e_w, 
                                                                    zeros_thx=self.overlap_thx_e - self.overlap_buffer_thx_e, grad_thx=self.overlap_buffer_thx_e, ones_thx=0,
                                                                    is_tailgrad=False)
                H_BldGradMap_0to1, H_BldGradMap_1to0, V_BldGradMap_0to1, V_BldGradMap_1to0, LT_BldGradMap_0to1, LT_BldGradMap_1to0 = BldGradMap_e_list

                # generate slice No.0
                if row_idx == 0 and col_idx == 0:
                    ObjFgSlice_e_t  = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    CloudFgSlice_e_t  = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    if not self.is_with_ControlNet:
                        chosen_obj_prompt = self._get_obj_prompt(first_obj_prompt_str, obj_prompt_str, noobj_prompt_str, ObjFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_ObjPrompt)
                    else:
                        chosen_obj_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)
                    
                    chosen_cloud_prompt = self._get_cloud_prompt(first_cloud_prompt_str, cloud_prompt_str_list, nocloud_prompt_str, CloudFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_CloudPrompt)

                    ObjFgSlice_e_t_list.append(ObjFgSlice_e_t)
                    CloudFgSlice_e_t_list.append(CloudFgSlice_e_t)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    ObjPrompt_list.append(chosen_obj_prompt)
                    CloudPrompt_list.append(chosen_cloud_prompt)

                # top edge
                elif row_idx == 0 and col_idx > 0:
                    
                    # compute Slice_e_0
                    ObjFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    ObjFgSlice_e_0[:, :, :, :self.overlap_thx_e] = ObjFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    CloudFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    CloudFgSlice_e_0[:, :, :, :self.overlap_thx_e] = CloudFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :, :self.overlap_thx_e] = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    
                    # compute Slice_e_t
                    ObjFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    CloudFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # compute BldMsk
                    BldMsk = V_BldGradMap_0to1
                    if not self.is_with_ControlNet:
                        chosen_obj_prompt = self._get_obj_prompt(chosen_obj_prompt, obj_prompt_str, noobj_prompt_str, ObjFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_ObjPrompt)
                    else:
                        chosen_obj_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)

                    chosen_cloud_prompt = self._get_cloud_prompt(chosen_cloud_prompt, cloud_prompt_str_list, nocloud_prompt_str, CloudFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_CloudPrompt)
                    
                    ObjFgSlice_e_0_list.append(ObjFgSlice_e_0)
                    ObjFgSlice_e_t_list.append(ObjFgSlice_e_t)
                    CloudFgSlice_e_0_list.append(CloudFgSlice_e_0)
                    CloudFgSlice_e_t_list.append(CloudFgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    ObjPrompt_list.append(chosen_obj_prompt)
                    CloudPrompt_list.append(chosen_cloud_prompt)

                # left edge
                elif row_idx > 0 and col_idx == 0:

                    # compute Slice_e_0
                    ObjFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    ObjFgSlice_e_0[:, :, :self.overlap_thx_e, :] = ObjFgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    CloudFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    CloudFgSlice_e_0[:, :, :self.overlap_thx_e, :] = CloudFgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :self.overlap_thx_e, :] = SynLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    
                    # compute Slice_e_t
                    ObjFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    CloudFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # compute msk
                    BldMsk = H_BldGradMap_0to1

                    if not self.is_with_ControlNet:
                        chosen_obj_prompt = self._get_obj_prompt(chosen_obj_prompt, obj_prompt_str, noobj_prompt_str, ObjFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_ObjPrompt)
                    else:
                        chosen_obj_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)
                    
                    chosen_cloud_prompt = self._get_cloud_prompt(chosen_cloud_prompt, cloud_prompt_str_list, nocloud_prompt_str, CloudFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_CloudPrompt)
                    
                    ObjFgSlice_e_0_list.append(ObjFgSlice_e_0)
                    ObjFgSlice_e_t_list.append(ObjFgSlice_e_t)
                    CloudFgSlice_e_0_list.append(CloudFgSlice_e_0)
                    CloudFgSlice_e_t_list.append(CloudFgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    ObjPrompt_list.append(chosen_obj_prompt)
                    CloudPrompt_list.append(chosen_cloud_prompt)

                else:

                    # compute Slice_e_0
                    ObjFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    ObjFgSlice_e_0[:, :, :self.overlap_thx_e, :] = ObjFgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    ObjFgSlice_e_0[:, :, :, :self.overlap_thx_e] = ObjFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    CloudFgSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    CloudFgSlice_e_0[:, :, :self.overlap_thx_e, :] = CloudFgLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    CloudFgSlice_e_0[:, :, :, :self.overlap_thx_e] = CloudFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                    SynSlice_e_0 = torch.zeros((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w), dtype=torch.float32).to(DEVICE)
                    SynSlice_e_0[:, :, :self.overlap_thx_e, :] = SynLargeImg_e[:, :, row_idx: row_idx + self.overlap_thx_e, col_idx: col_idx + self.ImgSlice_e_w]
                    SynSlice_e_0[:, :, :, :self.overlap_thx_e] = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.overlap_thx_e]
                  
                    # compute Slice_e_t
                    ObjFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    CloudFgSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    SynSlice_e_t = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)
                    
                    # compute msk
                    BldMsk = LT_BldGradMap_0to1

                    if not self.is_with_ControlNet:
                        chosen_obj_prompt = self._get_obj_prompt(chosen_obj_prompt, obj_prompt_str, noobj_prompt_str, ObjFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_ObjPrompt)
                    else:
                        chosen_obj_prompt = self._get_obj_prompt_forControlNet(obj_prompt_str, noobj_prompt_str, FgGen_conditional_element, row_idx, col_idx)
                    
                    chosen_cloud_prompt = self._get_cloud_prompt(chosen_cloud_prompt, cloud_prompt_str_list, nocloud_prompt_str, CloudFgLargeImg_e, row_idx, col_idx, detect_pixel_thx, is_fixed_prompt=self.is_Fixed_CloudPrompt)
             
                    ObjFgSlice_e_0_list.append(ObjFgSlice_e_0)
                    ObjFgSlice_e_t_list.append(ObjFgSlice_e_t)
                    CloudFgSlice_e_0_list.append(CloudFgSlice_e_0)
                    CloudFgSlice_e_t_list.append(CloudFgSlice_e_t)
                    SynSlice_e_0_list.append(SynSlice_e_0)
                    SynSlice_e_t_list.append(SynSlice_e_t)
                    BldMsk_list.append(BldMsk)
                    ObjPrompt_list.append(chosen_obj_prompt)
                    CloudPrompt_list.append(chosen_cloud_prompt)

                # FgGen uses the foreground mask; ImgSyn uses high-frequency features after object inpainting.
                if self.is_with_ControlNet:
                    ObjFgSlice_orimsk = FgGen_original_msk[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    ObjFgSlice_condele = FgGen_conditional_element[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    SynSlice_condele = ImgSyn_conditional_element[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: row_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_h, col_idx * SIDELENGTH_SCALE_FACTOR: col_idx * SIDELENGTH_SCALE_FACTOR + self.ImgSlice_w]
                    ObjFgSlice_orimsk_list.append(ObjFgSlice_orimsk)
                    ObjFgSlice_condele_list.append(ObjFgSlice_condele)
                    SynSlice_condele_list.append(SynSlice_condele)


            # start generating and write into ObjFgLargeImg_e, CloudFgLargeImg_e and SynLargeImg_e
            if row_idx == 0 and col_idx == 0: 
                print('ObjPrompt_list: ', ObjPrompt_list)
                print('CloudPrompt_list: ', CloudPrompt_list)

                if not self.is_with_ControlNet:
                    ObjFgSlice_e_set, ObjFgSlice_set, ObjMskSlice_e_set, ObjMskSlice_set = self.CFG_module.FgGen(ObjPrompt_list, FgGen_conditional_element=None,
                                                                                                                 is_openoperation=self.is_ObjFgGen_Use_OpenOperation, 
                                                                                                                 kernel_size=self.ObjOpenOperation_Kernel_Size, 
                                                                                                                 is_dilate=True)     
                    CloudFgSlice_e_set, CloudFgSlice_set_set, CloudMskSlice_e_set, CloudMskSlice_set = self.CFG_module.FgGen(CloudPrompt_list, FgGen_conditional_element=None, 
                                                                                                                             is_openoperation=self.is_CloudFgGen_Use_OpenOperation, 
                                                                                                                             kernel_size=self.CloudOpenOperation_Kernel_Size,
                                                                                                                             is_setminthreshold=self.is_CloudFgGen_Set_MinThreshold, 
                                                                                                                             min_threshold=self.CloudFgGen_MinThreshold, 
                                                                                                                             is_dilate=True) 
                
                    prompt_str_dict = {
                        'prompt_str_NoObj': ['NoObj' for _ in range(len(ObjPrompt_list))],
                        'prompt_str_Obj': ObjPrompt_list,
                        'prompt_str_NoCloud': ['NoCloud' for _ in range(len(ObjPrompt_list))],
                        'prompt_str_Cloud': CloudPrompt_list
                    }
                    zero_map = torch.zeros((ObjFgSlice_e_set.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float, device=DEVICE)
                    zero_map_e = self.CFG_module.Latent_model.encode(zero_map)
                    zero_map_e, _, _ = self.CFG_module.Latent_model.reparameterize(zero_map_e)
                    NoObjFgSlice_e_set = zero_map_e
                    NoCloudFgSlice_e_set = zero_map_e
                    fg_imgs_e_dict = {
                        'fg_imgs_NoObj_e': NoObjFgSlice_e_set,
                        'fg_imgs_Obj_e': ObjFgSlice_e_set,
                        'fg_imgs_NoCloud_e': NoCloudFgSlice_e_set,
                        'fg_imgs_Cloud_e': CloudFgSlice_e_set
                    }                  
                    SynSlice_e_set, SynSlice_set = self.CFG_module.ImgSyn_CFG(prompt_str_dict, fg_imgs_e_dict, CloudMskSlice_e_set, ImgSyn_conditional_element=None)
            
                else:
                    ObjFgSlice_e_set, ObjFgSlice_set, ObjMskSlice_e_set, ObjMskSlice_set = self.CFG_module.FgGen(ObjPrompt_list, FgGen_conditional_element=ObjFgSlice_condele_list[0],
                                                                                                                 is_openoperation=self.is_ObjFgGen_Use_OpenOperation,
                                                                                                                 kernel_size=self.ObjOpenOperation_Kernel_Size,
                                                                                                                 is_dilate=True)  
                    CloudFgSlice_e_set, CloudFgSlice_set_set, CloudMskSlice_e_set, CloudMskSlice_set = self.CFG_module.FgGen(CloudPrompt_list, FgGen_conditional_element=None, 
                                                                                                                             is_openoperation=self.is_CloudFgGen_Use_OpenOperation,
                                                                                                                             kernel_size=self.CloudOpenOperation_Kernel_Size,
                                                                                                                             is_setminthreshold=self.is_CloudFgGen_Set_MinThreshold,
                                                                                                                             min_threshold=self.CloudFgGen_MinThreshold,
                                                                                                                             is_dilate=True) 

                    # background ControlNet generation
                    SynSlice_condele_set = SynSlice_condele_list[0]
                    SynSlice_condele_e_set = self.CFG_module.Latent_model.encode(SynSlice_condele_set)
                    SynSlice_condele_e_set, _, _ = self.CFG_module.Latent_model.reparameterize(SynSlice_condele_e_set)                    

                    # first inpaint the Obj part in SynSlice_condele_set
                    FgSlice_orimsk_set = tensor_dilate(ObjFgSlice_orimsk_list[0]) 
                    FgSlice_orimsk_e_set = F.interpolate(FgSlice_orimsk_set, size=(IMG_SIZE // SIDELENGTH_SCALE_FACTOR, IMG_SIZE // SIDELENGTH_SCALE_FACTOR), mode='bilinear')
                    
                    SynSlice_e_t_set = torch.randn((1, LATENT_HIDDENCHANNEL, self.ImgSlice_e_h, self.ImgSlice_e_w)).to(DEVICE)

                    SynSlice_condele_e_set, SynSlice_condele_set = self.CFG_module.InPaint_inImgSyn(SynSlice_condele_e_set, SynSlice_e_t_set, FgSlice_orimsk_e_set, prompt_str=['NoObj' for _ in range(len(ObjFgSlice_e_set))])

                    # then separate the high-frequency information
                    SynSlice_condele_set = SynSlice_condele_set.squeeze(0)
                    SynSlice_Condele_HFreq_set = extract_high_frequency(
                        SynSlice_condele_set,
                        CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
                    )
                    SynSlice_Condele_HFreq_set = SynSlice_Condele_HFreq_set.unsqueeze(0)                 

                    # extract the Obj mask
                    ObjFgSlice_msk_set = self.CFG_module.FgSeg_model(ObjFgSlice_set)
                    ObjFgSlice_msk_set = F.sigmoid(ObjFgSlice_msk_set)
                    ObjFgSlice_msk_set = (ObjFgSlice_msk_set >= 0.5).float()      # [6, 1, 256, 256]  

                    SynSlice_Condele_HFreq_set = SynSlice_Condele_HFreq_set * (1.0 - ObjFgSlice_msk_set)                       

                    prompt_str_dict = {
                        'prompt_str_NoObj': ['NoObj' for _ in range(len(ObjPrompt_list))],
                        'prompt_str_Obj': ObjPrompt_list,
                        'prompt_str_NoCloud': ['NoCloud' for _ in range(len(ObjPrompt_list))],
                        'prompt_str_Cloud': CloudPrompt_list
                    }
                    zero_maps_e = torch.zeros((ObjFgSlice_e_set.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float, device=DEVICE)
                    zero_maps_e = self.CFG_module.Latent_model.encode(zero_maps_e)
                    zero_maps_e, _, _ = self.CFG_module.Latent_model.reparameterize(zero_maps_e)
                    NoObjFgSlice_e_set = zero_maps_e
                    NoCloudFgSlice_e_set = zero_maps_e
                    fg_imgs_e_dict = {
                        'fg_imgs_NoObj_e': NoObjFgSlice_e_set,
                        'fg_imgs_Obj_e': ObjFgSlice_e_set,
                        'fg_imgs_NoCloud_e': NoCloudFgSlice_e_set,
                        'fg_imgs_Cloud_e': CloudFgSlice_e_set
                    }   
                    SynSlice_e_set, SynSlice_set = self.CFG_module.ImgSyn_CFG(prompt_str_dict, fg_imgs_e_dict, CloudMskSlice_e_set, ImgSyn_conditional_element=SynSlice_Condele_HFreq_set)

            else:
                ObjFgSlice_e_0_set = torch.cat(ObjFgSlice_e_0_list)
                ObjFgSlice_e_t_set = torch.cat(ObjFgSlice_e_t_list)
                CloudFgSlice_e_0_set = torch.cat(CloudFgSlice_e_0_list)
                CloudFgSlice_e_t_set = torch.cat(CloudFgSlice_e_t_list)
                SynSlice_e_0_set = torch.cat(SynSlice_e_0_list)
                SynSlice_e_t_set = torch.cat(SynSlice_e_t_list)
                BldMsk_set = torch.cat(BldMsk_list)

                if self.is_with_ControlNet:
                    ObjFgSlice_orimsk_set = torch.cat(ObjFgSlice_orimsk_list)
                    ObjFgSlice_condele_set = torch.cat(ObjFgSlice_condele_list)
                    SynSlice_condele_set = torch.cat(SynSlice_condele_list)

                print('ObjPrompt_list: ', ObjPrompt_list)
                print('CloudPrompt_list: ', CloudPrompt_list)

                Slice_totalnum = len(ObjFgSlice_e_0_set)
                Loop_num = int(np.ceil(Slice_totalnum / self.batch_size))   

                ObjFgSlice_e_set = []
                CloudFgSlice_e_set = []
                SynSlice_e_set = []                
                for Loop_idx in range(Loop_num):

                    ObjFgSlice_e_0_batch = ObjFgSlice_e_0_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    ObjFgSlice_e_t_batch = ObjFgSlice_e_t_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    CloudFgSlice_e_0_batch = CloudFgSlice_e_0_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    CloudFgSlice_e_t_batch = CloudFgSlice_e_t_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]                    
                    SynSlice_e_0_batch = SynSlice_e_0_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    SynSlice_e_t_batch = SynSlice_e_t_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    BldMsk_batch = BldMsk_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    ObjPrompt_batch = ObjPrompt_list[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                    CloudPrompt_batch = CloudPrompt_list[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]

                    if not self.is_with_ControlNet:
                        ObjFgSlice_e_batch, ObjFgSlice_batch, ObjMskSlice_e_batch, ObjMskSlice_batch = self.CFG_module.InPaint_inFgGen(ObjFgSlice_e_0_batch, ObjFgSlice_e_t_batch, BldMsk_batch, ObjPrompt_batch,
                                                                                                                                       is_openoperation=self.is_ObjFgGen_Use_OpenOperation,
                                                                                                                                       kernel_size=self.ObjOpenOperation_Kernel_Size,
                                                                                                                                       is_dilate=True)
                        CloudFgSlice_e_batch, CloudFgSlice_batch, CloudMskSlice_e_batch, CloudMskSlice_batch = self.CFG_module.InPaint_inFgGen(CloudFgSlice_e_0_batch, CloudFgSlice_e_t_batch, BldMsk_batch, CloudPrompt_batch,
                                                                                                                                               is_openoperation=self.is_CloudFgGen_Use_OpenOperation,
                                                                                                                                               kernel_size=self.CloudOpenOperation_Kernel_Size,
                                                                                                                                               is_setminthreshold=self.is_CloudFgGen_Set_MinThreshold,
                                                                                                                                               min_threshold=self.CloudFgGen_MinThreshold,
                                                                                                                                               is_dilate=True) 

                        prompt_str_dict = {
                            'prompt_str_NoObj': ['NoObj' for _ in range(len(ObjPrompt_batch))],
                            'prompt_str_Obj': ObjPrompt_batch,
                            'prompt_str_NoCloud': ['NoCloud' for _ in range(len(CloudPrompt_batch))],
                            'prompt_str_Cloud': CloudPrompt_batch
                        }

                        zero_map_batch = torch.zeros((ObjFgSlice_e_0_batch.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float, device=DEVICE)
                        zero_map_e_batch = self.CFG_module.Latent_model.encode(zero_map_batch)
                        zero_map_e_batch, _, _ = self.CFG_module.Latent_model.reparameterize(zero_map_e_batch)
                        NoObjFgSlice_e_batch = zero_map_e_batch
                        NoCloudFgSlice_e_batch = zero_map_e_batch

                        fg_imgs_e_dict = {
                            'fg_imgs_NoObj_e': NoObjFgSlice_e_batch,
                            'fg_imgs_Obj_e': ObjFgSlice_e_batch,
                            'fg_imgs_NoCloud_e': NoCloudFgSlice_e_batch,
                            'fg_imgs_Cloud_e': CloudFgSlice_e_batch
                        }     
                        SynSlice_e_batch, SynSlice_batch = self.CFG_module.InPaint_inImgSyn_CFG(SynSlice_e_0_batch, SynSlice_e_t_batch, BldMsk_batch, prompt_str_dict, fg_imgs_e_dict, CloudMskSlice_e_batch)

                    else:
                        ObjFgSlice_orimsk_batch = ObjFgSlice_orimsk_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                        ObjFgSlice_condele_batch = ObjFgSlice_condele_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]
                        SynSlice_condele_batch = SynSlice_condele_set[Loop_idx * self.batch_size: (Loop_idx + 1) * self.batch_size]

                        ObjFgSlice_e_batch, ObjFgSlice_batch, ObjMskSlice_e_batch, ObjMskSlice_batch = self.CFG_module.InPaint_inFgGen(ObjFgSlice_e_0_batch, ObjFgSlice_e_t_batch, BldMsk_batch, ObjPrompt_batch, FgGen_conditional_element=ObjFgSlice_condele_batch,
                                                                                                                                       is_openoperation=self.is_ObjFgGen_Use_OpenOperation,
                                                                                                                                       kernel_size=self.ObjOpenOperation_Kernel_Size,
                                                                                                                                       is_dilate=True)
                        CloudFgSlice_e_batch, CloudFgSlice_batch, CloudMskSlice_e_batch, CloudMskSlice_batch = self.CFG_module.InPaint_inFgGen(CloudFgSlice_e_0_batch, CloudFgSlice_e_t_batch, BldMsk_batch, CloudPrompt_batch,
                                                                                                                                               is_openoperation=self.is_CloudFgGen_Use_OpenOperation,
                                                                                                                                               kernel_size=self.CloudOpenOperation_Kernel_Size,
                                                                                                                                               is_setminthreshold=self.is_CloudFgGen_Set_MinThreshold,
                                                                                                                                               min_threshold=self.CloudFgGen_MinThreshold,
                                                                                                                                               is_dilate=True) 
                        
                        SynSlice_condele_e_batch = self.CFG_module.Latent_model.encode(SynSlice_condele_batch)
                        SynSlice_condele_e_batch, _, _ = self.CFG_module.Latent_model.reparameterize(SynSlice_condele_e_batch)

                        # first inpaint the Obj part in SynSlice
                        ObjFgSlice_orimsk_e_batch = F.interpolate(ObjFgSlice_orimsk_batch, size=(IMG_SIZE // SIDELENGTH_SCALE_FACTOR, IMG_SIZE // SIDELENGTH_SCALE_FACTOR), mode='nearest')
                        ObjFgSlice_orimsk_e_batch = tensor_dilate(ObjFgSlice_orimsk_e_batch)
                        ObjFgSlice_orimsk_e_batch = gaussian_blur(ObjFgSlice_orimsk_e_batch, kernel_size=5, sigma=3)

                        SynSlice_condele_e_batch, SynSlice_condele_batch = self.CFG_module.InPaint_inImgSyn(SynSlice_condele_e_batch, SynSlice_e_t_batch, ObjFgSlice_orimsk_e_batch, prompt_str=['NoObj' for _ in range(len(ObjFgSlice_e_batch))])           

                        # then separate the high-frequency information
                        SynSlice_Condele_HFreq_list = []
                        for SynSlice_Condele in SynSlice_condele_batch:
                            SynSlice_Condele_HFreq = extract_high_frequency(
                                SynSlice_Condele,
                                CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
                            )
                            SynSlice_Condele_HFreq_list.append(SynSlice_Condele_HFreq.unsqueeze(0))
                        SynSlice_Condele_HFreq_batch = torch.cat(SynSlice_Condele_HFreq_list).to(DEVICE)                         

                        # extract the Obj mask
                        ObjFgSlice_msk_batch = self.CFG_module.FgSeg_model(ObjFgSlice_batch)
                        ObjFgSlice_msk_batch = F.sigmoid(ObjFgSlice_msk_batch)
                        ObjFgSlice_msk_batch = (ObjFgSlice_msk_batch >= 0.5).float()     # [6, 1, 256, 256]  

                        SynSlice_Condele_HFreq_batch = SynSlice_Condele_HFreq_batch * (1.0 - ObjFgSlice_msk_batch)   

                        # then run the denoising
                        prompt_str_dict = {
                            'prompt_str_NoObj': ['NoObj' for _ in range(len(ObjPrompt_batch))],
                            'prompt_str_Obj': ObjPrompt_batch,
                            'prompt_str_NoCloud': ['NoCloud' for _ in range(len(CloudPrompt_batch))],
                            'prompt_str_Cloud': CloudPrompt_batch
                        }   

                        zero_map_batch = torch.zeros((ObjFgSlice_e_0_batch.shape[0], INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.float, device=DEVICE)
                        zero_map_e_batch = self.CFG_module.Latent_model.encode(zero_map_batch)
                        zero_map_e_batch, _, _ = self.CFG_module.Latent_model.reparameterize(zero_map_e_batch)
                        NoObjFgSlice_e_batch = zero_map_e_batch
                        NoCloudFgSlice_e_batch = zero_map_e_batch

                        fg_imgs_e_dict = {
                            'fg_imgs_NoObj_e': NoObjFgSlice_e_batch,
                            'fg_imgs_Obj_e': ObjFgSlice_e_batch,
                            'fg_imgs_NoCloud_e': NoCloudFgSlice_e_batch,
                            'fg_imgs_Cloud_e': CloudFgSlice_e_batch
                        }    
                        SynSlice_e_batch, SynSlice_batch = self.CFG_module.InPaint_inImgSyn_CFG(SynSlice_e_0_batch, SynSlice_e_t_batch, BldMsk_batch, prompt_str_dict, fg_imgs_e_dict, CloudMskSlice_e_batch, ImgSyn_conditional_element=SynSlice_Condele_HFreq_batch)

                    ObjFgSlice_e_set.append(ObjFgSlice_e_batch)
                    CloudFgSlice_e_set.append(CloudFgSlice_e_batch)
                    SynSlice_e_set.append(SynSlice_e_batch)

                ObjFgSlice_e_set = torch.cat(ObjFgSlice_e_set)
                CloudFgSlice_e_set = torch.cat(CloudFgSlice_e_set)
                SynSlice_e_set = torch.cat(SynSlice_e_set)
                
            # write in
            for idx, (ObjFgSlice_e, CloudFgSlice_e, SynSlice_e) in enumerate(zip(ObjFgSlice_e_set, CloudFgSlice_e_set, SynSlice_e_set)):
                
                row_idx, col_idx = diagonal_points[idx]

                ObjFgSlice_e = ObjFgSlice_e.unsqueeze(0)
                CloudFgSlice_e = CloudFgSlice_e.unsqueeze(0)
                SynSlice_e = SynSlice_e.unsqueeze(0)

                MosGradMap_e_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                                    row_idx_max, col_idx_max,
                                                                    self.overlap_buffer_thx_e, self.ImgSlice_e_h, self.ImgSlice_e_w, 
                                                                    zeros_thx=0, grad_thx=self.overlap_buffer_thx_e, ones_thx=0,
                                                                    is_tailgrad=True)
                H_MosGradMap_0to1, H_MosGradMap_1to0, V_MosGradMap_0to1, V_MosGradMap_1to0, LT_MosGradMap_0to1, LT_MosGradMap_1to0 = MosGradMap_e_list

                if row_idx == 0 and col_idx == 0:
                    ObjFgLargeImg_e[:, :, :self.ImgSlice_e_h, :self.ImgSlice_e_w] = ObjFgSlice_e
                    CloudFgLargeImg_e[:, :, :self.ImgSlice_e_h, :self.ImgSlice_e_w] = CloudFgSlice_e
                    SynLargeImg_e[:, :, :self.ImgSlice_e_h, :self.ImgSlice_e_w] = SynSlice_e                  
                
                # top edge
                if row_idx == 0 and col_idx > 0:

                    new_row_idx = 0
                    new_col_idx = col_idx + self.overlap_nobuffer_thx_e

                    # reset the target area to 0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0

                    new_V_MosGradMap_0to1 = V_MosGradMap_0to1[:, :, :, :(self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)]
                    new_V_MosGradMap_1to0 = V_MosGradMap_1to0[:, :, :, :(self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)]

                    # Pair the cropped vertical weights with the matching right-side content.
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += ObjFg_CoverageArea * new_V_MosGradMap_1to0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += ObjFgSlice_e[:, :, :, self.overlap_nobuffer_thx_e:] * new_V_MosGradMap_0to1

                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += CloudFg_CoverageArea * new_V_MosGradMap_1to0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += CloudFgSlice_e[:, :, :, self.overlap_nobuffer_thx_e:] * new_V_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Syn_CoverageArea * new_V_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += SynSlice_e[:, :, :, self.overlap_nobuffer_thx_e:] * new_V_MosGradMap_0to1

                # left edge
                elif row_idx > 0 and col_idx == 0:
                    
                    new_row_idx = row_idx + self.overlap_nobuffer_thx_e
                    new_col_idx = 0

                    # reset the target area to 0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] = 0.0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] = 0.0
                    
                    # modify H_MosGradMap_1to0 and H_MosGradMap_0to1
                    new_H_MosGradMap_0to1 = H_MosGradMap_0to1[:, :, :(self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), :]
                    new_H_MosGradMap_1to0 = H_MosGradMap_1to0[:, :, :(self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), :]

                    # Pair the cropped horizontal weights with the matching lower content.
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)].clone()
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += ObjFg_CoverageArea * new_H_MosGradMap_1to0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += ObjFgSlice_e[:, :, self.overlap_nobuffer_thx_e:, :] * new_H_MosGradMap_0to1

                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)].clone()
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += CloudFg_CoverageArea * new_H_MosGradMap_1to0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += CloudFgSlice_e[:, :, self.overlap_nobuffer_thx_e:, :] * new_H_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)].clone()
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += Syn_CoverageArea * new_H_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w)] += SynSlice_e[:, :, self.overlap_nobuffer_thx_e:, :] * new_H_MosGradMap_0to1


                elif row_idx > 0 and col_idx > 0:

                    new_row_idx = row_idx + self.overlap_nobuffer_thx_e
                    new_col_idx = col_idx + self.overlap_nobuffer_thx_e

                    # reset the target area to 0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] = 0.0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] = 0.0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.overlap_buffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] = 0.0

                    if row_idx != row_idx_max:
                        # The buffered and unbuffered widths sum to overlap_thx_e.
                        ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                        CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                        SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0   
                    else:
                        # note: at row_idx_max there is no need to subtract self.overlap_buffer_thx_e again
                        ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
                        CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.overlap_buffer_thx_e)] = 0.0
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
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += ObjFg_CoverageArea * new_LT_MosGradMap_1to0
                    ObjFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += ObjFgSlice_e[:, :, self.overlap_nobuffer_thx_e:, self.overlap_nobuffer_thx_e:] * new_LT_MosGradMap_0to1

                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += CloudFg_CoverageArea * new_LT_MosGradMap_1to0
                    CloudFgLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += CloudFgSlice_e[:, :, self.overlap_nobuffer_thx_e:, self.overlap_nobuffer_thx_e:] * new_LT_MosGradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)].clone()              
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += Syn_CoverageArea * new_LT_MosGradMap_1to0
                    SynLargeImg_e[:, :, new_row_idx: (new_row_idx + self.ImgSlice_e_h - self.overlap_nobuffer_thx_e), new_col_idx: (new_col_idx + self.ImgSlice_e_w - self.overlap_nobuffer_thx_e)] += SynSlice_e[:, :, self.overlap_nobuffer_thx_e:, self.overlap_nobuffer_thx_e:] * new_LT_MosGradMap_0to1

            # update NoGrad_ObjFgLargeImg_e, NoGrad_CloudFgLargeImg_e and NoGrad_SynLargeImg_e
            NoGrad_ObjFgLargeImg_e = ObjFgLargeImg_e.clone()
            NoGrad_CloudFgLargeImg_e = CloudFgLargeImg_e.clone()
            NoGrad_SynLargeImg_e = SynLargeImg_e.clone()
        
        
        # decode and then write into the large images
        ObjFgLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        CloudFgLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        SynLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        ObjMskLargeImg = torch.zeros((1, UNET_OUTPUT_CHANNELS, LargeImg_h, LargeImg_w)  , dtype=torch.uint8).to(DEVICE)        
        CloudMskLargeImg = torch.zeros((1, UNET_OUTPUT_CHANNELS, LargeImg_h, LargeImg_w)  , dtype=torch.uint8).to(DEVICE)    

        NoGrad_ObjFgLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        NoGrad_CloudFgLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)
        NoGrad_SynLargeImg = torch.zeros((1, INPUT_CHANNELS, LargeImg_h, LargeImg_w), dtype=torch.float32).to(DEVICE)


        for diagonal_points in diagonals_points_list:
            for row_idx, col_idx in diagonal_points:     

                GradMap_list = self._compute_grad_of_crossarea(row_idx, col_idx, 
                                                               row_idx_max, col_idx_max,
                                                               self.overlap_thx, self.ImgSlice_h, self.ImgSlice_w, 
                                                               zeros_thx=0, grad_thx=self.overlap_thx, ones_thx=0,
                                                               is_tailgrad=True)
                H_GradMap_0to1, H_GradMap_1to0, V_GradMap_0to1, V_GradMap_1to0, LT_GradMap_0to1, LT_GradMap_1to0 = GradMap_list

                ObjFgSlice_e = ObjFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.ImgSlice_e_w]
                CloudFgSlice_e = CloudFgLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.ImgSlice_e_w]
                SynSlice_e = SynLargeImg_e[:, :, row_idx: row_idx + self.ImgSlice_e_h, col_idx: col_idx + self.ImgSlice_e_w]
                ObjFgSlice = self.CFG_module.Latent_model.decode(ObjFgSlice_e)  
                CloudFgSlice = self.CFG_module.Latent_model.decode(CloudFgSlice_e) 
                SynSlice = self.CFG_module.Latent_model.decode(SynSlice_e)  

                ObjMskSlice = self.CFG_module.FgSeg_model(ObjFgSlice)
                ObjMskSlice = F.sigmoid(ObjMskSlice)
                ObjMskSlice = (ObjMskSlice >= 0.5).float() 

                CloudMskSlice = self.CFG_module.FgSeg_model(CloudFgSlice)
                CloudMskSlice = F.sigmoid(CloudMskSlice)
                CloudMskSlice = (CloudMskSlice >= 0.5).float()           

                if row_idx == 0 and col_idx == 0:
                    ObjFgLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = ObjFgSlice
                    CloudFgLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = CloudFgSlice
                    SynLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = SynSlice
                    ObjMskLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = ObjMskSlice           
                    CloudMskLargeImg[:, :, :self.ImgSlice_h, :self.ImgSlice_w] = CloudMskSlice         
                
                # top edge
                elif row_idx == 0 and col_idx > 0:

                    # reset the target area to 0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0

                    # handle the rest
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFg_CoverageArea * V_GradMap_1to0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFgSlice * V_GradMap_0to1
                    
                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFg_CoverageArea * V_GradMap_1to0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFgSlice * V_GradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * V_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * V_GradMap_0to1                   

                    V_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    V_BldMskMap_1to0[:, :, :, :self.overlap_thx] = 1.0

                    ObjMsk_CoverageArea = ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * V_BldMskMap_1to0
                    ObjMsk_NewlyGened = ObjMskSlice * V_BldMskMap_1to0
                    ObjMskSlice = torch.min(ObjMsk_NewlyGened, ObjMsk_CoverageArea) + ObjMskSlice * (1.0 - V_BldMskMap_1to0)
                    ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = ObjMskSlice
                                    
                    CloudMsk_CoverageArea = CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * V_BldMskMap_1to0
                    CloudMsk_NewlyGened = CloudMskSlice * V_BldMskMap_1to0
                    CloudMskSlice = torch.min(CloudMsk_NewlyGened, CloudMsk_CoverageArea) + CloudMskSlice * (1.0 - V_BldMskMap_1to0)
                    CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = CloudMskSlice

                # left edge
                elif row_idx > 0 and col_idx == 0:

                    # reset the target area to 0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    
                    # handle the rest
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFg_CoverageArea * H_GradMap_1to0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFgSlice * H_GradMap_0to1

                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFg_CoverageArea * H_GradMap_1to0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFgSlice * H_GradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * H_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * H_GradMap_0to1

                    H_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    H_BldMskMap_1to0[:, :, :self.overlap_thx, :] = 1.0

                    ObjMsk_CoverageArea = ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * H_BldMskMap_1to0
                    ObjMsk_NewlyGened = ObjMskSlice * H_BldMskMap_1to0
                    ObjMskSlice = torch.min(ObjMsk_NewlyGened, ObjMsk_CoverageArea) + ObjMskSlice * (1.0 - H_BldMskMap_1to0)
                    ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = ObjMskSlice

                    CloudMsk_CoverageArea = CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * H_BldMskMap_1to0
                    CloudMsk_NewlyGened = CloudMskSlice * H_BldMskMap_1to0
                    CloudMskSlice = torch.min(CloudMsk_NewlyGened, CloudMsk_CoverageArea) + CloudMskSlice * (1.0 - H_BldMskMap_1to0)
                    CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = CloudMskSlice

                elif row_idx > 0 and col_idx > 0:                    

                     # reset the target area to 0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = 0.0
                    
                    if row_idx != row_idx_max:
                        ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h - self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0   
                    else:
                        ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0
                        SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.overlap_thx_e) * SIDELENGTH_SCALE_FACTOR] = 0.0                          
                                     
                    # handle the rest
                    ObjFg_CoverageArea = NoGrad_ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFg_CoverageArea * LT_GradMap_1to0
                    ObjFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += ObjFgSlice * LT_GradMap_0to1

                    CloudFg_CoverageArea = NoGrad_CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFg_CoverageArea * LT_GradMap_1to0
                    CloudFgLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += CloudFgSlice * LT_GradMap_0to1

                    Syn_CoverageArea = NoGrad_SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone()                  
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += Syn_CoverageArea * LT_GradMap_1to0
                    SynLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] += SynSlice * LT_GradMap_0to1

                    LT_BldMskMap_1to0 = torch.zeros((1, 1, self.ImgSlice_h, self.ImgSlice_w), dtype=torch.float, device=DEVICE)
                    LT_BldMskMap_1to0[:, :, :, :self.overlap_thx] = 1.0
                    LT_BldMskMap_1to0[:, :, :self.overlap_thx, :] = 1.0

                    ObjMsk_CoverageArea = ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * LT_BldMskMap_1to0
                    ObjMsk_NewlyGened = ObjMskSlice * LT_BldMskMap_1to0
                    ObjMskSlice = torch.min(ObjMsk_NewlyGened, ObjMsk_CoverageArea) + ObjMskSlice * (1.0 - LT_BldMskMap_1to0)
                    ObjMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = ObjMskSlice

                    CloudMsk_CoverageArea = CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR].clone() * LT_BldMskMap_1to0
                    CloudMsk_NewlyGened = CloudMskSlice * LT_BldMskMap_1to0
                    CloudMskSlice = torch.min(CloudMsk_NewlyGened, CloudMsk_CoverageArea) + CloudMskSlice * (1.0 - LT_BldMskMap_1to0)
                    CloudMskLargeImg[:, :, row_idx * SIDELENGTH_SCALE_FACTOR: (row_idx + self.ImgSlice_e_h) * SIDELENGTH_SCALE_FACTOR, col_idx * SIDELENGTH_SCALE_FACTOR: (col_idx + self.ImgSlice_e_w) * SIDELENGTH_SCALE_FACTOR] = CloudMskSlice
            

            # update NoGrad_ObjFgLargeImg, NoGrad_CloudFgLargeImg and NoGrad_SynLargeImg
            NoGrad_ObjFgLargeImg = ObjFgLargeImg.clone()
            NoGrad_CloudFgLargeImg = CloudFgLargeImg.clone()
            NoGrad_SynLargeImg = SynLargeImg.clone()

        if self.ImgSyn_RGB_savepath is not None:
            save_rgb_datas(prepare_rgb_vis_tensor(SynLargeImg).cpu(), nrow=3, 
                           savepath=self.ImgSyn_RGB_savepath, 
                           is_showminmax=False, is_makegrid=False)  

        if self.ObjFgGen_RGB_savepath is not None:
            save_rgb_datas(prepare_rgb_vis_tensor(ObjFgLargeImg).cpu(), nrow=3, 
                           savepath=self.ObjFgGen_RGB_savepath, 
                           is_showminmax=False, is_makegrid=False)  
            
        if self.CloudFgGen_RGB_savepath is not None:      
            save_rgb_datas(prepare_rgb_vis_tensor(CloudFgLargeImg).cpu(), nrow=3, 
                           savepath=self.CloudFgGen_RGB_savepath, 
                           is_showminmax=False, is_makegrid=False)    
            
        saved_proj = self.conditional_element_proj
        saved_geotrans = self.conditional_element_geotrans

        if self.ImgSyn_TIF_savepath is not None:
            _, saved_proj, saved_geotrans = save_tif_datas(SynLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                                                        savepath=self.ImgSyn_TIF_savepath, 
                                                        is_showminmax=False)  

        if self.ObjFgGen_TIF_savepath is not None:
            _, saved_proj, saved_geotrans = save_tif_datas(
                ObjFgLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                savepath=self.ObjFgGen_TIF_savepath, is_showminmax=False
            )

        if self.CloudFgGen_TIF_savepath is not None:
            _, saved_proj, saved_geotrans = save_tif_datas(
                CloudFgLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                savepath=self.CloudFgGen_TIF_savepath, is_showminmax=False
            )


        if self.ObjMsk_savepath is not None:
            save_msk_datas(ObjMskLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                           savepath=self.ObjMsk_savepath, 
                           is_showminmax=False, is_makegrid=False)      

        if self.CloudMsk_savepath is not None:
            save_msk_datas(CloudMskLargeImg.cpu(), projections=saved_proj, geotransforms=saved_geotrans,
                           savepath=self.CloudMsk_savepath, 
                           is_showminmax=False, is_makegrid=False)                         
                

        return ObjFgLargeImg, CloudFgLargeImg, SynLargeImg, ObjMskLargeImg, CloudMskLargeImg

if __name__=='__main__':
    from Multi_Condition_Generation.IMPGM_Multi_Condition_Inference import (
        DEFAULT_LAMBDA_CLOUD,
        DEFAULT_LAMBDA_OBJECT,
    )

    ObjFgGen_RGB_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_RGB/ObjFgGen.jpg'
    ObjFgGen_TIF_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_TIF/ObjFgGen.tif'
    CloudFgGen_RGB_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_RGB/CloudFgGen.jpg'
    CloudFgGen_TIF_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_TIF/CloudFgGen.tif'
    ImgSyn_RGB_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_RGB/ImgSyn.jpg'
    ImgSyn_TIF_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_TIF/ImgSyn.tif'
    ObjMsk_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_RGB/ObjMsk.tif'
    CloudMsk_savepath = r'./LargeImgs_CFG/LargeImgs_CFG_RGB/CloudMsk.tif'

    beta_t = build_beta_schedule_from_config(FGGEN_DIFFUSION_CONFIG)

    h_start_idx, h_end_idx =   0,  448
    w_start_idx, w_end_idx = 900, 1348

    # read FgGen_cond_ele
    FgMsk_data, _, _ = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_mask.tif')
    FgMsk_data = FgMsk_data[h_start_idx: h_end_idx, w_start_idx: w_end_idx]
    # FgMsk_data = FgMsk_data[:832, 900: 900 + 832]
    FgMsk_data = torch.from_numpy(FgMsk_data).to(device=DEVICE, dtype=torch.float32)
    FgMsk_data = FgMsk_data.unsqueeze(0).unsqueeze(0)

    # read ImgSyn_cond_ele
    Img_data, Img_projection, Img_geotransform = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_data.tif')
    Img_data = Img_data[:, h_start_idx: h_end_idx, w_start_idx: w_end_idx]
    # Img_data = Img_data[:, :832, 900: 900 + 832]
    Img_data = torch.from_numpy(Img_data).to(device=DEVICE, dtype=torch.float32)
    Img_data = Img_data.unsqueeze(0)

    image_mean = torch.as_tensor(IMAGE_MEAN, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
    image_std = torch.as_tensor(IMAGE_STD, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
    Img_data = (Img_data - image_mean) / image_std

    Img_geotransform = [Img_geotransform[0] + w_start_idx * Img_geotransform[1], Img_geotransform[1], Img_geotransform[2], 
                        Img_geotransform[3] + h_start_idx * Img_geotransform[5], Img_geotransform[4], Img_geotransform[5]]

    Generate_LargeImg_CFG(sampler_mode='ddim',
                          beta_t=beta_t,
                          overlap_rate=0.25,
                          is_with_ControlNet=True,
                          lambda_cloud=DEFAULT_LAMBDA_CLOUD,
                          lambda_object=DEFAULT_LAMBDA_OBJECT,
                          is_FgGenInpaint_Resample=True,
                          is_ImgSynInpaint_Resample=True,
                          ObjFgGen_RGB_savepath=ObjFgGen_RGB_savepath,
                          ObjFgGen_TIF_savepath=ObjFgGen_TIF_savepath,
                          CloudFgGen_RGB_savepath=CloudFgGen_RGB_savepath,
                          CloudFgGen_TIF_savepath=CloudFgGen_TIF_savepath,
                          ImgSyn_RGB_savepath=ImgSyn_RGB_savepath,
                          ImgSyn_TIF_savepath=ImgSyn_TIF_savepath,
                          ObjMsk_savepath=ObjMsk_savepath,
                          CloudMsk_savepath=CloudMsk_savepath,
                          conditional_element_proj=[Img_projection],
                          conditional_element_geotrans=[Img_geotransform],
                          ).main(LargeImg_h=448, LargeImg_w=448, 
                                 first_obj_prompt_str='Water', obj_prompt_str='Water', noobj_prompt_str='NoObj',
                                 first_cloud_prompt_str='LessCloud', cloud_prompt_str_list=['FewCloud', 'LessCloud', 'MoreCloud', 'ManyCloud'], nocloud_prompt_str='NoCloud',
                                 FgGen_original_msk=FgMsk_data,
                                 FgGen_conditional_element=FgMsk_data, ImgSyn_conditional_element=Img_data)
