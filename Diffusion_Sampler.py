from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from IMPGM_Config import *
import numpy as np
from IMPGM_Utils import (
    build_torch_generator,
    extract_diffusion_coefficient,
    randn_like as randn_like_with_generator,
)


DEFAULT_DDIM_STEPS = 500
DEFAULT_DDIM_ETA = 0.0


def _make_strictly_descending(time_steps, total_timesteps):
    """Resolve rounded candidates to an exact, strictly descending schedule."""
    requested_steps = int(time_steps.numel())
    resolved_steps = []
    previous_step = int(total_timesteps)

    for index, candidate in enumerate(time_steps.tolist()):
        # Keep enough lower indices available for every remaining DDIM update.
        lower_bound = requested_steps - index - 1
        upper_bound = previous_step - 1
        current_step = min(max(int(candidate), lower_bound), upper_bound)
        resolved_steps.append(current_step)
        previous_step = current_step

    return torch.tensor(
        resolved_steps,
        device=time_steps.device,
        dtype=torch.long,
    )


def build_ddim_time_steps(alpha_t_bar, steps=DEFAULT_DDIM_STEPS, sample_method="alpha_space"):
    """Build exactly ``steps`` selected training timesteps for DDIM sampling."""
    if alpha_t_bar.ndim != 1 or alpha_t_bar.numel() == 0:
        raise ValueError("Expected alpha_t_bar to be a non-empty 1D tensor.")
    steps = int(steps)
    if steps <= 0 or steps > int(alpha_t_bar.numel()):
        raise ValueError(
            f"Expected DDIM steps in [1, {int(alpha_t_bar.numel())}], got {steps}."
        )
    if sample_method not in {"alpha_space", "t_space"}:
        raise ValueError("Unsupported sample_method; use 'alpha_space' or 't_space'.")

    if sample_method == "alpha_space":
        alpha_bar_targets = torch.linspace(
            alpha_t_bar[-1],
            alpha_t_bar[0],
            steps,
            device=alpha_t_bar.device,
            dtype=alpha_t_bar.dtype,
        )
        time_steps = torch.bucketize(
            alpha_bar_targets.flip(0), alpha_t_bar.flip(0)
        )
        time_steps = (len(alpha_t_bar) - 1 - time_steps).clamp(
            min=0,
            max=len(alpha_t_bar) - 1,
        )
        time_steps = time_steps.flip(0)
    else:
        time_steps = torch.linspace(
            len(alpha_t_bar) - 1,
            0,
            steps,
            device=alpha_t_bar.device,
        ).round().long()

    time_steps = _make_strictly_descending(time_steps, len(alpha_t_bar))
    time_steps_prev = torch.cat([
        time_steps[1:],
        torch.zeros(1, device=time_steps.device, dtype=time_steps.dtype),
    ])
    return time_steps, time_steps_prev


class DDPMSampler(nn.Module):
    """Sample images with the DDPM reverse-diffusion process."""
    def __init__(self, model, beta_t, is_ImgSyn=False, is_with_ControlNet=False, is_Inference=False, generator=None):
        super().__init__()
        self.model = model
        self.T = len(beta_t)
        self.is_with_ControlNet = is_with_ControlNet
        self.is_ImgSyn = is_ImgSyn
        self.is_Inference = is_Inference
        self.generator = generator

        self.register_buffer("beta_t", beta_t)

        # Calculate the cumulative product of alpha, denoted alpha_bar_t in the paper.
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("alpha_t_bar", alpha_t_bar)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    def _resolve_generator(self, device, seed=None, generator=None):
        if generator is not None:
            self.generator = generator
        elif seed is not None:
            self.generator = build_torch_generator(seed, device)
        return self.generator

    def _randn_like(self, tensor):
        if isinstance(self.generator, (list, tuple)):
            if len(self.generator) != tensor.shape[0]:
                raise ValueError(
                    "The number of random generators must match the batch size: "
                    f"{len(self.generator)} vs {tensor.shape[0]}."
                )
            return torch.cat(
                [
                    randn_like_with_generator(
                        tensor[index:index + 1],
                        generator=generator,
                    )
                    for index, generator in enumerate(self.generator)
                ],
                dim=0,
            )
        return randn_like_with_generator(tensor, generator=self.generator)

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t, prompt_str, **kwargs):
        """
        Calculate the mean and variance for q(x_{t-1} | x_t, x_0) under v-prediction
        """
        if not self.is_ImgSyn and not self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str)
        elif self.is_ImgSyn and not self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str, kwargs['fg_imgs_e'])
        elif not self.is_ImgSyn and self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str, kwargs['conditional_element'])
        else:
            v_pred = self.model(x_t, t, prompt_str, kwargs['fg_imgs_e'], kwargs['conditional_element'])

        # Convert v-prediction to epsilon.
        alpha_bar_t = extract_diffusion_coefficient(self.alpha_t_bar, t, x_t.shape)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        epsilon_theta = sqrt_1_minus_alpha_bar_t * x_t + sqrt_alpha_bar_t * v_pred

        mean = extract_diffusion_coefficient(self.coeff_1, t, x_t.shape) * x_t - extract_diffusion_coefficient(self.coeff_2, t, x_t.shape) * epsilon_theta
        var = extract_diffusion_coefficient(self.posterior_variance, t, x_t.shape)  # var is constant
        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, prompt_str, **kwargs):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        
        mean, var = self.cal_mean_variance(x_t, t, prompt_str, **kwargs)

        z = self._randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).any():
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, prompt_str, is_record_process=False, **kwargs):
        self._resolve_generator(x_t.device, seed=kwargs.get("seed"), generator=kwargs.get("generator"))
        if is_record_process:
            x_list = [x_t]
        else:
            x_list = None

        if 'dataset_len' in kwargs:
            dataset_len = kwargs['dataset_len']
        else:
            dataset_len = None
        if 'batch_idx' in kwargs:
            batch_idx = kwargs['batch_idx']
        else:
            batch_idx = None
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = None
            
        if self.is_Inference and batch_idx is not None and batch_size is not None and dataset_len is not None:
            sample_num = min(batch_size * (batch_idx + 1), dataset_len)
            sampling_steps = tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, ncols=120, desc=f'>>> {sample_num} / {dataset_len}')
        else:
            sampling_steps = tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, ncols=80)

        for time_step in sampling_steps:
            x_t = self.sample_one_step(x_t, time_step, prompt_str, **kwargs)

            if x_list is not None:
                x_list.append(x_t)

            sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "batch_size": len(x_t)})

        if x_list is not None: 
            return x_t, x_list
        else:
            return x_t    # [batch_size, channels, height, width]
    

    @torch.no_grad()
    def q_sample(self, x_0, time_step, noise=None):
        if noise is None:
            noise = self._randn_like(x_0)
        sqrt_alphas_t_bar = torch.sqrt(self.alpha_t_bar)
        sqrt_one_minus_alphas_t_bar = torch.sqrt(1.0 - self.alpha_t_bar)
        noise_t_level = extract_diffusion_coefficient(sqrt_alphas_t_bar, time_step, x_0.shape) * x_0 + extract_diffusion_coefficient(sqrt_one_minus_alphas_t_bar, time_step, x_0.shape) * noise
        return noise_t_level    


    @torch.no_grad()
    def inpaint(self, x_0, x_t, msks, prompt_str, is_record_process=False, is_resample=True, **kwargs):

        self._resolve_generator(x_t.device, seed=kwargs.get("seed"), generator=kwargs.get("generator"))

        if is_record_process:
            x_list = [x_t]
        else:
            x_list = None

        if 'dataset_len' in kwargs:
            dataset_len = kwargs['dataset_len']
        else:
            dataset_len = None
        if 'batch_idx' in kwargs:
            batch_idx = kwargs['batch_idx']
        else:
            batch_idx = None
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = None     

        if is_resample:
            if 'resample_interval' in kwargs:
                resample_interval = kwargs['resample_interval']
            else:
                resample_interval = 5

            if 'jump_lens_list' in kwargs:
                jump_lens_list = kwargs['jump_lens_list']
            else:
                jump_lens_list = [1, 3, 5]

            if 'resample_timestep_range' in kwargs:
                resample_timestep_range = kwargs['resample_timestep_range']
            else:
                resample_timestep_range = [0.10 * self.T, 1.00 * self.T]
        else:
            resample_interval, jump_lens_list, resample_timestep_range = None, None, None
    
        if self.is_Inference and batch_idx is not None and batch_size is not None and dataset_len is not None:
            sample_num = min(batch_size * batch_idx, dataset_len)
            sampling_steps = tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, ncols=120, desc=f'>>> {sample_num} / {dataset_len}')
        else:
            sampling_steps = tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, ncols=80)

        for time_step in sampling_steps:

            t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)

        # Regenerate where mask is 1; preserve the blended context where it is 0.
            x_0_add_noise = self.q_sample(x_0, t, noise=self._randn_like(x_0))
            x_t = x_0_add_noise * (1.0 - msks) + msks * x_t 

            x_t = self.sample_one_step(x_t, time_step, prompt_str, **kwargs)

            # after one denoising step, the current timestep is t - 1
            t_minus_one = torch.clamp(t - 1, min=0)
            t_minus_one_scalar = int(t_minus_one[0])

            if is_resample and time_step % resample_interval == 0 and resample_timestep_range[0] <= time_step <= resample_timestep_range[1]:
                for jump_len in jump_lens_list:

                    # jump back jump_len steps
                    jump_t_scalar = min(t_minus_one_scalar + jump_len, self.T - 1)
                    jump_t = torch.full_like(t_minus_one, jump_t_scalar)

                    tmp_x_t = x_t.clone()
                    tmp_t = t_minus_one.clone()

                    cur_a  = extract_diffusion_coefficient(self.alpha_t_bar, tmp_t, tmp_x_t.shape)
                    jump_a = extract_diffusion_coefficient(self.alpha_t_bar, jump_t, tmp_x_t.shape)

                    tmp_x_t = torch.sqrt(jump_a / cur_a) * tmp_x_t + torch.sqrt(1.0 - (jump_a / cur_a)) * self._randn_like(tmp_x_t)
                    tmp_t = jump_t.clone()        

                    # sample normally from jump_t back to time_step
                    for t_forward in range(jump_t_scalar, t_minus_one_scalar, -1):

                        t_forward_tensor = torch.full((x_t.shape[0],), t_forward, device=tmp_x_t.device, dtype=torch.long)

                        x0_noise = self.q_sample(x_0, t_forward_tensor, noise=self._randn_like(x_0))
                        tmp_x_t = x0_noise * (1.0 - msks) + msks * tmp_x_t

                        tmp_x_t = self.sample_one_step(tmp_x_t, t_forward, prompt_str, **kwargs)
                    
                    x_t = tmp_x_t.clone()

            if x_list is not None:
                x_list.append(x_t)

            sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "batch_size": len(x_t)})

        if x_list is not None: 
            return x_t, x_list
        else:
            return x_t    # [batch_size, channels, height, width]



class DDIMSampler(nn.Module):
    """Sample images with the DDIM reverse-diffusion process."""
    def __init__(self, model, beta_t, is_ImgSyn=False, is_with_ControlNet=False, is_Inference=False, generator=None):
        super().__init__()
        self.model = model
        self.T = len(beta_t)
        self.is_with_ControlNet = is_with_ControlNet
        self.is_ImgSyn = is_ImgSyn
        self.is_Inference = is_Inference
        self.generator = generator

        self.register_buffer("beta_t", beta_t)

        # Calculate the cumulative product of alpha, denoted alpha_bar_t in the paper.
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("alpha_t_bar", alpha_t_bar)

    def _resolve_generator(self, device, seed=None, generator=None):
        if generator is not None:
            self.generator = generator
        elif seed is not None:
            self.generator = build_torch_generator(seed, device)
        return self.generator

    def _randn_like(self, tensor):
        if isinstance(self.generator, (list, tuple)):
            if len(self.generator) != tensor.shape[0]:
                raise ValueError(
                    "The number of random generators must match the batch size: "
                    f"{len(self.generator)} vs {tensor.shape[0]}."
                )
            return torch.cat(
                [
                    randn_like_with_generator(
                        tensor[index:index + 1],
                        generator=generator,
                    )
                    for index, generator in enumerate(self.generator)
                ],
                dim=0,
            )
        return randn_like_with_generator(tensor, generator=self.generator)

    @torch.no_grad()
    def sample_one_step(
        self,
        x_t,
        time_step,
        prev_time_step,
        eta,
        prompt_str,
        is_final_step=False,
        **kwargs,
    ):

        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        if prev_time_step < 0:
            raise ValueError(f"DDIM previous timestep must be non-negative, got {prev_time_step}.")

        if not self.is_ImgSyn and not self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str)
        elif self.is_ImgSyn and not self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str, kwargs['fg_imgs_e'])
        elif not self.is_ImgSyn and self.is_with_ControlNet:
            v_pred = self.model(x_t, t, prompt_str, kwargs['conditional_element'])
        else:
            v_pred = self.model(x_t, t, prompt_str, kwargs['fg_imgs_e'], kwargs['conditional_element'])
        
        # Convert v-prediction to epsilon.
        alpha_bar_t = extract_diffusion_coefficient(self.alpha_t_bar, t, x_t.shape)
        if is_final_step:
            # As in DDPM's t=0 update, alpha_bar_prev=1 denotes the clean endpoint.
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
        else:
            t_prev = torch.full(
                (x_t.shape[0],),
                prev_time_step,
                device=x_t.device,
                dtype=torch.long,
            )
            alpha_bar_t_prev = extract_diffusion_coefficient(
                self.alpha_t_bar,
                t_prev,
                x_t.shape,
            )
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        epsilon_theta = sqrt_1_minus_alpha_bar_t * x_t + sqrt_alpha_bar_t * v_pred

        sigma_variance = (
            (1 - alpha_bar_t_prev)
            / (1 - alpha_bar_t)
            * (1 - alpha_bar_t / alpha_bar_t_prev)
        ).clamp(min=0.0)
        sigma_t = eta * torch.sqrt(sigma_variance)
        direction_variance = (1 - alpha_bar_t_prev - sigma_t ** 2).clamp(min=0.0)
        z = self._randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_bar_t_prev / alpha_bar_t) * x_t +
                (torch.sqrt(direction_variance) - torch.sqrt((alpha_bar_t_prev * (1 - alpha_bar_t)) / alpha_bar_t)) * epsilon_theta +
                sigma_t * z
        )

        return x_t_minus_one
    

    @torch.no_grad()
    def forward(
        self,
        x_t,
        prompt_str,
        steps=DEFAULT_DDIM_STEPS,
        eta=DEFAULT_DDIM_ETA,
        is_record_process=False,
        **kwargs,
    ):

        self._resolve_generator(x_t.device, seed=kwargs.get("seed"), generator=kwargs.get("generator"))

        eta = float(eta)
        if eta < 0.0:
            raise ValueError(f"DDIM eta must be non-negative, got {eta}.")

        if is_record_process:
            x_list = [x_t]
        else:
            x_list = None

        if 'dataset_len' in kwargs:
            dataset_len = kwargs['dataset_len']
        else:
            dataset_len = None
        if 'batch_idx' in kwargs:
            batch_idx = kwargs['batch_idx']
        else:
            batch_idx = None
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = None

        sample_method = kwargs.get('sample_method', 'alpha_space')
        time_steps, time_steps_prev = build_ddim_time_steps(
            self.alpha_t_bar,
            steps=steps,
            sample_method=sample_method,
        )
        self.last_sampling_metadata = {
            "requested_steps": int(steps),
            "actual_steps": int(len(time_steps)),
            "eta": eta,
            "sample_method": sample_method,
        }

        if self.is_Inference and batch_idx is not None and batch_size is not None and dataset_len is not None:
            sample_num = min(batch_size * (batch_idx + 1), dataset_len)
            tqdm_bar = tqdm(zip(time_steps, time_steps_prev), colour="#6565b5", total=len(time_steps), ncols=120, desc=f'>>> {sample_num} / {dataset_len}')
        else:
            tqdm_bar = tqdm(zip(time_steps, time_steps_prev), colour="#6565b5", total=len(time_steps), ncols=80)


        for idx, (cur_t, cur_t_prev) in enumerate(tqdm_bar):
            x_t = self.sample_one_step(
                x_t,
                int(cur_t.item()),
                int(cur_t_prev.item()),
                eta,
                prompt_str,
                is_final_step=(idx == len(time_steps) - 1),
                **kwargs,
            )

            if x_list is not None:
                x_list.append(x_t)

            tqdm_bar.set_postfix(ordered_dict={"step": idx + 1, "batch_size": len(x_t)})

        if x_list is not None: 
            return x_t, x_list
        else:
            return x_t    # [batch_size, channels, height, width]


    @torch.no_grad()
    def q_sample(self, x_0, time_step, noise=None):
        if noise is None:
            noise = self._randn_like(x_0)
        sqrt_alphas_t_bar = torch.sqrt(self.alpha_t_bar)
        sqrt_one_minus_alphas_t_bar = torch.sqrt(1.0 - self.alpha_t_bar)
        noise_t_level = extract_diffusion_coefficient(sqrt_alphas_t_bar, time_step, x_0.shape) * x_0 + extract_diffusion_coefficient(sqrt_one_minus_alphas_t_bar, time_step, x_0.shape) * noise
        return noise_t_level    


    @torch.no_grad()
    def inpaint(self, x_0, x_t, msks, prompt_str, steps=500, eta=0.0, is_record_process=False, is_resample=True, **kwargs):

        self._resolve_generator(x_t.device, seed=kwargs.get("seed"), generator=kwargs.get("generator"))

        if is_record_process:
            x_list = [x_t]
        else:
            x_list = None

        if 'dataset_len' in kwargs:
            dataset_len = kwargs['dataset_len']
        else:
            dataset_len = None
        if 'batch_idx' in kwargs:
            batch_idx = kwargs['batch_idx']
        else:
            batch_idx = None
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = None

        if 'sample_method' in kwargs:
            sample_method = kwargs['sample_method']
            if sample_method != 'alpha_space' and sample_method != 't_space':
                raise Exception("Unsupported sample_method; use 'alpha_space' or 't_space'.")
        else:
            sample_method = 'alpha_space'

        if is_resample:
            if 'resample_interval' in kwargs:
                resample_interval = kwargs['resample_interval']
            else:
                resample_interval = 5
            
            if 'jump_lens_list' in kwargs:
                jump_lens_list = kwargs['jump_lens_list']
            else:
                jump_lens_list = [1, 3, 5]

            if 'resample_timestep_range' in kwargs:
                resample_timestep_range = kwargs['resample_timestep_range']
            else:
                resample_timestep_range = [0.10 * steps, 1.00 * steps]
        else:
            resample_interval, jump_lens_list, resample_timestep_range = None, None, None


        time_steps, time_steps_prev = build_ddim_time_steps(
            self.alpha_t_bar,
            steps=steps,
            sample_method=sample_method,
        )

        if self.is_Inference and batch_idx is not None and batch_size is not None and dataset_len is not None:
            sample_num = min(batch_size * batch_idx, dataset_len)
            tqdm_bar = tqdm(zip(time_steps, time_steps_prev), colour="#6565b5", total=len(time_steps), ncols=120, desc=f'>>> {sample_num} / {dataset_len}')
        else:
            tqdm_bar = tqdm(zip(time_steps, time_steps_prev), colour="#6565b5", total=len(time_steps), ncols=80)
        
        for idx, (cur_t, cur_t_prev) in enumerate(tqdm_bar):

            cur_t_tensor = torch.full((x_t.shape[0],), int(cur_t.item()), device=x_t.device, dtype=torch.long)
            
        # Regenerate where mask is 1; preserve the blended context where it is 0.
            x_0_add_noise = self.q_sample(x_0, cur_t_tensor, noise=self._randn_like(x_0))
            x_t = x_0_add_noise * (1.0 - msks) + msks * x_t            

            x_t = self.sample_one_step(
                x_t,
                int(cur_t.item()),
                int(cur_t_prev.item()),
                eta,
                prompt_str,
                is_final_step=(idx == len(time_steps) - 1),
                **kwargs,
            )

            # after one denoising step, the current step index is idx + 1
            idx_add_one = min(idx + 1, steps - 1)

            cur_step = steps - 1 - idx
            if is_resample and cur_step % resample_interval == 0 and resample_timestep_range[0] <= cur_step <= resample_timestep_range[1]:
                for jump_len in jump_lens_list:

                    # jump back jump_len steps
                    jump_idx = max(idx_add_one - jump_len, 0)

                    tmp_x_t = x_t.clone()
                    tmp_t = torch.full((x_t.shape[0],), int(time_steps[idx_add_one].item()), device=x_t.device, dtype=torch.long)

                    jump_t = torch.full_like(tmp_t, int(time_steps[jump_idx].item()))

                    cur_a  = extract_diffusion_coefficient(self.alpha_t_bar, tmp_t, tmp_x_t.shape)     
                    jump_a = extract_diffusion_coefficient(self.alpha_t_bar, jump_t, tmp_x_t.shape)   

                    tmp_x_t = torch.sqrt(jump_a / cur_a) * tmp_x_t + torch.sqrt(1.0 - (jump_a / cur_a)) * self._randn_like(tmp_x_t)
                    tmp_t = jump_t.clone()   

                    # sample normally from jump_t back to time_step
                    for idx_forward in range(jump_idx, idx_add_one):

                        t_forward_tensor = torch.full((x_t.shape[0],), int(time_steps[idx_forward].item()), device=x_t.device, dtype=torch.long)
                        
                        # re-inject the known region at each step
                        x0_noise = self.q_sample(x_0, t_forward_tensor, noise=self._randn_like(x_0))
                        tmp_x_t = x0_noise * (1.0 - msks) + msks * tmp_x_t
                        
                        tmp_x_t = self.sample_one_step(tmp_x_t, int(time_steps[idx_forward].item()), int(time_steps_prev[idx_forward].item()), eta, prompt_str, **kwargs)

                    x_t = tmp_x_t.clone()

            if x_list is not None:
                x_list.append(x_t)

            tqdm_bar.set_postfix(ordered_dict={"step": idx + 1, "batch_size": len(x_t)})

        if x_list is not None: 
            return x_t, x_list
        else:
            return x_t    # [batch_size, channels, height, width]


