"""Wan sampling and blockwise TACache velocity prediction."""
from contextlib import contextmanager
from dataclasses import replace
import gc
import math
import random
import sys
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import torch
from tacache.core import TACacheWrapper, TimeDecision, enable_tacache


def _block_perp(values, references, blocks, eps):
    """Project and normalize independently within each latent-time block."""
    outputs = []
    for value, reference in zip(values, references):
        value_f, reference_f = value.float(), reference.float()
        out = torch.zeros_like(value_f)
        bounds = torch.linspace(0, value.shape[1], blocks + 1, dtype=torch.long).tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            x = value_f[:, start:end].reshape(-1)
            v = reference_f[:, start:end].reshape(-1)
            perpendicular = x - ((x @ v) / (v @ v).clamp_min(eps)) * v
            out[:, start:end] = (perpendicular / perpendicular.norm().clamp_min(eps)).reshape_as(out[:, start:end])
        outputs.append(out.to(value.dtype))
    return outputs


class WanTACacheWrapper(TACacheWrapper):
    """Wan time-block prediction; calibration and table building are shared."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_history = None

    def _reset_time_selection(self):
        super()._reset_time_selection()
        self.time_decision = TimeDecision('sigma', 'native_t', 1.,
            'Use the actual scheduler sigma indexed by its native timestep; no linear t/1000 approximation')

    def _after_actual_forward(self, sample, stream, step_idx):
        """Update the direction and skip count after a real model forward."""
        current = [v.detach() for v in sample]
        previous = self.next_history if self.next_history is not None else self.v_state[stream]
        self.next_history = None
        self.geo_mode[stream] = 'video'
        self.skip_remaining[stream] = 0
        if previous is not None and step_idx not in (0, 1, self.steps_per_stream-2, self.steps_per_stream-1):
            h = self._get_h(stream, step_idx)
            if h > 1:
                delta = [v-p for v,p in zip(current, previous)]
                self.u_state[stream] = _block_perp(delta, previous, self.config.t_blocks, self.config.eps)
                self.skip_remaining[stream] = h-1
        self.v_state[stream] = current

    def _predict_velocity(self, velocity, direction, transition):
        """Predict the next velocity independently for each time block."""
        table = self.tables[0]
        direction = _block_perp(direction, velocity, self.config.t_blocks, self.config.eps)
        predictions = []
        for v, u in zip(velocity, direction):
            bounds = torch.linspace(0, v.shape[1], self.config.t_blocks+1, dtype=torch.long).tolist()
            out = torch.zeros_like(v)
            dt = torch.tensor(float(table['dt_update_used'][transition]), device=v.device, dtype=v.dtype)
            for block, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                vb, ub = v[:, start:end], u[:, start:end]
                k = torch.tensor(float(table['k_off_used'][transition, block]), device=v.device, dtype=v.dtype)
                alpha = torch.tensor(float(table['r_dt_over_v_off_used'][transition, block]), device=v.device, dtype=v.dtype)
                norm = vb.reshape(-1).norm().clamp_min(self.config.eps)
                out[:, start:end] = vb * torch.exp(k*dt) + alpha * norm * ub
            predictions.append(out)
        return predictions, direction

    def _advance_tacache_state(self, stream, step_idx):
        predicted, direction = self._predict_velocity(self.v_state[stream], self.u_state[stream], step_idx-1)
        self.v_state[stream], self.u_state[stream] = predicted, direction
        self.skip_remaining[stream] -= 1
        self.skip_counts[stream] += 1
        if self.skip_remaining[stream] == 0:
            self.next_history, self.u_state[stream] = self._predict_velocity(predicted, direction, step_idx)
        return predicted

def generate(self,
             input_prompt,
             mode,
             config,
             skip_table=None,
             size=(1280, 720),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             n_prompt="",
             seed=-1,
             offload_model=True):
    """Generate a Wan T2V video with an optional TACache."""
    from wan import text2video as wan_sampler

    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            scheduler = wan_sampler.FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1,
                use_dynamic_shifting=False)
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        else:
            scheduler = wan_sampler.FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1,
                use_dynamic_shifting=False)
            sigmas = wan_sampler.get_sampling_sigmas(sampling_steps, shift)
            wan_sampler.retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)

        def sigma_at_timestep(t):
            index = scheduler.index_for_timestep(t, scheduler.timesteps)
            return float(scheduler.sigmas[index])

        config = replace(config, scheduler=scheduler, sigma_fn=sigma_at_timestep)
        timesteps = scheduler.timesteps

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with enable_tacache(self.model, mode, config, skip_table=skip_table,
                layout="CTHW", wrapper_cls=WanTACacheWrapper) as cache:
            tacache = cache if mode == "infer" else None
            if tacache is not None:
                # Validate the whole native time grid once, with one CPU copy.
                for step_idx, value in enumerate(timesteps.detach().cpu().tolist()):
                    tacache._validate_runtime_time(value, step_idx)
                model_forward = tacache.original_method
            else:
                model_forward = self.model

            for step_idx, t in enumerate(tqdm(timesteps)):
                # Inference: predict a cached velocity or run both CFG branches.
                # Calibration runs both branches and records their combined velocity.
                if tacache is not None and tacache._can_skip(0, step_idx):
                    noise_pred = tacache._advance_tacache_state(0, step_idx)[0]
                else:
                    timestep = torch.stack([t])
                    self.model.to(self.device)
                    noise_pred_cond = model_forward(latents, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model_forward(latents, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                    if tacache is not None:
                        sample = tacache.extract_output([noise_pred])
                        tacache._after_actual_forward(sample, 0, step_idx)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                if tacache is not None:
                    # Account for both CFG branches; neither enters the hook.
                    tacache.sampler_calls += 2
                    tacache.cnt += 1
                    tacache._finish_forward_step()

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return (videos[0] if self.rank == 0 else None), cache

def save_video(video, path):
    """Save a C,T,H,W video using 16 FPS and libx264."""
    import imageio
    frames = video.clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    with imageio.get_writer(str(path), fps=16, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
