from contextlib import ExitStack
from functools import partial
import diffusers
from diffusers import AutoPipelineForText2Image, DDIMScheduler
import torch
import numpy as np
from typing import Optional, Callable, Any, overload
from PIL.Image import Image as PILImage
import PIL.Image
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
import re




def load_sdxl_lightning(config: dict, device: str, dtype: torch.dtype, local_files_only: bool):
    '''Load SDXL-Lightning model with specified number of steps.'''
    # SDXL-Lightning needs custom loading because it's designed replace only the unet of SDXL

    # load dependencies
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # config
    steps = config['steps']
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    if steps == 1:
        ckpt = "sdxl_lightning_1step_unet_x0.safetensors"
    elif steps in [2,4,8]:
        ckpt = f"sdxl_lightning_{steps}step_unet.safetensors"
    else:
        raise ValueError(f"Invalid number of steps: {steps}")

    # Load model
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet", local_files_only=local_files_only)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt, local_files_only=local_files_only), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", local_files_only=local_files_only).to(device)

    # fix sampler
    extra_kwargs = dict(prediction_type="sample") if steps == 1 else {}
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", **extra_kwargs, local_files_only=local_files_only)

    return pipe


@dataclass
class SDResult:
    prompt : str
    seed : int
    representations : 'SDRepresentation'
    images : list[PILImage]
    result_latent : torch.Tensor
    result_tensor : torch.Tensor
    result_image : PILImage
    def __repr__(self): return f'SDResult(prompt="{self.prompt}",seed={self.seed},...)'
    def to(self, device, dtype = None):
        '''Move tensors to device and cast to dtype. Does not do a deep copy.'''
        return SDResult(
            prompt = self.prompt,
            seed = self.seed,
            representations = self.representations.to(device, dtype),
            images = self.images,
            result_latent = self.result_latent.to(device, dtype),
            result_tensor = self.result_tensor.to(device, dtype),
            result_image = self.result_image,
        )


class SDRepresentation():
    '''Class to store representations extracted from SD.'''
    def __init__(self, data: dict[str, list[torch.Tensor]], seed: Optional[int] = None):
        # data.shape = (noise_steps, feature_dim, height, width)
        self.data = {k: torch.stack([v.squeeze(0) for v in vs]) for k, vs in data.items()}
        if not all(len(vs) == len(self.data[list(self.data.keys())[0]]) for vs in self.data.values()):
            raise ValueError('All representations must have the same number of timesteps.')
        self.seed = seed

    def __getattr__(self, key):
        if key in self.data: return self.data[key]
        raise AttributeError(f"'SDRepresentation' object has no attribute '{key}'")

    def __getitem__(self, key):
        return self.data[key]
    
    def __repr__(self):
        return 'SDRepresentation({' + ', '.join(f'"{k}": ...' for k in self.pos) + '})'
    
    def apply(self, fn: Callable, *args, **kwargs):
        '''Apply a function to all representations.'''
        return SDRepresentation({k: [fn(v, *args, **kwargs) for v in vs] for k, vs in self.data.items()})

    def _apply_binary(self, fn, other):
        if isinstance(other, SDRepresentation):
            assert self.pos == other.pos, 'Positions must be the same.'
            assert self.num_steps == other.num_steps, 'Number of steps must be the same.'
            return SDRepresentation({k: [getattr(v, fn)(o) for v, o in zip(vs, other[k])] for k, vs in self.data.items()})
        else:
            return self.apply(lambda x: getattr(x, fn)(other))

    def __add__(self, other): return self._apply_binary('__add__', other)
    def __radd__(self, other): return self._apply_binary('__radd__', other)
    def __sub__(self, other): return self._apply_binary('__sub__', other)
    def __rsub__(self, other): return self._apply_binary('__rsub__', other)
    def __mul__(self, other): return self._apply_binary('__mul__', other)
    def __rmul__(self, other): return self._apply_binary('__rmul__', other)
    def __truediv__(self, other): return self._apply_binary('__truediv__', other)
    def __rtruediv__(self, other): return self._apply_binary('__rtruediv__', other)
    def __floordiv__(self, other): return self._apply_binary('__floordiv__', other)
    def __rfloordiv__(self, other): return self._apply_binary('__rfloordiv__', other)
    def __mod__(self, other): return self._apply_binary('__mod__', other)
    def __rmod__(self, other): return self._apply_binary('__rmod__', other)
    def __pow__(self, other): return self._apply_binary('__pow__', other)
    def __rpow__(self, other): return self._apply_binary('__rpow__', other)

    @property
    def pos(self):
        return list(self.data.keys())

    @property
    def num_steps(self):
        return len(self[self.pos[0]])

    @property
    def device(self):
        return self.data[self.pos[0]][0].device

    @property
    def dtype(self):
        return self.data[self.pos[0]][0].dtype

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> 'SDRepresentation':
        '''Move representations to device and cast to dtype.'''
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        return SDRepresentation({k: [v.to(device=device, dtype=dtype) for v in vs] for k, vs in self.data.items()})

    def at(self, pos: list[str] | str | None = None, steps: list[int] | int | None = None) -> 'SDRepresentation':
        '''Select representations at specified positions and steps.'''
        if pos is None: pos = self.pos
        elif isinstance(pos, str): pos = [pos]
        if steps is None: steps = list(range(len(self[pos[0]])))
        elif isinstance(steps, int): steps = [steps]
        return SDRepresentation({k: [self[k][s] for s in steps] for k in pos})

    def concat(self) -> torch.Tensor:
        '''Concatenate all representations over positions and timesteps into a single tensor with the largest spatial size.'''
        # If the representation sizes are not multiples of each other, the bottom and right edges of the spatially larger representations will be 0-padded.
        max_spatial = np.array(max(self[x][0].shape[-2:] for x in self.pos))
        min_spatial = np.array(min(self[x][0].shape[-2:] for x in self.pos))
        spatial = min_spatial
        while (max_spatial > spatial).any(): spatial *= 2
        num_channels = sum(self[x][0].shape[0] for x in self.pos) * self.num_steps
        repr_full = torch.zeros((num_channels, *spatial), device=self.device, dtype=self.dtype)
        i = 0
        for p in self.pos:
            r = self[p].flatten(end_dim=1)  # merge timesteps into channel dimension
            c, w, h = r.shape
            tmp = r.repeat_interleave(spatial[0]//w, dim=-2).repeat_interleave(spatial[1]//h, dim=-1)
            repr_full[i:i+c, :tmp.shape[-2], :tmp.shape[-1]] = tmp
            i += c
        return repr_full

    def cosine_similarity(self, other: 'SDRepresentation') -> torch.Tensor:
        '''Compute cosine similarity between representations.'''
        assert self.pos == other.pos, 'Positions must be the same.'
        assert self.num_steps == other.num_steps, 'Number of steps must be the same.'
        a = self.concat().to(dtype=torch.float32)  # upcast to float32 to prevent overflow in dot product
        a = a / a.norm(dim=0, keepdim=True)
        b = other.concat().to(dtype=torch.float32, device=self.device)
        b = b / b.norm(dim=0, keepdim=True)
        return (a.flatten(1).T @ b.flatten(1)).reshape((*a.shape[1:], *b.shape[1:]))



class SD:
    '''
    Usage:
    ```
    sd = SD('SD1.5')
    result = sd('a cat')
    result.result_image
    ```

    In low-vram settings, and especially when using FLUX, you might want to use `sd.quantize()` to reduce VRAM usage.
    '''
    known_models = {
        'SD1.1': {
            'name': 'CompVis/stable-diffusion-v1-1',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD1.2': {
            'name': 'CompVis/stable-diffusion-v1-2',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD1.3': {
            'name': 'CompVis/stable-diffusion-v1-3',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD1.4': {
            'name': 'CompVis/stable-diffusion-v1-4',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD1.5': {
            # 'name': 'runwayml/stable-diffusion-v1-5',  # Runwayml deleted their repo
            'name': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD2.0': {
            'name': 'stabilityai/stable-diffusion-2',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD2.1': {
            'name': 'stabilityai/stable-diffusion-2-1',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD-Turbo': {
            'name': 'stabilityai/sd-turbo',
            'steps': 4,
            'guidance_scale': 0.0,
        },
        'SDXL': {
            'name': 'stabilityai/stable-diffusion-xl-base-1.0',
            'steps': 40,
            'guidance_scale': 5.0,  # TODO: is this correct?
        },
        'SDXL-Turbo': {
            'name': 'stabilityai/sdxl-turbo',
            'steps': 4,
            'guidance_scale': 0.0,
        },
        **{f'SDXL-Lightning-{steps}step': {
            'name': f'ByteDance/sdxl-lightning-{steps}step',
            'steps': steps,
            'guidance_scale': 0.0,
            'load_fn': load_sdxl_lightning,
        } for steps in [1,2,4,8]},
        'SD3': {
            'name': 'stabilityai/stable-diffusion-3-medium-diffusers',
            'steps': 28,
            'guidance_scale': 7.0,
        },
        'FLUX-dev': {
            'name': 'black-forest-labs/FLUX.1-dev',
            'steps': 28,
            'guidance_scale': 3.5,
        },
        'FLUX-schnell': {
            'name': 'black-forest-labs/FLUX.1-schnell',
            'steps': 4,
            'guidance_scale': 0.0,
        },
    }

    def __init__(
            self,
            model_name: str,
            device: str = 'auto',
            disable_progress_bar: bool = False,
            config: Optional[dict] = None,
            local_files_only : bool = False,
        ):
        '''
        Args:
            model_name: The name of the model to use.
            device: Device to use like 'cuda' or 'cpu'.
            disable_progress_bar: Disable tqdm progress bar during inference.
            config: Optional config dict to override the default config.
            local_files_only: Prevent downloading of the model weights.
        '''
        # fuzzy match model name to known models (e.g. 'sd15' -> 'SD1.5')
        self.model_name = {re.sub(r'\s|\.|-|_','',k).lower(): k for k in self.known_models.keys()}.get(re.sub(r'\s|\.|-|_','',model_name).lower(), model_name)

        # setup model config
        self.config = self.known_models.get(self.model_name, {'name': self.model_name}) if config is None else config

        # determine device and dtype
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32 if self.device == 'cpu' else torch.float16

        # setup pipeline
        progressbar_enabled = diffusers.utils.logging.is_progress_bar_enabled()
        if progressbar_enabled and disable_progress_bar: diffusers.utils.logging.disable_progress_bar()
        # TODO: also disable transformers progress bar, as some model submodules use it
        self.pipeline: 'diffusers.StableDiffusionPipeline | diffusers.StableDiffusionXLPipeline | diffusers.StableDiffusion3Pipeline | diffusers.FluxPipeline | Any'
        if 'load_fn' in self.config:
            self.pipeline = self.config['load_fn'](config=self.config, device=self.device, dtype=self.dtype, local_files_only=local_files_only)
        else:
            try:
                self.pipeline = AutoPipelineForText2Image.from_pretrained(self.config['name'], torch_dtype=torch.float16, local_files_only=local_files_only).to(self.device, dtype=self.dtype)
            except OSError as e:
                if 'is not a local folder and is not a valid model identifier' in str(e):
                    raise ValueError(f"Model `{self.config['name']}` isn't a known model and can also not be found on huggingface. Known models: \n" + '\n'.join(self.known_models.keys()))
                else:
                    raise e
        # restore progress bar status
        if progressbar_enabled and disable_progress_bar: diffusers.utils.logging.enable_progress_bar()

        # disable tqdm progress bar
        self.disable_progress_bar = disable_progress_bar
        if disable_progress_bar and hasattr(self.pipeline, 'set_progress_bar_config'):
            self.pipeline.set_progress_bar_config(disable=True)

        # upcast vae if necessary (SDXL models require float32)
        if hasattr(self.pipeline, 'upcast_vae') and self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast:
            self.pipeline.upcast_vae()

        # find out available extract positions
        self.available_extract_positions = []
        if hasattr(self.pipeline, 'unet'):
            for name, obj in self.pipeline.unet.named_children():
                if any(x in name for x in ['time_', '_norm', '_act', '_embedding']): continue
                self.available_extract_positions += [f'{name}[{i}]' for i in range(len(obj))] if isinstance(obj, torch.nn.ModuleList) else [name]
            self.available_extract_positions.sort(key=lambda x: ['conv_in', 'down_blocks', 'mid_block', 'up_blocks', 'conv_out'].index(x.split('[')[0]))
        elif hasattr(self.pipeline, 'transformer'):
            self.available_extract_positions = [f'transformer_blocks[{i}]' for i in range(len(self.pipeline.transformer.transformer_blocks))]
        else:
            print('WARNING: No unet/transformer found. Available extract positions will be empty.')

        # init cache variables
        self._representation_shapes = None
        self._ddim_scheduler = None
        self._cached_prompt_embeds = {}

    def get_representation_shapes(self) -> tuple[dict[str, tuple], tuple]:
        '''Return the shape of the representations at each extract position.'''
        if self._representation_shapes is None:
            if self.model_name not in self.known_models: raise ValueError(f'Cannot determine representation shapes for unknown model {self.model_name}.')
            if self.model_name in ['SD3', 'FLUX-dev', 'FLUX-schnell']:
                fake_img = PIL.Image.new('RGB', (1024, 1024))
                reprs = self.img2repr(fake_img, extract_positions=self.available_extract_positions, step=1)
                self._representation_shapes = {k: tuple(v.shape) for k, v in reprs.data.items()}, (3, 1024, 1024)
            else:
                result = self('', extract_positions=self.available_extract_positions, steps=1)
                def helper(t): return tuple(helper(t_) for t_ in t) if isinstance(t, tuple) else tuple(t.shape)
                self._representation_shapes = {k: helper(v[0]) for k,v in result.representations.data.items()}, tuple(result.result_tensor.shape)
        return self._representation_shapes
    
    def quantize(self, quantization_modules: list[str] | None = None, quantization_type: str = 'qfloat8', model_cpu_offload: bool = False, sequential_cpu_offload: bool = False):
        '''Optimize VRAM usage of the model.

        Args:
            quantization_modules: List of modules to quantize, e.g. 'transformer', 'unet', 'vae'.
            quantization_type: Type of quantization to use, e.g. 'qfloat8'.
            model_cpu_offload: Offload models to CPU.
            sequential_cpu_offload: Offload models on a submodule level (rather than model level).

        Example:
        ```
        flux = SD('FLUX-schnell')
        flux.quantize(['transformer', 'text_encoder_2'], model_cpu_offload=True)
        ```
        '''
        if quantization_modules:
            for name in quantization_modules:
                try:
                    from optimum.quanto import quantize, freeze
                except ImportError:
                    raise ImportError('Cannot optimize VRAM usage. Please install: `pip install optimum-quanto`.')
                quantize(getattr(self.pipeline, name), weights=quantization_type)
                freeze(getattr(self.pipeline, name))
        if model_cpu_offload:
            # offload models to CPU
            self.pipeline.enable_model_cpu_offload()
        if sequential_cpu_offload:
            # offloads modules on a submodule level (rather than model level)
            # this seems to interfere with img2repr
            self.pipeline.enable_sequential_cpu_offload()

    @torch.no_grad()
    def vae_decode(self, latents):
        vae = self.pipeline.vae
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype) / vae.config.scaling_factor
        image = vae.decode(latents).sample
        return image

    def __call__(
            self,
            prompt: str,
            steps: Optional[int] = None,
            guidance_scale: Optional[float] = None,
            seed: Optional[int] = None,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
            modification: Optional[Callable[[Any,Any,Any,str],Optional[torch.Tensor]]] = None,
            preserve_grad: bool = False,
            extract_positions: list[str] = [],
        ) -> 'SDResult':

        # use default values if not specified
        if steps is None:
            if 'steps' in self.config: steps = self.config['steps']
            else: raise ValueError('steps must be specified')
        if guidance_scale is None:
            if 'guidance_scale' in self.config: guidance_scale = self.config['guidance_scale']
            else: raise ValueError('guidance_scale must be specified')

        # random seed if not specified
        seed = seed if seed != None else int(torch.randint(0, 2**32, (1,)).item())

        if self.model_name in ['SD3', 'FLUX-dev', 'FLUX-schnell']:
            if preserve_grad or modification is not None or len(extract_positions) > 0:
                raise ValueError(f'{self.model_name} support for gradient preservation, modifications, or extract positions is not implemented yet.')
            result_images = self.pipeline(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, width=width, height=height)
            return SDResult(
                prompt=prompt,
                seed=seed,
                representations=None,
                images=None,
                result_latent=None,
                result_tensor=None,
                result_image=result_images.images[0],
            )
        else:
            # TODO: fix the following line for gradient preservation
            call_pipeline = (lambda *args, **kwargs: self.pipeline.__class__.__call__.__wrapped__(self.pipeline, *args, **kwargs)) if preserve_grad else self.pipeline

            # variables to store extracted results in
            representations = {pos: [] for pos in extract_positions}
            images = []

            def latents_callback(pipe, step_index, timestep, callback_kwargs):
                '''callback function to extract intermediate images'''
                latents = callback_kwargs['latents']
                image = (self.vae_decode(latents)[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                images.extend(self.pipeline.numpy_to_pil(image))
                return callback_kwargs

            # run pipeline
            with ExitStack() as stack, torch.no_grad():
                # setup hooks to extract representations
                for extract_position in self.available_extract_positions:
                    def get_repr(module, input, output, extract_position):
                        if extract_position in extract_positions:
                            if isinstance(output, tuple):
                                output = output[0]  # TODO: is it good to always take the first output and ignore the rest?
                            representations[extract_position].append(output)
                        if modification:
                            return modification(module, input, output, extract_position)
                    # eval is unsafe. Do not use in production.
                    stack.enter_context(eval(f'unet.{extract_position}', {'__builtins__': {}, 'unet': self.pipeline.unet}).register_forward_hook(partial(get_repr, extract_position=extract_position)))
                
                # run pipeline
                result = self.pipeline(
                    prompt,
                    width = width,
                    height = height,
                    num_inference_steps = steps,
                    guidance_scale = guidance_scale,
                    callback_on_step_end = latents_callback,
                    callback_on_step_end_tensor_inputs = ['latents'],
                    generator = torch.Generator(self.device).manual_seed(seed),
                    output_type = 'latent',
                )

        # cast images to same dtype as vae
        result_tensor = self.vae_decode(result.images)
        result_image = self.pipeline.image_processor.postprocess(result_tensor.detach(), output_type='pil')

        # return results
        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=SDRepresentation(representations, seed),
            images=images,
            result_latent=result.images[0],
            result_tensor=result_tensor[0],
            result_image=result_image[0],
        )

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        ''' Encode an image to latents.

        Args:
            img: The (PIL) image to encode.
        '''
        vae = self.pipeline.vae
        vae_dtype = next(vae.modules()).dtype
        if self.model_name in ['SD3', 'FLUX-dev', 'FLUX-schnell']:
            img_tensor = self.pipeline.image_processor.preprocess(images).to(device=self.device, dtype=vae_dtype)
            return (vae.encode(img_tensor).latent_dist.sample().to(dtype=self.dtype) - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            img_tensor = torch.tensor(np.array(images), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
            return vae.encode(img_tensor.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=self.dtype) * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        ''' Decode latents to an image.

        Args:
            latents: The latents to decode.
        '''
        pipe = self.pipeline
        if self.model_name in ['SD3', 'FLUX-dev', 'FLUX-schnell']:
            tmp = pipe.vae.decode((latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor, return_dict=False)
            return pipe.image_processor.postprocess(tmp)  # type: ignore
        else:
            return pipe.numpy_to_pil(self.vae_decode(latents).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy())

    @property
    def ddim_scheduler(self):
        if self._ddim_scheduler is None:
            self._ddim_scheduler = DDIMScheduler.from_pretrained(self.config['name'], subfolder='scheduler')
        return self._ddim_scheduler

    @torch.no_grad()
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompts: list[str], spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]:
        '''Convert image to representations at specified extract positions.

        Args:
            images: List of PIL images with same sizes to convert to extract representations for.
            extract_positions: List of extract positions to return representations for.
            step: Timestep determining the amount of noise to add. Noise increases with timestep. Must be in [0, 999]. Even step 0 might add noise, depending on the scheduler.
            prompt: Prompt to use for the image. If a list is provided, each image will be paired with the corresponding prompt.
            spatial_avg: If True, spatially average the representations.
            output_device: Device to move the representations to.
            seed: Seed for random number generation. If None, a random seed will be used.

        Returns:
            Dictionary with extract positions as keys and the corresponding representations as values.
        '''
        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # Set seed if provided
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)
        # It might be good to set `torch.use_deterministic_algorithms(True)`, but some operations are not available then. It also doesn't seem to make any difference.

        # encode image
        images_resized = [img.resize((resize, resize)) if resize is not None else img for img in images]
        latents = self.encode_latents(images_resized)  # this gives slightly different results for different batch sizes
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

        # TODO: maybe cache prompt_embeds (difficulty is that encode_prompt is different over models)

        if self.model_name == 'SD3':
            pipe.scheduler.set_timesteps(1000, device=self.device)
            timestep = pipe.scheduler.timesteps[999 - step]
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None)  # type: ignore
            latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)

            # extract representations
            with ExitStack() as stack, torch.no_grad():
                for extract_position in extract_positions:
                    def hook_fn(module, input, output, extract_position):
                            representations[extract_position] = output[1].to(output_device)
                    stack.enter_context(eval(f'model.{extract_position}', {'__builtins__': {}, 'model': pipe.transformer}).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
                pipe.transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]).to(device=self.device), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds)

            # fix representation shape
            num_tokens = next(iter(representations.values())).shape[1]
            potential_repr_shapes = [(i, num_tokens//i) for i in range(1, num_tokens) if num_tokens % i == 0]
            repr_shape = min(potential_repr_shapes, key=lambda x: abs(x[1]/x[0] - width / height))
            representations = {p: r.reshape(batch_size, *repr_shape, 1536).permute(0, 3, 1, 2) for p, r in representations.items()}

        elif 'FLUX' in self.model_name:
            # timestep
            pipe.scheduler.set_timesteps(1000, device=pipe.device)
            # We skip all the scheduler timestep calculation, as it seems to just result in `step`. So we just use this directly. +1 because the timesteps are 0-indexed.
            timestep = torch.tensor([step], device=pipe.device) + 1

            # prepare and noise latents
            latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompts, prompt_2=None)  # type: ignore
            latents, latent_image_ids = pipe.prepare_latents(batch_size, pipe.transformer.config.in_channels // 4, width, height, prompt_embeds.dtype, pipe.device, generator=torch.manual_seed(seed), latents=latents)
            latents = pipe._pack_latents(latents, *latents.shape)

            # extract representations
            with ExitStack() as stack, torch.no_grad():
                for extract_position in extract_positions:
                    def hook_fn(module, input, output, extract_position):
                        representations[extract_position] = output[1].to(output_device)
                    stack.enter_context(eval(f'model.{extract_position}', {'__builtins__': {}, 'model': pipe.transformer}).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
                pipe.transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]).to(latents.dtype)/1000, guidance=None, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids)

            # fix representation shape
            num_tokens = next(iter(representations.values())).shape[1]
            potential_repr_shapes = [(i, num_tokens//i) for i in range(1, num_tokens) if num_tokens % i == 0]
            repr_shape = min(potential_repr_shapes, key=lambda x: abs(x[1]/x[0] - width / height))
            representations = {p: r.reshape(batch_size, *repr_shape, 3072).permute(0, 3, 1, 2) for p, r in representations.items()}

        else:
            # apply noise
            timestep = torch.tensor(step, dtype=torch.long, device=self.device)
            latents = self.ddim_scheduler.add_noise(latents, noise, timestep)

            # scale latents
            # TODO: this is from SD1.5 (where it's not used), is it also necessary for other models?
            latents = pipe.scheduler.scale_model_input(latents, timestep)

            # create empty prompt embeddings
            prompt_embeds, *_ = self.pipeline.encode_prompt(prompt=prompts, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)  # type: ignore

            # setup unet config
            pipe.unet.config.addition_embed_type = 'nothing_at_all'

            with ExitStack() as stack, torch.no_grad():
                for extract_position in extract_positions:
                    def hook_fn(module, input, output, extract_position):
                        # print(extract_position, print_shape(output))
                        if isinstance(output, tuple):
                            output = output[0]  # TODO: is it good to always take the first output and ignore the rest?
                        if spatial_avg:
                            output = output.mean(dim=(2, 3))
                        representations[extract_position] = output.to(output_device)
                    # eval is unsafe. Do not use in production.
                    stack.enter_context(eval(f'model.{extract_position}', {'__builtins__': {}, 'model': pipe.unet}).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
                pipe.unet(latents, timestep, encoder_hidden_states=prompt_embeds)

        return [SDRepresentation({p: r[i,None,:,:,:] for p, r in representations.items()}, seed) for i in range(batch_size)]
        
    @overload
    def img2repr(self, data: PILImage | np.ndarray | str, extract_positions: list[str], step: int, resize: int | None = None, prompt: str = '', spatial_avg: bool = False, output_device: str = 'cpu', batch_size: int = 1, seed: Optional[int] = None) -> SDRepresentation: ...

    @overload
    def img2repr(self, data: list[PILImage | np.ndarray | str], extract_positions: list[str], step: int, resize: int | None = None, prompt: str | list[str] = '', spatial_avg: bool = False, output_device: str = 'cpu', batch_size: int = 1, seed: Optional[int] = None) -> list[SDRepresentation]: ...

    def img2repr(self, data: PILImage | np.ndarray | str | list[PILImage | np.ndarray | str], extract_positions: list[str], step: int, resize: int | None = None, prompt: str | list[str] = '', spatial_avg: bool = False, output_device: str = 'cpu', batch_size: int = 20, seed: Optional[int] = None):
        '''Convert image to representations at specified extract positions.

        Args:
            img: PIL image(s) to convert to extract representations for.
            extract_positions: List of extract positions to return representations for.
            step: Timestep determining the amount of noise to add. Noise increases with timestep. Must be in [0, 999]. Even step 0 might add noise, depending on the scheduler.
            prompt: Prompt to use for the image. If a list is provided, each image will be paired with the corresponding prompt.
            spatial_avg: If True, spatially average the representations.
            output_device: Device to move the representations to.
            batch_size: Number of images to process at once. The returned representations might differ slightly depending on the batch size. For best reproducability, set batch_size=1.
            seed: Seed for random number generation. If None, a random seed will be used.

        Returns:
            Dictionary with extract positions as keys and the corresponding representations as values.

        Examples:
        ```
        sd = SD('SD1.5')

        # extract h-space (mid_block repr.) for a single image with a specific seed
        hspace = sd.img2repr('./my_image.jpg', ['mid_block'], 30, seed=42)['mid_block']

        # extract representations for a full dataset
        dataset = sd.img2repr(
            datasets.load_dataset('cifar10'),
            extract_positions = sd.available_extract_positions,
            step = 100,
            spatial_avg = True,
        )
        dataset.save_to_disk('SD-Turbo_cifar10_reprs')
        '''
        single = not isinstance(data, list)
        data_list = [data] if single else data
        if len(data_list) == 0: return []
        if not all(isinstance(d, (PILImage, np.ndarray, str)) for d in data_list): raise ValueError(f'Unsupported type for data')
        images_pil = [PIL.Image.open(d) if isinstance(d, str) else PIL.Image.fromarray(d) if isinstance(d, np.ndarray) else d for d in data_list]
        images_rgb = [img.convert('RGB') for img in images_pil]
        if not isinstance(prompt, list): prompt = [prompt] * len(images_pil)
        if not len(prompt) == len(images_pil): raise ValueError('Number of prompts must match number of images')
        if not all(isinstance(p, str) for p in prompt): raise ValueError('Prompts must be strings')
        if not all(img.size == images_rgb[0].size for img in images_rgb): batch_size = 1
        representations = []
        for i in trange(0, len(images_rgb), batch_size, desc='Extracting representations for list of images', disable=single or self.disable_progress_bar):
            representations.extend(self._img2repr(images_rgb[i:i+batch_size], extract_positions, step, resize, prompt[i:i+batch_size], spatial_avg, output_device, seed))
        return representations[0] if single else representations
