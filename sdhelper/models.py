from contextlib import ExitStack
from functools import partial
import diffusers
from diffusers import AutoPipelineForText2Image, DDIMScheduler
import torch
import numpy as np
from typing import Optional, Callable, Any, overload
from PIL.Image import Image as PILImage
import PIL.Image
from tqdm.autonotebook import trange
import re
from abc import ABC, abstractmethod

from .data import SDRepresentation, SDResult


def _get_module_by_path(model: Any, path: str) -> Any:
    """Safely navigate to a module using a path string."""
    current = model
    for name, index in re.findall(r'(\w+)(?:\[(\d+)\])?', path):
        try:
            current = getattr(current, name)
            if index:
                current = current[int(index)]
        except AttributeError:
            raise AttributeError(f"Extract position `{path}` not found`: Attribute `{name}` not available.")
        except IndexError:
            raise IndexError(f"Extract position `{path}` not found`: Index `{index}` out of range.")
    return current


def _normalize_model_name(name: str) -> str:
    """Normalize model name for matching (lowercase, no separators)."""
    return re.sub(r'\s|\.|-|_', '', name).lower()


def _get_all_subclasses(cls: type) -> list[type]:
    """Recursively get all subclasses of a class."""
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_all_subclasses(subclass))
    return subclasses


def SD(name: str, device: str = 'auto', disable_progress_bar: bool = False, local_files_only: bool = False) -> 'SDBase':
    """Factory function to create a Stable Diffusion model instance by name.

    Args:
        name: Model name (e.g., 'SD1.5', 'FLUX-schnell', 'SDXL-Turbo').
              Name matching is case-insensitive and ignores separators.
        device: Device to run the model on (e.g., 'cuda', 'cpu', 'mps').
        disable_progress_bar: Whether to disable the progress bar.
        local_files_only: Whether to only use local files.

    Returns:
        An instance of the requested SD model.

    Raises:
        ValueError: If the model name is not found.

    Example:
        >>> model = SD('SD1.5', device='cuda')
        >>> model = SD('FLUX-schnell', disable_progress_bar=True)
    """
    target = _normalize_model_name(name)

    # Build registry from all SDBase subclasses
    registry: dict[str, type[SDBase]] = {}
    for cls in _get_all_subclasses(SDBase):
        if hasattr(cls, 'name') and isinstance(cls.name, str):
            normalized = _normalize_model_name(cls.name)
            registry[normalized] = cls

    if target not in registry:
        available = sorted(set(cls.name for cls in registry.values()))
        raise ValueError(f"Model `{name}` not found. Available models: {available}")

    return registry[target](device=device, disable_progress_bar=disable_progress_bar, local_files_only=local_files_only)


class SDBase:
    """Base class for (Stable) Diffusion models."""
    def __init__(self, device: str = 'auto', disable_progress_bar: bool = False, local_files_only: bool = False):
        self.local_files_only = local_files_only

        # determine device and dtype
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.dtype = torch.float32 if self.device == 'cpu' else torch.float16

        # setup pipeline
        progressbar_enabled = diffusers.utils.logging.is_progress_bar_enabled()
        if progressbar_enabled and disable_progress_bar: diffusers.utils.logging.disable_progress_bar()
        self.pipeline: 'diffusers.StableDiffusionPipeline | diffusers.StableDiffusionXLPipeline | diffusers.StableDiffusion3Pipeline | diffusers.FluxPipeline | diffusers.Flux2Pipeline | Any'
        if hasattr(self, '_load_pipeline'):
            self._load_pipeline()
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.full_name, local_files_only=local_files_only).to(self.device, dtype=self.dtype)
        # restore progress bar status
        if progressbar_enabled and disable_progress_bar: diffusers.utils.logging.enable_progress_bar()

        # disable tqdm progress bar
        self.disable_progress_bar = disable_progress_bar
        if disable_progress_bar and hasattr(self.pipeline, 'set_progress_bar_config'):
            self.pipeline.set_progress_bar_config(disable=True)

        # init cache variables
        self._representation_shapes = None
        self._ddim_scheduler = None
        self._cached_prompt_embeds = {}

    def __call__(self, prompt: str, steps: Optional[int] = None, guidance_scale: Optional[float] = None, seed: Optional[int] = None, *, width: Optional[int] = None, height: Optional[int] = None, modification: Optional[Callable[[Any,Any,Any,str],Optional[torch.Tensor]]] = None, extract_positions: list[str] = []) -> 'SDResult':
        if steps is None: steps = self.steps
        if guidance_scale is None: guidance_scale = self.guidance_scale
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        return self._generate(prompt, steps, guidance_scale, seed, width=width, height=height, modification=modification, extract_positions=extract_positions)

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

    @abstractmethod
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor: ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]: ...

    @abstractmethod
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompt: str, spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]: ...

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
            extract_positions = ['up_blocks[1]'],
            step = 100,
            spatial_avg = True,
        )
        dataset.save_to_disk('SD-Turbo_cifar10_up1_representations')
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


class SDUnet(SDBase, ABC):
    """base class for SD models with a U-Net architecture"""

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        vae = self.pipeline.vae
        vae_dtype = next(vae.modules()).dtype
        img_tensor = torch.tensor(np.array(images), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        return vae.encode(img_tensor.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=self.dtype) * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        ''' Decode latents to an image.

        Args:
            latents: The latents to decode.
        '''
        return self.pipeline.numpy_to_pil(self.vae_decode(latents).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy())

    @property
    def ddim_scheduler(self):
        if self._ddim_scheduler is None:
            self._ddim_scheduler = DDIMScheduler.from_pretrained(self.full_name, subfolder='scheduler')
        return self._ddim_scheduler

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification: Optional[Callable[[Any,Any,Any,str],Optional[torch.Tensor]]] = None, extract_positions: list[str] = []) -> 'SDResult':

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
            for extract_position in extract_positions:
                def get_repr(module, input, output, extract_position):
                    if isinstance(output, tuple):
                        output = output[0]  # TODO: is it good to always take the first output and ignore the rest?
                    representations[extract_position].append(output)
                    if modification:
                        return modification(module, input, output, extract_position)
                stack.enter_context(_get_module_by_path(self.pipeline.unet, extract_position).register_forward_hook(partial(get_repr, extract_position=extract_position)))

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
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompts: list[str], spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        representations = {}

        # Set seed if provided
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)

        # encode image
        images_resized = [img.resize((resize, resize)) if resize is not None else img for img in images]
        latents = self.encode_latents(images_resized)  # this gives slightly different results for different batch sizes
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

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
                stack.enter_context(_get_module_by_path(pipe.unet, extract_position).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
            pipe.unet(latents, timestep, encoder_hidden_states=prompt_embeds)

        return [SDRepresentation({p: r[i,None,:,:,:] for p, r in representations.items()}, seed) for i in range(batch_size)]


class SD1_1(SDUnet):
    name = 'SD1.1'
    full_name = 'CompVis/stable-diffusion-v1-1'
    steps = 50
    guidance_scale = 7.5


class SD1_2(SDUnet):
    name = 'SD1.2'
    full_name = 'CompVis/stable-diffusion-v1-2'
    steps = 50
    guidance_scale = 7.5


class SD1_3(SDUnet):
    name = 'SD1.3'
    full_name = 'CompVis/stable-diffusion-v1-3'
    steps = 50
    guidance_scale = 7.5


class SD1_4(SDUnet):
    name = 'SD1.4'
    full_name = 'CompVis/stable-diffusion-v1-4'
    steps = 50
    guidance_scale = 7.5


class SD1_5(SDUnet):
    name = 'SD1.5'
    # full_name = 'runwayml/stable-diffusion-v1-5',  # Runwayml deleted their repo
    full_name = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    steps = 50
    guidance_scale = 7.5


class SD2_0(SDUnet):
    name = 'SD2.0'
    full_name = 'stabilityai/stable-diffusion-2'
    steps = 50
    guidance_scale = 7.5


class SD2_1(SDUnet):
    name = 'SD2.1'
    full_name = 'stabilityai/stable-diffusion-2-1'
    steps = 50
    guidance_scale = 7.5


class SD_Turbo(SDUnet):
    name = 'SD-Turbo'
    full_name = 'stabilityai/sd-turbo'
    steps = 4
    guidance_scale = 0.0


class SDXL(SDUnet):
    name = 'SDXL'
    full_name = 'stabilityai/stable-diffusion-xl-base-1.0'
    steps = 40
    guidance_scale = 5.0  # TODO: is this correct?


class SDXL_Turbo(SDUnet):
    name = 'SDXL-Turbo'
    full_name = 'stabilityai/sdxl-turbo'
    steps = 4
    guidance_scale = 0.0


class SDXL_Lightning_base(SDUnet, ABC):
    """Base class for SDXL-Lightning models."""
    guidance_scale = 0.0

    def _load_pipeline(self):
        # SDXL-Lightning needs custom loading because it's designed to replace only the unet of SDXL
        # Based on: https://huggingface.co/ByteDance/SDXL-Lightning

        # load dependencies
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # config
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        if self.steps == 1:
            ckpt = "sdxl_lightning_1step_unet_x0.safetensors"
        elif self.steps in [2, 4, 8]:
            ckpt = f"sdxl_lightning_{self.steps}step_unet.safetensors"
        else:
            raise ValueError(f"Invalid number of steps: {self.steps}. Supported steps: 1, 2, 4, 8")

        # Load UNet model
        # Use load_config() followed by from_config() to avoid deprecation warning
        unet_config = UNet2DConditionModel.load_config(
            base,
            subfolder="unet",
            local_files_only=self.local_files_only
        )
        unet = UNet2DConditionModel.from_config(unet_config).to(self.device, torch.float16)

        # Load checkpoint
        checkpoint_path = hf_hub_download(repo, ckpt, local_files_only=self.local_files_only)
        unet.load_state_dict(load_file(checkpoint_path, device=self.device))

        # Load pipeline with custom UNet
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            dtype=torch.float16,
            variant="fp16",
            local_files_only=self.local_files_only
        ).to(self.device)

        # Configure scheduler
        # For 1-step model, use prediction_type="sample", otherwise use default
        scheduler_kwargs = {"prediction_type": "sample"} if self.steps == 1 else {}
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            **scheduler_kwargs
        )

        self.pipeline = pipe


class SDXL_Lightning_1step(SDXL_Lightning_base):
    name = 'SDXL-Lightning-1step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 1


class SDXL_Lightning_2step(SDXL_Lightning_base):
    name = 'SDXL-Lightning-2step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 2


class SDXL_Lightning_4step(SDXL_Lightning_base):
    name = 'SDXL-Lightning-4step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 4


class SDXL_Lightning_8step(SDXL_Lightning_base):
    name = 'SDXL-Lightning-8step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 8


class SDTransformer(SDBase, ABC):
    """base class for SD models with a transformer architecture"""

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        vae = self.pipeline.vae
        vae_dtype = next(vae.modules()).dtype
        img_tensor = self.pipeline.image_processor.preprocess(images).to(device=self.device, dtype=vae_dtype)
        return (vae.encode(img_tensor).latent_dist.sample().to(dtype=self.dtype) - vae.config.shift_factor) * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        ''' Decode latents to an image.

        Args:
            latents: The latents to decode.
        '''
        pipe = self.pipeline
        tmp = pipe.vae.decode((latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor, return_dict=False)
        return pipe.image_processor.postprocess(tmp)  # type: ignore

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification = None, extract_positions: list[str] = []) -> 'SDResult':
        if modification is not None or len(extract_positions) > 0:
            raise ValueError(f'{self.name} support for modifications, or extract positions is not implemented yet.')
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


class SD3Base(SDTransformer, ABC):
    """Base class for SD3 models."""
    latent_dim = 1536

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompts: list[str], spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]:

        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # Set seed if provided
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)

        # encode image
        images_resized = [img.resize((resize, resize)) if resize is not None else img for img in images]
        latents = self.encode_latents(images_resized)  # this gives slightly different results for different batch sizes
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

        pipe.scheduler.set_timesteps(1000, device=self.device)
        timestep = pipe.scheduler.timesteps[999 - step]
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None)  # type: ignore
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)

        # extract representations
        with ExitStack() as stack, torch.no_grad():
            for extract_position in extract_positions:
                def hook_fn(module, input, output, extract_position):
                        representations[extract_position] = output[1].to(output_device)
                stack.enter_context(_get_module_by_path(pipe.transformer, extract_position).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
            pipe.transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]).to(device=self.device), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds)

        # fix representation shape
        num_tokens = next(iter(representations.values())).shape[1]
        potential_repr_shapes = [(i, num_tokens//i) for i in range(1, num_tokens) if num_tokens % i == 0]
        repr_shape = min(potential_repr_shapes, key=lambda x: abs(x[1]/x[0] - width / height))
        representations = {p: r.reshape(batch_size, *repr_shape, self.latent_dim).permute(0, 3, 1, 2) for p, r in representations.items()}

        return [SDRepresentation({p: r[i,None,:,:,:] for p, r in representations.items()}, seed) for i in range(batch_size)]


class SD3(SD3Base):
    name = 'SD3'
    full_name = 'stabilityai/stable-diffusion-3-medium-diffusers'
    steps = 28
    guidance_scale = 7.0
    latent_dim = 1536


class SD3_5_Large(SD3Base):
    name = 'SD3.5-Large'
    full_name = 'stabilityai/stable-diffusion-3.5-large'
    steps = 28
    guidance_scale = 3.5
    latent_dim = 2432


class SD3_5_Medium(SD3Base):
    name = 'SD3.5-Medium'
    full_name = 'stabilityai/stable-diffusion-3.5-medium'
    steps = 28
    guidance_scale = 3.5
    latent_dim = 2432


class SD3_5_Large_Turbo(SD3Base):
    name = 'SD3.5-Large-Turbo'
    full_name = 'stabilityai/stable-diffusion-3.5-large-turbo'
    steps = 4
    guidance_scale = 1.0
    latent_dim = 2432


class FLUXBase(SDTransformer, ABC):
    """Base class for FLUX models."""
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompts: list[str], spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # Set seed if provided
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)

        # encode image
        images_resized = [img.resize((resize, resize)) if resize is not None else img for img in images]
        latents = self.encode_latents(images_resized)  # this gives slightly different results for different batch sizes
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

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
                stack.enter_context(_get_module_by_path(pipe.transformer, extract_position).register_forward_hook(partial(hook_fn, extract_position=extract_position)))
            pipe.transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]).to(latents.dtype)/1000, guidance=None, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids)

        # fix representation shape
        num_tokens = next(iter(representations.values())).shape[1]
        potential_repr_shapes = [(i, num_tokens//i) for i in range(1, num_tokens) if num_tokens % i == 0]
        repr_shape = min(potential_repr_shapes, key=lambda x: abs(x[1]/x[0] - width / height))
        representations = {p: r.reshape(batch_size, *repr_shape, 3072).permute(0, 3, 1, 2) for p, r in representations.items()}

        return [SDRepresentation({p: r[i,None,:,:,:] for p, r in representations.items()}, seed) for i in range(batch_size)]


class FLUX1_dev(FLUXBase):
    name = 'FLUX.1-dev'
    full_name = 'black-forest-labs/FLUX.1-dev'
    steps = 28
    guidance_scale = 3.5


class FLUX1_schnell(FLUXBase):
    name = 'FLUX.1-schnell'
    full_name = 'black-forest-labs/FLUX.1-schnell'
    steps = 4
    guidance_scale = 0.0


class FLUX1_Krea(FLUXBase):
    """FLUX.1 Krea [dev] is a FLUX [dev] variant tuned for strong aesthetics and photorealism. It works as a drop-in text-to-image replacement for FLUX.1-dev and uses the same FluxPipeline architecture, so FLUXBase's img2repr implementation continues to work."""
    name = "FLUX.1-Krea"
    full_name = "black-forest-labs/FLUX.1-Krea-dev"
    steps = 30
    guidance_scale = 4.5


class FLUX2_dev(SDBase):
    """FLUX.2-dev is a 32B parameter flow matching transformer model capable of generating and editing (multiple) images. It is initialized without the mistral-small text encoder to save memory."""
    name = 'FLUX.2-dev'
    # full_name = 'black-forest-labs/FLUX.2-dev'
    full_name = "diffusers/FLUX.2-dev-bnb-4bit"  # use quantized model by default
    steps = 28
    guidance_scale = 2.5  # according to https://fal.ai/models/fal-ai/flux-2/api#schema-input

    def _load_pipeline(self):
        try:
            from diffusers import Flux2Pipeline
        except ImportError:
            raise ImportError("Your diffusers package does not support Flux.2-dev, likely because it is too old. Version >= 0.36.0 is required.")

        self.pipeline = Flux2Pipeline.from_pretrained(
            self.full_name,
            text_encoder=None,
            torch_dtype=torch.bfloat16,
            local_files_only=self.local_files_only,
        ).to(self.device)


    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Convert latents from (B, C, H, W) to (B, C*4, H//2, W//2) by packing 2x2 patches."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Convert latents from (B, C*4, H, W) to (B, C, H*2, W*2) by unpacking 2x2 patches."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        """Encode PIL images to FLUX.2 latents.

        Args:
            images: List of PIL images to encode.

        Returns:
            Encoded latents tensor of shape (B, C*4, H//2, W//2) after patchification and batch normalization.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        # Preprocess images to tensor
        img_tensor = self.pipeline.image_processor.preprocess(images).to(device=self.device, dtype=vae_dtype)

        # Encode with VAE
        latents = vae.encode(img_tensor).latent_dist.mode()

        # Patchify latents (2x2 patches)
        latents = self._patchify_latents(latents)

        # Apply batch normalization
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std

        return latents.to(dtype=torch.bfloat16)  # FLUX.2 uses bfloat16

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        """Decode FLUX.2 latents to PIL images.

        Args:
            latents: Latents tensor of shape (B, C*4, H//2, W//2) (patchified and batch-normalized).

        Returns:
            List of decoded PIL images.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)

        # Reverse batch normalization
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        # Unpatchify latents
        latents = self._unpatchify_latents(latents)

        # Decode with VAE
        image = vae.decode(latents, return_dict=False)[0]

        # Postprocess to PIL images
        return self.pipeline.image_processor.postprocess(image, output_type="pil")

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        """Compute mu for dynamic shifting scheduler (from Flux2Pipeline)."""
        # see https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L251
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, resize: int | None, prompts: list[str], spatial_avg: bool, output_device: str, seed: Optional[int] = None) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # Set seed if provided
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)

        # Encode image to latents
        images_resized = [img.resize((resize, resize)) if resize is not None else img for img in images]
        latents = self.encode_latents(images_resized)
        noise = torch.randn_like(latents[None, 0]).expand(latents.shape)

        # Setup scheduler with dynamic shifting
        _, _, latent_h, latent_w = latents.shape
        num_steps = 1000
        mu = self._compute_empirical_mu(latent_h * latent_w, num_steps)
        pipe.scheduler.set_timesteps(num_steps, device=self.device, mu=mu)
        timestep = pipe.scheduler.timesteps[999 - step]

        # Prepare prompt embeddings (15360 = 3 text encoder layers × 5120 hidden dim)
        prompt_embeds = torch.randn((batch_size, 1, 15360), device=self.device, dtype=torch.bfloat16) * 0.01
        txt_ids = pipe._prepare_text_ids(prompt_embeds).to(self.device)

        # Add noise and prepare latents for transformer
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0).unsqueeze(0), noise=noise)
        latents, img_ids = pipe.prepare_latents(batch_size, pipe.transformer.config.in_channels // 4, width, height, torch.bfloat16, self.device, generator=torch.manual_seed(seed), latents=latents)

        # Prepare guidance
        guidance = torch.full([latents.shape[0]], self.guidance_scale, device=self.device, dtype=torch.bfloat16)

        # Extract representations via hooks
        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                def hook_fn(module, input, output, pos=pos):
                    out = output[1] if isinstance(output, tuple) and len(output) >= 2 else (output[0] if isinstance(output, tuple) else output)
                    representations[pos] = out.to(output_device)
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(hook_fn))
            pipe.transformer(
                hidden_states=latents,
                timestep=(timestep / 1000).expand(latents.shape[0]).to(torch.bfloat16),
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
            )

        # Reshape representations to spatial format
        num_tokens = next(iter(representations.values())).shape[1]
        potential_shapes = [(i, num_tokens // i) for i in range(1, num_tokens + 1) if num_tokens % i == 0]
        repr_shape = min(potential_shapes, key=lambda x: abs(x[1] / x[0] - width / height))
        representations = {p: r.reshape(batch_size, *repr_shape, 6144).permute(0, 3, 1, 2) for p, r in representations.items()}

        return [SDRepresentation({p: r[i, None] for p, r in representations.items()}, seed) for i in range(batch_size)]


class Playground_V2_5(SDUnet):
    """
    Playground v2.5 1024px aesthetic checkpoint.

    SDXL-style UNet + VAE, so we can reuse SDUnet's encode/decode/img2repr
    path exactly like SDXL.
    """
    name = "Playground-v2.5"
    full_name = "playgroundai/playground-v2.5-1024px-aesthetic"
    steps = 50
    guidance_scale = 3.0

    def _load_pipeline(self):
        from diffusers import StableDiffusionXLPipeline

        kwargs = {
            "torch_dtype": self.dtype,
            "local_files_only": self.local_files_only,
        }
        if self.dtype == torch.float16:
            kwargs["variant"] = "fp16"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.full_name,
            **kwargs,
        ).to(self.device)


class AuraFlow(SDTransformer):
    """AuraFlow v0.3 is a large rectified flow T2I model with a dedicated AuraFlowPipeline."""
    name = "AuraFlow"
    full_name = "fal/AuraFlow-v0.3"
    steps = 50
    guidance_scale = 3.5

    def _img2repr(self, *args, **kwargs) -> list[SDRepresentation]:
        raise NotImplementedError("img2repr is not implemented for AuraFlow yet. You can still use AuraFlow for text-to-image generation.")


class Kandinsky3(SDBase):
    """Kandinsky-3 is a U-Net based latent diffusion model with a Flan-UL2 text encoder and a MoVQ encoder/decoder."""
    name = "Kandinsky-3"
    full_name = "kandinsky-community/kandinsky-3"
    steps = 25
    guidance_scale = 4.0

    def _load_pipeline(self):
        kwargs = {
            "torch_dtype": self.dtype,
            "local_files_only": self.local_files_only,
        }
        if self.dtype == torch.float16:
            kwargs["variant"] = "fp16"

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self.full_name,
            **kwargs,
        ).to(self.device)

    def _img2repr(self, *args, **kwargs) -> list[SDRepresentation]:
        raise NotImplementedError("img2repr is not implemented for Kandinsky-3 yet. You can still use Kandinsky-3 for text-to-image generation.")
