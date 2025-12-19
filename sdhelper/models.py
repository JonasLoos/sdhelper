from contextlib import ExitStack
from functools import partial
import diffusers
from diffusers import AutoPipelineForText2Image
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
            raise AttributeError(f"Extract position `{path}` not found: Attribute `{name}` not available.")
        except IndexError:
            raise IndexError(f"Extract position `{path}` not found: Index `{index}` out of range.")
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


class SDBase(ABC):
    """Base class for (Stable) Diffusion models."""
    def __init__(self, device: str = 'auto', disable_progress_bar: bool = False, local_files_only: bool = False):
        self.local_files_only = local_files_only

        # determine device and dtype
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.dtype = torch.float32 if self.device == 'cpu' else torch.float16

        # setup pipeline
        progressbar_enabled = diffusers.utils.logging.is_progress_bar_enabled()
        if progressbar_enabled and disable_progress_bar: diffusers.utils.logging.disable_progress_bar()
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
        self._cached_prompt_embeds = {}

    def __call__(self, prompt: str, steps: Optional[int] = None, guidance_scale: Optional[float] = None, seed: Optional[int] = None, *, width: Optional[int] = None, height: Optional[int] = None, modification: Optional[Callable[[Any,Any,Any,str],Optional[torch.Tensor]]] = None, extract_positions: list[str] = []) -> 'SDResult':
        if steps is None: steps = self.steps
        if guidance_scale is None: guidance_scale = self.guidance_scale
        if seed is None: seed = int(torch.randint(0, 2**32, (1,)).item())
        return self._generate(prompt, steps, guidance_scale, seed, width=width, height=height, modification=modification, extract_positions=extract_positions)

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification = None, extract_positions: list[str] = []) -> 'SDResult':
        if modification is not None or len(extract_positions) > 0:
            raise ValueError(f'{self.name} support for modifications, or extract positions is not implemented yet.')
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result_images = self.pipeline(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, width=width, height=height, generator=generator)
        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=None,
            images=None,
            result_latent=None,
            result_tensor=None,
            result_image=result_images.images[0],
        )

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
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]: ...

    @overload
    def img2repr(self, data: PILImage | np.ndarray | str, extract_positions: list[str], step: int, prompt: str = '', batch_size: int = 1, seed: int = 42, extract_fn: Callable[[torch.Tensor],torch.Tensor] = lambda x: x.to('cpu'), raw: bool = False) -> SDRepresentation: ...

    @overload
    def img2repr(self, data: list[PILImage | np.ndarray | str], extract_positions: list[str], step: int, prompt: str | list[str] = '', batch_size: int = 1, seed: int = 42, extract_fn: Callable[[torch.Tensor],torch.Tensor] = lambda x: x.to('cpu'), raw: bool = False) -> list[SDRepresentation]: ...

    def img2repr(self, data: PILImage | np.ndarray | str | list[PILImage | np.ndarray | str], extract_positions: list[str], step: int, prompt: str | list[str] = '', batch_size: int = 1, seed: int = 42, extract_fn: Callable[[torch.Tensor],torch.Tensor] = lambda x: x.to('cpu'), raw: bool = False):
        '''Convert image to representations at specified extract positions.

        Args:
            img: PIL image(s) to convert to extract representations for.
            extract_positions: List of extract positions to return representations for.
            step: Timestep determining the amount of noise to add. Noise increases with timestep. Must be in [0, 999]. Even step 0 might add noise, depending on the scheduler.
            prompt: Prompt to use for the image. If a list is provided, each image will be paired with the corresponding prompt.
            batch_size: Number of images to process at once. The returned representations might differ slightly depending on the batch size. For best reproducability, set batch_size=1.
            seed: Seed for random number generation. If None, a random seed will be used.
            extract_fn: Function to apply to the representations. By default, the representations are moved to the CPU.
            raw: If True, return the raw representations without any processing like reshaping. This is applied before extract_fn.

        Returns:
            Dictionary with extract positions as keys and the corresponding representations as values.

        Examples:
        ```
        sd = SD('SD1.5')

        # extract h-space (mid_block repr.) for a single image with a specific seed
        hspace = sd.img2repr('./my_image.jpg', ['mid_block'], 30, seed=42)['mid_block']

        # extract representations for a full dataset
        repr_dataset = sd.img2repr(
            datasets.load_dataset('cifar10'),
            extract_positions = ['up_blocks[1]'],
            step = 100,
            spatial_avg = True,
        )
        repr_dataset.save_to_disk('SD15_cifar10_up1_representations')
        ```
        '''
        # check and preprocess args
        single = not isinstance(data, list)
        data_list = [data] if single else data
        if len(data_list) == 0: return []
        if not all(isinstance(d, (PILImage, np.ndarray, str)) for d in data_list): raise ValueError(f'Unsupported type for data')
        images_pil = [PIL.Image.open(d) if isinstance(d, str) else PIL.Image.fromarray(d) if isinstance(d, np.ndarray) else d for d in data_list]
        images_rgb = [img.convert('RGB') for img in images_pil]
        if not isinstance(prompt, list): prompt = [prompt] * len(images_pil)
        if not len(prompt) == len(images_pil): raise ValueError('Number of prompts must match number of images')
        if not all(isinstance(p, str) for p in prompt): raise ValueError('Prompts must be strings')
        if not all(img.size == images_rgb[0].size for img in images_rgb) and batch_size != 1: raise ValueError('All images must have the same size when batch_size != 1')

        # extract representations
        representations = []
        for i in trange(0, len(images_rgb), batch_size, desc='Extracting representations for list of images', disable=single or self.disable_progress_bar):
            torch.manual_seed(seed)
            representations.extend(self._img2repr(images_rgb[i:i+batch_size], extract_positions, step, prompt[i:i+batch_size], seed, extract_fn, raw))
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

        # extraction hook
        def hook_fn(module, input, output, pos):
            if isinstance(output, tuple):
                output = output[0]  # TODO: is it good to always take the first output and ignore the rest?
            representations[pos].append(output)
            if modification:
                return modification(module, input, output, pos)

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            # setup hooks to extract representations
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(self.pipeline.unet, pos).register_forward_hook(partial(hook_fn, pos=pos)))

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
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool = False) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        representations = {}

        # encode image
        latents = self.encode_latents(images)  # this gives slightly different results for different batch sizes
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

        # apply noise
        pipe.scheduler.set_timesteps(1000, device=self.device)
        timestep = torch.tensor([step], dtype=torch.long, device=self.device)
        latents = pipe.scheduler.add_noise(latents, noise, timestep)

        # scale latents
        # TODO: this is from SD1.5 (where it's not used), is it also necessary for other models?
        latents = pipe.scheduler.scale_model_input(latents, timestep)

        # create empty prompt embeddings
        prompt_embeds, *_ = self.pipeline.encode_prompt(prompt=prompts, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)

        # setup unet config
        pipe.unet.config.addition_embed_type = 'nothing_at_all'

        # extraction hook
        def hook_fn(module, input, output, pos):
            representations[pos] = extract_fn(output)

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.unet, pos).register_forward_hook(partial(hook_fn, pos=pos)))
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
    full_name = 'sd-research/stable-diffusion-2-base'
    steps = 50
    guidance_scale = 7.5


class SD2_1(SDUnet):
    name = 'SD2.1'
    full_name = 'sd-research/stable-diffusion-2-1-base'
    steps = 50
    guidance_scale = 7.5


class SD_Turbo(SDUnet):
    name = 'SD-Turbo'
    full_name = 'stabilityai/sd-turbo'
    steps = 4
    guidance_scale = 0.0


class SDXLBase(SDUnet, ABC):
    def _load_pipeline(self):
        self.pipeline = AutoPipelineForText2Image.from_pretrained(self.full_name, torch_dtype=torch.float16, local_files_only=self.local_files_only).to(self.device, dtype=self.dtype)

        # upcast vae to float32 to avoid precision issues
        self.pipeline.vae.to(dtype=torch.float32)


class SDXL(SDXLBase):
    name = 'SDXL'
    full_name = 'stabilityai/stable-diffusion-xl-base-1.0'
    steps = 40
    guidance_scale = 5.0


class SDXL_Turbo(SDXLBase):
    name = 'SDXL-Turbo'
    full_name = 'stabilityai/sdxl-turbo'
    steps = 4
    guidance_scale = 0.0


class SDXL_LightningBase(SDXLBase, ABC):
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
            torch_dtype=torch.float16,
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

        # upcast vae to float32 to avoid precision issues
        pipe.vae.to(dtype=torch.float32)

        self.pipeline = pipe


class SDXL_Lightning_1step(SDXL_LightningBase):
    name = 'SDXL-Lightning-1step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 1


class SDXL_Lightning_2step(SDXL_LightningBase):
    name = 'SDXL-Lightning-2step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 2


class SDXL_Lightning_4step(SDXL_LightningBase):
    name = 'SDXL-Lightning-4step'
    full_name = 'ByteDance/SDXL-Lightning'
    steps = 4


class SDXL_Lightning_8step(SDXL_LightningBase):
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
        output_tensor = pipe.vae.decode((latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(output_tensor)


class SD3Base(SDTransformer, ABC):
    """Base class for SD3 models."""

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]:

        pipe = self.pipeline
        batch_size = len(images)
        representations = {}

        # encode image
        latents = self.encode_latents(images)  # this gives slightly different results for different batch sizes
        h_lat, w_lat = latents.shape[2], latents.shape[3]
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

        pipe.scheduler.set_timesteps(1000, device=self.device)
        timestep = pipe.scheduler.timesteps[999 - step]
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None)
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)

        # setup hook
        def hook_fn(module, input, output, pos):
            if not raw:
                output = output[1].permute(0, 2, 1).reshape(batch_size, 1, -1, h_lat//2, w_lat//2)
            representations[pos] = extract_fn(output)

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos)))
            pipe.transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]).to(device=self.device), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds)

        return [SDRepresentation({p: r[i] for p, r in representations.items()}, seed) for i in range(batch_size)]


class SD3(SD3Base):
    name = 'SD3'
    full_name = 'stabilityai/stable-diffusion-3-medium-diffusers'
    steps = 28
    guidance_scale = 7.0


class SD3_5_Large(SD3Base):
    name = 'SD3.5-Large'
    full_name = 'stabilityai/stable-diffusion-3.5-large'
    steps = 28
    guidance_scale = 3.5


class SD3_5_Medium(SD3Base):
    name = 'SD3.5-Medium'
    full_name = 'stabilityai/stable-diffusion-3.5-medium'
    steps = 28
    guidance_scale = 3.5


class SD3_5_Large_Turbo(SD3Base):
    name = 'SD3.5-Large-Turbo'
    full_name = 'stabilityai/stable-diffusion-3.5-large-turbo'
    steps = 4
    guidance_scale = 1.0


class FLUXBase(SDTransformer, ABC):
    """Base class for FLUX models."""
    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # encode image
        latents = self.encode_latents(images)
        h_lat, w_lat = latents.shape[2], latents.shape[3]
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)  # expand to ensure each image is noised with the same noise/seed

        # timesteps / dynamic shifting
        cfg = pipe.scheduler.config

        set_kwargs = {}
        if bool(cfg.get("use_dynamic_shifting", False)):
            # diffusers.pipelines.flux.pipeline_flux.calculate_shift
            # image_seq_len is the packed latent sequence length used by Flux (2x2 packing => //4)
            image_seq_len = (h_lat * w_lat) // 4
            base_seq_len = cfg.get("base_image_seq_len", 256)
            max_seq_len  = cfg.get("max_image_seq_len", 4096)
            base_shift   = cfg.get("base_shift", 0.5)
            max_shift    = cfg.get("max_shift", 1.16)
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            set_kwargs["mu"] = float(mu)
        pipe.scheduler.set_timesteps(1000, device=pipe.device, **set_kwargs)
        timestep = pipe.scheduler.timesteps[999 - step]

        # prepare and noise latents
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompts, prompt_2=None)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        latents, latent_image_ids = pipe.prepare_latents(batch_size, pipe.transformer.config.in_channels // 4, width, height, prompt_embeds.dtype, pipe.device, generator=generator, latents=latents)
        latents = pipe._pack_latents(latents, *latents.shape)

        # extraction hook
        def hook_fn(module, input, output, pos):
            if not raw:
                output = output[1].permute(0, 2, 1).reshape(batch_size, 1, -1, h_lat//2, w_lat//2)
            representations[pos] = extract_fn(output)

        # Prepare guidance if required (for FLUX.1-dev and FLUX.1-Krea)
        guidance = torch.full([latents.shape[0]], self.guidance_scale * 1000.0, device=pipe.device, dtype=latents.dtype) if pipe.transformer.config.guidance_embeds else None

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos)))
            pipe.transformer(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0]).to(latents.dtype)/1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids
            )

        return [SDRepresentation({p: r[i] for p, r in representations.items()}, seed) for i in range(batch_size)]


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

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification = None, extract_positions: list[str] = []) -> 'SDResult':
        raise NotImplementedError("Flux.2-dev currently does not support image generation, as the text encoder is not automatically loaded. You can use `FLUX2_dev().pipeline(...)` instead.")

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

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]:
        if any(p != '' for p in prompts):
            raise NotImplementedError("FLUX.2 does not support prompt inputs yet")
        pipe = self.pipeline
        batch_size = len(images)
        width, height = images[0].size
        representations = {}

        # Encode image to latents
        latents = self.encode_latents(images)
        _, _, h_lat, w_lat = latents.shape
        noise = torch.randn_like(latents[None, 0]).expand(latents.shape)

        # Setup scheduler with dynamic shifting
        num_steps = 1000
        mu = self._compute_empirical_mu(h_lat * w_lat, num_steps)
        pipe.scheduler.set_timesteps(num_steps, device=self.device, mu=mu)
        timestep = pipe.scheduler.timesteps[999 - step]

        # Prepare prompt embeddings (15360 = 3 text encoder layers × 5120 hidden dim)
        # Use small random noise to avoid NaN values
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt_embeds = torch.randn((batch_size, 1, 15360), device=self.device, dtype=torch.bfloat16, generator=generator) * 0.01
        txt_ids = pipe._prepare_text_ids(prompt_embeds).to(self.device)

        # Add noise and prepare latents for transformer
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0).unsqueeze(0), noise=noise)
        latents, img_ids = pipe.prepare_latents(batch_size, pipe.transformer.config.in_channels // 4, width, height, torch.bfloat16, self.device, generator=generator, latents=latents)

        # Prepare guidance
        guidance = torch.full([latents.shape[0]], self.guidance_scale, device=self.device, dtype=torch.bfloat16)

        # extraction hook
        def hook_fn(_module, _input, output, pos):
            if not raw:
                output = output.permute(0, 2, 1).reshape(batch_size, 1, -1, h_lat//2, w_lat//2)
            representations[pos] = extract_fn(output)

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos)))
            pipe.transformer(
                hidden_states=latents,
                timestep=(timestep / 1000).expand(latents.shape[0]).to(torch.bfloat16),
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
            )

        return [SDRepresentation({p: r[i] for p, r in representations.items()}, seed) for i in range(batch_size)]


class Playground_V2_5(SDUnet):
    """
    Playground v2.5 1024px aesthetic checkpoint. Uses SDXL-style UNet + VAE.
    """
    name = "Playground-v2.5"
    full_name = "playgroundai/playground-v2.5-1024px-aesthetic"
    steps = 50
    guidance_scale = 3.0

    def _load_pipeline(self):
        from diffusers import StableDiffusionXLPipeline

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.full_name,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification = None, extract_positions: list[str] = []) -> 'SDResult':
        # The standard unet `_generate` method leads to grayish images, so we just use the pipeline directly.
        result = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            output_type="pil",
        )
        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=None,
            images=None,
            result_latent=None,
            result_tensor=None,
            result_image=result.images[0],
        )

class AuraFlow(SDBase):
    """AuraFlow v0.3 is a large rectified flow T2I model with a dedicated AuraFlowPipeline."""
    name = "AuraFlow"
    full_name = "fal/AuraFlow-v0.3"
    steps = 50
    guidance_scale = 3.5

    def _load_pipeline(self):
        from diffusers import AuraFlowPipeline

        self.pipeline = AuraFlowPipeline.from_pretrained(
            self.full_name,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)

        # upcast vae to float32 to avoid precision issues
        self.pipeline.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        vae = self.pipeline.vae
        img_tensor = self.pipeline.image_processor.preprocess(images).to(device=self.device, dtype=torch.float32)
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents.to(dtype=self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        vae = self.pipeline.vae
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents.to(dtype=torch.float32), return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *, width: Optional[int] = None, height: Optional[int] = None, modification = None, extract_positions: list[str] = []) -> 'SDResult':
        pipe = self.pipeline
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, width=width, height=height, generator=generator, output_type="latent").images
        image = self.decode_latents(latents)[0]
        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=None,
            images=None,
            result_latent=latents,
            result_tensor=None,
            result_image=image,
        )

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)
        representations = {}

        # encode image
        latents = self.encode_latents(images)
        _, _, h_lat, w_lat = latents.shape
        noise = torch.randn_like(latents[None,0]).expand(latents.shape)

        # timesteps
        pipe.scheduler.set_timesteps(1000, device=self.device)
        timestep_val = pipe.scheduler.timesteps[999 - step]

        # Scale noise (flow matching uses scale_noise instead of add_noise)
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep_val.unsqueeze(0), noise=noise)

        # encode prompts
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            do_classifier_free_guidance=False
        )

        # extraction hook
        def hook_fn(module, input, output, pos):
            if not raw:
                # cut and reshape output to spatial format
                output = output[:,:(h_lat*w_lat//4),:].permute(0, 2, 1).reshape(batch_size, 1, -1, h_lat//2, w_lat//2)
            representations[pos] = extract_fn(output)

        # Run transformer
        timestep_norm = timestep_val / 1000
        timestep_norm = timestep_norm.expand(latents.shape[0]).to(latents.device, dtype=latents.dtype)

        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos)))

            pipe.transformer(
                latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_norm,
                return_dict=False,
                attention_kwargs=pipe.attention_kwargs if hasattr(pipe, 'attention_kwargs') else None
            )

        return [SDRepresentation({p: r[i] for p, r in representations.items()}, seed) for i in range(batch_size)]


class ZImageTurbo(SDBase):
    name = "Z-Image-Turbo"
    full_name = "Tongyi-MAI/Z-Image-Turbo"
    steps = 9
    guidance_scale = 0.0

    def _load_pipeline(self):
        try:
            from diffusers import ZImagePipeline
        except ImportError:
            raise ImportError("Your diffusers package does not support Z-Image-Turbo, likely because it is too old. Version >= 0.36.0 is required.")

        self.pipeline = ZImagePipeline.from_pretrained(
            self.full_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            low_cpu_mem_usage=False,
            local_files_only=self.local_files_only,
        ).to(self.device)

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        img_tensor = self.pipeline.image_processor.preprocess(images).to(device=self.device, dtype=vae_dtype)
        return (vae.encode(img_tensor).latent_dist.sample().to(dtype=self.dtype) - vae.config.shift_factor) * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        latents = (latents.to(dtype=vae_dtype) / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int, prompts: list[str], seed: int, extract_fn: Callable[[torch.Tensor],torch.Tensor], raw: bool) -> list[SDRepresentation]:
        pipe = self.pipeline
        batch_size = len(images)

        # encode image
        latents = self.encode_latents(images)
        _, _, h_lat, w_lat = latents.shape

        # Generator for noise
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=latents.dtype)

        # timesteps
        image_seq_len = h_lat * w_lat // 4
        base_seq_len = pipe.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = pipe.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = pipe.scheduler.config.get("base_shift", 0.5)
        max_shift = pipe.scheduler.config.get("max_shift", 1.15)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b

        # handle timestep/noise
        pipe.scheduler.set_timesteps(1000, device=self.device, mu=mu)
        timestep = pipe.scheduler.timesteps[999 - step]
        latents = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)
        timestep_model_input = (1000 - timestep.expand(latents.shape[0])) / 1000

        # encode prompts
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            do_classifier_free_guidance=False,
        )

        latent_model_input = list(latents.to(pipe.transformer.dtype).unsqueeze(2).unbind(dim=0))

        representations = {}
        def hook_fn(module, input, output, pos):
            if not raw:
                output = output[:,:image_seq_len,:].permute(0, 2, 1).reshape(batch_size, 1, -1, h_lat//2, w_lat//2)
            representations[pos] = extract_fn(output)

        with ExitStack() as stack, torch.no_grad():
            for pos in extract_positions:
                stack.enter_context(_get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos)))

            pipe.transformer(
                latent_model_input,
                timestep_model_input.to(dtype=pipe.transformer.dtype),
                prompt_embeds,
                return_dict=False
            )

        return [SDRepresentation({p: r[i] for p, r in representations.items()}, seed) for i in range(batch_size)]


class QwenImage(SDBase):
    name = "QwenImage"
    full_name = "Qwen/Qwen-Image"
    steps = 50
    guidance_scale = 4.0

    def _load_pipeline(self):
        from diffusers import DiffusionPipeline

        # Qwen-Image examples use bfloat16 on CUDA
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.full_name,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)

        # helpful for big images
        if hasattr(self.pipeline, "vae") and hasattr(self.pipeline.vae, "enable_tiling"):
            self.pipeline.vae.enable_tiling()

    def _vae_stats(self):
        # Qwen VAE uses per-channel mean/std
        vae = self.pipeline.vae
        mean = getattr(vae.config, "latents_mean", None)
        std = getattr(vae.config, "latents_std", None)
        if mean is None or std is None:
            return None, None
        mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(std, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        return mean, std

    @torch.no_grad()
    def encode_latents(self, images: list[PILImage]) -> torch.Tensor:
        pipe = self.pipeline
        vae = pipe.vae

        # VAE
        x = pipe.image_processor.preprocess(images).to(device=self.device, dtype=torch.float32)
        z = vae.tiled_encode(x)
        mean, std = self._vae_stats()
        if mean is not None:
            z = (z - mean) / std

        return z.to(dtype=self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[PILImage]:
        pipe = self.pipeline
        vae = pipe.vae

        z = latents.to(device=self.device, dtype=torch.float32)
        mean, std = self._vae_stats()
        if mean is not None:
            z = z * std + mean

        if hasattr(vae, "tiled_decode"):
            x = vae.tiled_decode(z).sample
        else:
            out = vae.decode(z, return_dict=True)
            x = out.sample if hasattr(out, "sample") else out[0]

        return pipe.image_processor.postprocess(x, output_type="pil")

    @staticmethod
    def _pack_latents_2x2(latents: torch.Tensor) -> torch.Tensor:
        # packing: patch_size=2 => 2x2 pack, channels *= 4, spatial //= 2
        # (B, C, H, W) -> (B, (H//2)*(W//2), C*4)
        b, c, h, w = latents.shape
        lat = latents.view(b, c, h // 2, 2, w // 2, 2)
        lat = lat.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H//2, W//2, C, 2, 2)
        lat = lat.view(b, (h // 2) * (w // 2), c * 4)
        return lat

    @staticmethod
    def _unpack_latents_2x2(packed: torch.Tensor, h_lat: int, w_lat: int, c: int) -> torch.Tensor:
        # (B, (H//2)*(W//2), C*4) -> (B, C, H, W)
        b, seq, c4 = packed.shape
        assert c4 == c * 4
        lat = packed.view(b, h_lat // 2, w_lat // 2, c, 2, 2)
        lat = lat.permute(0, 3, 1, 4, 2, 5).contiguous()
        lat = lat.view(b, c, h_lat, w_lat)
        return lat

    def _generate(self, prompt: str, steps: int, guidance_scale: float, seed: int, *,
                  width: Optional[int] = None, height: Optional[int] = None,
                  modification=None, extract_positions: list[str] = []) -> "SDResult":
        if modification is not None or extract_positions:
            raise ValueError("QwenImage: modifications/extract_positions during generation not implemented.")

        pipe = self.pipeline
        g = torch.Generator(device=self.device).manual_seed(seed)

        out = pipe(
            prompt=prompt,
            negative_prompt="",
            true_cfg_scale=guidance_scale,
            num_inference_steps=int(steps),
            width=width,
            height=height,
            generator=g,
            output_type="pil",
        )

        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=None,
            images=None,
            result_latent=None,
            result_tensor=None,
            result_image=out.images[0],
        )

    def _img2repr(self, images: list[PILImage], extract_positions: list[str], step: int,
                  prompts: list[str], seed: int,
                  extract_fn: Callable[[torch.Tensor], torch.Tensor],
                  raw: bool) -> list["SDRepresentation"]:
        pipe = self.pipeline
        batch_size = len(images)

        # encode -> (B, C, H, W)
        latents = self.encode_latents(images)
        b, c, h_lat, w_lat = latents.shape

        gen = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(latents.shape, generator=gen, device=self.device, dtype=latents.dtype)

        # scheduler timestep selection (train-time grid of 1000)
        cfg = pipe.scheduler.config
        set_kwargs = {}
        if bool(cfg.get("use_dynamic_shifting", False)):
            # same dynamic-shift idea used across FlowMatch schedulers: shift depends on image sequence length
            image_seq_len = (h_lat * w_lat) // 4
            base_seq_len = cfg.get("base_image_seq_len", 256)
            max_seq_len = cfg.get("max_image_seq_len", 8192)
            base_shift = cfg.get("base_shift", 0.5)
            max_shift = cfg.get("max_shift", 1.15)
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b0 = base_shift - m * base_seq_len
            set_kwargs["mu"] = float(image_seq_len * m + b0)

        pipe.scheduler.set_timesteps(1000, device=self.device, **set_kwargs)
        timestep = pipe.scheduler.timesteps[999 - step]

        noised = pipe.scheduler.scale_noise(latents, timestep=timestep.unsqueeze(0), noise=noise)

        # pack for transformer: (B, seq, in_channels)
        packed = self._pack_latents_2x2(noised).to(dtype=self.dtype)

        # prompt encoding
        prompt_embeds, prompt_mask = pipe.encode_prompt(prompts, device=self.device)
        txt_seq_lens = prompt_mask.to(torch.long).sum(dim=1).tolist()
        t_in = timestep.expand(batch_size).to(device=self.device, dtype=torch.long)

        # capture hooks
        representations: dict[str, torch.Tensor] = {}
        def hook_fn(_module, _inp, output, pos: str):
            print(output)
            print(type(output))
            print(output[0].shape)
            print(output[1].shape)
            if not raw:
                output = output[1].permute(0, 2, 1).contiguous().view(batch_size, 1, -1, h_lat // 2, w_lat // 2)
            representations[pos] = extract_fn(output)

        with ExitStack() as stack:
            for pos in extract_positions:
                stack.enter_context(
                    _get_module_by_path(pipe.transformer, pos).register_forward_hook(partial(hook_fn, pos=pos))
                )

            pipe.transformer(
                hidden_states=packed,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                timestep=t_in,
                img_shapes=[(1, h_lat // 2, w_lat // 2)] * batch_size,
                txt_seq_lens=txt_seq_lens,
                guidance=None,
                return_dict=False,
            )

        return [
            SDRepresentation({p: representations[p][i] for p in representations.keys()}, seed)
            for i in range(batch_size)
        ]


# Potential models to add:
# - stabilityai/stable-cascade
# - tencent/HunyuanImage-3.0
# - ...
