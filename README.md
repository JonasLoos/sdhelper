[![PyPI](https://img.shields.io/pypi/v/sdhelper?logo=pypi)](https://pypi.org/project/sdhelper/)
[![Downloads](https://img.shields.io/pypi/dm/sdhelper)](https://pypi.org/project/sdhelper/)

# `sdhelper` - Simple Representation Extraction for Diffusion Models

A helper package for working with image diffusion models, for example with **S**table **D**iffusion. Allows for easy extraction of U-Net and transformer representations, i.e. the intermediate activations in the latent space of the denoising model.


## Installation

```bash
pip install sdhelper
```


## Usage

```python
from sdhelper import SD

# load model
sd = SD('SD-1.5')

# generate image
img = sd('a beautiful landscape').result_image

# extract representations from the `up[1]` block at time step 50
r = sd.img2repr(img, extract_positions=['up_blocks[1]'], step=50)

# compute similarity between all pairs of tokens in `r`
similarities = r.cosine_similarity(r)
```

Available models:

* [`SD1.1`](https://huggingface.co/CompVis/stable-diffusion-v1-1)
* [`SD1.2`](https://huggingface.co/CompVis/stable-diffusion-v1-2)
* [`SD1.3`](https://huggingface.co/CompVis/stable-diffusion-v1-3)
* [`SD1.4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
* [`SD1.5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)†
* [`SD2.0`](https://huggingface.co/sd-research/stable-diffusion-2-base)†
* [`SD2.1`](https://huggingface.co/sd-research/stable-diffusion-2-1-base)†
* [`SD-Turbo`](https://huggingface.co/stabilityai/sd-turbo)
* [`SDXL`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
* [`SDXL-Turbo`](https://huggingface.co/stabilityai/sdxl-turbo)
* [`SDXL-Lightning-1step`](https://huggingface.co/ByteDance/SDXL-Lightning)
* [`SDXL-Lightning-2step`](https://huggingface.co/ByteDance/SDXL-Lightning)
* [`SDXL-Lightning-4step`](https://huggingface.co/ByteDance/SDXL-Lightning)
* [`SDXL-Lightning-8step`](https://huggingface.co/ByteDance/SDXL-Lightning)
* [`SD3`](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
* [`SD3.5-Large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
* [`SD3.5-Medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
* [`SD3.5-Large-Turbo`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
* [`FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev)
* [`FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
* [`FLUX.1-Krea`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
* [`FLUX.2-dev`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit)
* [`Playground-v2.5`](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
* [`AuraFlow`](https://huggingface.co/fal/AuraFlow-v0.3)
* [`Kandinsky-3`](https://huggingface.co/kandinsky-community/kandinsky-3)

Especially for FLUX models, it might make sense to quantize the weights and enable CPU offloading:

```python
flux = SD('FLUX-schnell')
flux.quantize(['transformer', 'text_encoder_2'], model_cpu_offload=True)
```

> [!NOTE]
> † The official original repositories of [`SD1.5`](https://huggingface.co/runwayml/stable-diffusion-v1-5), [`SD2.0`](https://huggingface.co/stabilityai/stable-diffusion-2), and [`SD2.1`](https://huggingface.co/stabilityai/stable-diffusion-2-1) were deleted. Alternative unofficial repositories are used instead.
