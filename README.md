# Stable Diffusion Helper

A helper package for working with stable diffusion models. Allows for easy extraction of U-Net (and transformer) representations.


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

* SD1.1
* SD1.2
* SD1.3
* SD1.4
* SD1.5
* SD2.0
* SD2.1
* SD-Turbo
* SDXL
* SDXL-Turbo
* SDXL-Lightning-1step
* SDXL-Lightning-2step
* SDXL-Lightning-4step
* SDXL-Lightning-8step
* SD3
* FLUX-dev
* FLUX-schnell

Especially for FLUX models, it might make sense to quantize the weights and enable CPU offloading:

```python
flux = SD('FLUX-schnell')
flux.quantize(['transformer', 'text_encoder_2'], model_cpu_offload=True)
```
