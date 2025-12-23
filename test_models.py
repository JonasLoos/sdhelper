import torch
import gc
import numpy as np
import pathlib
import traceback
import PIL.Image

import sdhelper


test_dir = pathlib.Path('test')
test_dir.mkdir(parents=True, exist_ok=True)

all_models = [cls for cls in sdhelper.models._get_all_subclasses(sdhelper.models.SDBase) if hasattr(cls, 'name') and isinstance(cls.name, str) and 'base' not in cls.name.lower()]
# all_models = [sdhelper.models.AuraFlow, sdhelper.models.Playground_V2_5]
# all_models = [cls for cls in all_models if not issubclass(cls, sdhelper.models.SDUnet) and not issubclass(cls, sdhelper.models.SD3Base)]
# all_models = [sdhelper.models.FLUX1_dev, sdhelper.models.FLUX1_schnell, sdhelper.models.FLUX1_Krea]
all_models = [m for m in all_models if m != sdhelper.models.ZImageTurbo]

print(all_models)


results = {}


def get_extract_position(f):
    p = f.pipeline
    if hasattr(p, 'unet'):
        if hasattr(p.unet, 'up_blocks'):
            return "up_blocks[1]"
        else:
            raise ValueError(f'Unknown unet architecture: {p.unet.__class__.__name__}')
    elif hasattr(p, 'transformer'):
        if hasattr(p.transformer, 'single_transformer_blocks'):
            return "single_transformer_blocks[12]"
        elif hasattr(p.transformer, 'transformer_blocks'):
            return "transformer_blocks[6]"
        elif hasattr(p.transformer, 'layers'):
            return "layers[12]"
        else:
            raise ValueError(f'Unknown transformer architecture: {p.transformer.__class__.__name__}')
    else:
        raise ValueError(f'Unknown pipeline architecture: {p.__class__.__name__}')


def test_generation(model: sdhelper.models.SDBase, dir: pathlib.Path):
    model('a large tree').result_image.save(dir / 'tree.jpg')


def test_vae(model: sdhelper.models.SDBase, dir: pathlib.Path):
    latents = model.encode_latents([PIL.Image.open('sample.jpg')])
    images = model.decode_latents(latents)
    images[0].save(dir / 'vae_decoded.jpg')


def test_extraction(model: sdhelper.models.SDBase, dir: pathlib.Path):
    r = model.img2repr('sample.jpg', [get_extract_position(model)], 100)
    assert torch.isfinite(r.concat()).all(), f'{model.name} returned NaN or Inf: {r.concat()}'
    sim = r.cosine_similarity(r)
    n = sim.shape[0]
    sim = sim.cpu().numpy()[n//2,n//2,:,:]/2 + 0.5
    sim_img = PIL.Image.fromarray((sim*255).astype(np.uint8))
    sim_img.save(dir / 'similarity.jpg')


def test_extraction_raw(model: sdhelper.models.SDBase, dir: pathlib.Path):
    p = get_extract_position(model)
    r = model.img2repr('sample.jpg', [p], 100, raw=True)
    assert isinstance(r, dict), f'{model.name} returned {type(r)}: {r}'
    assert p in r, f'{model.name} did not return representations for {p}'
    assert torch.isfinite(r[p]).all(), f'{model.name} returned NaN or Inf: {r[p]}'
    r.cosine_similarity(r)


test_functions = {
    'gen': test_generation,
    'vae': test_vae,
    'repr': test_extraction,
    'repr_raw': test_extraction_raw,
}

try:
    for model_cls in all_models:
        print()
        print("#"*100)
        print(model_cls.name)
        print("#"*100)
        dir = test_dir / model_cls.name
        dir.mkdir(parents=True, exist_ok=True)
        try:
            f = model_cls(local_files_only=False)
            results[model_cls.name] = {}
            results[model_cls.name]['initialization'] = "✅"
            for test_name, test_function in test_functions.items():
                try:
                    test_function(f, dir)
                    results[model_cls.name][test_name] = "✅"
                except Exception as e:
                    print(f"Failed to {test_name}")
                    print(f'Error: {e}')
                    print(traceback.format_exc())
                    results[model_cls.name][test_name] = "❌"
        except Exception as e:
            if "huggingface_hub.errors.LocalEntryNotFoundError:" not in str(traceback.format_exc()):
                results[model_cls.name] = {
                    'initialization': "❌",
                    **{test_name: " " for test_name in test_functions}
                }
                print(f"Failed to initialize {model_cls.name}")
                print(f'Error: {e}')
            else:
                print(f"{model_cls.name} not available locally")
        finally:
            if 'f' in locals():
                del f
            gc.collect()
            torch.cuda.empty_cache()
finally:
    # print results
    print('='*100)
    print('Results')
    print('='*100)
    def print_row(model_name, results):
        print(f'{model_name:<20} {results.get("initialization", " "):<10}', end=' ')
        for test_name in test_functions.keys():
            print(f'{results.get(test_name, " "):<10}', end=' ')
        print()

    print_row('Model', {test_name: test_name for test_name in test_functions.keys()})
    print('-'*100)
    for model_name, result in results.items():
        print_row(model_name, result)
    print('='*100)
