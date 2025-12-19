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
# all_models = [cls for cls in all_models if not issubclass(cls, sdhelper.models.SDUnet)]
# all_models = [sdhelper.models.FLUX1_dev, sdhelper.models.FLUX1_schnell, sdhelper.models.FLUX1_Krea]

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
        if hasattr(p.transformer, 'transformer_blocks'):
            return "transformer_blocks[12]"
        elif hasattr(p.transformer, 'single_transformer_blocks'):
            return "single_transformer_blocks[12]"
        elif hasattr(p.transformer, 'layers'):
            return "layers[12]"
        else:
            raise ValueError(f'Unknown transformer architecture: {p.transformer.__class__.__name__}')
    else:
        raise ValueError(f'Unknown pipeline architecture: {p.__class__.__name__}')

try:
    for model in all_models:
        print()
        print("#"*100)
        print(model.name)
        print("#"*100)
        dir = test_dir / model.name
        dir.mkdir(parents=True, exist_ok=True)
        try:
            f = model(local_files_only=False)
            results[model.name] = {}
            results[model.name]['initialization'] = "✅"
            # image generation test
            try:
                f('a large tree').result_image.save(dir / 'tree.jpg')
                f('a cat on a mat').result_image.save(dir / 'cat.jpg')
                results[model.name]['image_generation'] = "✅"
            except Exception as e:
                print("Failed to generate images")
                print(f'Error: {e}')
                print(traceback.format_exc())
                results[model.name]['image_generation'] = "❌"

            # vae test
            try:
                latents = f.encode_latents([PIL.Image.open('sample.jpg')])
                images = f.decode_latents(latents)
                images[0].save(dir / 'vae_decoded.jpg')
                results[model.name]['vae'] = "✅"
            except Exception as e:
                print("Failed to encode/decode latents")
                print(f'Error: {e}')
                print(traceback.format_exc())
                results[model.name]['vae'] = "❌"

            # extraction test
            try:
                r = f.img2repr('sample.jpg', [get_extract_position(f)], 100)
                sim = r.cosine_similarity(r)
                n = sim.shape[0]
                sim = sim.cpu().numpy()[n//2,n//2,:,:]/2 + 0.5
                sim_img = PIL.Image.fromarray((sim*255).astype(np.uint8))
                sim_img.save(dir / 'similarity.jpg')
                sim_normalized = (sim - sim.min()) / (sim.max() - sim.min())
                sim_img = PIL.Image.fromarray((sim_normalized*255).astype(np.uint8))
                sim_img.save(dir / 'similarity_normalized.jpg')
                results[model.name]['extraction'] = "✅"
            except Exception as e:
                print("Failed to extract representations")
                print(f'Error: {e}')
                print(traceback.format_exc())
                results[model.name]['extraction'] = "❌"
        except Exception as e:
            if "huggingface_hub.errors.LocalEntryNotFoundError:" not in str(traceback.format_exc()):
                results[model.name] = {
                    'initialization': "❌",
                    'image_generation': " ",
                    'vae': " ",
                    'extraction': " "
                }
                print("Failed to initialize model")
                print(f'Error: {e}')
            else:
                print(f"Model {model.name} not available locally")
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
    print(f'{"Model":<20} {"Init":<10} {"Image gen":<10} {"VAE":<10} {"Extraction":<10}')
    print('-'*100)
    for model_name, result in results.items():
        print(f'{model_name:<20} {result.get("initialization", " "):<10} {result.get("image_generation", " "):<10} {result.get("vae", " "):<10} {result.get("extraction", " "):<10}')
    print('='*100)
