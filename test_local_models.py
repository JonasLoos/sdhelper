"""
Test script to verify that all locally available models load correctly,
generate images without NaN values, and extract representations properly.

Usage:
    python tests/test_local_models.py

Options:
    --model NAME    Test only a specific model by name
    --skip-gen      Skip generation tests (faster but less thorough)
    --verbose       Show verbose output
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image


def check_for_nan(tensor: torch.Tensor, name: str) -> list[str]:
    """Check a tensor for NaN values and return list of issues."""
    issues = []
    if tensor is None:
        return issues
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        issues.append(f"{name}: {nan_count}/{total} NaN values detected")
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        issues.append(f"{name}: {inf_count}/{total} Inf values detected")
    return issues


def get_extract_positions(model) -> list[str]:
    """Get appropriate extract positions based on model type."""
    from sdhelper.models import SDUnet, SD3Base, FLUXBase, FLUX2_dev
    
    if isinstance(model, SDUnet):
        # U-Net based models
        return ['mid_block', 'up_blocks[0]']
    elif isinstance(model, SD3Base):
        # SD3 transformer models
        return ['transformer_blocks[0]']
    elif isinstance(model, FLUX2_dev):
        # FLUX.2 models
        return ['transformer_blocks[0]']
    elif isinstance(model, FLUXBase):
        # FLUX.1 models
        return ['transformer_blocks[0]']
    else:
        return []


def create_test_image(width: int = 512, height: int = 512) -> Image.Image:
    """Create a simple test image with gradient pattern."""
    # Create a gradient pattern
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = int(255 * x / width)      # Red gradient
            arr[y, x, 1] = int(255 * y / height)     # Green gradient
            arr[y, x, 2] = 128                        # Constant blue
    return Image.fromarray(arr)


def test_model(model_name: str, skip_gen: bool = False, verbose: bool = False) -> dict:
    """Test a single model for loading, generation, and repr extraction."""
    from sdhelper.models import SD, SDBase, _get_all_subclasses, _normalize_model_name
    
    result = {
        'name': model_name,
        'load': {'status': 'pending', 'issues': []},
        'generate': {'status': 'skipped', 'issues': []},
        'img2repr': {'status': 'skipped', 'issues': []},
    }
    
    # Try to load the model
    try:
        if verbose:
            print(f"  Loading {model_name}...")
        model = SD(model_name, disable_progress_bar=True, local_files_only=True)
        result['load']['status'] = 'passed'
        if verbose:
            print(f"  ✓ Loaded successfully")
    except Exception as e:
        result['load']['status'] = 'failed'
        result['load']['issues'].append(str(e))
        if verbose:
            print(f"  ✗ Load failed: {e}")
        return result
    
    # Test generation
    if not skip_gen:
        try:
            if verbose:
                print(f"  Testing generation...")
            gen_result = model("a test image", steps=1, seed=42)
            issues = []
            
            # Check result image
            if gen_result.result_image is None:
                issues.append("result_image is None")
            else:
                img_arr = np.array(gen_result.result_image)
                if np.isnan(img_arr).any():
                    issues.append("result_image contains NaN")
            
            # Check latents and tensors
            if gen_result.result_latent is not None:
                issues.extend(check_for_nan(gen_result.result_latent, 'result_latent'))
            if gen_result.result_tensor is not None:
                issues.extend(check_for_nan(gen_result.result_tensor, 'result_tensor'))
            
            # Check representations if available
            if gen_result.representations is not None:
                for pos, tensor in gen_result.representations.data.items():
                    issues.extend(check_for_nan(tensor, f'repr[{pos}]'))
            
            result['generate']['status'] = 'passed' if not issues else 'warning'
            result['generate']['issues'] = issues
            if verbose:
                if issues:
                    print(f"  ⚠ Generation completed with warnings: {issues}")
                else:
                    print(f"  ✓ Generation passed")
        except Exception as e:
            result['generate']['status'] = 'failed'
            result['generate']['issues'].append(str(e))
            if verbose:
                print(f"  ✗ Generation failed: {e}")
    
    # Test img2repr (representation extraction)
    extract_positions = get_extract_positions(model)
    if extract_positions:
        try:
            if verbose:
                print(f"  Testing img2repr with positions {extract_positions}...")
            
            # Create a test image
            test_img = create_test_image(512, 512)
            
            # Try to extract representations
            repr_result = model.img2repr(test_img, extract_positions, step=10, seed=42)
            
            issues = []
            for pos in extract_positions:
                if pos in repr_result.data:
                    tensor = repr_result.data[pos]
                    issues.extend(check_for_nan(tensor, f'img2repr[{pos}]'))
                else:
                    issues.append(f"Position {pos} not in result")
            
            result['img2repr']['status'] = 'passed' if not issues else 'warning'
            result['img2repr']['issues'] = issues
            if verbose:
                if issues:
                    print(f"  ⚠ img2repr completed with warnings: {issues}")
                else:
                    print(f"  ✓ img2repr passed")
        except NotImplementedError as e:
            result['img2repr']['status'] = 'skipped'
            result['img2repr']['issues'].append(f"Not implemented: {e}")
            if verbose:
                print(f"  ⊘ img2repr not implemented: {e}")
        except Exception as e:
            result['img2repr']['status'] = 'failed'
            result['img2repr']['issues'].append(str(e))
            if verbose:
                print(f"  ✗ img2repr failed: {e}")
    else:
        result['img2repr']['status'] = 'skipped'
        result['img2repr']['issues'].append("No extract positions defined for this model type")
        if verbose:
            print(f"  ⊘ img2repr skipped (no extract positions)")
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def get_locally_available_models() -> list[str]:
    """Get list of model names that are available locally."""
    from sdhelper.models import SD, SDBase, _get_all_subclasses
    
    available = []
    
    # Get all model classes
    for cls in _get_all_subclasses(SDBase):
        if not hasattr(cls, 'name') or not isinstance(cls.name, str):
            continue
        if not hasattr(cls, 'full_name'):
            continue
        
        model_name = cls.name
        full_name = cls.full_name
        
        # Try to check if model is locally available
        try:
            from huggingface_hub import try_to_load_from_cache
            
            # Check if at least the model_index.json or config file exists locally
            cache_result = try_to_load_from_cache(full_name, "model_index.json")
            if cache_result is not None:
                available.append(model_name)
                continue
            
            # Try config.json as fallback
            cache_result = try_to_load_from_cache(full_name, "config.json")
            if cache_result is not None:
                available.append(model_name)
                continue
                
        except Exception:
            pass
    
    return sorted(available)


def main():
    parser = argparse.ArgumentParser(description='Test locally available SD models')
    parser.add_argument('--model', type=str, help='Test only a specific model by name')
    parser.add_argument('--skip-gen', action='store_true', help='Skip generation tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list', action='store_true', help='Only list locally available models')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDHelper Local Model Test Suite")
    print("=" * 60)
    
    # Get locally available models
    print("\nScanning for locally available models...")
    if args.model:
        models_to_test = [args.model]
        print(f"Testing specific model: {args.model}")
    else:
        models_to_test = get_locally_available_models()
        print(f"Found {len(models_to_test)} locally available models:")
        for m in models_to_test:
            print(f"  - {m}")
    
    if args.list:
        return 0
    
    if not models_to_test:
        print("\nNo locally available models found!")
        print("Download some models first to run tests.")
        return 1
    
    # Run tests
    print("\n" + "-" * 60)
    print("Running Tests")
    print("-" * 60)
    
    results = []
    for model_name in models_to_test:
        print(f"\n[{model_name}]")
        result = test_model(model_name, skip_gen=args.skip_gen, verbose=args.verbose)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    warnings = 0
    failed = 0
    skipped = 0
    
    for r in results:
        name = r['name']
        
        # Determine overall status
        statuses = [r['load']['status'], r['generate']['status'], r['img2repr']['status']]
        if 'failed' in statuses:
            status = '✗ FAILED'
            failed += 1
        elif 'warning' in statuses:
            status = '⚠ WARNING'
            warnings += 1
        elif all(s == 'skipped' for s in statuses[1:]) and statuses[0] == 'passed':
            status = '○ LOAD OK'
            passed += 1
        elif 'passed' in statuses:
            status = '✓ PASSED'
            passed += 1
        else:
            status = '⊘ SKIPPED'
            skipped += 1
        
        # Print line
        load_s = {'passed': '✓', 'failed': '✗', 'skipped': '⊘', 'pending': '?'}[r['load']['status']]
        gen_s = {'passed': '✓', 'failed': '✗', 'skipped': '⊘', 'warning': '⚠', 'pending': '?'}[r['generate']['status']]
        repr_s = {'passed': '✓', 'failed': '✗', 'skipped': '⊘', 'warning': '⚠', 'pending': '?'}[r['img2repr']['status']]
        
        print(f"  {name:25} Load:{load_s} Gen:{gen_s} Repr:{repr_s}  {status}")
        
        # Print issues if any
        all_issues = r['load']['issues'] + r['generate']['issues'] + r['img2repr']['issues']
        if all_issues and (args.verbose or 'failed' in statuses):
            for issue in all_issues:
                print(f"    → {issue}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(results)} models | Passed: {passed} | Warnings: {warnings} | Failed: {failed} | Skipped: {skipped}")
    print("-" * 60)
    
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
