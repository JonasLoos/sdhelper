from dataclasses import dataclass
from PIL.Image import Image as PILImage
from typing import Optional, Callable
import torch
import numpy as np

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
