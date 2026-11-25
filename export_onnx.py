"""Export CLIDD to a single end-to-end ONNX model.

    image (B, C, H, W) float32 in [0, 1], C = 1 or 3
      -> keypoints   (B, K, 2) float32, (x, y) in input image pixels
         scores      (B, K)    float32, sorted in descending order
         descriptors (B, K, D) float32, L2-normalized

The graph contains the whole pipeline of `CLIDD.forward` in clidd.py: resizing to a multiple of 32, the network,
keypoint selection (NMS, border, score threshold, top-k) and descriptor sampling. The Triton kernel used by
`Model.sample` cannot be exported, so sampling is re-implemented here with `grid_sample` and matrix products,
which computes the same values.

ONNX needs a fixed number of outputs, so K equals --top_k. When fewer points pass the threshold, the remaining
entries are padding with score -inf and zero descriptors; keep the entries with `scores > -inf` (they come first).
The (resized) input must have at least K pixels.

Usage:
    python export_onnx.py M64                       # -> weights/M64.onnx, dynamic batch and image size
    python export_onnx.py M64 --top_k 2048 --static --H 480 --W 640 --output clidd_m64_480x640.onnx
"""
import argparse
import importlib.util
import inspect
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

if importlib.util.find_spec('triton') is None:
    # model/model.py imports the Triton kernel at module level. It is only used by Model.sample, which the
    # export does not call, so a placeholder lets the export run where Triton is unavailable (e.g. CPU only).
    def _unavailable(*args, **kwargs):
        raise RuntimeError('Triton is not installed')
    sys.modules['model.triton_plugin'] = types.SimpleNamespace(deformable_sample_project=_unavailable)

from model.model import Model  # noqa: E402

# Same table as CLIDD.cfgs. Not imported from clidd.py, which requires Triton.
CFGS = {
    'A48': {'c1': 4, 'c2': 4, 'c3': 4, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 48, 'cdetect': 4, 'M': 4, 'max_offset': 128},
    'N64': {'c1': 8, 'c2': 8, 'c3': 8, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 8, 'max_offset': 128},
    'T64': {'c1': 8, 'c2': 16, 'c3': 24, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 8, 'max_offset': 128},
    'S64': {'c1': 8, 'c2': 24, 'c3': 32, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
    'M64': {'c1': 16, 'c2': 32, 'c3': 48, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
    'L64': {'c1': 16, 'c2': 48, 'c3': 96, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
    'G128': {'c1': 16, 'c2': 64, 'c3': 256, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
    'E128': {'c1': 16, 'c2': 64, 'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
    'U128': {'c1': 32, 'c2': 128, 'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
}

OUTPUT_NAMES = ['keypoints', 'scores', 'descriptors']


def stack_weight(weight, channels):
    """Conv weight (O, sum(channels), 1, M) -> (sum over levels of M * C, O), matching `gather` below."""
    blocks = [w[:, :, 0, :].permute(2, 1, 0).reshape(-1, weight.shape[0]) for w in weight.split(channels, 1)]
    return torch.cat(blocks, 0)


def gather(levels, grids):
    """Bilinear samples of each level at its grid (B, N, M, 2), concatenated -> (B, N, sum over levels of M * C)."""
    columns = []
    for level, grid in zip(levels, grids):
        sampled = F.grid_sample(level, grid, mode='bilinear', padding_mode='zeros', align_corners=False)  # B, C, N, M
        columns.append(sampled.permute(0, 2, 3, 1).flatten(2))
    return torch.cat(columns, -1)


class CLIDDExport(nn.Module):
    """CLIDD.forward as a traceable module with a fixed number of keypoints."""

    def __init__(self, model: Model, top_k: int, radius: int = 2, score: float = -5):
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.radius = radius
        self.score_thresh = score
        self.max_offset = float(model.max_offset)

        channels = [m.weight.shape[1] for m in (model.conv1, model.conv2, model.conv3)]
        self.register_buffer('offset_weight', stack_weight(model.dcn.weight.detach(), channels))
        self.register_buffer('offset_bias', model.dcn.bias.detach().clone())
        self.register_buffer('post_weight', stack_weight(model.post_conv.weight.detach(), channels))
        self.register_buffer('post_bias', model.post_conv.bias.detach().clone())

    def sample(self, levels, kpts, size):
        """Same as Model.sample. kpts (B, K, 1, 2) normalized to [-1, 1], size = (W, H) of the network input."""
        offset = gather(levels, [kpts] * 3) @ self.offset_weight + self.offset_bias
        offset = offset.unflatten(-1, (-1, 2)).clamp(-self.max_offset, self.max_offset)
        offset_1, offset_2, offset_3 = (offset / size * 2).chunk(3, -2)
        desc = gather(levels, [kpts + offset_1, kpts + offset_2, kpts + offset_3]) @ self.post_weight + self.post_bias
        return F.normalize(desc, 2, -1)

    def forward(self, x):
        oH, oW = x.shape[-2:]
        nH = oH // 32 * 32
        nW = oW // 32 * 32
        # Unconditional: with equal sizes, align_corners=True resizing returns the input unchanged.
        x = F.interpolate(x, (nH, nW), mode='bilinear', align_corners=True)
        x = x.expand(-1, 3, -1, -1)  # 1 channel -> 3 channels; no-op for 3 channels

        raw_desc, raw_detect = self.model(x)

        detect = raw_detect > self.score_thresh
        if self.radius > 0:
            k = self.radius * 2 + 1
            detect = detect & (raw_detect == F.max_pool2d(raw_detect, k, 1, self.radius))
        raw_detect, detect = raw_detect[:, 0], detect[:, 0]  # B, H, W
        H, W = raw_detect.shape[-2:]
        ys = torch.arange(H, device=x.device)
        xs = torch.arange(W, device=x.device)
        inside = ((ys >= 4) & (ys < H - 4))[:, None] & ((xs >= 4) & (xs < W - 4))[None, :]
        detect = detect & inside

        masked = raw_detect.masked_fill(~detect, float('-inf')).flatten(1)
        scores, idx = masked.topk(self.top_k, dim=1)
        kpts = torch.stack([idx % W, torch.div(idx, W, rounding_mode='floor')], -1).to(x.dtype)  # B, K, 2

        size = torch.stack([nW, nH]).to(x.dtype) if isinstance(nW, torch.Tensor) else \
            torch.tensor([nW, nH], dtype=x.dtype, device=x.device)
        desc = self.sample(list(raw_desc), ((kpts + 0.5) / size * 2 - 1)[:, :, None], size)
        desc = desc * (scores > float('-inf')).to(desc.dtype)[..., None]

        scale = torch.stack([oW / nW, oH / nH]).to(x.dtype) if isinstance(nW, torch.Tensor) else \
            torch.tensor([oW / nW, oH / nH], dtype=x.dtype, device=x.device)
        return kpts * scale, scores, desc


def build(cfg, weights, top_k, radius, score):
    model = Model(**CFGS[cfg])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return CLIDDExport(model.eval(), top_k, radius, score).eval()


def export(module, path, H, W, batch, static, opset):
    dummy = torch.rand(batch, 3, H, W)
    dynamic_axes = None
    if not static:
        dynamic_axes = {'image': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}}
        dynamic_axes.update({name: {0: 'batch'} for name in OUTPUT_NAMES})
    kwargs = {}
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False  # the TorchScript exporter handles the data-dependent sizes used here
    with torch.no_grad():
        torch.onnx.export(module, (dummy,), path, input_names=['image'], output_names=OUTPUT_NAMES,
                          dynamic_axes=dynamic_axes, opset_version=opset, **kwargs)


def check(module, path, H, W, batch, static):
    """Compare ONNX Runtime with the PyTorch module, also at other sizes when exported with dynamic axes."""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    shapes = [(batch, 3, H, W)]
    if not static:
        shapes += [(batch + 1, 1, H // 2 + 7, W // 2 + 13)]
    generator = torch.Generator().manual_seed(0)
    for shape in shapes:
        image = torch.rand(shape, generator=generator)
        # Smooth random image, so there are clear local maxima as in a natural image.
        image = F.interpolate(F.interpolate(image, scale_factor=1 / 8, mode='area'), shape[-2:], mode='bilinear')
        with torch.no_grad():
            expected = [t.numpy() for t in module(image)]
        actual = session.run(None, {'image': image.numpy()})
        e_kpts, e_scores, e_desc = expected
        a_kpts, a_scores, a_desc = actual
        valid = e_scores > float('-inf')
        same = (a_kpts == e_kpts).all(-1) & valid
        print(f'  input {tuple(shape)}: {int(valid.sum())} keypoints, '
              f'{same.sum() / max(valid.sum(), 1) * 100:.2f}% identical positions, '
              f'max |score diff| {abs(a_scores[same] - e_scores[same]).max(initial=0):.2e}, '
              f'max |descriptor diff| {abs(a_desc[same] - e_desc[same]).max(initial=0):.2e}')


def main():
    parser = argparse.ArgumentParser(description='Export CLIDD to an end-to-end ONNX model.')
    parser.add_argument('model', choices=CFGS.keys())
    parser.add_argument('--weights', help='default: weights/<model>.pth')
    parser.add_argument('--output', help='default: weights/<model>.onnx')
    parser.add_argument('--top_k', type=int, default=4096, help='number of keypoints in the output')
    parser.add_argument('--radius', type=int, default=2, help='NMS radius, 0 disables NMS')
    parser.add_argument('--score', type=float, default=-5, help='score threshold')
    parser.add_argument('--H', type=int, default=480, help='input height used for tracing (and the fixed height with --static)')
    parser.add_argument('--W', type=int, default=640, help='input width used for tracing (and the fixed width with --static)')
    parser.add_argument('--batch', type=int, default=1, help='batch size used for tracing (and the fixed one with --static)')
    parser.add_argument('--static', action='store_true', help='fixed input shape instead of dynamic batch, channels, height and width')
    parser.add_argument('--opset', type=int, default=17, help='at least 16 (GridSample)')
    parser.add_argument('--no_check', action='store_true', help='skip the comparison with ONNX Runtime')
    args = parser.parse_args()

    if args.opset < 16:
        parser.error('--opset must be at least 16 for GridSample')
    if (args.H // 32 * 32) * (args.W // 32 * 32) < args.top_k:
        parser.error('--top_k must not exceed the number of pixels of the (resized) input')
    weights = args.weights or f'weights/{args.model}.pth'
    output = args.output or f'weights/{args.model}.onnx'

    module = build(args.model, weights, args.top_k, args.radius, args.score)
    export(module, output, args.H, args.W, args.batch, args.static, args.opset)
    print(f'Exported {output}')

    if not args.no_check:
        try:
            check(module, output, args.H, args.W, args.batch, args.static)
        except ImportError:
            print('onnxruntime is not installed, skipping the check')


if __name__ == '__main__':
    main()
