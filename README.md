# CLIDD: Cross-Layer Independent Deformable Description for Efficient and Discriminative Local Feature Representation

We introduce Cross-Layer Independent Deformable Description (CLIDD), a high-performance local feature representation method designed for scalability. It provides a diverse range of models, from an ultra-compact 4,252 (0.004M) parameter variant to high-performance configurations exceeding 200 FPS on edge devices, delivering a robust and efficient solution for image description and matching.

<img src="assets/visualization.png" width='1000'>

The visualization below shows the A48 configuration tested on the KITTI dataset.

<img src="assets/KITTI-A48.gif" width='1000'>

## Model Zoo

Available model variants are listed below. FPS results are measured on an NVIDIA Jetson Orin-NX.

| Model | Dim | MP | FPS |
| :--- | :--- | :--- | :--- |
| A48 | 48 | 0.004 | 881.1 |
| N64 | 64 | 0.019 | 842.7 |
| T64 | 64 | 0.043 | 803.5 |
| S64 | 64 | 0.100 | 746.4 |
| M64 | 64 | 0.168 | 625.1 |
| L64 | 64 | 0.347 | 591.4 |
| G128 | 128 | 2.254 | 475.4 |
| E128 | 128 | 3.508 | 431.3 |
| U128 | 128 | 4.400 | 281.4 |

## Install Dependencies

Set up the environment by installing the required packages:

```shell
pip install -r requirements.txt
```

## Usage

Run the demo by specifying the input source and model:

```shell
python demo_seq.py [camera / PATH_TO_VIDEO_FILE] [Model]
```

Example for camera input with the A48 model:

```shell
python demo_seq.py camera A48
```

**Note:** Initial execution or significant changes in the number of points may cause brief stutters as Triton re-optimizes the kernels.

## ONNX Export

Export a model to a single end-to-end ONNX file (network, keypoint selection and descriptor sampling). Triton is not required for the export; the deformable sampling is expressed with standard ONNX operators (`GridSample`, `MatMul`).

```shell
pip install onnx onnxruntime
python export_onnx.py M64                  # -> weights/M64.onnx
```

| | Name | Shape | Description |
| :--- | :--- | :--- | :--- |
| Input | `image` | (B, 1 or 3, H, W) | float32 in [0, 1] |
| Output | `keypoints` | (B, K, 2) | (x, y) in input pixels |
| | `scores` | (B, K) | sorted in descending order, `-inf` for padding |
| | `descriptors` | (B, K, D) | L2-normalized, zeros for padding |

K equals `--top_k` (default 4096). If fewer keypoints pass the threshold, the remaining entries are padding; keep the ones with `scores > -inf`. Batch size and image size are dynamic by default (`--static` fixes them to `--batch`, `--H`, `--W`). Like `CLIDD.forward`, images whose size is not a multiple of 32 are resized inside the model and the keypoints are scaled back. `--radius` and `--score` set the NMS radius and the score threshold. After exporting, the script compares ONNX Runtime with PyTorch.

```python
import onnxruntime as ort

session = ort.InferenceSession('weights/M64.onnx')
keypoints, scores, descriptors = session.run(None, {'image': image})  # image: (1, 3, H, W) float32 in [0, 1]
valid = scores[0] > -float('inf')
keypoints, descriptors = keypoints[0][valid], descriptors[0][valid]
```
