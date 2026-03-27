# Luggage Watch Inference

DeepStream inference pipeline for YOLO-based suspicious luggage detection
on an Embedded System, NVIDIA Jetson Orin Nano.

## Quick Start

These are instructions for running the DeepStream pipeline on a Jetson
Orin Nano, the steps may differ for other platforms.

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6.x
- DeepStream SDK 7.1 installed ([installation guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html))
- Python 3.10+
- [DeepStream YOLO Parser](https://github.com/marcoslucianops/DeepStream-Yolo) from marcoslucianops

### Setup

```bash
# Create and activate a virtual environment
python3 -m venv source/.venv
source source/.venv/bin/activate

# Install system dependencies (required for PyGObject)
sudo apt install -y libcairo2-dev libgirepository1.0-dev pkg-config python3-dev

# Install DeepStream Python bindings (pyds)
# Download the matching wheel from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases
pip install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# Install Python dependencies
pip install -r source/requirements.txt
```

#### Compiling the YOLO Parser Library

```bash
# Clone the repository
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo

# Compile the library for the matching CUDA version
# For Jetpack 6.x and DeepStream 7.1
export CUDA_VER=12.6 
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

# Move the library to the inference config folder
cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /path/to/your/luggage-watch-inference/config/
```

**Note**: the `make` command may fail searching for a specific directory `-I/opt/nvidia/deepstream/deepstream/sources/includes`. If you installed DeepStream using `apt` this can happen as it is installed in a versioned folder (e.g., `/opt/nvidia/deepstream/deepstream-7.1`). In order to fix this, create a symlink between the versioned and non-versioned paths.

#### Downloading the Re-ID Model

```bash
sudo wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' -P /path/to/your/luggage-watch-inference/model/
```

### Run (Local)

```bash
# To force max performance out of the Jetson
sudo nvpmodel -m 0
sudo jetson_clocks

source source/.venv/bin/activate

# Single video
python source/app.py --input /path/to/video.mp4 --output-dir ./outputs/mot

# Dataset directory (one .txt MOT file per video)
python source/app.py --input /path/to/dataset --recursive --output-dir ./outputs/mot
```

### MOT Output Format

Each output file is in MOT-style CSV rows:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
```

Class IDs:
- `0` = person
- `1` = luggage

### Headless Docker Batch Export (for Ground Truth Pre-Annotation)

Build the image:

```bash
docker build -t luggage-watch-inference:latest .
```

The Docker build compiles the custom YOLO parser library and downloads the
Re-ID model automatically, so no separate host-side setup is required.

Run on a mounted dataset directory:

```bash
docker run --rm --runtime nvidia \
	-v /path/to/dataset:/data:ro \
	-v /path/to/output:/out \
	luggage-watch-inference:latest \
	--input /data \
	--recursive \
	--output-dir /out \
	--no-overlay
```

For a single input video in the mounted dataset:

```bash
docker run --rm --runtime nvidia \
	-v /path/to/dataset:/data:ro \
	-v /path/to/output:/out \
	luggage-watch-inference:latest \
	--input /data/example.mp4 \
	--output-dir /out
```

On first run, TensorRT will build an engine file from the ONNX model.
This takes ~10-15 minutes but only happens once; subsequent runs load
the cached `.engine` file in seconds.
