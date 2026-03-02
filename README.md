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

### Setup

```bash
# Create and activate a virtual environment
python3 -m venv source/.venv
source source/.venv/bin/activate

# Install system dependencies (required for PyGObject)
sudo apt install -y libcairo2-dev libgirepository1.0-dev pkg-config python3-dev

# Install DeepStream Python bindings (pyds)
# Download the matching wheel from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases
pip install pyds-<version>-py3-none-linux_aarch64.whl

# Install Python dependencies
pip install -r source/requirements.txt
```

### Run

```bash
source source/.venv/bin/activate
python source/app.py
```

On first run, TensorRT will build an engine file from the ONNX model.
This takes ~10-15 minutes but only happens once — subsequent runs load
the cached `.engine` file in seconds.
