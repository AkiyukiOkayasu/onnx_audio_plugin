# ONNX Audio Plugin

[![CI](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml/badge.svg)](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml)

This is a minimal example of using ONNX in an audio plugin. There are three versions: using [Tract](https://github.com/sonos/tract), [Burn](https://github.com/tracel-ai/burn), [ort](https://github.com/pykeio/ort) for ONNX execution.  

This project is not production-ready and is not practical. However, it is intended for anyone who wants to learn about a minimal audio plugin using ONNX. If you are looking for practicality then you may want to check out [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

## Features

- Uses ONNX models for audio processing
- Three implementations: using [Tract](https://github.com/sonos/tract), [Burn](https://github.com/tracel-ai/burn), [ort](https://github.com/pykeio/ort)
- Supports VST3 and CLAP plugin formats

## Installation

You can download the VST3 and CLAP plug-ins from the [Release](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/releases/latest).

## ONNX

### linear.onnx

[linear.onnx](onnx/linear.onnx) is an ONNX model that performs phase inversion using minimal elements. This model is used by default. It was created using [linear.py](onnx/linear.py).

### tanh.onnx

[tanh.onnx](onnx/tanh.onnx) is an ONNX model that mimics the hyperbolic tangent (tanh) function. This model is used for non-linear audio processing. It was created using [tanh.py](onnx/tanh.py).

### Visualizing ONNX Models

To visualize ONNX graphs, [Netron](https://netron.app/) is a quick and easy way to do so. Netron allows you to visually inspect the structure of ONNX models.

## Build (For Developers)

If you want to build manually, follow these steps after installing [Rust](https://rustup.rs/):

```shell
cargo xtask bundle -p onnx_plug_tract -p onnx_plug_burn -p onnx_plug_ort --release
```

### Install build plugin

#### macOS

```shell
PLUGIN_NAME="Onnx Plug Tract"
rsync -ahv --delete target/bundled/${PLUGIN_NAME}.clap/ ~/Library/Audio/Plug-Ins/CLAP/${PLUGIN_NAME}.clap
rsync -ahv --delete target/bundled/${PLUGIN_NAME}.vst3/ ~/Library/Audio/Plug-Ins/VST3/${PLUGIN_NAME}.vst3
```

```shell
PLUGIN_NAME="Onnx Plug Burn"
rsync -ahv --delete target/bundled/"${PLUGIN_NAME}".clap/ ~/Library/Audio/Plug-Ins/CLAP/"${PLUGIN_NAME}".clap
rsync -ahv --delete target/bundled/${PLUGIN_NAME}.vst3/ ~/Library/Audio/Plug-Ins/VST3/${PLUGIN_NAME}.vst3
```

```shell
PLUGIN_NAME="Onnx Plug Ort"
rsync -ahv --delete target/bundled/"${PLUGIN_NAME}".clap/ ~/Library/Audio/Plug-Ins/CLAP/"${PLUGIN_NAME}".clap
rsync -ahv --delete target/bundled/${PLUGIN_NAME}.vst3/ ~/Library/Audio/Plug-Ins/VST3/${PLUGIN_NAME}.vst3
```

### Validation

#### CLAP

```shell
PLUGIN_NAME="Onnx Plug Tract"
clap-validator validate target/bundled/"${PLUGIN_NAME}".clap
```

```shell
PLUGIN_NAME="Onnx Plug Burn"
clap-validator validate target/bundled/"${PLUGIN_NAME}".clap
```

```shell
PLUGIN_NAME="Onnx Plug Ort"
clap-validator validate target/bundled/"${PLUGIN_NAME}".clap
```

#### VST3

```shell
PLUGIN_NAME="Onnx Plug Tract"
pluginval --verbose --strictness-level 5 target/bundled/${PLUGIN_NAME}.vst3
```

```shell
PLUGIN_NAME="Onnx Plug Burn"
pluginval --verbose --strictness-level 5 target/bundled/${PLUGIN_NAME}.vst3
```

```shell
PLUGIN_NAME="Onnx Plug Ort"
pluginval --verbose --strictness-level 5 target/bundled/${PLUGIN_NAME}.vst3
```
