# Onnx Audio Plugin

[![CI](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml/badge.svg)](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml)

Audio plug-in minimum example using ONNX.  
This project is not product ready and is not practical. However, it is for anyone who wants to know about a minimum audio plug-in using ONNX.  

If you are looking for practicality then you may want to check out [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

## Install

You can download the VST3 and CLAP plug-ins from the [Release](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/releases/latest).

## Building

After installing [Rust](https://rustup.rs/), you can compile Onnx Audio Plugin as follows:

```shell
cargo xtask bundle onnx_audio_plugin --release
```

## ONNX

[linear.onnx](linear.onnx) is a ONNX that does phase inversion using a minimum element.  
![linearonnx](https://user-images.githubusercontent.com/6957368/223927260-67f8b17d-13da-4b6b-a651-b9e236d3bc17.png)  

To check ONNX graphs, [Netron](https://netron.app/) is a quick and easy way to do so.  
linear.onnx has been created with [LinearONNX.ipynb](LinearONNX.ipynb).
