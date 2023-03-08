# Onnx Audio Plugin

[![CI](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml/badge.svg)](https://github.com/AkiyukiOkayasu/onnx_audio_plugin/actions/workflows/ci.yaml)

Audio plug-in example using ONNX.  
This project has just started and most of the features have not yet been implemented or do not work properly.  

## Building

After installing [Rust](https://rustup.rs/), you can compile Onnx Test as follows:

```shell
cargo xtask bundle onnx_audio_plugin --release
```

```shell
pluginval target/bundled/onnx_audio_plugin.vst3whi
```

## ONNX

linear.onnx is phase invertor.  
[Netron](https://netron.app/)  
