use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::io::BufReader;
use tract_onnx::prelude::*;

fn pre_process(c: &mut Criterion) {
    let input_vec = tract_ndarray::Array4::<f32>::zeros((1, 1, 1, 1)); //tractの入出力は4階tensorを前提としている
    c.bench_function("Pre process", |b| {
        b.iter(|| {
            let tensor = input_vec.clone().into_tensor();
        })
    });
}

fn onnx(c: &mut Criterion) {
    let onnx_model = include_bytes!("../linear.onnx");
    let model = onnx()
        .model_for_read(&mut BufReader::new(&onnx_model[..]))
        .unwrap()
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1, 1, 1, 1]),
        )
        .unwrap()
        .with_output_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1, 1, 1, 1]),
        )
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    c.bench_function("Run ONNX model", |b| {
        b.iter(|| {
            // for sample in 0..48000 {
            //     for channel in 0..pcm_specs.num_channels as u32 {
            //         let _s = reader.read_sample(channel, sample).unwrap();
            //     }
            // }
        })
    });
}

fn post_process(c: &mut Criterion) {
    let output: SmallVec<[TValue; 4]> = tvec!([0, 0, 0, 0]);
    c.bench_function("Post process", |b| b.iter(|| {}));
}

criterion_group!(benches, pre_process, onnx, post_process);
criterion_main!(benches);
