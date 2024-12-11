use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../onnx/linear.onnx")
        .out_dir("model/")
        .run_from_script();

    ModelGen::new()
        .input("../onnx/tanh.onnx")
        .out_dir("model/")
        .run_from_script();
}
