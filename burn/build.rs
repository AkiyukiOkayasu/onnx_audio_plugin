use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../onnx/linear/linear.onnx")
        .out_dir("model/")
        .run_from_script();
}
