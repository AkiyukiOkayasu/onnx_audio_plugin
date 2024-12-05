//! build.rsでONNXから変換したモデルをmoduleにしたもの
//!

/// ONNXから変換したモデル
pub mod model {
    // tanh
    include!(concat!(env!("OUT_DIR"), "/model/linear.rs"));
    // TODO linearやsinなどの他のモデルも追加する
}
