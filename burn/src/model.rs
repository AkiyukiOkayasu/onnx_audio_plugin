//! build.rsでONNXから変換したモデルをmoduleにしたもの
//!

include!(concat!(env!("OUT_DIR"), "/model/linear.rs"));
// TODO tanhやsinなどの他のモデルも追加する
