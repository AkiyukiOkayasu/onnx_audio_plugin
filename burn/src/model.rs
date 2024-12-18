//! build.rsでONNXから変換したモデルをmoduleにしたもの
//!

pub mod tanh {
    include!(concat!(env!("OUT_DIR"), "/model/tanh.rs"));
}

pub mod linear {
    include!(concat!(env!("OUT_DIR"), "/model/linear.rs"));
}
