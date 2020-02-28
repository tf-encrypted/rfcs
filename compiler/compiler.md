# TFE Compiler

- logical computations defined by protobuf + MLIR dialect
- each scheme has its own protobuf and MLIR dialect

## Resources

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)
- [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR repo](https://github.com/llvm/llvm-project/tree/master/mlir)
- [MLIR in TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir), including dialects
  - [GraphDef to MLIR dialect](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/tensorflow/translate)
  - [TF Graph transformations moved to MLIR](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/tensorflow/transforms)
- [Cross-language optimizations between Rust and C++ via LLVM](http://blog.llvm.org/2019/09/closing-gap-cross-language-lto-between.html)