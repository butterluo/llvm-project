// RUN: toyc-ch4 %s -emit=mlir -opt 2>&1 | FileCheck %s

// Check the result of inlining+shape inference on an input module.

toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}

// CHECK-NOT: toy.func private @multiply_transpose
// CHECK-NOT: tensor<*xf64>

// CHECK-LABEL: toy.func @main()
// CHECK:         [[VAL_0:%.*]] = toy.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
// CHECK:         [[VAL_1:%.*]] = toy.transpose([[VAL_0]] : tensor<2x3xf64>) to tensor<3x2xf64>
// CHECK:         [[VAL_2:%.*]] = toy.mul [[VAL_1]], [[VAL_1]] : tensor<3x2xf64>
// CHECK:         toy.print [[VAL_2]] : tensor<3x2xf64>
// CHECK:         toy.return



// 参考: https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg
//在Chapter3里面我们学到了如何在MLIR里面实现表达式重写，但上面也有一个非常明显的问题：我们为Toy语言实现的Pass在其它的Dialect抽象中没办法重用，
//因为这里只是针对Toy语言的一些Operation的特化操作，如果为每种Dialect实现每种转化会导致大量重复代码。所以，这一节以两个例子为例讲解如何在MLIR中实现泛化的表达式。
// examples\toy\CMakeLists.txt
//  > examples\toy\Ch4\CMakeLists.txt <-提供*.td生成的cpp examples\toy\Ch4\include\toy\CMakeLists.txt
// ./toyc-ch4 ../../mlir/test\Examples\Toy\Ch4\shape_inference.mlir -emit=mlir -mlir-print-debuginfo
// toyc.cpp.main()->dumpMLIR()->mlirGen()->'mlirGen(ExprAST'

//内联Pass
// mlir/Dialect.cpp 实现和注册内联pass接口
// Ch5/include/toy/Ops.td)文件中加入 CallInterfaces.td 以定义GenericCallOp用于指明IR在toy.generic_call的地方调用了其它函数
// 在Ch5/mlir/Dialect.cpp中客户化GenericCallOp生成的cpp中的某些函数,比如CallOpInterface.getCallableForCallee()
// 添加cast操作:Ops.td加入CastOp ,在mlir/Dialect.cpp中实现CastOpInterface.areCastCompatible()
//      这是因为在函数调用时，输入张量的类型是确定的。但在函数定义的时候，输入张量的类型是不确定的（泛化类型，这一点可以从上面的原始版本MLIR表达式中看出来）。因此在调用的时候就需要一个隐藏的数据类型转换，否则无法进行内联操作
// 在Ch5/mlir/Dialect.cpp中实现DialectInlinerInterface.materializeCallConversion(),使得在inline函数的参数与输入的数据的类型不一致时,会尝试cast,以便能顺利inline
// toyc.cpp 将内联Pass添加到优化pipline中 'pm.addPass(mlir::createInlinerPass())'

//推断shape,以避免无谓的shape type的cast
//  在examples\toy\Ch4\include\toy\ShapeInferenceInterface.td定义ShapeInferenceOpInterface接口(或特征)
//  Ops.td中需要做shape推断的op都需要添加ShapeInferenceOpInterface特征,比如MulOp, CastOp
//  mlir/Dialect.cpp中针对添加了ShapeInferenceOpInterface特征的op,实现ShapeInferenceOpInterface.inferShapes(),比如MulOp, CastOp
//  examples\toy\Ch4\mlir\ShapeInferencePass.cpp创建一个pass对有ShapeInferenceInterface特征的op进行shape推断,重点看runOnOperation()
// toyc.cpp 将shape推断Pass添加到优化pipline中 'optPM.addPass(mlir::toy::createShapeInferencePass());'