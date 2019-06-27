/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file buffer_op.cc
 * \brief placeholder for ring buffer operator
 */
#include <nnvm/top/tensor.h>
#include <nnvm/op.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

struct RingBufferParam : public dmlc::Parameter<RingBufferParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(RingBufferParam) {
    DMLC_DECLARE_FIELD(axis)
      .describe("The axis along which to buffer previous inputs");
  }
};

DMLC_REGISTER_PARAMETER(RingBufferParam);

inline bool RingBufferShape(const NodeAttrs& attrs,
                            std::vector<TShape>* in_attrs,
                            std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& input_shape = in_attrs->at(0);
  const TShape& buffer_shape = in_attrs->at(1);

  const RingBufferParam& param = get<RingBufferParam>(attrs.parsed);
  const size_t bufferAxis = static_cast<size_t>(param.axis < 0
    ? static_cast<int>(buffer_shape.ndim()) + param.axis : param.axis);
  CHECK_LT(bufferAxis, buffer_shape.ndim())
    << "Buffer axis out of range: " << param.axis;
  CHECK_EQ(buffer_shape.ndim(), input_shape.ndim())
    << "buffer and input must have same number of dimensions, buffer_shape = "
    << buffer_shape << ", input_shape = " << input_shape;
  for (size_t i = 0; i < buffer_shape.ndim(); ++i) {
    if (i != bufferAxis) {
      CHECK_EQ(input_shape[i], buffer_shape[i])
        << "Input shape inconsistent with buffer shape in dimension " << i;
    }
  }
  CHECK_LE(input_shape[bufferAxis], buffer_shape[bufferAxis])
    << "Input must be shorter than buffer in dimension " << param.axis;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, buffer_shape);

  return true;
}

NNVM_REGISTER_OP(ring_buffer)
.describe(R"code(Implements a ring buffer, in which a set number of past
inputs are internally cached. The output of the buffer operator is the latest
[length_buffer] outputs.)code" NNVM_ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RingBufferParam>)
.set_attr<FMutateInputs>(
  "FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{1};
  })
.set_attr<FInferShape>("FInferShape", RingBufferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCorrectLayout>(
  "FCorrectLayout", [](const NodeAttrs& attrs,
                       std::vector<Layout>* in_layouts,
                       const std::vector<Layout>* last_in_layouts,
                       std::vector<Layout>* out_layouts) {
  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, Layout("NCHW"));
  return true;
})
.add_argument("data", "NDArray-or-Symbol", "Latest input")
.add_argument("buffer", "NDArray-or-Symbol",
              "Buffer storing latest [length_buffer] inputs")
.add_arguments(RingBufferParam::__FIELDS__())
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    /* dummy; will be replaced by a call to nnvm.top.register_compute() */
    LOG(FATAL) << "Reached a dummy implementation; "
               << "must supply a TVM implementation with nnvm.top.register_compute()";
    return Array<Tensor>{ inputs[0] };
})
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
