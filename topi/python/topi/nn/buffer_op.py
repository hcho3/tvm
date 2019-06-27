# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Ring buffer op"""
from __future__ import absolute_import as _abs
import tvm
from ..util import equal_const_int
from .. import tag

@tvm.tag_scope(tag=tag.INJECTIVE+",ring_buffer")
def ring_buffer(input, buffer, axis):
    assert len(input.shape) == len(buffer.shape), \
        'buffer and input must have same number of dimensions, ' + \
        'buffer.shape = {}, input.shape = {}'.format(buffer.shape, input.shape)
    assert axis >= 0 and axis < len(buffer.shape), 'buffer axis out of range'
    for i in range(len(input.shape)):
        if i == axis:
            assert int(str(input.shape[i])) <= int(str(buffer.shape[i]))
        else:
            assert int(str(input.shape[i])) == int(str(buffer.shape[i]))

    # for now only 4D and 5D are supported
    assert len(buffer.shape) == 4 or len(buffer.shape) == 5

    buflen = buffer.shape[axis]
    input_size = input.shape[axis]

    if len(buffer.shape) == 4:
        if axis == 0:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                                   tvm.if_then_else(i < buflen - input_size,
                                       buffer[i + input_size, j, k, l],
                                       input[i - buflen + input_size, j, k, l]),
                               name='new_buffer')
        elif axis == 1:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                                   tvm.if_then_else(j < buflen - input_size,
                                       buffer[i, j + input_size, k, l],
                                       input[i, j - buflen + input_size, k, l]),
                               name='new_buffer')
        elif axis == 2:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                                   tvm.if_then_else(k < buflen - input_size,
                                       buffer[i, j, k + input_size, l],
                                       input[i, j, k - buflen + input_size, l]),
                               name='new_buffer')
        elif axis == 3:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                                   tvm.if_then_else(l < buflen - input_size,
                                       buffer[i, j, k, l + input_size],
                                       input[i, j, k, l - buflen + input_size]),
                               name='new_buffer')
        else:
            assert False, "shouldn't get here"
    elif len(buffer.shape) == 5:
        if axis == 0:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                                   tvm.if_then_else(i < buflen - input_size,
                                       buffer[i + input_size, j, k, l, m],
                                       input[i - buflen + input_size, j, k, l, m]),
                               name='new_buffer')
        elif axis == 1:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                                   tvm.if_then_else(j < buflen - input_size,
                                       buffer[i, j + input_size, k, l, m],
                                       input[i, j - buflen + input_size, k, l, m]),
                               name='new_buffer')
        elif axis == 2:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                                   tvm.if_then_else(k < buflen - input_size,
                                       buffer[i, j, k + input_size, l, m],
                                       input[i, j, k - buflen + input_size, l, m]),
                               name='new_buffer')
        elif axis == 3:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                                   tvm.if_then_else(l < buflen - input_size,
                                       buffer[i, j, k, l + input_size, m],
                                       input[i, j, k, l - buflen + input_size, m]),
                               name='new_buffer')
        elif axis == 4:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                                   tvm.if_then_else(m < buflen - input_size,
                                       buffer[i, j, k, l, m + input_size],
                                       input[i, j, k, l, m - buflen + input_size]),
                               name='new_buffer')
        else:
            assert False, "shouldn't get here"
    else:
        assert 'Only 4D and 5D supported'
