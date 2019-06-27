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
from .. import tag

@tvm.tag_scope(tag=tag.INJECTIVE+",ring_buffer")
def ring_buffer(data, buffer, axis):
    """
    Implements a ring buffer, in which a set number of past inputs are internally cached
    """
    assert len(data.shape) == len(buffer.shape), \
        'buffer and data must have same number of dimensions, ' + \
        'buffer.shape = {}, data.shape = {}'.format(buffer.shape, data.shape)
    assert 0 <= axis < len(buffer.shape), 'buffer axis out of range'
    for i in range(len(data.shape)):
        if i == axis:
            assert int(str(data.shape[i])) <= int(str(buffer.shape[i]))
        else:
            assert int(str(data.shape[i])) == int(str(buffer.shape[i]))

    # for now only 4D and 5D are supported
    assert len(buffer.shape) == 4 or len(buffer.shape) == 5

    buflen = buffer.shape[axis]
    data_size = data.shape[axis]

    if len(buffer.shape) == 4:
        if axis == 0:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                               tvm.if_then_else(i < buflen - data_size,
                                                buffer[i + data_size, j, k, l],
                                                data[i - buflen + data_size, j, k, l]),
                               name='new_buffer')
        if axis == 1:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                               tvm.if_then_else(j < buflen - data_size,
                                                buffer[i, j + data_size, k, l],
                                                data[i, j - buflen + data_size, k, l]),
                               name='new_buffer')
        if axis == 2:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                               tvm.if_then_else(k < buflen - data_size,
                                                buffer[i, j, k + data_size, l],
                                                data[i, j, k - buflen + data_size, l]),
                               name='new_buffer')
        if axis == 3:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l:
                               tvm.if_then_else(l < buflen - data_size,
                                                buffer[i, j, k, l + data_size],
                                                data[i, j, k, l - buflen + data_size]),
                               name='new_buffer')
        assert False, "shouldn't get here"
    elif len(buffer.shape) == 5:
        if axis == 0:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                               tvm.if_then_else(i < buflen - data_size,
                                                buffer[i + data_size, j, k, l, m],
                                                data[i - buflen + data_size, j, k, l, m]),
                               name='new_buffer')
        if axis == 1:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                               tvm.if_then_else(j < buflen - data_size,
                                                buffer[i, j + data_size, k, l, m],
                                                data[i, j - buflen + data_size, k, l, m]),
                               name='new_buffer')
        if axis == 2:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                               tvm.if_then_else(k < buflen - data_size,
                                                buffer[i, j, k + data_size, l, m],
                                                data[i, j, k - buflen + data_size, l, m]),
                               name='new_buffer')
        if axis == 3:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                               tvm.if_then_else(l < buflen - data_size,
                                                buffer[i, j, k, l + data_size, m],
                                                data[i, j, k, l - buflen + data_size, m]),
                               name='new_buffer')
        if axis == 4:
            return tvm.compute(buffer.shape,
                               lambda i, j, k, l, m:
                               tvm.if_then_else(m < buflen - data_size,
                                                buffer[i, j, k, l, m + data_size],
                                                data[i, j, k, l, m - buflen + data_size]),
                               name='new_buffer')
        assert False, "shouldn't get here"
    else:
        assert 'Only 4D and 5D supported'
    return None
