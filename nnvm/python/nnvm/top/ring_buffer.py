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

# pylint: disable=invalid-name, unused-argument
"""Definition of ring buffer op"""
from __future__ import absolute_import

import topi
import tvm
from . import registry as reg

@reg.register_compute('ring_buffer', level=100)
def compute_ring_buffer(attrs, inputs, _):
    return topi.nn.ring_buffer(inputs[0], inputs[1], axis=attrs.get_int('axis'))

@reg.register_schedule('ring_buffer')
def schedule_ring_buffer(_, outs, target):
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)
