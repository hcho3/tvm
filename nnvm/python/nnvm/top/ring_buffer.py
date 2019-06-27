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
