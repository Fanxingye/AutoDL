# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
import torch
import signal
import torch.distributed as dist


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function


def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d


class TimeoutHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig
        self.device = torch.device("cuda")
    @property
    def interrupted(self):
        if not dist.is_initialized():
            return self._interrupted

        interrupted = torch.tensor(self._interrupted).int().to(self.device)
        dist.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def __enter__(self):
        self._interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            self.release()
            self._interrupted = True
            print(f"Received SIGTERM")

        def ignoring_handler(signum, frame):
            self.release()
            print("Received SIGTERM, ignoring")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            signal.signal(self.sig, master_handler)
        else:
            signal.signal(self.sig, ignoring_handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time
