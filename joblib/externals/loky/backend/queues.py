###############################################################################
# Queue and SimpleQueue implementation for loky
#
# authors: Thomas Moreau, Olivier Grisel
#
# based on multiprocessing/queues.py (16/02/2017)
# * Add some custom reducers for the Queues/SimpleQueue to tweak the
#   pickling process. (overload Queue._feed/SimpleQueue.put)
#
import os
import sys
import errno
import weakref
import threading
from multiprocessing import util
from multiprocessing.queues import (
    Full,
    Queue as mp_Queue,
    SimpleQueue as mp_SimpleQueue,
    _sentinel,
)
from multiprocessing.context import assert_spawning

from .reduction import dumps


__all__ = ["Queue", "SimpleQueue", "Full"]


class Queue(mp_Queue):
    def __init__(self, maxsize=0, reducers=None, ctx=None):
        super().__init__(maxsize=maxsize, ctx=ctx)
        self._reducers = reducers

    # Use custom queue set/get state to be able to reduce the custom reducers
    def __getstate__(self):
        assert_spawning(self)
        return (
            self._ignore_epipe,
            self._maxsize,
            self._reader,
            self._writer,
            self._reducers,
            self._rlock,
            self._wlock,
            self._sem,
            self._opid,
        )

    def __setstate__(self, state):
        (
            self._ignore_epipe,
            self._maxsize,
            self._reader,
            self._writer,
            self._reducers,
            self._rlock,
            self._wlock,
            self._sem,
            self._opid,
        ) = state
        if sys.version_info >= (3, 9):
            self._reset()
        else:
            self._after_fork()

    # Overload _start_thread to correctly call our custom _feed
    def _start_thread(self):
        pass

    # Overload the _feed methods to use our custom pickling strategy.
    @staticmethod
    def _feed(
        buffer,
        notempty,
        send_bytes,
        writelock,
        close,
        reducers,
        ignore_epipe,
        onerror,
        queue_sem,
    ):
        pass

    def _on_queue_feeder_error(self, e, obj):
        """
        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        """
        pass


class SimpleQueue(mp_SimpleQueue):
    def __init__(self, reducers=None, ctx=None):
        super().__init__(ctx=ctx)

        # Add possiblity to use custom reducers
        self._reducers = reducers

    def close(self):
        self._reader.close()
        self._writer.close()

    # Use custom queue set/get state to be able to reduce the custom reducers
    def __getstate__(self):
        assert_spawning(self)
        return (
            self._reader,
            self._writer,
            self._reducers,
            self._rlock,
            self._wlock,
        )

    def __setstate__(self, state):
        (
            self._reader,
            self._writer,
            self._reducers,
            self._rlock,
            self._wlock,
        ) = state

    # Overload put to use our customizable reducer
    def put(self, obj):
        # serialize the data before acquiring the lock
        obj = dumps(obj, reducers=self._reducers)
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)
