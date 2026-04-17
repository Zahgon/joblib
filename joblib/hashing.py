"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import decimal
import hashlib
import io
import pickle
import struct
import sys
import types

Pickler = pickle._Pickler


class _ConsistentSet(object):
    """Class used to ensure the hash of Sets is preserved
    whatever the order of its items.
    """

    def __init__(self, set_sequence):
        # Forces order of elements in set to ensure consistent hash.
        try:
            # Trying first to order the set assuming the type of elements is
            # consistent and orderable.
            # This fails on python 3 when elements are unorderable
            # but we keep it in a try as it's faster.
            self._sequence = sorted(set_sequence)
        except (TypeError, decimal.InvalidOperation):
            # If elements are unorderable, sorting them using their hash.
            # This is slower but works in any case.
            self._sequence = sorted((hash(e) for e in set_sequence))


class _MyHash(object):
    """Class used to hash objects that won't normally pickle"""

    def __init__(self, *args):
        self.args = args


class Hasher(Pickler):
    """A subclass of pickler, to do cryptographic hashing, rather than
    pickling. This is used to produce a unique hash of the given
    Python object that is not necessarily cryptographically secure.
    """

    def __init__(self, hash_name="md5"):
        self.stream = io.BytesIO()
        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = 3
        Pickler.__init__(self, self.stream, protocol=protocol)
        # Initialise the hash obj
        self._hash = hashlib.new(hash_name, usedforsecurity=False)

    def hash(self, obj, return_digest=True):
        pass

    def save(self, obj):
        pass

    def memoize(self, obj):
        # We want hashing to be sensitive to value instead of reference.
        # For example we want ['aa', 'aa'] and ['aa', 'aaZ'[:2]]
        # to hash to the same value and that's why we disable memoization
        # for strings
        pass

    # The dispatch table of the pickler is not accessible in Python
    # 3, as these lines are only bugware for IPython, we skip them.
    def save_global(self, obj, name=None, pack=struct.pack):
        # We have to override this method in order to deal with objects
        # defined interactively in IPython that are not injected in
        # __main__
        pass

    dispatch = Pickler.dispatch.copy()
    # builtin
    dispatch[type(len)] = save_global
    # type
    dispatch[type(object)] = save_global
    # classobj
    dispatch[type(Pickler)] = save_global
    # function
    dispatch[type(pickle.dump)] = save_global

    # We use *args in _batch_setitems signature because _batch_setitems has an
    # additional 'obj' argument in Python 3.14
    def _batch_setitems(self, items, *args):
        # forces order of keys in dict to ensure consistent hash.
        pass

    def save_set(self, set_items):
        # forces order of items in Set to ensure consistent hash
        pass

    dispatch[type(set())] = save_set


class NumpyHasher(Hasher):
    """Special case the hasher for when numpy is loaded."""

    def __init__(self, hash_name="md5", coerce_mmap=False):
        """
        Parameters
        ----------
        hash_name: string
            The hash algorithm to be used
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
            objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np

        self.np = np
        if hasattr(np, "getbuffer"):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

    def save(self, obj):
        """Subclass the save method, to hash ndarray subclass, rather
        than pickling them. Off course, this is a total abuse of
        the Pickler class.
        """
        pass


def hash(obj, hash_name="md5", coerce_mmap=False):
    """Quick calculation of a hash to identify uniquely Python objects
    containing numpy arrays.

    Parameters
    ----------
    hash_name: 'md5' or 'sha1'
        Hashing algorithm used. sha1 is supposedly safer, but md5 is
        faster.
    coerce_mmap: boolean
        Make no difference between np.memmap and np.ndarray
    """
    pass
