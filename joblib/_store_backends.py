"""Storage providers backends for Memory caching."""

import collections
import datetime
import json
import operator
import os
import os.path
import re
import shutil
import threading
import time
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from pickle import PicklingError

from . import numpy_pickle
from .backports import concurrency_safe_rename
from .disk import memstr_to_bytes, mkdirp, rm_subdirs
from .logger import format_time

CacheItemInfo = collections.namedtuple("CacheItemInfo", "path size last_access")


class CacheWarning(Warning):
    """Warning to capture dump failures except for PicklingError."""

    pass


def concurrency_safe_write(object_to_write, filename, write_func):
    """Writes an object into a unique file in a concurrency-safe way."""
    # Temporary name is composed of UUID, process_id and thread_id to avoid
    # collisions due to concurrent write.
    # UUID is unique across nodes and time and help avoid collisions, even if
    # the cache folder is shared by several Python processes with the same pid and
    # thread id on different nodes of a cluster for instance.
    thread_id = id(threading.current_thread())
    temporary_filename = f"{filename}.{uuid.uuid4().hex}-{os.getpid()}-{thread_id}"

    write_func(object_to_write, temporary_filename)

    return temporary_filename


class StoreBackendBase(metaclass=ABCMeta):
    """Helper Abstract Base Class which defines all methods that
    a StorageBackend must implement."""

    location = None

    @abstractmethod
    def _open_item(self, f, mode):
        """Opens an item on the store and return a file-like object.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        f: a file-like object
            The file-like object where an item is stored and retrieved
        mode: string, optional
            the mode in which the file-like object is opened allowed valued are
            'rb', 'wb'

        Returns
        -------
        a file-like object
        """

    @abstractmethod
    def _item_exists(self, location):
        """Checks if an item location exists in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        location: string
            The location of an item. On a filesystem, this corresponds to the
            absolute path, including the filename, of a file.

        Returns
        -------
        True if the item exists, False otherwise
        """

    @abstractmethod
    def _move_item(self, src, dst):
        """Moves an item from src to dst in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        src: string
            The source location of an item
        dst: string
            The destination location of an item
        """

    @abstractmethod
    def create_location(self, location):
        """Creates a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory.
        """

    @abstractmethod
    def clear_location(self, location):
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """

    @abstractmethod
    def get_items(self):
        """Returns the whole list of items available in the store.

        Returns
        -------
        The list of items identified by their ids (e.g filename in a
        filesystem).
        """

    @abstractmethod
    def configure(self, location, verbose=0, backend_options=dict()):
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Contains a dictionary of named parameters used to configure the
            store backend.
        """


class StoreBackendMixin(object):
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        """Load an item from the store given its id as a list of str."""
        full_path = os.path.join(self.location, *call_id)

        if verbose > 1:
            ts_string = (
                "{: <16}".format(format_time(time.time() - timestamp))
                if timestamp is not None
                else ""
            )
            signature = os.path.basename(call_id[0])
            if metadata is not None and "input_args" in metadata:
                kwargs = ", ".join(
                    "{}={}".format(*item) for item in metadata["input_args"].items()
                )
                signature += "({})".format(kwargs)
            msg = "[Memory]{}: Loading {}".format(ts_string, signature)
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        mmap_mode = None if not hasattr(self, "mmap_mode") else self.mmap_mode

        filename = os.path.join(full_path, "output.pkl")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        # file-like object cannot be used when mmap_mode is set
        if mmap_mode is None:
            with self._open_item(filename, "rb") as f:
                item = numpy_pickle.load(f)
        else:
            item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return item

    def dump_item(self, call_id, item, verbose=1):
        """Dump an item in the store at the id given as a list of str."""
        pass

    def clear_item(self, call_id):
        """Clear the item at the id, given as a list of str."""
        item_path = os.path.join(self.location, *call_id)
        if self._item_exists(item_path):
            self.clear_location(item_path)

    def contains_item(self, call_id):
        """Check if there is an item at the id, given as a list of str."""
        pass

    def get_item_info(self, call_id):
        """Return information about item."""
        pass

    def get_metadata(self, call_id):
        """Return actual metadata of an item."""
        try:
            item_path = os.path.join(self.location, *call_id)
            filename = os.path.join(item_path, "metadata.json")
            with self._open_item(filename, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except:  # noqa: E722
            return {}

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation."""
        pass

    def contains_path(self, call_id):
        """Check cached function is available in store."""
        pass

    def clear_path(self, call_id):
        """Clear all items with a common path in the store."""
        func_path = os.path.join(self.location, *call_id)
        if self._item_exists(func_path):
            self.clear_location(func_path)

    def store_cached_func_code(self, call_id, func_code=None):
        """Store the code of the cached function."""
        func_path = os.path.join(self.location, *call_id)
        if not self._item_exists(func_path):
            self.create_location(func_path)

        if func_code is not None:
            filename = os.path.join(func_path, "func_code.py")
            with self._open_item(filename, "wb") as f:
                f.write(func_code.encode("utf-8"))

    def get_cached_func_code(self, call_id):
        """Store the code of the cached function."""
        pass

    def get_cached_func_info(self, call_id):
        """Return information related to the cached function if it exists."""
        pass

    def clear(self):
        """Clear the whole store content."""
        self.clear_location(self.location)

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Remove the store's oldest files to enforce item, byte, and age limits.
        """
        pass

    def _get_items_to_delete(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Get items to delete to keep the store under size, file, & age limits.
        """
        pass

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        pass

    def __repr__(self):
        """Printable representation of the store location."""
        return '{class_name}(location="{location}")'.format(
            class_name=self.__class__.__name__, location=self.location
        )


class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, "output.pkl")
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        # The directory has already been deleted
                        continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        if backend_options is None:
            backend_options = {}

        # setup location directory
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)

        # Automatically add `.gitignore` file to the cache folder.
        # XXX: the condition is necessary because in `Memory.__init__`, the user
        # passed `location` param is modified to be either `{location}` or
        # `{location}/joblib` depending on input type (`pathlib.Path` vs `str`).
        # The proper resolution of this inconsistency is tracked in:
        # https://github.com/joblib/joblib/issues/1684
        cache_directory = (
            os.path.dirname(location)
            if os.path.dirname(location) and os.path.basename(location) == "joblib"
            else location
        )
        gitignore = os.path.join(cache_directory, ".gitignore")
        if not os.path.exists(gitignore):
            try:
                with open(gitignore, "w") as file:
                    file.write("# Created by joblib automatically.\n")
                    file.write("*\n")
            except OSError as e:
                warnings.warn(f"Unable to write {gitignore}. Exception: {e}.")

        # item can be stored compressed for faster I/O
        self.compress = backend_options.get("compress", False)

        # FileSystemStoreBackend can be used with mmap_mode options under
        # certain conditions.
        mmap_mode = backend_options.get("mmap_mode")
        if self.compress and mmap_mode is not None:
            warnings.warn(
                "Compressed items cannot be memmapped in a "
                "filesystem store. Option will be ignored.",
                stacklevel=2,
            )

        self.mmap_mode = mmap_mode
        self.verbose = verbose
