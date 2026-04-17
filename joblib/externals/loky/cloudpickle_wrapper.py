import inspect
from functools import partial
from cloudpickle import dumps, loads


WRAP_CACHE = {}


class CloudpickledObjectWrapper:
    def __init__(self, obj, keep_wrapper=False):
        self._obj = obj
        self._keep_wrapper = keep_wrapper

    def __reduce__(self):
        _pickled_object = dumps(self._obj)
        if not self._keep_wrapper:
            return loads, (_pickled_object,)

        return _reconstruct_wrapper, (_pickled_object, self._keep_wrapper)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seemlessly as the
        # previous object.
        if attr not in ["_obj", "_keep_wrapper"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)


# Make sure the wrapped object conserves the callable property
class CallableObjectWrapper(CloudpickledObjectWrapper):
    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


def _wrap_non_picklable_objects(obj, keep_wrapper):
    if callable(obj):
        return CallableObjectWrapper(obj, keep_wrapper=keep_wrapper)
    return CloudpickledObjectWrapper(obj, keep_wrapper=keep_wrapper)


def _reconstruct_wrapper(_pickled_object, keep_wrapper):
    pass


def _wrap_objects_when_needed(obj):
    # Function to introspect an object and decide if it should be wrapped or
    # not.
    pass


def wrap_non_picklable_objects(obj, keep_wrapper=True):
    """Wrapper for non-picklable object to use cloudpickle to serialize them.

    Note that this wrapper tends to slow down the serialization process as it
    is done with cloudpickle which is typically slower compared to pickle. The
    proper way to solve serialization issues is to avoid defining functions and
    objects in the main scripts and to implement __reduce__ functions for
    complex classes.
    """
    # If obj is a  class, create a CloudpickledClassWrapper which instantiates
    # the object internally and wrap it directly in a CloudpickledObjectWrapper
    if inspect.isclass(obj):

        class CloudpickledClassWrapper(CloudpickledObjectWrapper):
            def __init__(self, *args, **kwargs):
                self._obj = obj(*args, **kwargs)
                self._keep_wrapper = keep_wrapper

        CloudpickledClassWrapper.__name__ = obj.__name__
        return CloudpickledClassWrapper

    # If obj is an instance of a class, just wrap it in a regular
    # CloudpickledObjectWrapper
    return _wrap_non_picklable_objects(obj, keep_wrapper=keep_wrapper)
