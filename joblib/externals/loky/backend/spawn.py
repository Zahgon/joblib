###############################################################################
# Prepares and processes the data to setup the new process environment
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/spawn.py (17/02/2017)
#  * Improve logging data
#
import os
import sys
import runpy
import textwrap
import types
from multiprocessing import process, util


if sys.platform != "win32":
    WINEXE = False
    WINSERVICE = False
else:
    import msvcrt
    from multiprocessing.reduction import duplicate

    WINEXE = sys.platform == "win32" and getattr(sys, "frozen", False)
    WINSERVICE = sys.executable.lower().endswith("pythonservice.exe")

if WINSERVICE:
    _python_exe = os.path.join(sys.exec_prefix, "python.exe")
else:
    _python_exe = sys.executable


def get_executable():
    return _python_exe


def _check_not_importing_main():
    if getattr(process.current_process(), "_inheriting", False):
        raise RuntimeError(
            textwrap.dedent(
                """\
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable."""
            )
        )


def get_preparation_data(name, init_main_module=True):
    """Return info about parent needed by child to unpickle process object."""
    _check_not_importing_main()
    d = dict(
        log_to_stderr=util._log_to_stderr,
        authkey=bytes(process.current_process().authkey),
        name=name,
        sys_argv=sys.argv,
        orig_dir=process.ORIGINAL_DIR,
        dir=os.getcwd(),
    )

    # Send sys_path and make sure the current directory will not be changed
    d["sys_path"] = [p if p != "" else process.ORIGINAL_DIR for p in sys.path]

    # Make sure to pass the information if the multiprocessing logger is active
    if util._logger is not None:
        d["log_level"] = util._logger.getEffectiveLevel()
        if util._logger.handlers:
            h = util._logger.handlers[0]
            d["log_fmt"] = h.formatter._fmt

    # Tell the child how to communicate with the resource_tracker
    from .resource_tracker import _resource_tracker

    _resource_tracker.ensure_running()
    if sys.platform == "win32":
        d["tracker_fd"] = msvcrt.get_osfhandle(_resource_tracker._fd)
    else:
        d["tracker_fd"] = _resource_tracker._fd

    if os.name == "posix":
        # joblib/loky#242: allow loky processes to retrieve the resource
        # tracker of their parent in case the child processes depickles
        # shared_memory objects, that are still tracked by multiprocessing's
        # resource_tracker by default.
        # XXX: this is a workaround that may be error prone: in the future, it
        # would be better to have loky subclass multiprocessing's shared_memory
        # to force registration of shared_memory segments via loky's
        # resource_tracker.
        from multiprocessing.resource_tracker import (
            _resource_tracker as mp_resource_tracker,
        )

        # multiprocessing's resource_tracker must be running before loky
        # process is created (othewise the child won't be able to use it if it
        # is created later on)
        mp_resource_tracker.ensure_running()
        d["mp_tracker_fd"] = mp_resource_tracker._fd

    # Figure out whether to initialise main in the subprocess as a module
    # or through direct execution (or to leave it alone entirely)
    if init_main_module:
        main_module = sys.modules["__main__"]
        try:
            main_mod_name = getattr(main_module.__spec__, "name", None)
        except BaseException:
            main_mod_name = None
        if main_mod_name is not None:
            d["init_main_from_name"] = main_mod_name
        elif sys.platform != "win32" or (not WINEXE and not WINSERVICE):
            main_path = getattr(main_module, "__file__", None)
            if main_path is not None:
                if (
                    not os.path.isabs(main_path)
                    and process.ORIGINAL_DIR is not None
                ):
                    main_path = os.path.join(process.ORIGINAL_DIR, main_path)
                d["init_main_from_path"] = os.path.normpath(main_path)

    return d


#
# Prepare current process
#
old_main_modules = []


def prepare(data, parent_sentinel=None):
    """Try to get current process ready to unpickle process object."""
    pass


# Multiprocessing module helpers to fix up the main module in
# spawned subprocesses
def _fixup_main_from_name(mod_name):
    # __main__.py files for packages, directories, zip archives, etc, run
    # their "main only" code unconditionally, so we don't even try to
    # populate anything in __main__, nor do we make any changes to
    # __main__ attributes
    pass


def _fixup_main_from_path(main_path):
    # If this process was forked, __main__ may already be populated
    pass
