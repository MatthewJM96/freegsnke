import logging
import mmap
from pathlib import Path
import re
from subprocess import Popen, run, PIPE, TimeoutExpired

from posix_ipc import Semaphore, SharedMemory, O_CREAT, O_EXCL, unlink_semaphore, unlink_shared_memory, BusyError, ExistentialError, SEMAPHORE_TIMEOUT_SUPPORTED

from .vc_provider import VirtualCircuitProvider

_logger = logging.getLogger(__name__)


_SHARED_MEMORY_NAME = "/rtvc_shm"
_SHARED_MEMORY_SIZE = 1024
_SEM_READY_NAME = "/rtvc_inf_req"
_SEM_DONE_NAME = "/rtvc_inf_done"
_SEM_QUIT_NAME = "/rtvc_quit"

_RTVC_VERSION_PATTERN = r"^rtvc v(\d+\.\d+\.\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"


if not SEMAPHORE_TIMEOUT_SUPPORTED:
    _logger.warning(
        "Semaphore timeout is not supported on this system, as such a failure of the "
        "RTVC server to start up will result in this program stalling."
    )


class RealTimeVirtualCircuitProvider(VirtualCircuitProvider):
    """
    Provides methods to set up a real-time virtual circuits server and obtain virtual
    circuits from this.
    """

    def __init__(self, model_specs: list[Path], rtvc_binary: Path | None = None, start_rtvc_now: bool = True):
        self._started = False
        self._rtvc_process: Popen | None = None

        # Set default RTVC binary path.
        if rtvc_binary is None:
            rtvc_binary = Path("./rtvc")

        self._rtvc_binary = rtvc_binary

        # Check RTVC binary exists and looks the way it should. Note that logging is
        # done from within this method, we simply do an early exit here.
        if not self._validate_rtvc_binary():
            return

        # Validate all model specs provided are at least existing files.
        if not all([model_spec.is_file() for model_spec in model_specs]):
            self._model_specs = model_specs

        # Start RTVC server if requested to start now.
        if start_rtvc_now:
            if self.start_up():
                self._started = True
            else:
                _logger.error("Failed to start RTVC server.")

    @property
    def started(self) -> bool:
        return self._started

    def start_up(self) -> bool:
        """
        Creates the IPC resources needed to communicate with the RTVC server, and then
        starts that server up.
        """

        if self.started:
            _logger.warning("Tried to start RTVC server when it's already started.")
            return False

        # Create a shared memory segment with name _SHARED_MEMORY_NAME and size
        # _SHARED_MEMORY_SIZE. Under the hood, posix_ipc uses `shm_open` and `ftruncate`
        # to achieve this.

        try:
            self._shared_memory = SharedMemory(name=_SHARED_MEMORY_NAME, flags=O_CREAT | O_EXCL, size=_SHARED_MEMORY_SIZE)
        except ExistentialError:
            _logger.error("Shared memory segment already exists for RTVC comms.")
            return False

        # Map the shared memory segment just created into the address space of this
        # process. This gives us a pointer at which location we can set data to pass to
        # the RTVC server for VC prediction.

        self._shared_memory_ptr = mmap.mmap(self._shared_memory.fd, _SHARED_MEMORY_SIZE)

        # Create a set of semaphores used for communicating between this process and the
        # RTVC server. These semaphores have the following responsibilities:
        #   Ready:  when signalled with a call to `Semaphore.release` by this process,
        #           the RTVC server is signalled to predict a VC given the data
        #           presently stored in the shared memory segment.
        #   Done:   when a signal is recieved with a call to `Semaphore.acquire` in this
        #           process, the RealTimeVirtualCircuitProvider class is informed that
        #           the RTVC server has completed prediction and the new VC is available
        #           to be read from the shared memory segment. Note that we also use
        #           a call to `Semaphore.acquire` on the Done semaphore to receive the
        #           alive signal from the RTVC server during its start-up.
        #   Quit:   when signalled with a call to `Semaphore.release` by this process,
        #           the RTVC server is signalled to shutdown. Note that a final shutdown
        #           actually requires one more call to `Semaphore.release` on the Ready
        #           Semaphore to avoid a race condition.

        try:
            self._sem_ready = Semaphore(name=_SEM_READY_NAME, flags=O_CREAT | O_EXCL, initial_value=0)
            self._sem_done = Semaphore(name=_SEM_DONE_NAME, flags=O_CREAT | O_EXCL, initial_value=0)
            self._sem_quit = Semaphore(name=_SEM_QUIT_NAME, flags=O_CREAT | O_EXCL, initial_value=0)
        except ExistentialError:
            _logger.error("Semaphores already exist for RTVC comms.")
            return False

        # TODO(Matthew): Start RTVC server.

        # Await an alive signal from the RTVC server, if not received in the timeout
        # period we log an error as we consider the RTVC server dead.
        try:
            # Timeout of 10 seconds for RTVC server to start up, more than necessary for
            # present RTVC implementation.
            self._sem_done.acquire(timeout=10)
        except BusyError:
            _logger.error(
                "Could not communicate with RTVC server, likely it failed to start up."
            )
            return False

        return True

    def _validate_rtvc_binary(self) -> bool:
        """
        Validates the RTVC binary of this instance. Checks first that the binary exists,
        # and secondly that it can return version info as expected.
        """

        if not self._rtvc_binary.is_file():
            _logger.error(f"Provided RTVC binary does not exist: {self._rtvc_binary}")
            return False

        # Obtain RTVC version, with a timeout of 10 seconds.
        try:
            result = run([str(self._rtvc_binary), "--version"], stdout=PIPE, timeout=10)
        except TimeoutExpired:
            _logger.error(
                f"RTVC binary hung when requesting version info: {self._rtvc_binary}"
            )
            return False

        version_string = result.stdout.decode("utf-8")

        if not re.match(_RTVC_VERSION_PATTERN, version_string):
            _logger.error(
                f"RTVC binary does not look right, a call to:\n    {self._rtvc_binary} "
                f"--version\nresulted in:\n    {version_string}"
            )
            return False

        return True
