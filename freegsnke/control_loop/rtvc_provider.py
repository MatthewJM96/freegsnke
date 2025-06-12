import logging
import re
import struct
from mmap import mmap
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired, run
from typing import NamedTuple

import numpy as np
from posix_ipc import (
    O_CREAT,
    O_EXCL,
    SEMAPHORE_TIMEOUT_SUPPORTED,
    BusyError,
    ExistentialError,
    Semaphore,
    SharedMemory,
    unlink_semaphore,
    unlink_shared_memory,
)
from yaml import YAMLError, safe_load

from freegsnke.observable_registry import ObservableRegistry
from freegsnke.virtual_circuits import VirtualCircuit

from .vc_provider import VirtualCircuitProvider

_logger = logging.getLogger(__name__)


# Handle names for shared memory and semaphore, plus size of shared memory segment.
_SHARED_MEMORY_NAME = "/rtvc_shm"
_SHARED_MEMORY_SIZE = 1024  # 1kB
_SEM_READY_NAME = "/rtvc_inf_req"
_SEM_DONE_NAME = "/rtvc_inf_done"
_SEM_QUIT_NAME = "/rtvc_quit"

# Regex pattern that allows only "rtvc v<VER>" where <VER> is a valid Semantic
# Versioning string.
_RTVC_VERSION_PATTERN = (
    r"^rtvc v(\d+\.\d+\.\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))"
    r"?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


if not SEMAPHORE_TIMEOUT_SUPPORTED:
    _logger.warning(
        "Semaphore timeout is not supported on this system, as such a failure of the "
        "RTVC server to start up will result in this program stalling."
    )


class ModelSpec(NamedTuple):
    """
    A model spec holds metadata regarding a model that is pertinent to preparing for and
    processing after inference.
    """

    data_file: Path
    inputs: list[str]
    outputs: list[str]

    @staticmethod
    def from_filepath(filepath: Path) -> "ModelSpec | None":
        """
        Loads a model spec from the listed filepath. If the model spec could not be
        loaded, then None is returned.

        Parameters
        ----------
        filepath : pathlib.Path
            The filepath to the model spec to load.
        """
        # Ensure a file exists at the provided filepath, only way it could possibly be
        # a model spec!

        if not filepath.is_file():
            _logger.warning(f"No model spec exists at: {filepath}")
            return None

        # Load YAML into `data`, notet hat we use safe_load as it guarantees no funny
        # business, and we have an invalid model spec should PyYAML throw an exception
        # for any reason.

        with open(filepath, "r") as f:
            try:
                data = safe_load(filepath)
            except YAMLError as e:
                _logger.warning(f"Invalid model spec at: {filepath}\n{e}")
                return None

        # We expect a dictionary with three keys, "data_file", "inputs", and "outputs".

        if not isinstance(data, dict):
            _logger.warning(f"Invalid model spec at: {filepath}\n{e}")
            return None

        if len(set("data_file", "inputs", "outputs") - set(data)) != 0:
            _logger.warning(f"Invalid model spec at: {filepath}\n{e}")
            return None

        # Validate the filepath provided by "data_file" is a valid file.

        if not isinstance(data["data_file"], str):
            _logger.warning(
                f"Invalid data filepath provided in model spec at: {filepath}"
            )
            return None

        data_filepath = Path(data["data_file"])
        if not data_filepath.is_file():
            _logger.warning(
                "Data filepath provided by model spec is not a file, filepath was: "
                f"{data_filepath}"
            )
            return None

        # TODO(Matthew): validate with tflite lib? RTVC will fail to start up if it
        #                isn't a valid model so in any case this will be caught but it
        #                would let us provide a nicer warning to do it here.

        # Validate that inputs and outputs are lists of strings.

        inputs = data["inputs"]
        outputs = data["outputs"]
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            _logger.warning(
                "One of inputs or outputs was not a list as provided by model spec at: "
                f"{filepath}"
            )
            return None

        if not all([isinstance(x, str) for x in inputs]):
            _logger.warning(
                "At least one input was not a string as provided by model spec at: "
                f"{filepath}"
            )
            return None

        if not all([isinstance(x, str) for x in outputs]):
            _logger.warning(
                "At least one input was not a string as provided by model spec at: "
                f"{filepath}"
            )
            return None

        return ModelSpec(data_filepath, inputs, outputs)


class RealTimeVirtualCircuitProvider(VirtualCircuitProvider):
    """
    Provides methods to set up a real-time virtual circuits server and obtain virtual
    circuits from this.
    """

    def __init__(
        self,
        model_spec_paths: list[Path],
        controllable_coils: list[str],
        rtvc_binary: Path | None = None,
        start_rtvc_now: bool = True,
        observable_registry: ObservableRegistry | None = None,
    ):
        """
        Initialises the RealTimeVirtualCircuitProvider instance, by default starting up
        the RTVC server it will communicate with to obtain virtual circuit predictions.

        Parameters
        ----------
        model_spec_paths : list[pathlib.Path]
            List of model spec filepaths to use for initialising the RTVC server.
        controllable_coils : list[str]
            List of coil names that are used for shape control.
        rtvc_binary : pathlib.Path | None (default: None)
            Path to an RTVC binary, if None then path is taken to be "./rtvc".
        start_rtvc_now : bool (default: True)
            If True, the RTVC server and associated IPC resources are instantiated as
            part of RealTimeVirtualCircuitProvider initialisation. If False, then
            `RealTimeVirtualCircuitProvider.start_up` must be called later to do this
            instead; this is required in order to get virtual circuits from this
            provider.
        observable_registry : ObservableRegistry | None (default: None)
            The observable registry to set the provider to use.
        """
        self._started = False
        self._shared_memory: SharedMemory | None = None
        self._shared_memory_map: mmap | None = None
        self._sem_ready: Semaphore | None = None
        self._sem_done: Semaphore | None = None
        self._sem_quit: Semaphore | None = None
        self._rtvc_process: Popen | None = None

        # Set default RTVC binary path.
        if rtvc_binary is None:
            rtvc_binary = Path("./rtvc")

        self._rtvc_binary = rtvc_binary

        # Check RTVC binary exists and looks the way it should. Note that logging is
        # done from within this method, we simply do an early exit here.
        if not self._validate_rtvc_binary():
            return

        # Load and validate all model specs.

        if len(model_spec_paths) < 1:
            _logger.error(
                "At least one model spec must be supplied to "
                "RealTimeVirtualCircuitProvider."
            )
            return

        self._model_specs: list[ModelSpec] = []
        for path in model_spec_paths:
            model_spec = ModelSpec.from_filepath(path)

            if model_spec is None:
                _logger.error(f"Provided invalid model spec at {path}.")
                return

            self._model_specs.append(model_spec)

        # Validate that all model specs have the same inputs and outputs.
        # NOTE(Matthew): If we wish to support any notion of heterogenous consensus, we
        #                would need to relax this check.

        for idx in range(1, len(self._model_specs)):
            if self._model_specs[idx].inputs != self._model_specs[0].inputs:
                _logger.error(
                    f"The {idx+1}th model spec provided does not conform in inputs to "
                    "that of the first model spec."
                )
                return
            if self._model_specs[idx].outputs != self._model_specs[0].outputs:
                _logger.error(
                    f"The {idx+1}th model spec provided does not conform in outputs to "
                    "that of the first model spec."
                )
                return

        # Validate that all controllable coils exist in the inputs of the models.
        missing_coils = [
            coil not in self._model_specs[0].inputs for coil in controllable_coils
        ]
        if len(missing_coils) != 0:
            _logger.error(
                "Some coils specified as controllable are missing from inputs of models"
                f" in RealTimeVirtualCircuitsProvider: {missing_coils}"
            )
            return
        self._coil_indices = [
            self._model_specs[0].inputs.index(coil) for coil in controllable_coils
        ]

        # Start RTVC server if requested to start now.
        if start_rtvc_now:
            if self.start_up():
                self._started = True
            else:
                _logger.error("Failed to start RTVC server.")

        super().__init__(observable_registry=observable_registry)

    def __del__(self):
        """
        On destruction of this object, make sure we clean up IPC resources if needed.
        """
        # unstarted_okay=True avoids a log warning if we were in an unstarted state (aka
        # no resources to clean up).
        self.shutdown(unstarted_okay=True)

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
            self._shared_memory = SharedMemory(
                name=_SHARED_MEMORY_NAME,
                flags=O_CREAT | O_EXCL,
                size=_SHARED_MEMORY_SIZE,
            )
        except ExistentialError:
            _logger.error("Shared memory segment already exists for RTVC comms.")
            return False

        # Map the shared memory segment just created into the address space of this
        # process. This gives us a pointer at which location we can set data to pass to
        # the RTVC server for VC prediction.

        self._shared_memory_map = mmap(self._shared_memory.fd, _SHARED_MEMORY_SIZE)

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
            self._sem_ready = Semaphore(
                name=_SEM_READY_NAME, flags=O_CREAT | O_EXCL, initial_value=0
            )
            self._sem_done = Semaphore(
                name=_SEM_DONE_NAME, flags=O_CREAT | O_EXCL, initial_value=0
            )
            self._sem_quit = Semaphore(
                name=_SEM_QUIT_NAME, flags=O_CREAT | O_EXCL, initial_value=0
            )
        except ExistentialError:
            _logger.error("Semaphores already exist for RTVC comms.")
            return False

        # Construct the command to invoke the RTVC server with.
        #   NOTE(Matthew): For now we pass a --jacobian flag to obtain the full
        #                  sensitivity matrix. As this involves an inversion we cannot
        #                  then filter out columns corresponding to targets we don't
        #                  want to change. As such it may be desirable to obtain the
        #                  uninverted matrix, and then do inversion here after cropping
        #                  to the scheduled target space.
        rtvc_cmd = [str(self._rtvc_binary), "--jacobian"]
        rtvc_cmd.extend(
            [f"--model-spec={model_spec.data_file}" for model_spec in self._model_specs]
        )
        # Invoke the RTVC process, we will now only communicate with the RTVC server
        # using the IPC resources until such a time as we come to shutdown where we will
        # use this process handle accordingly.
        self._rtvc_process = Popen(rtvc_cmd)

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

    def shutdown(self, unstarted_okay: bool = False) -> bool:
        """
        Shuts down the RTVC server, subsequently cleaning up IPC resources used for
        communication with the server.

        Parameters
        ----------
        unstarted_okay : bool (default: False)
            If True, then should this function finding this
            RealTimeVirtualCircuitProvider to be in an unstarted state does not result
            in a warning being logged. If False, then a warning will be logged in this
            scenario - the behaviour most desirable on explicit calls to this function.
        """

        if not self.started:
            if not unstarted_okay:
                _logger.warning(
                    "Tried to shutdown RTVC server when it wasn't started yet."
                )
            return False

        # Send quit signal to the RTVC server, note the ready signal is part of this
        # to avoid a race condition.
        self._sem_quit.release()
        self._sem_ready.release()

        # Give RTVC server 10 seconds to shutdown gracefully.
        try:
            self._rtvc_process.wait(timeout=10)
        except TimeoutExpired:
            # SIGKILL the RTVC server - this is ugly but all handles on IPC resources
            # are guaranteed to be freed and the RTVC server is stateless, so ultimately
            # not so bad.
            self._rtvc_process.kill()

        # Close our handles on IPC resources.
        self._shared_memory_map.close()
        self._shared_memory.close_fd()
        self._sem_ready.close()
        self._sem_done.close()
        self._sem_quit.close()

        # Unlink (aka destroy) IPC resources as existing in the OS.
        unlink_shared_memory(self._SHARED_MEMORY_NAME)
        unlink_semaphore(_SEM_READY_NAME)
        unlink_semaphore(_SEM_DONE_NAME)
        unlink_semaphore(_SEM_QUIT_NAME)

        # For completeness, unset our IPC fields.
        self._shared_memory = None
        self._shared_memory_map = None
        self._sem_ready = None
        self._sem_done = None
        self._sem_quit = None

        # We are no longer in a started state.
        self._started = False

        return True

    def get_vc(self, timestamp: float, targets: list[str]) -> VirtualCircuit | None:
        """
        Gets a Virtual Circuit for the given timestamp and observables requested from
        the registry.

        Parameters
        ----------
        timestamp : float (4 decimal places)
            time stamp of the virtual circuit to be retrieved
        targets : list[str]
            list of targets to get a virtual circuit for

        Returns
        -------
        vc : VirtualCircuit | None
            virtual circuit object to be used by the control voltages class or None if
            no virtual circuit could be obtained or constructed.
        """

        # NOTE(Matthew): We are assuming all models require the same inputs, this
        #                restriction concerns the note concerning validation of the
        #                assumption in RealTimeVirtualCircuitsProvider.__init__.

        # Make sure we have an observable registry, which in turn implies it is
        # validated to support all inputs out models require.

        if self._observable_registry is None:
            _logger.error(
                "RealTimeVirtualCircuitProvider cannot generate a virtual circuit "
                "unless a valid observable registry is supplied."
            )
            return None

        # Check that the targets requested are supported by models this instance of RTVC
        # has loaded.

        unsupported_targets = [
            target not in self._model_specs[0].inputs for target in targets
        ]
        if len(unsupported_targets) != 0:
            _logger.error(
                "Unsupported targets supplied to RealTimeVirtualCircuitProvider.get_vc:"
                f" {unsupported_targets}"
            )
            return None

        # Get all input values from the observable registry.

        input_data: list[float] = []
        for input in self._model_specs[0].inputs:
            input_val = self._observable_registry.get(input, timestamp)
            if input_val is None:
                _logger.error(
                    f"Could not retrieve {input} to calculate VC at time {timestamp}"
                )
                return None
            input_data.append(input_val)

        # Predict the shape matrix by sending a request to the RTVC server.

        shape_matrix = self._predict_shape_matrix(input_data)
        if shape_matrix is None:
            _logger.error(
                f"Failed to obtain shape matrix at time {timestamp} from the RTVC "
                "server."
            )
            return None

        # Subset the shape matrix by targets specified, and then invert to get VC
        # matrix.

        target_indices = [
            self._model_specs[0].outputs.index(target) for target in targets
        ]
        subsetted_shape_matrix = shape_matrix[target_indices, :][:, self._coil_indices]

        vc_matrix = np.linalg.pinv(subsetted_shape_matrix)

        return VirtualCircuit(shape_matrix=shape_matrix, VCs_matrix=vc_matrix)

    def _predict_shape_matrix(self, input_data: np.ndarray) -> np.ndarray | None:
        """
        Predict a shape circuit by invoking inference on the RTVC server. A shape matrix
        is the Jacobian of output targets with respect to input data. Input data mainly
        consists of currents, but can also include physical parameters like li and
        betap. Output targets are geometric properties of the plasma.

        Parameters
        ----------
        input_data : list[float]
            The input data on which the RTVC server should infer the virtual circuit
            matrix.
        """
        # Ensure data fits in the shared memory. Note the factor of 4 is reflects that
        # we are working in 32-bit precision and so 4 bytes per input value.
        if input_data.dtype != np.float32:
            try:
                input_data = input_data.astype(np.float32)
            except Exception:
                _logger.error("Invalid type of input data supplied, expect float32.")
                return None

        if len(input_data.shape) != 1:
            _logger.error(
                "Invalid input data shape provided to RealTimeVirtualCircuitProvider."
                f"_predict_shape_matrix: {input_data.shape}"
            )
            return None

        if input_data.shape[0] * 4 > _SHARED_MEMORY_SIZE:
            _logger.error("Input data too large for shared memory.")
            return None

        # Write input data to shared memory.
        struct.pack_into(f"{len(input_data)}f", self._shared_memory_map, 0, *input_data)

        # Signal that we are ready for an inference task to be performed.
        self._sem_ready.release()

        # Await result of inference. If we timeout, then log an error and fail up.
        try:
            self._sem_done.acquire(timeout=10)
        except BusyError:
            _logger.error("RTVC server timedout inferring a virtual circuit.")
            return None

        # Read result and return it.
        # TODO(Matthew): Dynamically calculate the size of the returned matrix.
        return (
            np.array(struct.unpack_from("195f", self._shared_memory_map, 0))
            .reshape((15, 13))
            .astype(np.float32)
        )

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

        # Decode version string provided in stdout by RTVC binary.
        version_string = result.stdout.decode("utf-8")

        # Validate that the version string obtained matches "rtvc v<VER>" where <VER> is
        # a valid Semantic Versioning string.
        if not re.match(_RTVC_VERSION_PATTERN, version_string):
            _logger.error(
                f"RTVC binary does not look right, a call to:\n    {self._rtvc_binary} "
                f"--version\nresulted in:\n    {version_string}"
            )
            return False

        return True

    def _validate_observable_registry(
        self, observable_registry: ObservableRegistry
    ) -> bool:
        """
        Determine if the provided observable registry satisfies the necessary
        requirements for get_vc to be executed correctly. E.g. does it provide access to
        all the physical parameters of an equilibrium needed by a model.

        Parameters
        ----------
        observable_registry : ObservableRegistry
            The observable registry to validate.
        """

        # NOTE(Matthew): We are assuming all models require the same inputs, this
        #                restriction concerns the note concerning validation of the
        #                assumption in RealTimeVirtualCircuitsProvider.__init__.
        missing_inputs = [
            not observable_registry.has(input) for input in self._model_specs[0].inputs
        ]

        if len(missing_inputs) != 0:
            _logger.warning(
                "RealTimeVirtualCircuitProvider supplied an observable registry that is"
                f" missing support for the following observables: {missing_inputs}"
            )
            return False

        return True
