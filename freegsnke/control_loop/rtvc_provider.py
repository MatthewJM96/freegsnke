import logging
from pathlib import Path

from .vc_provider import VirtualCircuitProvider

_logger = logging.getLogger(__name__)


class RealTimeVirtualCircuitProvider(VirtualCircuitProvider):
    """
    Provides methods to set up a real-time virtual circuits server and obtain virtual
    circuits from this.
    """

    def __init__(self, rtvc_binary: Path | None = None, start_rtvc_now: bool = True):
        self._started = False

        # Set default RTVC binary path.
        if rtvc_binary is None:
            rtvc_binary = Path("./rtvc")

        self._rtvc_binary = rtvc_binary

        # Check RTVC binary exists and looks the way it should. Note that logging is
        # done from within this method, we simply do an early exit here.
        if not self._validate_rtvc_binary():
            return

        if start_rtvc_now:
            if self.start_up():
                self._started = True
            else:
                _logger.error("Failed to start RTVC server.")

