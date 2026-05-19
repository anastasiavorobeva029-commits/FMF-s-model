from threading import Lock
from typing import Dict, Any

_CACHE_LOCK = Lock()
_SIMULATION_CACHE: Dict[int, Dict[str, Any]] = {}
