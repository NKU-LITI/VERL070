# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
from collections.abc import Iterable


_ATTACHED_KEYS: set[str] = set()
_LISTENING = False


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _first_env(names: Iterable[str], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def maybe_wait_for_debugger(env_name: str, default_port: int, label: str, aliases: tuple[str, ...] = ()) -> None:
    env_names = (env_name, *aliases)
    if not any(_env_enabled(name) for name in env_names):
        return

    key = "|".join(env_names)
    if key in _ATTACHED_KEYS:
        try:
            import debugpy
        except ImportError:
            print(f"[debugpy] {label}: debugpy is not installed; skip debugger breakpoint")
            return
        print(f"[debugpy] {label}: breaking in attached process, host={socket.gethostname()}, pid={os.getpid()}")
        debugpy.breakpoint()
        return
    _ATTACHED_KEYS.add(key)

    global _LISTENING
    port_names = tuple(f"{name}_PORT" for name in env_names)
    wait_names = tuple(f"{name}_WAIT" for name in env_names)
    host_names = tuple(f"{name}_HOST" for name in env_names)
    port = int(_first_env(port_names, str(default_port)))
    host = _first_env(host_names, "127.0.0.1")
    wait_for_client = _first_env(wait_names, "1").lower() in {"1", "true", "yes", "on"}

    try:
        import debugpy
    except ImportError:
        print(f"[debugpy] {label}: debugpy is not installed; skip debugger attach")
        return

    if not _LISTENING:
        debugpy.listen((host, port))
        _LISTENING = True
        print(f"[debugpy] {label}: listening on {host}:{port}, host={socket.gethostname()}, pid={os.getpid()}")
    else:
        print(f"[debugpy] {label}: debugpy is already listening in this process")

    if wait_for_client:
        print(f"[debugpy] {label}: waiting for VS Code attach")
        debugpy.wait_for_client()
    debugpy.breakpoint()
