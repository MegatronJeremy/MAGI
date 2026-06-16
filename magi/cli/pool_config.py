"""CLI helpers for backend pool configuration."""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from magi.llm import BackendPool


DEFAULT_SCAN_PORTS = 8
DEFAULT_OLLAMA_STARTUP_TIMEOUT = 15.0


@dataclass(frozen=True)
class GpuTarget:
    vendor: str
    index: int
    name: str


def _parse_host(host: str):
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "localhost"
    start_port = parsed.port or 11434
    return scheme, hostname, start_port


def _auto_ollama_entries(host: str, model: str, count: int) -> list[dict]:
    scheme, hostname, start_port = _parse_host(host)
    return [
        {
            "name": f"ollama-{port}",
            "host": f"{scheme}://{hostname}:{port}",
            "model": model,
        }
        for port in range(start_port, start_port + max(1, count))
    ]


def _run_command(args: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _detect_nvidia_gpus() -> list[GpuTarget]:
    command = shutil.which("nvidia-smi")
    if command:
        output = _run_command([
            command,
            "--query-gpu=index,name",
            "--format=csv,noheader",
        ])
        gpus = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            gpus.append(GpuTarget("nvidia", int(parts[0]), parts[1]))
        if gpus:
            return gpus
    names = [
        name
        for name in _windows_video_controller_names()
        if "nvidia" in name.casefold()
    ]
    return [GpuTarget("nvidia", index, name) for index, name in enumerate(names)]


def _windows_video_controller_names() -> list[str]:
    if platform.system() != "Windows":
        return []
    powershell = shutil.which("powershell")
    if powershell:
        output = _run_command([
            powershell,
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ])
        names = [line.strip() for line in output.splitlines() if line.strip()]
        if names:
            return names
    output = _run_command(["wmic", "path", "win32_VideoController", "get", "Name", "/value"])
    names = []
    for line in output.splitlines():
        if line.startswith("Name="):
            names.append(line.split("=", 1)[1].strip())
    return [name for name in names if name]


def _detect_amd_gpus_from_rocm() -> list[GpuTarget]:
    command = shutil.which("rocm-smi")
    if not command:
        return []
    output = _run_command([command, "--showproductname"])
    gpus = []
    seen = set()
    for line in output.splitlines():
        if "GPU[" not in line:
            continue
        start = line.find("GPU[") + len("GPU[")
        end = line.find("]", start)
        if end <= start:
            continue
        raw_index = line[start:end].strip()
        if not raw_index.isdigit() or raw_index in seen:
            continue
        seen.add(raw_index)
        name = line.split(":", 1)[-1].strip() or f"AMD GPU {raw_index}"
        gpus.append(GpuTarget("amd", int(raw_index), name))
    return gpus


def _detect_amd_gpus_from_windows() -> list[GpuTarget]:
    names = []
    for name in _windows_video_controller_names():
        lowered = name.casefold()
        if "amd" in lowered or "radeon" in lowered:
            names.append(name)
    return [GpuTarget("amd", index, name) for index, name in enumerate(names)]


def detect_gpus() -> list[GpuTarget]:
    nvidia = _detect_nvidia_gpus()
    amd = _detect_amd_gpus_from_rocm() or _detect_amd_gpus_from_windows()
    return nvidia + amd


def _ollama_command(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    command = shutil.which("ollama")
    if command:
        return command
    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidate = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
            if os.path.exists(candidate):
                return candidate
    return None


def _gpu_env(target: GpuTarget) -> dict[str, str]:
    if target.vendor == "nvidia":
        return {
            "CUDA_VISIBLE_DEVICES": str(target.index),
            "NVIDIA_VISIBLE_DEVICES": str(target.index),
        }
    if target.vendor == "amd":
        return {
            "HIP_VISIBLE_DEVICES": str(target.index),
            "ROCR_VISIBLE_DEVICES": str(target.index),
            "GPU_DEVICE_ORDINAL": str(target.index),
        }
    return {}


def _instance_port(instance) -> int | None:
    if not instance.host:
        return None
    return urlparse(instance.host).port


def _spawn_ollama_server(
    command: str,
    target: GpuTarget,
    hostname: str,
    port: int,
    warn: Callable[[str], None],
) -> bool:
    env = os.environ.copy()
    env.update(_gpu_env(target))
    env["OLLAMA_HOST"] = f"{hostname}:{port}"
    popen_kwargs = {}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        popen_kwargs["start_new_session"] = True
    try:
        subprocess.Popen(
            [command, "serve"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs,
        )
    except OSError as exc:
        warn(f"WARNING could not spawn Ollama for {target.name} on port {port}: {exc}")
        return False
    warn(f"spawned Ollama on {hostname}:{port} for {target.vendor} GPU {target.index}: {target.name}")
    return True


async def _health_check_scan(
    args: argparse.Namespace,
    warn: Callable[[str], None],
    *,
    warn_empty: bool = False,
    warn_live: bool = True,
) -> BackendPool:
    pool = BackendPool.from_entries(
        _auto_ollama_entries(args.host, args.model, getattr(args, "scan_ports", DEFAULT_SCAN_PORTS)),
        assignment=getattr(args, "assignment", "pooled"),
        default_backend=args.backend,
        default_model=args.model,
        warn=warn,
    )
    await pool.health_check(warn_dead=False, warn_empty=warn_empty, warn_live=warn_live)
    return pool


async def _wait_for_spawned_servers(
    args: argparse.Namespace,
    warn: Callable[[str], None],
    expected_count: int,
) -> BackendPool:
    timeout = getattr(args, "ollama_startup_timeout", DEFAULT_OLLAMA_STARTUP_TIMEOUT)
    deadline = asyncio.get_running_loop().time() + timeout
    pool = await _health_check_scan(args, warn, warn_live=False)
    while asyncio.get_running_loop().time() < deadline:
        if len(pool.instances) >= expected_count:
            return pool
        await asyncio.sleep(0.5)
        pool = await _health_check_scan(args, warn, warn_live=False)
    return pool


async def build_backend_pool(
    args: argparse.Namespace,
    warn: Callable[[str], None] | None = None,
) -> BackendPool:
    warn = warn or print
    if args.backend == "ollama" and getattr(args, "auto_instances", True):
        pool = await _health_check_scan(args, warn)
        live_count = len(pool.instances)
        if getattr(args, "auto_spawn_ollama", True):
            gpus = detect_gpus()
            scan_ports = getattr(args, "scan_ports", DEFAULT_SCAN_PORTS)
            if gpus and live_count < min(len(gpus), scan_ports):
                command = _ollama_command(getattr(args, "ollama_command", None))
                if command:
                    _, hostname, start_port = _parse_host(args.host)
                    live_ports = {
                        port
                        for port in (_instance_port(instance) for instance in pool.instances)
                        if port is not None
                    }
                    available_ports = [
                        port
                        for port in range(start_port, start_port + scan_ports)
                        if port not in live_ports
                    ]
                    missing_targets = gpus[live_count:live_count + len(available_ports)]
                    spawned = 0
                    for target, port in zip(missing_targets, available_ports):
                        if _spawn_ollama_server(command, target, hostname, port, warn):
                            spawned += 1
                    if spawned:
                        expected_count = min(len(gpus), scan_ports, live_count + spawned)
                        pool = await _wait_for_spawned_servers(args, warn, expected_count)
                        live_count = len(pool.instances)
                else:
                    warn("WARNING could not find ollama executable; skipping auto-spawn")
            elif not gpus:
                warn("WARNING no GPUs detected for Ollama auto-spawn; using discovered servers only")

        if live_count:
            return pool
        warn(f"WARNING no Ollama servers found while scanning from {args.host}; using --host only")

    pool = BackendPool.default_local(
        backend_name=args.backend,
        host=args.host,
        model=args.model,
        assignment=args.assignment,
        warn=warn,
    )
    await pool.health_check()
    return pool


def route_agents_for_downstream(agents: list, pool: BackendPool) -> None:
    """Assign stable backends/models for synthesis helpers, option derivation, and voting."""

    for index, agent in enumerate(agents):
        instance = pool.backend_for(agent, index)
        agent.backend = instance.backend
        if instance.model:
            agent.model = instance.model
