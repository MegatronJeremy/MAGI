"""CLI helpers for backend pool configuration."""

from __future__ import annotations

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
DEFAULT_OLLAMA_STARTUP_TIMEOUT = 30.0
DEFAULT_MANAGED_PORT_BASE = 11500


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
            # Prevent this instance from also claiming the AMD GPU.
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
        }
    if target.vendor == "amd":
        return {
            "HIP_VISIBLE_DEVICES": str(target.index),
            "ROCR_VISIBLE_DEVICES": str(target.index),
            "GPU_DEVICE_ORDINAL": str(target.index),
            # Prevent this instance from also claiming NVIDIA GPUs.
            "CUDA_VISIBLE_DEVICES": "-1",
            "NVIDIA_VISIBLE_DEVICES": "-1",
        }
    return {}


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


async def _scan_managed_pool(
    scheme: str,
    hostname: str,
    port_base: int,
    count: int,
    model: str,
    assignment: str,
    backend: str,
    warn: Callable[[str], None],
    *,
    warn_live: bool = True,
) -> BackendPool:
    """Health-check the MAGI-managed port range and return only live instances."""
    entries = [
        {
            "name": f"magi-{port_base + i}",
            "host": f"{scheme}://{hostname}:{port_base + i}",
            "model": model,
        }
        for i in range(count)
    ]
    pool = BackendPool.from_entries(
        entries,
        assignment=assignment,
        default_backend=backend,
        default_model=model,
        warn=warn,
    )
    await pool.health_check(warn_dead=False, warn_empty=False, warn_live=warn_live)
    return pool


async def _wait_for_managed_pool(
    scan_fn: Callable,
    expected_count: int,
    timeout: float,
    warn: Callable[[str], None],
) -> BackendPool:
    warn(f"waiting for {expected_count} Ollama server(s) to start (timeout {timeout:g}s)...")
    deadline = asyncio.get_running_loop().time() + timeout
    pool = await scan_fn(warn_live=False)
    while asyncio.get_running_loop().time() < deadline:
        if len(pool.instances) >= expected_count:
            warn(f"all {expected_count} Ollama server(s) ready")
            return pool
        await asyncio.sleep(0.5)
        pool = await scan_fn(warn_live=False)
    found = len(pool.instances)
    if found < expected_count:
        warn(
            f"WARNING only {found}/{expected_count} Ollama server(s) responded within {timeout:g}s; "
            "try --ollama-startup-timeout to increase the wait"
        )
    return pool


async def build_backend_pool(
    args,
    warn: Callable[[str], None] | None = None,
) -> BackendPool:
    warn = warn or print

    # Option B: explicit --backends bypasses all discovery and auto-spawn.
    explicit_backends = getattr(args, "backends", None)
    if explicit_backends:
        entries = [
            {
                "name": f"backend-{i}",
                "host": raw if "://" in raw else f"http://{raw}",
                "model": args.model,
            }
            for i, raw in enumerate(explicit_backends)
        ]
        pool = BackendPool.from_entries(
            entries,
            assignment=args.assignment,
            default_backend=args.backend,
            default_model=args.model,
            warn=warn,
        )
        await pool.health_check()
        return pool

    # Option A: MAGI-managed port range, isolated from user-run Ollama instances.
    if args.backend == "ollama" and getattr(args, "auto_instances", True):
        scheme, hostname, _ = _parse_host(args.host)
        managed_base = getattr(args, "managed_port_base", DEFAULT_MANAGED_PORT_BASE)
        scan_ports = getattr(args, "scan_ports", DEFAULT_SCAN_PORTS)

        async def scan(warn_live: bool = True) -> BackendPool:
            return await _scan_managed_pool(
                scheme, hostname, managed_base, scan_ports,
                args.model, args.assignment, args.backend, warn,
                warn_live=warn_live,
            )

        pool = await scan()
        live_count = len(pool.instances)

        if getattr(args, "auto_spawn_ollama", True) and live_count == 0:
            gpus = detect_gpus()
            if gpus:
                command = _ollama_command(getattr(args, "ollama_command", None))
                if command:
                    spawned = 0
                    for i, target in enumerate(gpus[:scan_ports]):
                        if _spawn_ollama_server(command, target, hostname, managed_base + i, warn):
                            spawned += 1
                    if spawned:
                        timeout = getattr(args, "ollama_startup_timeout", DEFAULT_OLLAMA_STARTUP_TIMEOUT)
                        pool = await _wait_for_managed_pool(scan, spawned, timeout, warn)
                        live_count = len(pool.instances)
                else:
                    warn("WARNING could not find ollama executable; skipping auto-spawn")
            else:
                warn("WARNING no GPUs detected for Ollama auto-spawn; using discovered servers only")

        if live_count:
            return pool
        warn(
            f"WARNING no Ollama servers found on managed port range {managed_base}–"
            f"{managed_base + scan_ports - 1}; falling back to --host"
        )

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
