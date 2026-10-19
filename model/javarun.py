"""
Runs the Java simulator from Python: one JVM per run, several in parallel.

CloudSim and iFogSim keep their state in statics, so every simulation needs its own
JVM; the simulator also resolves ./dataset/... against its working directory, so each
JVM runs inside ifogsim/.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

from design import IFOGSIM

CLASSPATH = "build/classes:jars/*:jars/commons-math3-3.5/*"


def ensure_built() -> None:
    marker = IFOGSIM / "build" / "classes" / "org" / "fog" / "ft" / "FtExperiment.class"
    if not marker.exists():
        subprocess.run(["bash", "build.sh"], cwd=IFOGSIM, check=True,
                       stdout=subprocess.DEVNULL)


def java(main: str, args: Sequence[str], props: dict[str, str] | None = None,
         capture: bool = True, timeout: float = 900) -> subprocess.CompletedProcess:
    cmd = ["java"]
    for k, v in (props or {}).items():
        cmd.append(f"-D{k}={v}")
    cmd += ["-cp", CLASSPATH, f"org.fog.ft.{main}", *map(str, args)]
    return subprocess.run(cmd, cwd=IFOGSIM, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def run_many(jobs: Iterable[tuple[str, Sequence[str]]], workers: int | None = None,
             on_done: Callable[[int, int, subprocess.CompletedProcess], None] | None = None
             ) -> list[subprocess.CompletedProcess]:
    """Runs (main, args) jobs in parallel JVMs; raises if any run fails."""
    jobs = list(jobs)
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    results: list[subprocess.CompletedProcess | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(java, main, args): i for i, (main, args) in enumerate(jobs)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            results[i] = res
            done += 1
            if res.returncode != 0:
                tail = (res.stdout or "")[-2000:]
                raise RuntimeError(f"simulation {jobs[i]} failed (exit {res.returncode}):\n{tail}")
            if on_done:
                on_done(done, len(jobs), res)
    return results  # type: ignore[return-value]


def relpath_from_ifogsim(p: Path) -> str:
    """Paths handed to the JVM are resolved from ifogsim/."""
    return os.path.relpath(Path(p).resolve(), IFOGSIM.resolve())
