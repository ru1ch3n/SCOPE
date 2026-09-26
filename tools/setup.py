"""One command creates an isolated environment, installs dependencies and obtains data."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pde", choices=("poisson", "darcy", "helmholtz", "ns-nonbounded", "ns-bounded", "all")
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--environment", type=Path, default=Path(".venv"))
    parser.add_argument("--cpu", action="store_true", help="CPU-only environment for software tests")
    parser.add_argument("--env-only", action="store_true", help="Install without downloading data")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        parser.error("Use Python 3.11 or 3.12")
    if not args.env_only and args.pde is None:
        parser.error("Specify --pde, or use --env-only")
    root = Path(__file__).resolve().parents[1]
    env = args.environment.resolve()
    if env.exists() and not (env / "pyvenv.cfg").is_file():
        parser.error("Environment directory exists but is not a venv; preserving it")
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def call(*command):
        subprocess.run([str(python), *command], cwd=root, check=True)

    call("-m", "pip", "install", "--upgrade", "pip")
    index = "https://download.pytorch.org/whl/" + ("cpu" if args.cpu else "cu130")
    call("-m", "pip", "install", "torch==2.12.1", "--index-url", index)
    call("-m", "pip", "install", "-e", ".[test]")
    if not args.env_only:
        call("-m", "scope.data", "--pde", args.pde, "--split", args.split, "--data-root", args.data_root)
    print(f"Ready. Python: {python}")


if __name__ == "__main__":
    main()
