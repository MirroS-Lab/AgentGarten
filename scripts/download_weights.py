"""Download the AgentGarten weights and point the repository's .env at them.

Layout under <weights_dir>, matching the README:

    AgentGarten-renderer/  renderer checkpoint (student, text tower, tokenizer)
    Wan2.2-TI2V-5B/        Wan2.2_VAE.pth
    Cosmos3-Nano/          base model, only with --with-base (training starts here)

A Hugging Face token, if a repo asks for one, is read from the HF_TOKEN
environment variable by huggingface_hub and is never printed here.
"""

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

RENDERER_REPO = "MirroS-Lab/AgentGarten-renderer"
BASE_REPO = "nvidia/Cosmos3-Nano"
BASE_PATTERNS = ["transformer/*", "text_tokenizer/*", "assets/negative_prompt.json"]
VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
VAE_FILE = "Wan2.2_VAE.pth"


def set_env(lines: list[str], key: str, value: str) -> None:
    """Set key=value in the .env lines.

    Fills a commented template line and keeps a value the user already set.
    """
    active = re.compile(rf"^{re.escape(key)}=")
    template = re.compile(rf"^#\s*{re.escape(key)}=")
    if any(active.match(line) for line in lines):
        print(f".env: kept your existing {key}")
        return
    for index, line in enumerate(lines):
        if template.match(line):
            lines[index] = f"{key}={value}"
            print(f".env: set {key}={value}")
            return
    lines.append(f"{key}={value}")
    print(f".env: added {key}={value}")


def write_env(env_file: Path, env_example: Path, values: dict[str, str]) -> None:
    if not env_file.exists():
        shutil.copyfile(env_example, env_file)
        print(f".env: created from {env_example.name}")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    for key, value in values.items():
        set_env(lines, key, value)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument(
        "--with-base",
        action="store_true",
        help="also download Cosmos3-Nano, which only training needs",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--env-example", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, required=True, help="interpreter to record as WM_PYTHON"
    )
    args = parser.parse_args()
    weights = args.weights_dir.resolve()

    vae = Path(
        hf_hub_download(VAE_REPO, VAE_FILE, local_dir=weights / "Wan2.2-TI2V-5B")
    )
    renderer = Path(
        snapshot_download(RENDERER_REPO, local_dir=weights / "AgentGarten-renderer")
    )
    print(f"renderer: {renderer}")
    print(f"vae:      {vae}")
    values = {
        "WM_PYTHON": args.python.resolve().as_posix(),
        "WM_WAN22_VAE": vae.as_posix(),
    }

    if args.with_base:
        base = Path(
            snapshot_download(
                BASE_REPO,
                allow_patterns=BASE_PATTERNS,
                local_dir=weights / "Cosmos3-Nano",
            )
        )
        print(f"base:     {base}")
        values["WM_COSMOS3_NANO"] = base.as_posix()

    write_env(args.env_file, args.env_example, values)


if __name__ == "__main__":
    main()
