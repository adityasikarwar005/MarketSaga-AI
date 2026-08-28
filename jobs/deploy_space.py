"""Publish the dashboard as a Hugging Face Space.

    python jobs/deploy_space.py --check          what would go, and what would not
    python jobs/deploy_space.py --space USER/NAME

The workspace is four static files and a README, so this is a static Space and
needs no runtime. `pip install -e ".[deploy]"` first.

**A free Space is public.** Everything on the page becomes readable by anyone
with the URL: the approach, the numbers, and the two gates that are failing.
That is a deliberate thing to publish rather than an accident of the free tier,
so `--check` prints what would become public and the upload asks before it runs.

Authentication is not handled here and is not read from a prompt. Run
`huggingface-cli login` yourself; the token belongs in the machine's credential
store, not in a script, a config file or a chat window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "services" / "dashboard"

# Only these are published. An allowlist rather than an ignore list: a new file
# dropped into the directory is excluded until somebody adds it here, which is
# the safer direction for a folder that gets uploaded to a public URL.
PUBLISHED = ("README.md", "index.html", "app.css", "app.js", "data.js")


def files() -> list[Path]:
    missing = [n for n in PUBLISHED if not (SPACE_DIR / n).is_file()]
    if missing:
        raise SystemExit(f"  missing from {SPACE_DIR.name}/: {', '.join(missing)}")
    return [SPACE_DIR / n for n in PUBLISHED]


def check() -> int:
    print(f"\n  Space source: {SPACE_DIR.relative_to(ROOT)}\n")
    total = 0
    for f in files():
        size = f.stat().st_size
        total += size
        print(f"    {f.name:<14}{size:>8,} bytes")
    print(f"    {'total':<14}{total:>8,} bytes")

    extra = sorted(p.name for p in SPACE_DIR.iterdir() if p.is_file() and p.name not in PUBLISHED)
    if extra:
        print(f"\n  not published (not on the allowlist): {', '.join(extra)}")

    print(
        "\n  What becomes public:\n"
        "    - the approach, the panel layout, and every number on the page\n"
        "    - Gate G4 and G7 shown failing, which is the current measured state\n"
        "    - the session snapshot in data.js, which is synthetic throughout\n"
        "\n  What does not: no credentials, no broker keys, no recorded market\n"
        "  data, no experiment log. The Space is the five files above and nothing\n"
        "  else in this repository."
    )
    return 0


def deploy(space: str, *, private: bool, yes: bool) -> int:
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        raise SystemExit('  huggingface_hub is missing. pip install -e ".[deploy]"') from None

    try:
        me = whoami()
    except Exception:
        raise SystemExit(
            "  not logged in to Hugging Face.\n"
            "  Run `huggingface-cli login` yourself and paste the token there - it\n"
            "  belongs in the credential store, not in this script or a chat window."
        ) from None

    print(f"\n  logged in as {me.get('name', '?')}")
    print(f"  target: {space}  ({'private' if private else 'PUBLIC'})")
    check()

    if not yes:
        print(
            "\n  Nothing has been uploaded. Re-run with --yes to publish.\n"
            "  A free Space is public, and the page carries real conclusions about\n"
            "  a real project - worth deciding rather than defaulting into."
        )
        return 0

    api = HfApi()
    api.create_repo(
        repo_id=space, repo_type="space", space_sdk="static", private=private, exist_ok=True
    )
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        allow_patterns=list(PUBLISHED),
        commit_message="MarketSage Desk - monitoring workspace",
    )
    print(f"\n  published: https://huggingface.co/spaces/{space}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--space", help="target repo, as USER/NAME")
    p.add_argument("--check", action="store_true", help="show what would go, upload nothing")
    p.add_argument("--private", action="store_true", help="private Space instead of public")
    p.add_argument("--yes", action="store_true", help="actually upload")
    args = p.parse_args(argv)

    if args.check or not args.space:
        if not args.space and not args.check:
            print("  give --space USER/NAME, or --check to see what would be published")
        return check()
    return deploy(args.space, private=args.private, yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
