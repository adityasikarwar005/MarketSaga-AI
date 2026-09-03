"""Publish the dashboard as a Hugging Face Space.

    python jobs/deploy_space.py --check          what would go, and what would not
    python jobs/deploy_space.py --space USER/NAME

The workspace is a handful of static files, so this is a static Space and
needs no runtime. `pip install -e ".[deploy]"` first.

**A free Space is public.** Everything on the page becomes readable by anyone
with the URL: the approach, the numbers, and the two gates that are failing.
That is a deliberate thing to publish rather than an accident of the free tier,
so `--check` prints what would become public and the upload asks before it runs.

Authentication comes from `HF_TOKEN` in `.env`, or from a prior `hf auth login`,
in that order. Never from a prompt or a command-line argument - a token passed
as an argument lands in the shell history of whoever ran it, and in the process
list of everyone on the machine while it runs.

A secret in a file is fine exactly as long as the file stays out of the
repository, so that is checked rather than trusted: the script refuses to read a
token from a `.env` that `git check-ignore` does not cover. That check is the
whole difference between a convenient config and a leak waiting for one careless
`git add -A`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "services" / "dashboard"

# Only these are published. An allowlist rather than an ignore list: a new file
# dropped into the directory is excluded until somebody adds it here, which is
# the safer direction for a folder that gets uploaded to a public URL.
PUBLISHED = (
    "README.md",
    "index.html",
    "site.css",
    "desk.html",
    "chart.html",
    "chart_index.js",
    "app.css",
    "app.js",
    "data.js",
)

# The one exception to the allowlist, and it is a narrow one. The chain payload
# is split per expiry now - NIFTY alone lists eighteen - so the names are not
# knowable in advance and listing them here would mean editing this file every
# time an expiry rolls. The pattern is anchored and specific: an underlying in
# capitals, a date, nothing else. It matches what jobs/chart_history.py writes
# and would not match a stray file somebody dropped in the directory, which is
# the property the allowlist exists for.
PUBLISHED_PATTERN = re.compile(r"^chart_[A-Z][A-Z0-9]*_\d{8}\.js$")

# `live.html` is deliberately absent. It polls `/api/live`, which is served by
# the local dashboard process reading the recorder's own partitions - a static
# Space has neither. Published, it would render an endpoint error to everyone
# who opened it: correct behaviour for that page, and a useless thing to host.
HELD_BACK = {"live.html": "needs the local /api/live endpoint and recorded data"}


def missing_dependencies() -> list[str]:
    """Local files a published page needs that are not themselves published.

    An allowlist fails safe for secrets and silently for correctness. When M91
    split the dashboard in two, `index.html` became a front door referencing
    `site.css` and `desk.html`, and neither was added here - so the Space went
    on serving the old single page, and the next upload would have put the new
    front door up without its stylesheet or the page it links to. Nothing would
    have errored; the page would simply have arrived unstyled.

    Only relative references count. A font on a CDN is not this script's
    business, and treating it as one would make the check noisy enough to skip.
    """
    wanted: list[str] = []
    for name in PUBLISHED:
        page = SPACE_DIR / name
        if page.suffix != ".html" or not page.exists():
            continue
        text = page.read_text(encoding="utf-8", errors="ignore")
        for ref in re.findall(r'(?:href|src)="([^"]+)"', text):
            if ref.startswith(("http://", "https://", "//", "#", "data:", "mailto:")):
                continue
            target = ref.split("?")[0].split("#")[0].strip()
            if target in ("./", "/") or not target:
                continue
            if target in PUBLISHED or PUBLISHED_PATTERN.match(target):
                continue
            wanted.append(f"{name} -> {target}")
    return sorted(set(wanted))


def payloads() -> list[Path]:
    """The per-expiry chain files, in the order the index lists them.

    Sorted so a diff of two deploys is readable, and gathered by pattern rather
    than by reading the index: an index that names a file which is not there is
    a page that waits forever for a global, and the check below is what catches
    that - it cannot catch it if the file list is derived from the same index.
    """
    return sorted(
        (f for f in SPACE_DIR.iterdir() if f.is_file() and PUBLISHED_PATTERN.match(f.name)),
        key=lambda f: f.name,
    )


def index_is_covered() -> list[str]:
    """Every file the index tells the page to load must be one this deploy sends.

    The failure this prevents is silent on both sides: the page requests a
    script that 404s, `onerror` fires into a promise nobody is awaiting on the
    happy path, and the pane stays empty with the ladder still drawn beside it.
    """
    index = SPACE_DIR / "chart_index.js"
    if not index.is_file():
        return []
    text = index.read_text(encoding="utf-8", errors="ignore")
    here = {f.name for f in payloads()} | set(PUBLISHED)
    return sorted({m for m in re.findall(r'"file":\s*"([^"]+)"', text) if m not in here})


def _token_from_env() -> str | None:
    """HF_TOKEN out of `.env`, but only if git genuinely cannot see the file.

    Keeping a secret in a file is fine exactly as long as the file stays out of
    the repository, so that is checked rather than trusted. A `.env` that git
    tracks is one commit away from being public, and the check costs one
    subprocess.
    """
    import os
    import subprocess

    env_file = ROOT / ".env"
    if not env_file.exists():
        return os.environ.get("HF_TOKEN") or None

    ignored = (
        subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=ROOT, capture_output=True, check=False
        ).returncode
        == 0
    )
    if not ignored:
        raise SystemExit(
            "  .env is NOT gitignored, so a token in it is one commit away from\n"
            "  being public. Add `.env` to .gitignore before putting a secret\n"
            "  there. Refusing to read it."
        )

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip("\"'") or None
    return os.environ.get("HF_TOKEN") or None


def files() -> list[Path]:
    missing = [n for n in PUBLISHED if not (SPACE_DIR / n).is_file()]
    if missing:
        raise SystemExit(f"  missing from {SPACE_DIR.name}/: {', '.join(missing)}")
    orphans = index_is_covered()
    if orphans:
        raise SystemExit(
            f"  chart_index.js names {len(orphans)} payload(s) that are not in "
            f"{SPACE_DIR.name}/: {', '.join(orphans[:4])}"
            + ("..." if len(orphans) > 4 else "")
            + "\n  The page would load the index, draw the ladder, and wait "
            "forever for candles that 404. Re-run jobs/chart_history.py."
        )
    return [SPACE_DIR / n for n in PUBLISHED] + payloads()


def check() -> int:
    print(f"\n  Space source: {SPACE_DIR.relative_to(ROOT)}\n")
    total = 0
    for f in files():
        size = f.stat().st_size
        total += size
        print(f"    {f.name:<28}{size:>10,} bytes")
    print(f"    {'total':<28}{total:>10,} bytes")

    extra = sorted(p.name for p in SPACE_DIR.iterdir() if p.is_file() and p.name not in PUBLISHED)
    if extra:
        print(f"\n  not published (not on the allowlist): {', '.join(extra)}")
    for name, why in HELD_BACK.items():
        if (SPACE_DIR / name).exists():
            print(f"    {name}: held back on purpose - {why}")

    broken = missing_dependencies()
    if broken:
        print("\n  BLOCKED: a published page references a file that is not published:")
        for line in broken:
            print(f"    {line}")
        print(
            "  Uploading this would put a page up without something it needs.\n"
            "  Add the file to PUBLISHED, or stop referencing it."
        )
        return 1

    print(
        "\n  What becomes public:\n"
        "    - the approach, the panel layout, and every number on the page\n"
        "    - Gate G4 and G7 shown failing, which is the current measured state\n"
        "    - the session snapshot in data.js, which is synthetic throughout\n"
        "\n  What does not: no credentials, no broker keys, no recorded market\n"
        "  data, no experiment log. The Space is the files above and nothing\n"
        "  else in this repository."
    )
    return 0


def deploy(space: str, *, private: bool, yes: bool) -> int:
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        raise SystemExit('  huggingface_hub is missing. pip install -e ".[deploy]"') from None

    token = _token_from_env()
    try:
        me = whoami(token=token) if token else whoami()
    except Exception:
        raise SystemExit(
            "  not authenticated to Hugging Face. Either:\n"
            "    - put HF_TOKEN=... in .env, a Write token from\n"
            "      huggingface.co/settings/tokens, or\n"
            "    - run `hf auth login` yourself.\n"
            "  A Read token authenticates fine and then fails at create_repo,\n"
            "  which reads as a permissions bug rather than a wrong token type."
        ) from None

    source = ".env" if token else "credential store"
    print(f"\n  logged in as {me.get('name', '?')}  (token from {source})")
    print(f"  target: {space}  ({'private' if private else 'PUBLIC'})")
    if check() != 0:
        return 1

    if not yes:
        print(
            "\n  Nothing has been uploaded. Re-run with --yes to publish.\n"
            "  A free Space is public, and the page carries real conclusions about\n"
            "  a real project - worth deciding rather than defaulting into."
        )
        return 0

    api = HfApi(token=token)
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
