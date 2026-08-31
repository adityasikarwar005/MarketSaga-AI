"""The Space allowlist, and the check that keeps it honest.

An allowlist fails safe for secrets and silently for correctness. When M91 split
the dashboard into a front door and a workspace, `index.html` began referencing
`site.css` and `desk.html` and neither was added to the list - so the deployed
Space went on serving the old single page, and the next upload would have put
the new front door up without its stylesheet. Nothing would have errored.
"""

from __future__ import annotations

import pytest

from jobs.deploy_space import HELD_BACK, PUBLISHED, SPACE_DIR, missing_dependencies


def test_every_published_page_has_what_it_references():
    broken = missing_dependencies()
    assert not broken, (
        "a page on the allowlist references a local file that is not on it, so an "
        f"upload would publish it incomplete: {broken}"
    )


def test_the_check_would_notice_a_missing_dependency(tmp_path, monkeypatch):
    """A detector nobody has watched fire is not a detector."""
    import jobs.deploy_space as mod

    page = tmp_path / "index.html"
    page.write_text('<link rel="stylesheet" href="nowhere.css">', encoding="utf-8")
    monkeypatch.setattr(mod, "SPACE_DIR", tmp_path)
    monkeypatch.setattr(mod, "PUBLISHED", ("index.html",))
    assert mod.missing_dependencies() == ["index.html -> nowhere.css"]


def test_remote_and_anchor_references_are_not_dependencies(tmp_path, monkeypatch):
    """A font on a CDN is not this script's business, and flagging it would make
    the check noisy enough that somebody starts skipping it."""
    import jobs.deploy_space as mod

    page = tmp_path / "index.html"
    page.write_text(
        '<a href="#gates">g</a><a href="./">home</a>'
        '<link href="https://fonts.googleapis.com/css2?family=X" rel="stylesheet">'
        '<img src="data:image/png;base64,AAAA">',
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "SPACE_DIR", tmp_path)
    monkeypatch.setattr(mod, "PUBLISHED", ("index.html",))
    assert mod.missing_dependencies() == []


def test_the_front_door_and_the_workspace_are_both_published():
    """The two pages the dashboard actually consists of. Publishing one without
    the other is what the Space was doing."""
    assert "index.html" in PUBLISHED
    assert "desk.html" in PUBLISHED
    assert "site.css" in PUBLISHED


def test_the_live_page_is_held_back_with_a_reason():
    """`live.html` polls a local endpoint that a static Space does not have. It
    is excluded on purpose, and the reason is recorded so the next person does
    not read the absence as an oversight and 'fix' it."""
    assert "live.html" not in PUBLISHED
    assert "live.html" in HELD_BACK
    assert (SPACE_DIR / "live.html").exists(), "the file exists; only its publication is declined"


def test_nothing_held_back_is_also_published():
    assert not set(HELD_BACK) & set(PUBLISHED)


@pytest.mark.parametrize("name", PUBLISHED)
def test_every_allowlisted_file_exists(name: str):
    assert (SPACE_DIR / name).is_file(), f"{name} is on the allowlist and not in the directory"
