"""The Space allowlist, and the check that keeps it honest.

An allowlist fails safe for secrets and silently for correctness. When M91 split
the dashboard into a front door and a workspace, `index.html` began referencing
`site.css` and `desk.html` and neither was added to the list - so the deployed
Space went on serving the old single page, and the next upload would have put
the new front door up without its stylesheet. Nothing would have errored.
"""

from __future__ import annotations

import sys
import types

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


class TestChainPayloads:
    """The chain ships as one file per expiry, so the allowlist grew a pattern.

    An allowlist is the right shape for a directory that gets uploaded to a
    public URL - a new file is excluded until somebody says otherwise. Naming
    thirty-odd expiry payloads in it would mean editing this file every time an
    expiry rolls, which is how allowlists become ignore lists. The pattern is
    the compromise, and it is only worth anything if it stays narrow.
    """

    def test_the_pattern_matches_what_the_packer_writes(self):
        from jobs.deploy_space import PUBLISHED_PATTERN

        for name in ("chart_NIFTY_20260908.js", "chart_BANKNIFTY_20270629.js"):
            assert PUBLISHED_PATTERN.match(name), f"{name} would not be published"

    @pytest.mark.parametrize(
        "name",
        [
            ".env",
            "chart_index.js",
            "notes.js",
            "chart_nifty_20260908.js",
            "chart_NIFTY_2026.js",
            "chart_NIFTY_20260908.js.bak",
            "secrets_NIFTY_20260908.js",
        ],
    )
    def test_the_pattern_does_not_widen_into_an_ignore_list(self, name: str):
        from jobs.deploy_space import PUBLISHED_PATTERN

        assert not PUBLISHED_PATTERN.match(name), (
            f"{name} would be uploaded to a public URL by a rule meant for " "expiry payloads"
        )

    def test_an_index_naming_a_file_that_is_not_there_stops_the_deploy(self, tmp_path, monkeypatch):
        """The failure that is silent on both sides.

        The page loads the index, draws the whole ladder from it, requests the
        candles, and gets a 404. `onerror` rejects a promise on a path nobody
        awaits, so the console shows one failed request and the pane simply
        stays empty next to a ladder that looks complete.
        """
        import jobs.deploy_space as mod

        (tmp_path / "chart_index.js").write_text(
            'window.CHART_INDEX = {"expiries":[{"file": "chart_NIFTY_20260908.js"}]};',
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "SPACE_DIR", tmp_path)
        assert mod.index_is_covered() == ["chart_NIFTY_20260908.js"]

        (tmp_path / "chart_NIFTY_20260908.js").write_text("window.X = 1;", encoding="utf-8")
        assert mod.index_is_covered() == []

    def test_the_payloads_are_gathered_from_disk_not_from_the_index(self, tmp_path, monkeypatch):
        """Deriving the file list from the index would make the check above vacuous."""
        import jobs.deploy_space as mod

        (tmp_path / "chart_NIFTY_20260908.js").write_text("window.X = 1;", encoding="utf-8")
        (tmp_path / "chart_index.js").write_text("window.CHART_INDEX = {};", encoding="utf-8")
        monkeypatch.setattr(mod, "SPACE_DIR", tmp_path)
        assert [f.name for f in mod.payloads()] == ["chart_NIFTY_20260908.js"]


class TestWhatIsActuallyUploaded:
    """The summary and the upload have to be the same list, by construction.

    They were not. `check()` built its list from `files()`, which honours the
    expiry-payload pattern; `upload_folder` was passed `PUBLISHED` alone, which
    does not. So a deploy printed 6.9 MB of candles, uploaded nine files, and
    reported "published" - truthfully, about the nine. The index went up naming
    eighteen payloads that were never sent, and the live page would have drawn a
    complete ladder over eighteen 404s.

    A summary printed to be read before saying yes, that describes something
    other than what happens after yes, is worse than no summary.
    """

    def _spy(self, monkeypatch):
        calls = {}

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, **kw):
                calls["create"] = kw

            def upload_folder(self, **kw):
                calls["upload"] = kw

        # The vendor package is stood up in sys.modules rather than imported.
        # `huggingface_hub` lives in the `[deploy]` extra and CI installs
        # `[dev,api,infra]`, so importing it here passed locally and failed on
        # the runner - and skipping when it is absent would retire this guard in
        # the only place that runs unattended. `deploy()` imports it inside the
        # function, so a stand-in in sys.modules is all it ever sees.
        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            types.SimpleNamespace(HfApi=FakeApi, whoami=lambda **kw: {"name": "someone"}),
        )
        import jobs.deploy_space as mod

        monkeypatch.setattr(mod, "_token_from_env", lambda: "tok")
        return calls

    def test_the_upload_sends_exactly_what_the_check_listed(self, monkeypatch, capsys):
        import jobs.deploy_space as mod

        calls = self._spy(monkeypatch)
        assert mod.deploy("someone/space", private=False, yes=True) == 0
        capsys.readouterr()

        assert calls["upload"]["allow_patterns"] == [
            f.name for f in mod.files()
        ], "the upload and the printed summary are two different lists again"

    def test_the_expiry_payloads_are_in_it(self, monkeypatch, capsys):
        """The concrete regression: they were not, and nothing said so."""
        import jobs.deploy_space as mod

        calls = self._spy(monkeypatch)
        mod.deploy("someone/space", private=False, yes=True)
        capsys.readouterr()

        sent = calls["upload"]["allow_patterns"]
        payloads = [n for n in sent if mod.PUBLISHED_PATTERN.match(n)]
        assert payloads, (
            "not one expiry payload was uploaded; the page would draw its ladder "
            "from the index and 404 every set of candles behind it"
        )
        assert "chart_index.js" in sent

    def test_superseded_payloads_are_removed_from_the_Space(self, monkeypatch, capsys):
        """A leftover remote payload is the same hazard as a leftover local one.

        Worse, in fact: it is invisible from here. A file from a previous shape
        of the page - yesterday's expiries, or the per-underlying files that
        preceded the split - is a valid script defining a valid global, sitting
        next to an index that no longer names it.
        """
        import jobs.deploy_space as mod

        calls = self._spy(monkeypatch)
        mod.deploy("someone/space", private=False, yes=True)
        capsys.readouterr()

        assert calls["upload"].get("delete_patterns") == ["chart_*.js"]
