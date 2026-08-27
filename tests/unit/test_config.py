"""M01 verify - config registry."""

from __future__ import annotations

import warnings
from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from libs.ms_core import config as cfgmod


@pytest.fixture
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # unverified-instrument warning
        return cfgmod.load()


def test_loads(cfg):
    assert cfg.config_hash
    assert len(cfg.instruments.instruments) >= 3


def test_hash_stable(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        again = cfgmod.load()
    assert cfg.config_hash == again.config_hash


def test_hash_changes_when_config_changes(tmp_path, cfg):
    for name in cfgmod.CONFIG_FILES:
        src = (cfgmod.CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        (tmp_path / f"{name}.yaml").write_text(src, encoding="utf-8")

    risk = yaml.safe_load((tmp_path / "risk_limits.yaml").read_text(encoding="utf-8"))
    risk["max_trades_per_day"] += 1
    (tmp_path / "risk_limits.yaml").write_text(yaml.safe_dump(risk), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        changed = cfgmod.load(tmp_path)
    assert changed.config_hash != cfg.config_hash


def test_rejects_unknown_key(tmp_path):
    """A typo in a YAML key must be loud, not silently ignored."""
    for name in cfgmod.CONFIG_FILES:
        src = (cfgmod.CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        (tmp_path / f"{name}.yaml").write_text(src, encoding="utf-8")

    risk = yaml.safe_load((tmp_path / "risk_limits.yaml").read_text(encoding="utf-8"))
    risk["max_trades_per_dya"] = 99  # typo
    (tmp_path / "risk_limits.yaml").write_text(yaml.safe_dump(risk), encoding="utf-8")

    with pytest.raises(ValidationError, match=r"max_trades_per_dya|Extra inputs"):
        cfgmod.load(tmp_path, warn=False)


def test_config_is_frozen(cfg):
    with pytest.raises(ValidationError):
        cfg.risk_limits.max_trades_per_day = 999  # type: ignore[misc]


def test_cost_is_point_in_time(cfg):
    """Applying today's statutory rates to an old trade is not a backtest."""
    early = cfg.costs.on(date(2020, 6, 1))
    assert early.effective_from <= date(2020, 6, 1)
    with pytest.raises(LookupError):
        cfg.costs.on(date(1999, 1, 1))


def test_lot_size_is_a_lookup_not_an_attribute(cfg):
    fut = cfg.instrument("NIFTY-FUT")
    assert fut.lot_size_on(date(2024, 1, 1)) > 0
    with pytest.raises(LookupError):
        fut.lot_size_on(date(1999, 1, 1))


def test_warns_while_instruments_unverified(cfg):
    """Sizing on unverified lot sizes is how positions come out 3x too large."""
    with pytest.warns(UserWarning, match="Unverified instrument"):
        cfgmod.load()
    assert cfg.unverified_instruments(), (
        "Once reference data is checked against the exchange circular and "
        "verified: true is set, delete this assertion."
    )


def test_low_credibility_source_cannot_trigger_directional(tmp_path):
    """The AP-hack rule, enforced at config load time."""
    for name in cfgmod.CONFIG_FILES:
        src = (cfgmod.CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        (tmp_path / f"{name}.yaml").write_text(src, encoding="utf-8")

    sources = yaml.safe_load((tmp_path / "sources.yaml").read_text(encoding="utf-8"))
    for s in sources["sources"]:
        if s["credibility"] < 0.5:
            s["may_trigger_directional"] = True
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources), encoding="utf-8")

    with pytest.raises(ValidationError, match=r"may only|credibility"):
        cfgmod.load(tmp_path, warn=False)


def test_taxonomy_has_noise_family(cfg):
    """'Nothing here' is a real class the classifier must learn, not an absence."""
    names = cfg.taxonomy.class_names()
    assert any(n.startswith("noise.") for n in names)
    assert cfg.taxonomy.get("noise.opinion").tradeable is False


def test_taxonomy_lookup_rejects_unknown(cfg):
    with pytest.raises(KeyError):
        cfg.taxonomy.get("monetary.not_a_real_class")


def test_high_severity_classes_require_two_sources(cfg):
    military = cfg.taxonomy.get("geopolitical.military_action")
    assert military.require_sources >= 2
