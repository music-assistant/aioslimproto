"""Tests for player model classification helpers."""

from aioslimproto.models import min_sync_adjust_for


def test_min_sync_adjust_for_software_uses_base_default() -> None:
    """Software players and unknown devices keep the wider Player.pm deadband."""
    # squeezelite reports dev_id 12 -> "squeezeplay"
    assert min_sync_adjust_for("squeezeplay") == 30
    assert min_sync_adjust_for("softsqueeze") == 30
    assert min_sync_adjust_for("softsqueeze3") == 30
    assert min_sync_adjust_for("softboom") == 30
    assert min_sync_adjust_for("unknown device") == 30


def test_min_sync_adjust_for_hardware_uses_squeezebox2_default() -> None:
    """Squeezebox2-class hardware uses the tighter Squeezebox2.pm deadband."""
    # the Squeezebox Radio/Touch report dev_id 9 -> "controller"
    assert min_sync_adjust_for("controller") == 10
    assert min_sync_adjust_for("squeezebox2") == 10
    assert min_sync_adjust_for("transporter") == 10
    assert min_sync_adjust_for("receiver") == 10
    assert min_sync_adjust_for("boom") == 10
