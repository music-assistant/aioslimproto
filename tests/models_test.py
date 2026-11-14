"""Tests for player model classification helpers."""

from aioslimproto.models import min_sync_adjust_for


def test_min_sync_adjust_for_squeezebox2_classes() -> None:
    """Types LMS drives with Squeezebox2 or a subclass use its tighter deadband."""
    # squeezelite reports dev_id 12 -> "squeezeplay"
    assert min_sync_adjust_for("squeezeplay") == 10
    # the Squeezebox Radio/Touch report dev_id 9 -> "controller"
    assert min_sync_adjust_for("controller") == 10
    assert min_sync_adjust_for("squeezeesp32") == 10
    assert min_sync_adjust_for("softsqueeze") == 10
    assert min_sync_adjust_for("softsqueeze3") == 10
    assert min_sync_adjust_for("softboom") == 10
    assert min_sync_adjust_for("squeezebox2") == 10
    assert min_sync_adjust_for("transporter") == 10
    assert min_sync_adjust_for("receiver") == 10
    assert min_sync_adjust_for("boom") == 10


def test_min_sync_adjust_for_other_players_uses_base_default() -> None:
    """The SB1, SqueezeSlave and unknown devices keep the Player.pm deadband."""
    assert min_sync_adjust_for("squeezebox") == 30
    assert min_sync_adjust_for("squeezeslave") == 30
    assert min_sync_adjust_for("unknown device") == 30
