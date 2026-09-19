"""Tests for the LMS-ported sync logic (clock map, play points, sync group)."""

import asyncio
from collections.abc import Callable
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from aioslimproto.client import SlimClient
from aioslimproto.sync import SyncGroup

TRUE_OFFSET = 1_700_000_000.0
BASE_JIFFIES = 1_000_000

MakeClient = Callable[..., SlimClient]


@pytest.fixture
def writer() -> Mock:
    """Create a mocked player connection writer."""
    result = Mock()
    result.drain = AsyncMock()
    result.is_closing.return_value = False
    result.can_write_eof.return_value = False
    result.wait_closed = AsyncMock()
    result.get_extra_info.return_value = None
    return result


@pytest.fixture
def make_client(writer: Mock) -> MakeClient:
    """Return a factory for SlimClients with a stubbed strm sender."""

    def _factory(player_id: str = "p1") -> SlimClient:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        client = SlimClient(reader, writer, Mock())
        client._send_strm = AsyncMock()  # noqa: SLF001
        client._player_id = player_id  # noqa: SLF001
        client.packet_latency = 0.0
        return client

    return _factory


def _set_play_point(client: SlimClient, status_time: float, apparent: float) -> None:
    """Force a play point on the client (epoch 0, no packet latency)."""
    client._jiffies_epoch = 0.0  # noqa: SLF001
    client.packet_latency = 0.0
    client._jiffies = int(status_time * 1000)  # noqa: SLF001
    client._elapsed_milliseconds = int((status_time - apparent) * 1000)  # noqa: SLF001


class TestJiffiesEpoch:
    """The min-latency clock estimator (port of trackJiffiesEpoch)."""

    async def test_converges_to_minimum_offset(self, make_client: MakeClient) -> None:
        """The smallest observed offset wins, ignoring queueing latency."""
        client = make_client()
        latencies = [0.050, 0.020, 0.0, 0.010, 0.0, 0.030, 0.005, 0.0, 0.040, 0.015]
        latencies *= 6  # 60 samples keeps a zero-latency sample in every window
        for index, latency in enumerate(latencies):
            jiffies = BASE_JIFFIES + index * 1000
            client.track_jiffies_epoch(jiffies, jiffies / 1000 + TRUE_OFFSET + latency)

        jiffies = BASE_JIFFIES + (len(latencies) - 1) * 1000
        expected = jiffies / 1000 + TRUE_OFFSET
        assert client.jiffies_to_timestamp(jiffies) == pytest.approx(
            expected, abs=0.001
        )

    async def test_follows_slow_drift(self, make_client: MakeClient) -> None:
        """A steadily growing minimum offset is tracked upward."""
        client = make_client()
        drift_per_sample = 0.0005
        samples = 200
        for index in range(samples):
            jiffies = BASE_JIFFIES + index * 1000
            timestamp = jiffies / 1000 + TRUE_OFFSET + index * drift_per_sample
            client.track_jiffies_epoch(jiffies, timestamp)

        # epoch should have climbed with the drift and the mapping error stay bounded
        assert client.jiffies_epoch > TRUE_OFFSET + 0.05
        jiffies = BASE_JIFFIES + (samples - 1) * 1000
        expected = jiffies / 1000 + TRUE_OFFSET + (samples - 1) * drift_per_sample
        assert client.jiffies_to_timestamp(jiffies) == pytest.approx(expected, abs=0.05)


class TestPlayPoint:
    """Play points normalize per-player buffer/decoder differences."""

    async def test_computes_status_and_apparent_start(
        self, make_client: MakeClient
    ) -> None:
        """A play point is (status_time, status_time - elapsed)."""
        client = make_client()
        client._jiffies_epoch = 0.0  # noqa: SLF001
        client._jiffies = 10_000_000  # noqa: SLF001
        client._elapsed_milliseconds = 2000  # noqa: SLF001

        assert client.play_point == (10_000.0, 9_998.0)

    async def test_none_without_elapsed(self, make_client: MakeClient) -> None:
        """No play point is available before the player reports playback."""
        client = make_client()
        client._jiffies_epoch = 0.0  # noqa: SLF001
        client._jiffies = 10_000_000  # noqa: SLF001
        client._elapsed_milliseconds = 0  # noqa: SLF001

        assert client.play_point is None


class TestSyncStart:
    """Coordinated start (port of _syncStart)."""

    async def test_aligns_all_players_to_one_instant(
        self, make_client: MakeClient
    ) -> None:
        """All players start at the same server time, honouring their delays."""
        first = make_client("a")
        second = make_client("b")
        first._jiffies_epoch = 1000.0  # noqa: SLF001
        second._jiffies_epoch = 1000.0  # noqa: SLF001
        first.start_delay, first.play_delay = 100, 50
        second.start_delay, second.play_delay = 0, 0

        now = 2_000_000.0
        await SyncGroup().start([first, second], now=now)

        start_calls = {
            "a": first._send_strm.call_args,  # noqa: SLF001
            "b": second._send_strm.call_args,  # noqa: SLF001
        }
        instants = {}
        for key, delay in (("a", 0.150), ("b", 0.0)):
            call_args = start_calls[key]
            assert call_args.args[0] == b"u"
            interval = call_args.kwargs["replay_gain"]
            instants[key] = interval / 1000 + 1000.0 + delay

        # startAt = now + (max delay 0.150 + 0.200 headroom)
        expected = now + 0.350
        assert instants["a"] == pytest.approx(expected, abs=0.002)
        assert instants["b"] == pytest.approx(expected, abs=0.002)

    async def test_noop_for_single_player(self, make_client: MakeClient) -> None:
        """A single player is not a group and is left untouched."""
        only = make_client("a")
        await SyncGroup().start([only], now=1000.0)
        only._send_strm.assert_not_called()  # noqa: SLF001


class TestCheckSync:
    """Steady-state correction (port of _CheckSync)."""

    def _group(
        self,
        make_client: MakeClient,
        apparent_a: float,
        apparent_b: float,
        now: float,
    ) -> tuple[SlimClient, SlimClient]:
        first = make_client("a")
        second = make_client("b")
        for client, apparent in ((first, apparent_a), (second, apparent_b)):
            client.min_sync_adjust = 10
            _set_play_point(client, now - 0.2, apparent)
        return first, second

    async def test_bails_on_stale_play_point(self, make_client: MakeClient) -> None:
        """No correction when a member has not reported recently."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.2, now)
        # make play points older than the 3s recency gate
        _set_play_point(first, now - 5, 8000.0)
        _set_play_point(second, now - 5, 8000.2)

        await SyncGroup().check([first, second], now=now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_ignores_deviation_below_min_adjust(
        self, make_client: MakeClient
    ) -> None:
        """A deviation smaller than min_sync_adjust is left alone."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.005, now)

        await SyncGroup().check([first, second], now=now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_ignores_absurd_deviation(self, make_client: MakeClient) -> None:
        """A deviation above MAX_DEVIATION_ADJUST is ignored."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8030.0, now)

        await SyncGroup().check([first, second], now=now)

        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_skips_ahead_the_lagging_player(
        self, make_client: MakeClient
    ) -> None:
        """The player that started later (behind) is skipped ahead."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)

        await SyncGroup().check([first, second], now=now)

        second._send_strm.assert_any_call(b"a", replay_gain=250)  # noqa: SLF001
        first._send_strm.assert_not_called()  # noqa: SLF001

    async def test_pauses_player_ahead_of_nonskippable_reference(
        self, make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-skippable player is the reference; an ahead player is paused."""
        monkeypatch.setattr(
            SlimClient,
            "can_skip_ahead",
            property(lambda self: self.player_id != "a"),
        )
        now = 9000.0
        first, second = self._group(make_client, 8000.25, 8000.0, now)

        await SyncGroup().check([first, second], now=now)

        second._send_strm.assert_any_call(b"p", replay_gain=250)  # noqa: SLF001
        first._send_strm.assert_not_called()  # noqa: SLF001

    async def test_throttles_repeated_checks(self, make_client: MakeClient) -> None:
        """A second check within the interval does nothing."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.2, now)
        group = SyncGroup()

        await group.check([first, second], now=now)
        first._send_strm.reset_mock()  # noqa: SLF001
        second._send_strm.reset_mock()  # noqa: SLF001
        await group.check([first, second], now=now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_noop_for_single_player(self, make_client: MakeClient) -> None:
        """A single player is not a group and is left untouched."""
        only = make_client("a")
        _set_play_point(only, 9000.0 - 0.2, 8000.0)
        await SyncGroup().check([only], now=9000.0)
        only._send_strm.assert_not_called()  # noqa: SLF001

    async def test_bails_without_play_point(self, make_client: MakeClient) -> None:
        """No correction while a member has no play point yet."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        second._elapsed_milliseconds = 0  # noqa: SLF001

        await SyncGroup().check([first, second], now=now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_logs_play_points_at_debug(
        self, make_client: MakeClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The comparison is logged when debug logging is on."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)

        with caplog.at_level(logging.DEBUG, logger="aioslimproto.sync"):
            await SyncGroup().check([first, second], now=now)

        assert "playPoints:" in caplog.text
