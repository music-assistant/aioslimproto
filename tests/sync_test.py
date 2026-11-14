"""Tests for the LMS-ported sync logic (clock map, play points, sync group)."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
import logging
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from aioslimproto.client import SlimClient
from aioslimproto.const import CHECK_SYNC_INTERVAL
from aioslimproto.models import PlayerState
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


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake the wall and monotonic clock of the sync code; both start out equal."""
    clock = SimpleNamespace(wall=9_000.0, mono=9_000.0)
    fake_time = SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono)
    monkeypatch.setattr("aioslimproto.client.time", fake_time)
    monkeypatch.setattr("aioslimproto.sync.time", fake_time)
    return clock


def _set_play_point(client: SlimClient, status_time: float, apparent: float) -> None:
    """Force a play point on the client (epoch 0, no packet latency)."""
    client._jiffies_epoch = 0.0  # noqa: SLF001
    client.packet_latency = 0.0
    client._jiffies = int(status_time * 1000)  # noqa: SLF001
    client._elapsed_milliseconds = int((status_time - apparent) * 1000)  # noqa: SLF001


def _shift_reading(client: SlimClient, millis: int) -> None:
    """Move the client's last report in time, keeping its apparent start."""
    client._jiffies += millis  # noqa: SLF001
    client._elapsed_milliseconds += millis  # noqa: SLF001


async def _check_twice(group: SyncGroup, clients: list[SlimClient], now: float) -> None:
    """Run two consecutive sync checks on new reports of unchanged play points."""
    step = round(CHECK_SYNC_INTERVAL * 1000)
    for client in clients:
        _shift_reading(client, -step)
    await group.check(clients, now=now - CHECK_SYNC_INTERVAL)
    for client in clients:
        _shift_reading(client, step)
    await group.check(clients, now=now)


begin_group = pytest.mark.parametrize(
    "begin",
    [
        lambda group, clients, now: group.start(clients, now=now),
        lambda group, clients, now: group.resume(clients, 1000.0, now=now),
    ],
    ids=["start", "resume"],
)


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

    async def test_first_sample_sets_the_epoch(self, make_client: MakeClient) -> None:
        """A player clock reading close to the server clock is mapped right away."""
        client = make_client()
        # e.g. a player on the server host, which shares its monotonic clock
        client.track_jiffies_epoch(1_000_000, 1_000.5)

        await client.start_at(1_002.5)

        client._send_strm.assert_called_once_with(b"u", replay_gain=1_002_000)  # noqa: SLF001


class TestSyncClock:
    """Sync timing runs on the monotonic clock, timed at the socket read."""

    @pytest.fixture
    async def live_client(self, writer: Mock) -> AsyncIterator[SlimClient]:
        """Create a SlimClient that reads its packets from a live socket reader."""
        client = SlimClient(asyncio.StreamReader(), writer, Mock())
        client.packet_latency = 0.0
        yield client
        client._reader_task.cancel()  # noqa: SLF001

    @staticmethod
    async def _heartbeat(client: SlimClient, jiffies: int) -> None:
        """Let the player send an STMt with its clock at jiffies; wait until handled."""
        handled = asyncio.Event()
        client.callback = Mock(side_effect=lambda *_args: handled.set())
        payload = b"STMt" + struct.pack(
            "!BBBLLLLHLLLLHLL", 0, 0, 0, 0, 0, 0, 0, 0, jiffies, 0, 0, 0, 0, 0, 0
        )
        client._reader.feed_data(  # noqa: SLF001
            b"STAT" + struct.pack("!I", len(payload)) + payload
        )
        await asyncio.wait_for(handled.wait(), 1)

    async def test_epoch_survives_a_wall_clock_step(
        self, live_client: SlimClient, clock: SimpleNamespace
    ) -> None:
        """A wall clock step (e.g. an NTP correction) leaves the clock mapping alone."""
        await self._heartbeat(live_client, 5_000_000)
        epoch = live_client.jiffies_epoch
        clock.mono += 1
        clock.wall += 3_601

        await self._heartbeat(live_client, 5_001_000)

        assert live_client.jiffies_epoch == epoch
        assert live_client.jiffies_to_timestamp(5_001_000) == clock.mono

    async def test_arrival_is_timed_at_the_socket_read(
        self,
        live_client: SlimClient,
        clock: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Event-loop delay between the read and its handling stays out of the map."""
        read_at = clock.mono

        def create_task_later(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
            # the handler only gets to run half a second after the read
            clock.mono += 0.5
            clock.wall += 0.5
            return asyncio.create_task(coro)

        monkeypatch.setattr("aioslimproto.client.create_task", create_task_later)

        await self._heartbeat(live_client, 5_000_000)

        assert live_client.jiffies_epoch == read_at - 5_000

    async def test_song_position_extrapolates_on_the_monotonic_clock(
        self, make_client: MakeClient, clock: SimpleNamespace
    ) -> None:
        """A playing player's song position counts on in monotonic time."""
        client = make_client()
        client._state = PlayerState.PLAYING  # noqa: SLF001
        client._jiffies_epoch = 0.0  # noqa: SLF001
        client._jiffies = int(clock.mono * 1000)  # noqa: SLF001
        client._elapsed_milliseconds = 42_000  # noqa: SLF001
        clock.mono += 0.25
        clock.wall += 3_600

        assert client.song_elapsed_seconds == pytest.approx(42.25)

    @pytest.mark.parametrize(
        "begin",
        [
            lambda group, clients: group.start(clients),
            lambda group, clients: group.resume(clients, 42.0),
        ],
        ids=["start", "resume"],
    )
    async def test_group_starts_on_the_monotonic_clock(
        self, make_client: MakeClient, clock: SimpleNamespace, begin: Callable
    ) -> None:
        """Without a time override the start instant is taken on the monotonic clock."""
        clients = [make_client("a"), make_client("b")]
        for client in clients:
            client._jiffies_epoch = 0.0  # noqa: SLF001
        clock.wall += 3_600

        await begin(SyncGroup(), clients)

        for client in clients:
            client._send_strm.assert_called_once_with(  # noqa: SLF001
                b"u", replay_gain=int((clock.mono + 0.2) * 1000)
            )

    async def test_group_checks_on_the_monotonic_clock(
        self, make_client: MakeClient, clock: SimpleNamespace
    ) -> None:
        """Without a time override play points are aged on the monotonic clock."""
        first, second = make_client("a"), make_client("b")
        for client, apparent in ((first, 8000.0), (second, 8000.25)):
            client.min_sync_adjust = 10
            client._state = PlayerState.PLAYING  # noqa: SLF001
            _set_play_point(client, clock.mono - 0.2, apparent)
        clock.wall += 3_600
        group = SyncGroup()

        await group.check([first, second])
        clock.mono += CHECK_SYNC_INTERVAL
        for client in (first, second):
            _shift_reading(client, round(CHECK_SYNC_INTERVAL * 1000))
        await group.check([first, second])

        second._send_strm.assert_any_call(b"a", replay_gain=250)  # noqa: SLF001


class TestPlayPoint:
    """Play points normalize per-player buffer/decoder differences."""

    async def test_computes_status_and_apparent_start(
        self, make_client: MakeClient
    ) -> None:
        """A play point is (status_time, status_time - elapsed, song_elapsed)."""
        client = make_client()
        client._jiffies_epoch = 0.0  # noqa: SLF001
        client._jiffies = 10_000_000  # noqa: SLF001
        client._elapsed_milliseconds = 2000  # noqa: SLF001
        client._elapsed_seconds = 2  # noqa: SLF001

        assert client.play_point == (10_000.0, 9_998.0, 2.0)

    async def test_song_elapsed_is_zero_while_stopped(
        self, make_client: MakeClient
    ) -> None:
        """A stopped player reports no song position, a paused one its last."""
        client = make_client()
        client._elapsed_milliseconds = 42_000  # noqa: SLF001
        client._elapsed_seconds = 42  # noqa: SLF001

        client._state = PlayerState.PAUSED  # noqa: SLF001
        assert client.song_elapsed_seconds == 42.0
        client._state = PlayerState.STOPPED  # noqa: SLF001
        assert client.song_elapsed_seconds == 0.0

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

    async def test_holds_off_the_sync_check_after_a_start(
        self, make_client: MakeClient
    ) -> None:
        """No correction runs until two seconds after the common start instant."""
        now = 9000.0
        first = make_client("a")
        second = make_client("b")
        for client, apparent in ((first, 8000.0), (second, 8000.25)):
            client.min_sync_adjust = 10
            client._state = PlayerState.PLAYING  # noqa: SLF001
            _set_play_point(client, now + 2.25, apparent)
        group = SyncGroup()
        await group.start([first, second], now=now)
        second._send_strm.reset_mock()  # noqa: SLF001

        await group.check([first, second], now=now + 1.0)
        await group.check([first, second], now=now + 2.1)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await group.check([first, second], now=now + 2.3)
        call_args = second._send_strm.call_args  # noqa: SLF001
        assert call_args.args[0] == b"a"
        assert call_args.kwargs["replay_gain"] == pytest.approx(250, abs=1)

    async def test_start_instant_wraps_with_the_player_clock(
        self, make_client: MakeClient
    ) -> None:
        """A start instant past the 32-bit jiffies wrap is sent wrapped."""
        client = make_client("a")
        client._jiffies_epoch = 0.0  # noqa: SLF001

        await client.start_at((2**32 + 500) / 1000)

        client._send_strm.assert_called_once_with(b"u", replay_gain=500)  # noqa: SLF001

    async def test_start_at_without_clock_unpauses_plainly(
        self, make_client: MakeClient
    ) -> None:
        """A player whose clock is not known yet is unpaused right away."""
        client = make_client("a")

        await client.start_at(1000.0)

        client._send_strm.assert_called_once_with(b"u", replay_gain=0)  # noqa: SLF001


class TestSyncResume:
    """Group resume (port of _Resume)."""

    NOW = 1_000_000.0
    PAUSED_AT = 42.0

    @staticmethod
    def _set_paused_position(client: SlimClient, elapsed_ms: int) -> None:
        """Give the client a clock epoch of 0 and a reported song position."""
        client._jiffies_epoch = 0.0  # noqa: SLF001
        client._jiffies = int((TestSyncResume.NOW - 1) * 1000)  # noqa: SLF001
        client._elapsed_milliseconds = elapsed_ms  # noqa: SLF001

    @staticmethod
    def _start_instant(client: SlimClient) -> float:
        """Return the server time a client was told to start at (epoch 0)."""
        call_args = client._send_strm.call_args  # noqa: SLF001
        assert call_args.args[0] == b"u"
        return float(call_args.kwargs["replay_gain"]) / 1000

    async def test_delays_each_player_by_its_lead_over_the_pause_point(
        self, make_client: MakeClient
    ) -> None:
        """A player paused past the pause point starts that much later."""
        ahead = make_client("a")
        behind = make_client("b")
        self._set_paused_position(ahead, 42_050)
        self._set_paused_position(behind, 41_980)

        await SyncGroup().resume([ahead, behind], self.PAUSED_AT, now=self.NOW)

        base = self.NOW + 0.200
        assert self._start_instant(ahead) == pytest.approx(base + 0.050, abs=0.002)
        assert self._start_instant(behind) == pytest.approx(base, abs=0.002)

    async def test_keeps_a_play_delay_lead_across_the_resume(
        self, make_client: MakeClient
    ) -> None:
        """A player with a play delay starts that much earlier, as on a start."""
        delayed = make_client("a")
        plain = make_client("b")
        delayed.play_delay = 50
        # in sync, the delayed player runs its play delay ahead in song position
        self._set_paused_position(delayed, 42_050)
        self._set_paused_position(plain, 42_000)

        await SyncGroup().resume([delayed, plain], self.PAUSED_AT, now=self.NOW)

        base = self.NOW + 0.200
        assert self._start_instant(delayed) == pytest.approx(base, abs=0.002)
        assert self._start_instant(plain) == pytest.approx(base, abs=0.002)

    async def test_player_without_play_point_starts_at_the_base_instant(
        self, make_client: MakeClient
    ) -> None:
        """A player without a play point gets no extra delay."""
        first = make_client("a")
        second = make_client("b")
        self._set_paused_position(first, 42_050)
        self._set_paused_position(second, 0)

        await SyncGroup().resume([first, second], self.PAUSED_AT, now=self.NOW)

        base = self.NOW + 0.200
        assert self._start_instant(second) == pytest.approx(base, abs=0.002)

    async def test_single_player_is_unpaused_plainly(
        self, make_client: MakeClient
    ) -> None:
        """A single player resumes right away without a start instant."""
        only = make_client("a")
        self._set_paused_position(only, 42_050)

        await SyncGroup().resume([only], self.PAUSED_AT, now=self.NOW)

        only._send_strm.assert_called_once_with(b"u", replay_gain=0)  # noqa: SLF001

    async def test_holds_off_the_sync_check_after_a_resume(
        self, make_client: MakeClient
    ) -> None:
        """No correction runs until two seconds after the common start instant."""
        now = 9000.0
        first = make_client("a")
        second = make_client("b")
        self._set_paused_position(first, 42_000)
        self._set_paused_position(second, 42_000)
        group = SyncGroup()
        await group.resume([first, second], self.PAUSED_AT, now=now)
        second._send_strm.reset_mock()  # noqa: SLF001
        for client in (first, second):
            client._state = PlayerState.PLAYING  # noqa: SLF001
        # the second player is now 250ms behind
        _set_play_point(first, now + 2.25, 8000.0)
        _set_play_point(second, now + 2.25, 8000.25)

        await group.check([first, second], now=now + 1.0)
        await group.check([first, second], now=now + 2.1)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await group.check([first, second], now=now + 2.3)
        call_args = second._send_strm.call_args  # noqa: SLF001
        assert call_args.args[0] == b"a"
        assert call_args.kwargs["replay_gain"] == pytest.approx(250, abs=1)


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
            client._state = PlayerState.PLAYING  # noqa: SLF001
            _set_play_point(client, now - 0.2, apparent)
        return first, second

    async def test_bails_on_stale_play_point(self, make_client: MakeClient) -> None:
        """No correction when a member has not reported recently."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.2, now)
        # make play points older than the 3s recency gate
        _set_play_point(first, now - 5, 8000.0)
        _set_play_point(second, now - 5, 8000.2)

        await _check_twice(SyncGroup(), [first, second], now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_ignores_deviation_below_min_adjust(
        self, make_client: MakeClient
    ) -> None:
        """A deviation smaller than min_sync_adjust is left alone."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.005, now)

        await _check_twice(SyncGroup(), [first, second], now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_ignores_absurd_deviation(self, make_client: MakeClient) -> None:
        """A deviation above MAX_DEVIATION_ADJUST is ignored."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8030.0, now)

        await _check_twice(SyncGroup(), [first, second], now)

        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_skips_ahead_the_lagging_player(
        self, make_client: MakeClient
    ) -> None:
        """The player that started later (behind) is skipped ahead."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)

        await _check_twice(SyncGroup(), [first, second], now)

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

        await _check_twice(SyncGroup(), [first, second], now)

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

    @pytest.mark.parametrize("state", [PlayerState.PAUSED, PlayerState.STOPPED])
    async def test_leaves_a_group_alone_while_a_player_is_not_playing(
        self, make_client: MakeClient, state: PlayerState
    ) -> None:
        """A group with a paused or stopped player is not corrected."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        first._state = state  # noqa: SLF001

        await _check_twice(SyncGroup(), [first, second], now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_catches_up_a_player_that_plays_on_after_an_underrun(
        self, make_client: MakeClient
    ) -> None:
        """A player still marked as rebuffering after an underrun is corrected."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        second._state = PlayerState.BUFFERING  # noqa: SLF001

        await _check_twice(SyncGroup(), [first, second], now)

        second._send_strm.assert_called_once_with(b"a", replay_gain=250)  # noqa: SLF001

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

        await _check_twice(SyncGroup(), [first, second], now)

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

    async def test_a_single_deviating_check_does_not_correct(
        self, make_client: MakeClient
    ) -> None:
        """One play point off the reference is not enough for a correction."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)

        await SyncGroup().check([first, second], now=now)

        first._send_strm.assert_not_called()  # noqa: SLF001
        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_two_deviating_checks_correct_by_the_latest_deviation(
        self, make_client: MakeClient
    ) -> None:
        """A deviation seen on two checks in a row is corrected by its latest size."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await group.check([first, second], now=now)

        await self._check_again(group, first, second, 8000.125)

        second._send_strm.assert_called_once()  # noqa: SLF001
        call_args = second._send_strm.call_args  # noqa: SLF001
        assert call_args.args[0] == b"a"
        assert call_args.kwargs["replay_gain"] == pytest.approx(125, abs=1)
        first._send_strm.assert_not_called()  # noqa: SLF001

    async def test_a_repeated_reading_does_not_confirm(
        self, make_client: MakeClient
    ) -> None:
        """A second check on the same report of the player is no confirmation."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await group.check([first, second], now=now)
        _shift_reading(first, round(CHECK_SYNC_INTERVAL * 1000))

        await group.check([first, second], now=group.next_check)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_called_once()  # noqa: SLF001

    async def test_a_repeated_reference_reading_does_not_confirm(
        self, make_client: MakeClient
    ) -> None:
        """A second check on the same report of the reference is no confirmation."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await group.check([first, second], now=now)
        _shift_reading(second, round(CHECK_SYNC_INTERVAL * 1000))

        await group.check([first, second], now=group.next_check)

        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_a_deviation_that_changes_side_is_not_confirmed(
        self, make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A player behind the reference and then ahead of it is left alone."""
        monkeypatch.setattr(
            SlimClient,
            "can_skip_ahead",
            property(lambda self: self.player_id != "a"),
        )
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await group.check([first, second], now=now)

        await self._check_again(group, first, second, 7999.75)

        second._send_strm.assert_not_called()  # noqa: SLF001

    async def test_a_correction_needs_a_fresh_confirmation(
        self, make_client: MakeClient
    ) -> None:
        """After a correction, a remaining deviation is confirmed again first."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await _check_twice(group, [first, second], now)
        second._send_strm.assert_called_once_with(b"a", replay_gain=250)  # noqa: SLF001
        second._send_strm.reset_mock()  # noqa: SLF001

        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_called_once()  # noqa: SLF001

    async def test_a_player_back_in_sync_drops_its_deviation(
        self, make_client: MakeClient
    ) -> None:
        """A deviation interrupted by a check inside the deadband starts over."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await group.check([first, second], now=now)

        await self._check_again(group, first, second, 8000.005)
        await self._check_again(group, first, second, 8000.25)

        second._send_strm.assert_not_called()  # noqa: SLF001

    @begin_group
    async def test_first_check_after_a_start_corrects_at_once(
        self, make_client: MakeClient, begin: Callable
    ) -> None:
        """A player found off on the first check after a start is corrected at once."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await begin(group, [first, second], now)
        first._send_strm.reset_mock()  # noqa: SLF001
        second._send_strm.reset_mock()  # noqa: SLF001

        await self._check_after_holdoff(group, first, second, 8000.25)

        call_args = second._send_strm.call_args  # noqa: SLF001
        assert call_args.args[0] == b"a"
        assert call_args.kwargs["replay_gain"] == pytest.approx(250, abs=1)
        first._send_strm.assert_not_called()  # noqa: SLF001

    @begin_group
    @pytest.mark.parametrize(
        "reported", [-0.2, 0.5], ids=["before-start", "first-second"]
    )
    async def test_first_check_after_a_start_skips_early_play_points(
        self, make_client: MakeClient, begin: Callable, reported: float
    ) -> None:
        """A play point from before the holdoff ended is not used for a correction."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await begin(group, [first, second], now)
        second._send_strm.reset_mock()  # noqa: SLF001
        later = group.next_check + 0.3
        _set_play_point(first, later - 0.2, 8000.0)
        # still recent, but taken while paused or in the first second of playback
        _set_play_point(second, now + reported, 8000.25)

        await group.check([first, second], now=later)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_called_once()  # noqa: SLF001

    @begin_group
    async def test_later_checks_after_a_start_need_a_confirmation(
        self, make_client: MakeClient, begin: Callable
    ) -> None:
        """After the first check following a start, a deviation is confirmed first."""
        now = 9000.0
        first, second = self._group(make_client, 8000.0, 8000.25, now)
        group = SyncGroup()
        await begin(group, [first, second], now)
        second._send_strm.reset_mock()  # noqa: SLF001

        await self._check_after_holdoff(group, first, second, 8000.005)
        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_not_called()  # noqa: SLF001

        await self._check_again(group, first, second, 8000.25)
        second._send_strm.assert_called_once()  # noqa: SLF001

    @staticmethod
    async def _check_after_holdoff(
        group: SyncGroup, first: SlimClient, second: SlimClient, apparent_b: float
    ) -> None:
        """Run the first check after a start, on play points from after its holdoff."""
        later = group.next_check + 0.3
        _set_play_point(first, later - 0.2, 8000.0)
        _set_play_point(second, later - 0.2, apparent_b)
        await group.check([first, second], now=later)

    @staticmethod
    async def _check_again(
        group: SyncGroup, first: SlimClient, second: SlimClient, apparent_b: float
    ) -> None:
        """Run the next due check, with fresh play points and b at apparent_b."""
        later = group.next_check
        _set_play_point(first, later - 0.2, 8000.0)
        _set_play_point(second, later - 0.2, apparent_b)
        await group.check([first, second], now=later)
