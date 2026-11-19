"""Tests for SlimClient's stream start and next-media promotion."""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import struct
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, call

import pytest

from aioslimproto.cli import SlimProtoCLI
from aioslimproto.client import SlimClient
from aioslimproto.models import (
    ButtonCode,
    EventType,
    MediaDetails,
    PlayerState,
    PlayerStatus,
    RemoteCode,
)
from aioslimproto.server import SlimServer

# a cont frame as LMS sends it: metaint (no ICY metadata), loop and guid count
_CONT_PAYLOAD = struct.pack("!IBH", 0, 0, 0)
_TRACK_URL = "http://127.0.0.1:8080/track.wav"
_RESP_OK = b"HTTP/1.0 200 OK\r\nContent-Type: audio/wav\r\n\r\n"


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
async def client(writer: Mock) -> SlimClient:
    """Create a SlimClient whose socket already reports EOF, and a stubbed play_url."""
    reader = asyncio.StreamReader()
    reader.feed_eof()
    slim_client = SlimClient(reader, writer, Mock())
    slim_client.play_url = AsyncMock()
    return slim_client


@pytest.fixture
async def live_client(writer: Mock) -> SlimClient:
    """Create a SlimClient with a real play_url and a stubbed strm sender."""
    reader = asyncio.StreamReader()
    reader.feed_eof()
    slim_client = SlimClient(reader, writer, Mock())
    slim_client._send_strm = AsyncMock()  # noqa: SLF001
    slim_client._powered = True  # noqa: SLF001
    return slim_client


def _enqueue_next_media(client: SlimClient) -> MediaDetails:
    """Simulate a track already enqueued via play_url(enqueue=True)."""
    media = MediaDetails(url="http://example.com/next.mp3")
    client._next_media = media  # noqa: SLF001
    return media


def _frame(operation: bytes, payload: bytes) -> bytes:
    """Build a packet as the player sends it: operation, payload length, payload."""
    return operation + struct.pack("!I", len(payload)) + payload


class TestPromoteNextMediaOnStmu:
    """STMu (decoder underrun) can race ahead of the STMd that normally promotes."""

    @pytest.mark.asyncio
    async def test_promotes_enqueued_media_instead_of_dropping_it(
        self, client: SlimClient
    ) -> None:
        """A pre-enqueued track should be started rather than discarded."""
        media = _enqueue_next_media(client)

        await client._process_stat_stmu(b"")  # noqa: SLF001

        client.play_url.assert_called_once_with(
            url=media.url,
            mime_type=media.mime_type,
            metadata=media.metadata,
            transition=media.transition,
            transition_duration=media.transition_duration,
            enqueue=False,
            autostart=True,
            send_flush=False,
            stream_threshold=media.stream_threshold,
            output_threshold=media.output_threshold,
        )
        assert client.next_media is None

    @pytest.mark.asyncio
    async def test_stops_normally_without_enqueued_media(
        self, client: SlimClient
    ) -> None:
        """With nothing enqueued, STMu should still stop playback as before."""
        await client._process_stat_stmu(b"")  # noqa: SLF001

        client.play_url.assert_not_called()
        assert client.state == PlayerState.STOPPED


class TestDecoderReadyPromotesAtEnqueueTime:
    """With large buffers STMd can pass before the enqueue; don't wait for STMu."""

    @pytest.mark.asyncio
    async def test_enqueue_after_stmd_starts_immediately(
        self, live_client: SlimClient
    ) -> None:
        """A track enqueued after STMd should start right away (gapless handoff)."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._process_stat_stmd(b"")  # noqa: SLF001

        await live_client.play_url(
            url="http://127.0.0.1:8080/next.mp3",
            enqueue=True,
            send_flush=False,
        )

        assert live_client._send_strm.call_args.kwargs["command"] == b"s"  # noqa: SLF001
        assert live_client.next_media is None
        assert live_client.state == PlayerState.PLAYING

    @pytest.mark.asyncio
    async def test_enqueue_without_decoder_ready_is_stored(
        self, live_client: SlimClient
    ) -> None:
        """Without a preceding STMd, an enqueued track is stored for later promotion."""
        await live_client.play_url(
            url="http://127.0.0.1:8080/next.mp3",
            enqueue=True,
            send_flush=False,
        )

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.next_media is not None

    @pytest.mark.asyncio
    async def test_stop_clears_decoder_readiness(self, live_client: SlimClient) -> None:
        """An enqueue after stop() must be stored, not started from a stale STMd."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._process_stat_stmd(b"")  # noqa: SLF001
        await live_client.stop()
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client.play_url(
            url="http://127.0.0.1:8080/next.mp3",
            enqueue=True,
            send_flush=False,
        )

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.next_media is not None

    @pytest.mark.asyncio
    async def test_late_stmd_after_stop_does_not_rearm(
        self, live_client: SlimClient
    ) -> None:
        """An STMd arriving after stop() must not let an enqueue restart playback."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        await live_client.stop()
        live_client._send_strm.reset_mock()  # noqa: SLF001
        live_client._process_stat_stmd(b"")  # noqa: SLF001

        await live_client.play_url(
            url="http://127.0.0.1:8080/next.mp3",
            enqueue=True,
            send_flush=False,
        )

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.next_media is not None


class TestGaplessHandoffState:
    """The current track plays on from the output buffer until the next one starts."""

    @staticmethod
    async def _hand_off(client: SlimClient) -> None:
        """Enqueue the next track and let STMd and STMc start its stream."""
        await client.play_url(
            url="http://127.0.0.1:8080/next.mp3",
            enqueue=True,
            send_flush=False,
        )
        client._process_stat_stmd(b"")  # noqa: SLF001
        await asyncio.sleep(0)  # let the promotion task send the strm-s
        assert client._send_strm.call_args.kwargs["command"] == b"s"  # noqa: SLF001
        client._process_stat_stmc(b"")  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_playing_player_keeps_playing_until_the_next_track_starts(
        self, live_client: SlimClient
    ) -> None:
        """The player reports playing throughout the handoff, not buffering."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001

        await self._hand_off(live_client)
        assert live_client.state == PlayerState.PLAYING

        await live_client._process_stat_stms(b"")  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING
        assert live_client.current_media is not None
        assert live_client.current_media.url == "http://127.0.0.1:8080/next.mp3"

    @pytest.mark.asyncio
    async def test_pause_key_pauses_during_the_handoff(
        self, live_client: SlimClient
    ) -> None:
        """Toggling pause while the next track is queued pauses the current one."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        await self._hand_off(live_client)

        await live_client.toggle_pause()

        live_client._send_strm.assert_called_with(b"p")  # noqa: SLF001
        assert live_client.state == PlayerState.PAUSED

    @pytest.mark.asyncio
    async def test_paused_player_stays_paused_and_can_resume(
        self, live_client: SlimClient
    ) -> None:
        """A handoff while paused (decoder done near the track end) keeps it paused."""
        live_client._state = PlayerState.PAUSED  # noqa: SLF001

        await self._hand_off(live_client)
        assert live_client.state == PlayerState.PAUSED

        await live_client.play()
        live_client._send_strm.assert_called_with(b"u", flags=0)  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("state", "send_flush"),
        [(PlayerState.PLAYING, True), (PlayerState.STOPPED, False)],
    )
    async def test_new_stream_without_audio_ahead_reports_buffering(
        self, live_client: SlimClient, state: PlayerState, *, send_flush: bool
    ) -> None:
        """A flushed or stopped player has nothing to play until the stream starts."""
        live_client._state = state  # noqa: SLF001

        await live_client.play_url(url=_TRACK_URL, send_flush=send_flush)
        live_client._process_stat_stmc(b"")  # noqa: SLF001

        assert live_client.state == PlayerState.BUFFERING


def _press(client: SlimClient, kind: str, key: str, timestamp: int = 1000) -> None:
    """Let the player report a press of play or pause, on a button or the remote."""
    if kind == "butn":
        client._process_butn(struct.pack("!LL", timestamp, ButtonCode[key]))  # noqa: SLF001
    else:
        client._process_ir(struct.pack("!LxxL", timestamp, RemoteCode[key]))  # noqa: SLF001


def _button_events(client: SlimClient) -> list[dict[str, int | str]]:
    """Return the button events the host was given."""
    callback: Mock = client.callback  # type: ignore[assignment]
    return [
        event_call.args[2]
        for event_call in callback.call_args_list
        if event_call.args[1] == EventType.PLAYER_BTN_EVENT
    ]


@pytest.mark.parametrize("kind", ["butn", "ir"])
class TestTransportKeys:
    """The player acts on play and pause itself where it can, the host gets the rest."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("state", "key", "command", "after"),
        [
            (PlayerState.PLAYING, "PAUSE", call(b"p"), PlayerState.PAUSED),
            (PlayerState.PAUSED, "PAUSE", call(b"u", flags=0), PlayerState.PLAYING),
            (PlayerState.PAUSED, "PLAY", call(b"u", flags=0), PlayerState.PLAYING),
        ],
    )
    async def test_a_key_the_player_can_follow_stays_with_the_player(
        self,
        live_client: SlimClient,
        kind: str,
        state: PlayerState,
        key: str,
        command: object,
        after: PlayerState,
    ) -> None:
        """Pause on a playing player and either key on a paused one need no host."""
        live_client._state = state  # noqa: SLF001

        _press(live_client, kind, key)
        await asyncio.sleep(0)

        assert live_client._send_strm.call_args_list == [command]  # noqa: SLF001
        assert live_client.state == after
        assert _button_events(live_client) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("state", "key"),
        [
            (PlayerState.STOPPED, "PLAY"),
            (PlayerState.STOPPED, "PAUSE"),
            (PlayerState.BUFFERING, "PLAY"),
            (PlayerState.BUFFERING, "PAUSE"),
            (PlayerState.BUFFER_READY, "PLAY"),
            (PlayerState.BUFFER_READY, "PAUSE"),
            (PlayerState.PLAYING, "PLAY"),
        ],
    )
    async def test_a_key_the_player_can_not_follow_goes_to_the_host(
        self, live_client: SlimClient, kind: str, state: PlayerState, key: str
    ) -> None:
        """A stopped player has nothing to resume: the host decides what plays."""
        live_client._state = state  # noqa: SLF001

        _press(live_client, kind, key)
        await asyncio.sleep(0)

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.state == state
        if kind == "butn":
            expected = {"type": "butn", "timestamp": 1000, "button": ButtonCode[key]}
        else:
            expected = {"type": "ir", "timestamp": 1000, "code": RemoteCode[key]}
        assert _button_events(live_client) == [expected]

    @pytest.mark.asyncio
    async def test_a_pause_is_not_turned_into_a_resume_by_a_pause_before_it(
        self, live_client: SlimClient, kind: str
    ) -> None:
        """A press is decided on the state it was made in, not what its task finds."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001

        _press(live_client, kind, "PAUSE")
        # a pause of the host lands before the task of the press runs
        live_client._state = PlayerState.PAUSED  # noqa: SLF001
        await asyncio.sleep(0)

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.state == PlayerState.PAUSED


class TestEnqueueNeverFlushes:
    """An enqueued track goes after the current one, whatever send_flush says."""

    @pytest.mark.asyncio
    async def test_stored_enqueue_keeps_the_current_track(
        self, live_client: SlimClient
    ) -> None:
        """An enqueue with the default send_flush leaves the playing track alone."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._elapsed_milliseconds = 42_000  # noqa: SLF001

        await live_client.play_url(url="http://127.0.0.1:8080/next.mp3", enqueue=True)

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.next_media is not None
        assert live_client.state == PlayerState.PLAYING
        assert live_client.elapsed_milliseconds >= 42_000

    @pytest.mark.asyncio
    async def test_enqueue_after_stmd_starts_without_flushing(
        self, live_client: SlimClient
    ) -> None:
        """An enqueue that starts right away only adds the next stream."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._elapsed_milliseconds = 42_000  # noqa: SLF001
        live_client._process_stat_stmd(b"")  # noqa: SLF001

        await live_client.play_url(url="http://127.0.0.1:8080/next.mp3", enqueue=True)

        live_client._send_strm.assert_called_once()  # noqa: SLF001
        assert live_client._send_strm.call_args.kwargs["command"] == b"s"  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING
        assert live_client.elapsed_milliseconds >= 42_000


class TestStopInvalidatesEnqueuedMedia:
    """An explicit stop() must not be overridden by a late STMu."""

    @pytest.mark.asyncio
    async def test_late_stmu_after_stop_does_not_resume(
        self, client: SlimClient
    ) -> None:
        """A STMu that arrives after stop() should not resume the enqueued track."""
        _enqueue_next_media(client)

        await client.stop()
        assert client.next_media is None

        await client._process_stat_stmu(b"")  # noqa: SLF001

        client.play_url.assert_not_called()
        assert client.state == PlayerState.STOPPED


class TestStreamThresholds:
    """The buffer thresholds given to play_url reach the player's strm-s."""

    @pytest.mark.asyncio
    async def test_play_url_uses_default_thresholds(
        self, live_client: SlimClient
    ) -> None:
        """Without thresholds, the player buffers 200 KB and 2 seconds."""
        await live_client.play_url(
            url="http://127.0.0.1:8080/track.wav",
            mime_type="audio/wav",
        )

        strm = live_client._send_strm.call_args.kwargs  # noqa: SLF001
        assert (strm["threshold"], strm["output_threshold"]) == (200, 20)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("start", ["direct", "stmu", "next"])
    async def test_custom_thresholds_reach_the_player(
        self, live_client: SlimClient, *, start: str
    ) -> None:
        """Custom thresholds apply however the url gets started, enqueued or not."""
        await live_client.play_url(
            url="http://127.0.0.1:8080/radio.flac",
            enqueue=start != "direct",
            send_flush=False,
            stream_threshold=64,
            output_threshold=1,
        )
        if start == "stmu":
            await live_client._process_stat_stmu(b"")  # noqa: SLF001
        elif start == "next":
            await live_client.next()

        strm = live_client._send_strm.call_args.kwargs  # noqa: SLF001
        assert strm["command"] == b"s"
        assert (strm["threshold"], strm["output_threshold"]) == (64, 1)


class TestStreamBodyWaitsForCont:
    """The player must not read the stream body before our codc resets its buffer."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("autostart", "expected"), [(False, b"2"), (True, b"3")])
    async def test_play_url_holds_body_until_cont(
        self, live_client: SlimClient, *, autostart: bool, expected: bytes
    ) -> None:
        """Both start modes make the player wait for cont before reading the body."""
        await live_client.play_url(
            url="http://127.0.0.1:8080/track.wav",
            mime_type="audio/wav",
            autostart=autostart,
        )

        strm = live_client._send_strm.call_args.kwargs  # noqa: SLF001
        assert strm["command"] == b"s"
        assert strm["autostart"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("autostart", [False, True])
    async def test_resp_sends_cont_after_codc(
        self, client: SlimClient, *, autostart: bool
    ) -> None:
        """The cont that releases the body always follows the codc."""
        client._auto_play = autostart  # noqa: SLF001
        client.send_frame = AsyncMock()

        await client._process_resp(  # noqa: SLF001
            b"HTTP/1.0 200 OK\r\n"
            b"Content-Type: audio/wav;rate=44100;bitrate=24;channels=2\r\n\r\n"
        )

        assert client.send_frame.await_args_list == [
            call(b"codc", b"p2321"),
            call(b"cont", _CONT_PAYLOAD),
        ]

    @pytest.mark.asyncio
    async def test_error_response_still_releases_the_player(
        self, client: SlimClient
    ) -> None:
        """A player holding back an error body would otherwise wait forever."""
        client.send_frame = AsyncMock()

        await client._process_resp(b"HTTP/1.0 404 Not Found\r\n\r\n")  # noqa: SLF001

        client.send_frame.assert_awaited_once_with(b"cont", _CONT_PAYLOAD)
        assert client.state == PlayerState.STOPPED

    @pytest.mark.asyncio
    async def test_resp_without_content_type_still_sends_cont(
        self, client: SlimClient
    ) -> None:
        """Without a codc to wait for, the body is released right away."""
        client.send_frame = AsyncMock()

        await client._process_resp(b"HTTP/1.0 200 OK\r\n\r\n")  # noqa: SLF001

        client.send_frame.assert_awaited_once_with(b"cont", _CONT_PAYLOAD)


class TestCodcSampleRate:
    """The codc of a PCM stream carries the slimproto code of its sample rate."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rate", "code"),
        [(8000, b"5"), (11025, b"0"), (22050, b"1"), (32000, b"2"), (44100, b"3")],
    )
    async def test_resp_sends_rate_code(
        self, client: SlimClient, *, rate: int, code: bytes
    ) -> None:
        """Each rate gets its slimproto code, then the cont follows."""
        client.send_frame = AsyncMock()

        await client._process_resp(  # noqa: SLF001
            b"HTTP/1.0 200 OK\r\n"
            b"Content-Type: audio/wav;rate=%d;bitrate=16;channels=2\r\n\r\n" % rate
        )

        assert client.send_frame.await_args_list == [
            call(b"codc", b"p1" + code + b"21"),
            call(b"cont", _CONT_PAYLOAD),
        ]


class TestStaleRespIsIgnored:
    """A RESP of a stream the player already dropped must not reach the new stream."""

    @pytest.mark.asyncio
    async def test_resp_before_stmc_is_ignored(self, live_client: SlimClient) -> None:
        """A RESP read between strm-s and its STMc belongs to the dropped stream."""
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        await live_client.play_url(url=_TRACK_URL, mime_type="audio/wav")
        live_client.send_frame = AsyncMock()

        await live_client._process_resp(_RESP_OK)  # noqa: SLF001

        live_client.send_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_resp_after_stmc_is_handled(self, live_client: SlimClient) -> None:
        """The new stream's own RESP, which follows its STMc, gets codc and cont."""
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        await live_client.play_url(url=_TRACK_URL, mime_type="audio/wav")
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        live_client.send_frame = AsyncMock()

        await live_client._process_resp(_RESP_OK)  # noqa: SLF001

        assert live_client.send_frame.await_args_list == [
            call(b"codc", b"p1321"),
            call(b"cont", _CONT_PAYLOAD),
        ]

    @pytest.mark.asyncio
    async def test_resp_read_together_with_its_stmc_is_handled(
        self, writer: Mock
    ) -> None:
        """Packets from one read are handled in the order the player sent them."""
        reader = asyncio.StreamReader()
        slim_client = SlimClient(reader, writer, Mock())
        slim_client._send_strm = AsyncMock()  # noqa: SLF001
        slim_client._powered = True  # noqa: SLF001
        slim_client._process_stat_stmc(b"")  # noqa: SLF001
        await slim_client.play_url(url=_TRACK_URL, mime_type="audio/wav")
        cont_sent = asyncio.Event()

        async def send_frame(command: bytes, _data: bytes) -> None:
            if command == b"cont":
                cont_sent.set()

        slim_client.send_frame = AsyncMock(side_effect=send_frame)

        # STMc with its status fields zeroed, directly followed by the RESP
        reader.feed_data(
            _frame(b"STAT", b"STMc" + bytes(49)) + _frame(b"RESP", _RESP_OK)
        )

        try:
            await asyncio.wait_for(cont_sent.wait(), 1)
        finally:
            slim_client._reader_task.cancel()  # noqa: SLF001
        assert slim_client.send_frame.await_args_list == [
            call(b"codc", b"p1321"),
            call(b"cont", _CONT_PAYLOAD),
        ]

    @pytest.mark.asyncio
    async def test_stored_enqueue_keeps_current_resp(
        self, live_client: SlimClient
    ) -> None:
        """A stored enqueue sends no strm-s, so the current stream keeps its RESP."""
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        await live_client.play_url(url=_TRACK_URL, mime_type="audio/wav")
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        await live_client.play_url(
            url="http://127.0.0.1:8080/next.wav", enqueue=True, send_flush=False
        )
        live_client.send_frame = AsyncMock()

        await live_client._process_resp(_RESP_OK)  # noqa: SLF001

        assert live_client.send_frame.await_args_list == [
            call(b"codc", b"p1321"),
            call(b"cont", _CONT_PAYLOAD),
        ]

    @pytest.mark.asyncio
    async def test_player_without_stmc_still_gets_cont(
        self, live_client: SlimClient
    ) -> None:
        """A player that never sent STMc can't be guarded, so its RESP is handled."""
        await live_client.play_url(url=_TRACK_URL, mime_type="audio/wav")
        live_client.send_frame = AsyncMock()

        await live_client._process_resp(_RESP_OK)  # noqa: SLF001

        assert live_client.send_frame.await_args_list == [
            call(b"codc", b"p1321"),
            call(b"cont", _CONT_PAYLOAD),
        ]


class TestStaleEndOfPlaybackIsIgnored:
    """The end of a stream the player already dropped must not end the new stream."""

    @staticmethod
    async def _report(client: SlimClient, *events: bytes) -> None:
        """Let the player report the given STAT events, in order."""
        for event in events:
            await client._process_stat(event + bytes(47), received=0.0)  # noqa: SLF001

    @classmethod
    async def _play_first_track(cls, client: SlimClient) -> None:
        """Start a first track the way the player reports it."""
        await client.play_url(
            url="http://127.0.0.1:8080/one.wav", metadata={"title": "One"}
        )
        await cls._report(client, b"STMc", b"STMs")

    @staticmethod
    async def _listed_tracks(client: SlimClient) -> list[str]:
        """Return the tracks the menu-mode CLI status of the player lists."""
        server = SimpleNamespace(
            name="testserver",
            logger=logging.getLogger(),
            get_player=lambda _player_id: client,
        )
        cli = SlimProtoCLI(cast("SlimServer", server))
        status = await cli._handle_status(client.player_id, "-", 10, menu="menu")  # noqa: SLF001
        return [item["text"] for item in status.get("item_loop", [])]

    @pytest.mark.asyncio
    async def test_a_late_end_of_the_flushed_stream_keeps_the_new_stream(
        self, live_client: SlimClient
    ) -> None:
        """An STMd and STMu sent before the new stream's STMc leave its media."""
        await self._play_first_track(live_client)

        await live_client.play_url(url=_TRACK_URL, metadata={"title": "Two"})
        await self._report(live_client, b"STMd", b"STMu")
        await self._report(live_client, b"STMc", b"STMs")

        assert live_client.state == PlayerState.PLAYING
        assert live_client.current_media is not None
        assert live_client.current_media.url == _TRACK_URL
        assert await self._listed_tracks(live_client) == ["Two"]

    @pytest.mark.asyncio
    async def test_the_end_of_a_stream_that_never_started_stops_the_player(
        self, live_client: SlimClient
    ) -> None:
        """A stream that was told to play but carried no audio ends without STMs."""
        await self._play_first_track(live_client)

        await live_client.play_url(
            url=_TRACK_URL, metadata={"title": "Two"}, autostart=False
        )
        await self._report(live_client, b"STMc")
        await live_client.unpause_at(0)
        await self._report(live_client, b"STMr", b"STMd", b"STMu")

        assert live_client.state == PlayerState.STOPPED
        assert live_client.current_media is None
        assert await self._listed_tracks(live_client) == []


class TestRedirect:
    """A redirect restarts the stream being set up at the new location."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("autostart", [False, True])
    async def test_redirect_keeps_media_and_start_mode(
        self, client: SlimClient, *, autostart: bool
    ) -> None:
        """The new stream keeps the details and start mode it was requested with."""
        media = MediaDetails(
            url=_TRACK_URL,
            mime_type="audio/wav",
            metadata={"title": "Track"},
            stream_threshold=100,
            output_threshold=5,
        )
        client._buffering_media = media  # noqa: SLF001
        client._auto_play = autostart  # noqa: SLF001

        await client._process_resp(  # noqa: SLF001
            b"HTTP/1.0 302 Found\r\nLocation: http://127.0.0.1:8081/track.wav\r\n\r\n"
        )

        client.play_url.assert_awaited_once_with(
            "http://127.0.0.1:8081/track.wav",
            media.mime_type,
            media.metadata,
            media.transition,
            media.transition_duration,
            send_flush=False,
            stream_threshold=media.stream_threshold,
            output_threshold=media.output_threshold,
            autostart=autostart,
        )

    @pytest.mark.asyncio
    async def test_redirect_during_gapless_handoff_keeps_playing(
        self, live_client: SlimClient
    ) -> None:
        """Redirecting the next track's stream doesn't stop the current track."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._elapsed_milliseconds = 42_000  # noqa: SLF001
        await live_client.play_url(url=_TRACK_URL, send_flush=False)
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client._process_resp(  # noqa: SLF001
            b"HTTP/1.0 302 Found\r\nLocation: http://127.0.0.1:8081/track.wav\r\n\r\n"
        )

        live_client._send_strm.assert_called_once()  # noqa: SLF001
        assert live_client._send_strm.call_args.kwargs["command"] == b"s"  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING
        assert live_client.elapsed_milliseconds >= 42_000


class TestUpdateMetadata:
    """A stream that plays on can be given the metadata of what it carries now."""

    @pytest.mark.asyncio
    async def test_replaces_the_metadata_of_the_playing_media(
        self, live_client: SlimClient
    ) -> None:
        """The media that plays reports the new metadata, and the display shows it."""
        await live_client.play_url(
            url=_TRACK_URL, metadata={"title": "One", "artist": "First"}
        )
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        live_client.callback.reset_mock()
        live_client.display_control.set_lines = AsyncMock()
        live_client.extra_data["playlist_timestamp"] = 0

        await live_client.update_metadata({"title": "Two", "artist": "Second"})

        assert live_client.current_media is not None
        assert live_client.current_media.url == _TRACK_URL
        assert live_client.current_media.metadata == {
            "title": "Two",
            "artist": "Second",
        }
        # clients that follow the playlist are told that it changed
        assert live_client.extra_data["playlist_timestamp"] > 0
        live_client.callback.assert_called_once_with(
            live_client, EventType.PLAYER_UPDATED
        )
        live_client.display_control.set_lines.assert_awaited_once_with("Second", "Two")

    @pytest.mark.asyncio
    async def test_replaces_the_metadata_of_a_stream_that_is_about_to_play(
        self, live_client: SlimClient
    ) -> None:
        """A stream that buffers starts with the metadata it was given last."""
        await live_client.play_url(url=_TRACK_URL, metadata={"title": "Old"})
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        await live_client.play_url(
            url="http://127.0.0.1:8080/next.wav", metadata={"title": "One"}
        )

        await live_client.update_metadata({"title": "Two"})

        # the stream that still plays keeps its own
        assert live_client.current_media is not None
        assert live_client.current_media.metadata == {"title": "Old"}
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        assert live_client.current_media.url == "http://127.0.0.1:8080/next.wav"
        assert live_client.current_media.metadata == {"title": "Two"}

    @pytest.mark.asyncio
    async def test_the_same_metadata_is_no_update(
        self, live_client: SlimClient
    ) -> None:
        """Metadata that did not change is not reported again."""
        await live_client.play_url(url=_TRACK_URL, metadata={"title": "One"})
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        live_client.callback.reset_mock()
        live_client.extra_data["playlist_timestamp"] = 0

        await live_client.update_metadata({"title": "One"})

        live_client.callback.assert_not_called()
        assert live_client.extra_data["playlist_timestamp"] == 0

    @pytest.mark.asyncio
    async def test_a_player_without_a_stream_takes_no_metadata(
        self, live_client: SlimClient
    ) -> None:
        """Metadata belongs to a stream: without one there is nothing to update."""
        await live_client.update_metadata({"title": "One"})

        assert live_client.current_media is None
        updated = call(live_client, EventType.PLAYER_UPDATED)
        assert updated not in live_client.callback.mock_calls


class TestClearMedia:
    """A stream that is over for the server can be forgotten by the player."""

    @pytest.mark.asyncio
    async def test_forgets_the_media_a_stop_keeps(
        self, live_client: SlimClient
    ) -> None:
        """After a stop the media that played is still reported, until it is cleared."""
        await live_client.play_url(
            url=_TRACK_URL, metadata={"title": "One", "artist": "First"}
        )
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        await live_client.stop()
        assert live_client.current_media is not None
        live_client.callback.reset_mock()
        live_client.display_control.set_lines = AsyncMock()
        live_client.extra_data["playlist_timestamp"] = 0

        await live_client.clear_media()

        assert live_client.current_media is None
        assert live_client.current_url is None
        assert live_client.extra_data["playlist_timestamp"] > 0
        live_client.callback.assert_called_once_with(
            live_client, EventType.PLAYER_UPDATED
        )
        # an idle display: the clock, not the track
        (shown,) = live_client.display_control.set_lines.await_args_list
        assert shown.args == ()
        assert set(shown.kwargs) == {"fullscreen"}

    @pytest.mark.asyncio
    async def test_forgets_a_stream_that_did_not_start(
        self, live_client: SlimClient
    ) -> None:
        """A stream that was requested and never played leaves no media either."""
        await live_client.play_url(url=_TRACK_URL, metadata={"title": "One"})
        _enqueue_next_media(live_client)

        await live_client.clear_media()

        # a late start of that stream brings nothing back
        await live_client._process_stat_stms(b"")  # noqa: SLF001
        assert live_client.current_media is None
        assert live_client.next_media is None

    @pytest.mark.asyncio
    async def test_a_late_redirect_of_a_forgotten_stream_starts_nothing(
        self, live_client: SlimClient
    ) -> None:
        """A redirect that arrives after its stream was forgotten is ignored."""
        await live_client.play_url(url=_TRACK_URL, metadata={"title": "One"})
        live_client._process_stat_stmc(b"")  # noqa: SLF001
        await live_client.stop()
        await live_client.clear_media()
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client._process_resp(  # noqa: SLF001
            b"HTTP/1.0 302 Found\r\nLocation: http://127.0.0.1:8081/track.wav\r\n\r\n"
        )

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.current_media is None

    @pytest.mark.asyncio
    async def test_a_player_without_media_has_nothing_to_forget(
        self, live_client: SlimClient
    ) -> None:
        """Clearing twice reports the change once."""
        live_client.extra_data["playlist_timestamp"] = 0

        await live_client.clear_media()

        updated = call(live_client, EventType.PLAYER_UPDATED)
        assert updated not in live_client.callback.mock_calls
        assert live_client.extra_data["playlist_timestamp"] == 0


class TestOutputUnderrun:
    """A player that ran out of audio (STMo) is resumed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("autostart", [False, True])
    async def test_resumes_the_player(
        self, live_client: SlimClient, *, autostart: bool
    ) -> None:
        """The player is unpaused, however its stream was started."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        live_client._auto_play = autostart  # noqa: SLF001

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_awaited_once_with(b"u", replay_gain=0)  # noqa: SLF001
        underrun = call(live_client, EventType.PLAYER_OUTPUT_UNDERRUN)
        assert underrun in live_client.callback.mock_calls
        assert live_client.state == PlayerState.PLAYING

    @pytest.mark.asyncio
    async def test_resumes_the_player_during_a_gapless_handoff(
        self, live_client: SlimClient
    ) -> None:
        """An underrun while the next track buffers resumes the player too."""
        await live_client.play_url(url=_TRACK_URL, send_flush=False)
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_awaited_once_with(b"u", replay_gain=0)  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_ignores_the_flushed_stream(self, live_client: SlimClient) -> None:
        """A late underrun of the flushed stream doesn't start the new one early."""
        await live_client.play_url(url=_TRACK_URL, autostart=False, send_flush=True)
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        underrun = call(live_client, EventType.PLAYER_OUTPUT_UNDERRUN)
        assert underrun not in live_client.callback.mock_calls


class TestStopAfterUnderrun:
    """A player the host stops after an underrun stays stopped."""

    @staticmethod
    def _react_to_underrun(
        client: SlimClient, reaction: Callable[[], Awaitable[None]]
    ) -> list[asyncio.Task[None]]:
        """Run the reaction as a task on each underrun, as an event subscriber does."""
        tasks: list[asyncio.Task[None]] = []

        def callback(_client: SlimClient, event: EventType, _data: Any = None) -> None:
            if event == EventType.PLAYER_OUTPUT_UNDERRUN:
                tasks.append(asyncio.create_task(reaction()))

        client.callback = Mock(side_effect=callback)
        return tasks

    @pytest.mark.asyncio
    async def test_the_resume_confirmation_does_not_restart_a_stopped_player(
        self, live_client: SlimClient
    ) -> None:
        """The answer to the unpause that went out before the stop changes nothing."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        tasks = self._react_to_underrun(live_client, live_client.stop)

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await asyncio.gather(*tasks)
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        assert live_client._send_strm.await_args_list == [  # noqa: SLF001
            call(b"u", replay_gain=0),
            call(b"q", flags=0),
        ]
        assert live_client.state == PlayerState.STOPPED

    @pytest.mark.asyncio
    async def test_an_underrun_of_a_stopped_player_does_not_unpause_it(
        self, live_client: SlimClient
    ) -> None:
        """An underrun reported after the stop is passed on but not resumed."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        await live_client.stop()
        live_client._send_strm.reset_mock()  # noqa: SLF001

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        underrun = call(live_client, EventType.PLAYER_OUTPUT_UNDERRUN)
        assert underrun in live_client.callback.mock_calls
        assert live_client.state == PlayerState.STOPPED

    @pytest.mark.asyncio
    async def test_powering_off_after_an_underrun_keeps_the_player_stopped(
        self, live_client: SlimClient
    ) -> None:
        """A power off in reaction to the underrun stops the player for good."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        tasks = self._react_to_underrun(
            live_client, lambda: live_client.power(powered=False)
        )

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await asyncio.gather(*tasks)
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        assert not live_client.powered
        assert live_client.state == PlayerState.STOPPED

    @pytest.mark.asyncio
    async def test_a_new_stream_requested_after_an_underrun_starts_normally(
        self, live_client: SlimClient
    ) -> None:
        """Underruns of the flushed stream are not resumed; the new stream plays."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001
        tasks = self._react_to_underrun(
            live_client, lambda: live_client.play_url(url=_TRACK_URL)
        )

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await asyncio.gather(*tasks)
        live_client._send_strm.reset_mock()  # noqa: SLF001
        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await live_client._process_stat(b"STMs" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_not_called()  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING
        assert live_client.current_url == _TRACK_URL

    @pytest.mark.asyncio
    async def test_an_underrun_the_host_ignores_is_resumed(
        self, live_client: SlimClient
    ) -> None:
        """Without a reaction from the host, the player is unpaused and plays on."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001

        await live_client._process_stat(b"STMo" + bytes(47), received=0.0)  # noqa: SLF001
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        live_client._send_strm.assert_awaited_once_with(b"u", replay_gain=0)  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING

    @pytest.mark.asyncio
    async def test_a_paused_player_is_resumed_by_play(
        self, live_client: SlimClient
    ) -> None:
        """A normal pause and unpause end in the playing state."""
        live_client._state = PlayerState.PLAYING  # noqa: SLF001

        await live_client.pause()
        await live_client._process_stat(_stmp_payload(200_000, 10_000), received=0.0)  # noqa: SLF001
        assert live_client.state == PlayerState.PAUSED
        await live_client.play()
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        assert live_client.state == PlayerState.PLAYING

    @pytest.mark.asyncio
    async def test_a_stream_started_at_buffer_ready_is_playing(
        self, live_client: SlimClient
    ) -> None:
        """The resume of a stream that waited for its buffer marks it playing."""
        await live_client.play_url(url=_TRACK_URL, autostart=False)
        live_client._process_stat_stml(b"")  # noqa: SLF001
        assert live_client.state == PlayerState.BUFFER_READY

        await live_client.unpause_at(0)
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001

        assert live_client.state == PlayerState.PLAYING


def _stmt_payload(jiffies: int, elapsed_ms: int = 0) -> bytes:
    """Build an STMt STAT payload with the given player clock and song position."""
    return b"STMt" + struct.pack(
        "!BBBLLLLHLLLLHLL",
        *(0, 0, 0, 0, 0, 0, 0, 0, jiffies, 0, 0, elapsed_ms // 1000, 0, elapsed_ms, 0),
    )


def _stmp_payload(jiffies: int, elapsed_ms: int) -> bytes:
    """Build an STMp STAT payload with the given player clock and song position."""
    return b"STMp" + struct.pack(
        "!BBBLLLLHLLLLHLL",
        *(0, 0, 0, 0, 0, 0, 0, 0, jiffies, 0, 0, elapsed_ms // 1000, 0, elapsed_ms, 0),
    )


class TestJiffies:
    """jiffies follows the player clock from its last STMt, whatever the stream does."""

    @pytest.fixture
    def now(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Freeze the client's clock; bump now[0] to let time pass."""
        clock = [1_000_000.0]
        monkeypatch.setattr("aioslimproto.client.time", Mock(time=lambda: clock[0]))
        return clock

    @pytest.mark.asyncio
    async def test_flush_keeps_player_clock(
        self, live_client: SlimClient, now: list[float]
    ) -> None:
        """A flush (new track or seek) doesn't move the player clock."""
        await live_client._process_stat(_stmt_payload(40_000), received=0.0)  # noqa: SLF001
        now[0] += 0.5
        await live_client.play_url(url=_TRACK_URL, autostart=False, send_flush=True)
        now[0] += 0.25

        assert live_client.jiffies == 40_750

    @pytest.mark.asyncio
    async def test_stream_start_keeps_player_clock(
        self, live_client: SlimClient, now: list[float]
    ) -> None:
        """A stream start (STMs) doesn't move the player clock."""
        await live_client._process_stat(_stmt_payload(40_000), received=0.0)  # noqa: SLF001
        now[0] += 0.8
        await live_client._process_stat(b"STMs" + bytes(47), received=0.0)  # noqa: SLF001
        now[0] += 0.1

        assert live_client.jiffies == 40_900


class TestByeCommand:
    """A BYE! packet must disconnect the client just like reaching EOF does."""

    @pytest.mark.asyncio
    async def test_bye_disconnects_without_eof(self, writer: Mock) -> None:
        """BYE! must end the read loop and disconnect, with no socket EOF."""
        reader = asyncio.StreamReader()
        disconnected = asyncio.Event()

        def callback(_client: SlimClient, event: EventType, *_args: object) -> None:
            if event is EventType.PLAYER_DISCONNECTED:
                disconnected.set()

        slim_client = SlimClient(reader, writer, callback)
        slim_client._connected = True  # noqa: SLF001

        # real players always send a 1-byte reason (0x00 normal, 0x01 upgrade)
        reader.feed_data(_frame(b"BYE!", b"\x00"))

        await asyncio.wait_for(disconnected.wait(), 1)
        await asyncio.wait_for(slim_client._reader_task, 1)  # noqa: SLF001

        assert slim_client.connected is False


class TestSendToLostConnection:
    """A send that finds the socket gone must still report the disconnect."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lost", ["eof", "closing", "fails_on_drain"])
    async def test_send_before_reader_notices_signals_disconnect(
        self, writer: Mock, lost: str
    ) -> None:
        """The player is reported disconnected when a send sees the loss first."""
        reader = asyncio.StreamReader()
        callback = Mock()
        slim_client = SlimClient(reader, writer, callback)
        slim_client._connected = True  # noqa: SLF001
        # the transport hands the loss to the reader before the reader task runs;
        # a failed write reaches it as a broken pipe
        if lost == "eof":
            reader.feed_eof()
        else:
            reader.set_exception(BrokenPipeError())
        if lost == "closing":
            writer.is_closing.return_value = True
        if lost == "fails_on_drain":
            writer.drain.side_effect = BrokenPipeError()

        await slim_client.send_frame(b"vers", b"7.9")
        await asyncio.wait([slim_client._reader_task], timeout=1)  # noqa: SLF001

        assert slim_client.connected is False
        assert call(slim_client, EventType.PLAYER_DISCONNECTED) in callback.mock_calls
        writer.close.assert_called()


class TestRestoreOnHelo:
    """A connecting player gets its volume and power state, even when unchanged."""

    _HELO = bytes([12, 0]) + b"\xaa\xbb\xcc\xdd\xee\xff" + bytes(28)

    async def _connect(self, client: SlimClient, command: bytes) -> list[bytes]:
        """Process a squeezelite HELO; return the payloads sent for command."""
        client.send_frame = AsyncMock()
        await client._process_helo(self._HELO + b"Model=squeezelite,flc")  # noqa: SLF001
        client.disconnect()
        return [
            frame.args[1]
            for frame in client.send_frame.await_args_list
            if frame.args[0] == command
        ]

    @pytest.mark.asyncio
    async def test_helo_sends_default_volume(self, client: SlimClient) -> None:
        """Skipping the unchanged default level must not leave the player silent."""
        audg_payloads = await self._connect(client, b"audg")

        assert len(audg_payloads) == 1
        new_gain = client.volume_control.new_gain()
        assert struct.unpack("!LLBBLL", audg_payloads[0])[4:] == (new_gain, new_gain)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("powered", [True, False])
    async def test_helo_sends_power_state(
        self, client: SlimClient, *, powered: bool
    ) -> None:
        """The player gets the cached power state, which power() would skip."""
        client._powered = powered  # noqa: SLF001
        aude_payloads = await self._connect(client, b"aude")

        assert aude_payloads == [struct.pack("2B", int(powered), 1)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("initial", "expected"), [(11, 11), (None, 50)])
    async def test_helo_sends_initial_volume(
        self, writer: Mock, initial: int | None, expected: int
    ) -> None:
        """The first audg carries the consumer's initial volume, else the default."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        initial_volume = AsyncMock(return_value=initial)
        slim_client = SlimClient(reader, writer, Mock(), initial_volume=initial_volume)

        audg_payloads = await self._connect(slim_client, b"audg")

        initial_volume.assert_awaited_once_with("aa:bb:cc:dd:ee:ff")
        assert slim_client.volume_level == expected
        assert len(audg_payloads) == 1
        new_gain = slim_client.volume_control.new_gain()
        assert struct.unpack("!LLBBLL", audg_payloads[0])[4:] == (new_gain, new_gain)


def _gains(audg_payload: bytes) -> tuple[int, int]:
    """Return the (old, new) left-channel gain of an audg payload."""
    old_gain, _, _, _, new_gain, _ = struct.unpack("!LLBBLL", audg_payload)
    return old_gain, new_gain


class TestMute:
    """Mute sends zero gain, since aude would switch the player's output off."""

    @staticmethod
    def _frames(client: SlimClient, command: bytes) -> list[bytes]:
        """Return the payloads sent for command."""
        send_frame = cast("AsyncMock", client.send_frame)
        return [
            frame.args[1]
            for frame in send_frame.await_args_list
            if frame.args[0] == command
        ]

    @pytest.mark.asyncio
    async def test_mute_sends_zero_gain(self, client: SlimClient) -> None:
        """Muting sends a zero-gain audg and no aude."""
        client.send_frame = AsyncMock()

        await client.mute(True)  # noqa: FBT003

        assert client.muted is True
        assert [_gains(p) for p in self._frames(client, b"audg")] == [(0, 0)]
        assert self._frames(client, b"aude") == []

    @pytest.mark.asyncio
    async def test_unmute_restores_volume(self, client: SlimClient) -> None:
        """Unmuting sends the stored volume's gain again."""
        client.send_frame = AsyncMock()
        await client.mute(True)  # noqa: FBT003

        await client.mute(False)  # noqa: FBT003

        expected = (
            client.volume_control.old_gain(),
            int(client.volume_control.new_gain()),
        )
        assert client.muted is False
        assert _gains(self._frames(client, b"audg")[-1]) == expected

    @pytest.mark.asyncio
    async def test_volume_set_while_muted(self, client: SlimClient) -> None:
        """A volume change while muted stays silent and is applied on unmute."""
        client.send_frame = AsyncMock()
        await client.mute(True)  # noqa: FBT003

        await client.volume_set(20)

        assert client.muted is True
        assert client.volume_level == 20
        assert _gains(self._frames(client, b"audg")[-1]) == (0, 0)

        await client.mute(False)  # noqa: FBT003

        expected = (
            client.volume_control.old_gain(),
            int(client.volume_control.new_gain()),
        )
        assert expected != (0, 0)
        assert _gains(self._frames(client, b"audg")[-1]) == expected


class TestSetdPlayerName:
    """SETD 0 carries the player name, with or without a NUL terminator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"\x00Kitchen\x00", "Kitchen"),  # squeezelite, SqueezeESP32
            (b"\x00Kitchen", "Kitchen"),  # SqueezePlay (Radio/Touch/Controller)
            (b"\x00Kitchen\x00\xff\xff", "Kitchen"),
            (b"\x00K\xfcche\x00", "K�che"),
        ],
    )
    async def test_name_is_parsed(
        self, client: SlimClient, payload: bytes, expected: str
    ) -> None:
        """The name ends at the first NUL, or at the end of the payload."""
        client._process_setd(payload)  # noqa: SLF001

        assert client.name == expected
        client.callback.assert_called_with(
            client, EventType.PLAYER_NAME_RECEIVED, expected
        )

    @pytest.mark.parametrize(
        ("capabilities", "expected"),
        [
            ({"Model": "baby", "ModelName": "Squeezebox Radio"}, "Squeezebox Radio"),
            # no model name to go by: type and id, as before
            ({"Model": "squeezelite"}, "squeezelite: aa:bb:cc:dd:ee:ff"),
        ],
    )
    def test_name_without_setd_is_the_model(
        self, client: SlimClient, capabilities: dict, expected: str
    ) -> None:
        """Without a reported name the player is called by its model, like LMS."""
        client._capabilities = capabilities  # noqa: SLF001
        client._device_type = "squeezelite"  # noqa: SLF001
        client._player_id = "aa:bb:cc:dd:ee:ff"  # noqa: SLF001

        assert client.name == expected


class TestSocketReader:
    """The socket reader handles every complete packet it has buffered."""

    @staticmethod
    async def _read(writer: Mock, data: bytes) -> list[bytes]:
        """Feed data followed by EOF and return the DSCO payloads that were handled."""
        reader = asyncio.StreamReader()
        slim_client = SlimClient(reader, writer, Mock())
        handled: list[bytes] = []
        slim_client._process_dsco = handled.append  # noqa: SLF001
        reader.feed_data(data)
        reader.feed_eof()
        await asyncio.wait_for(slim_client._reader_task, 1)  # noqa: SLF001
        await asyncio.sleep(0)
        return handled

    @pytest.mark.asyncio
    async def test_packets_from_one_read_are_all_handled_in_order(
        self, writer: Mock
    ) -> None:
        """Several packets arriving together are all handled, in arrival order."""
        payloads = [bytes([i]) for i in range(30)]

        handled = await self._read(
            writer, b"".join(_frame(b"DSCO", p) for p in payloads)
        )

        assert handled == payloads

    @pytest.mark.asyncio
    async def test_packet_without_payload_is_handled(self, writer: Mock) -> None:
        """A packet that consists of only its header is handled too."""
        handled = await self._read(writer, _frame(b"DSCO", b""))

        assert handled == [b""]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bye", [_frame(b"BYE!", b""), None])
    async def test_helo_followed_by_disconnect_does_not_connect(
        self, writer: Mock, bye: bytes | None
    ) -> None:
        """A player that is gone right after hello is not reported connected."""
        reader = asyncio.StreamReader()
        callback = Mock()
        slim_client = SlimClient(reader, writer, callback)
        reader.feed_data(_frame(b"HELO", bytes([12, 0]) + bytes(6)))
        if bye:
            reader.feed_data(bye)
        else:
            reader.feed_eof()

        await asyncio.wait_for(slim_client._reader_task, 1)  # noqa: SLF001
        for _ in range(10):
            await asyncio.sleep(0)

        assert not slim_client.connected
        assert call(slim_client, EventType.PLAYER_CONNECTED) not in callback.mock_calls


class TestPositionAcrossPauseAndResume:
    """The position neither steps back at a pause nor jumps forward at a resume."""

    @staticmethod
    def _clock(monkeypatch: pytest.MonkeyPatch, now: float) -> list[float]:
        clock = [now]
        # only the client module sees the fake clock, not the rest of the process
        monkeypatch.setattr(
            "aioslimproto.client.time", SimpleNamespace(time=lambda: clock[0])
        )
        return clock

    @staticmethod
    def _playing(client: SlimClient, *, elapsed_ms: int, reported_at: float) -> None:
        client._state = PlayerState.PLAYING  # noqa: SLF001
        client._elapsed_milliseconds = elapsed_ms  # noqa: SLF001
        client._last_timestamp = reported_at  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_pausing_keeps_the_position_reached_and_it_stays_put(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seven tenths of a second after the last report, the pause holds it."""
        clock = self._clock(monkeypatch, 100.7)
        self._playing(live_client, elapsed_ms=10_000, reported_at=100.0)
        await live_client.pause()
        assert live_client.state == PlayerState.PAUSED
        assert live_client.elapsed_milliseconds == 10_700
        clock[0] = 105.0
        assert live_client.elapsed_milliseconds == 10_700

    @pytest.mark.asyncio
    async def test_resuming_counts_on_from_the_resume_not_from_before_the_pause(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a three second pause the position has not moved when play starts."""
        clock = self._clock(monkeypatch, 103.0)
        live_client._state = PlayerState.PAUSED  # noqa: SLF001
        live_client._elapsed_milliseconds = 10_700  # noqa: SLF001
        live_client._last_timestamp = 100.0  # noqa: SLF001
        await live_client.play()
        assert live_client.state == PlayerState.PLAYING
        assert live_client.elapsed_milliseconds == 10_700
        clock[0] = 104.0
        assert live_client.elapsed_milliseconds == 11_700

    @pytest.mark.asyncio
    async def test_a_pause_made_on_the_player_holds_the_position_too(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pause the player started itself holds the position it reports."""
        clock = self._clock(monkeypatch, 100.4)
        self._playing(live_client, elapsed_ms=10_000, reported_at=100.0)
        await live_client._process_stat(_stmp_payload(200_000, 10_350), received=0.0)  # noqa: SLF001
        clock[0] = 108.0
        assert live_client.state == PlayerState.PAUSED
        assert live_client.elapsed_milliseconds == 10_350

    @pytest.mark.asyncio
    async def test_the_pause_confirmation_sets_the_play_point(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The paused position the player reports is the play point's song position."""
        self._clock(monkeypatch, 100.4)
        self._playing(live_client, elapsed_ms=10_000, reported_at=100.0)
        live_client._jiffies_epoch = 0.0  # noqa: SLF001
        live_client.packet_latency = 0.0
        await live_client.pause()
        await live_client._process_stat(_stmp_payload(200_000, 10_350), received=0.0)  # noqa: SLF001
        assert live_client.play_point == pytest.approx((200.0, 189.65, 10.35))

    @pytest.mark.asyncio
    async def test_a_heartbeat_while_paused_keeps_the_paused_position(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A heartbeat during the pause moves the player clock, not the position."""
        self._clock(monkeypatch, 100.4)
        self._playing(live_client, elapsed_ms=10_000, reported_at=100.0)
        live_client._jiffies_epoch = 0.0  # noqa: SLF001
        live_client.packet_latency = 0.0
        await live_client.pause()
        await live_client._process_stat(_stmp_payload(200_000, 10_350), received=0.0)  # noqa: SLF001
        # squeezelite extrapolates its frozen position from a stale status copy
        await live_client._process_stat(_stmt_payload(203_000, 10_440), received=0.0)  # noqa: SLF001
        assert live_client.jiffies == 203_000
        assert live_client.elapsed_milliseconds == 10_350
        assert live_client.play_point is not None
        assert live_client.play_point[2] == pytest.approx(10.35)

    @pytest.mark.asyncio
    async def test_a_heartbeat_after_the_resume_sets_the_position_again(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the player confirms the resume, its heartbeats report the position."""
        self._clock(monkeypatch, 100.4)
        self._playing(live_client, elapsed_ms=10_000, reported_at=100.0)
        live_client._jiffies_epoch = 0.0  # noqa: SLF001
        live_client.packet_latency = 0.0
        await live_client.pause()
        await live_client._process_stat(_stmp_payload(200_000, 10_350), received=0.0)  # noqa: SLF001
        await live_client._process_stat(b"STMr" + bytes(47), received=0.0)  # noqa: SLF001
        await live_client._process_stat(_stmt_payload(206_000, 11_350), received=0.0)  # noqa: SLF001
        assert live_client.elapsed_milliseconds == 11_350
        assert live_client.play_point is not None
        assert live_client.play_point[2] == pytest.approx(11.35)

    @pytest.mark.asyncio
    async def test_a_pause_during_a_stream_switch_keeps_the_position_at_zero(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pause confirmation still reporting the flushed stream is not shown."""
        self._clock(monkeypatch, 100.4)
        await live_client.play_url(url=_TRACK_URL, autostart=False, send_flush=True)
        await live_client._process_stat(_stmp_payload(200_000, 10_350), received=0.0)  # noqa: SLF001
        assert live_client.elapsed_milliseconds == 0

    @pytest.mark.asyncio
    async def test_a_resume_made_on_the_player_restarts_the_clock(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the player resumed by itself, the position counts on from there."""
        clock = self._clock(monkeypatch, 109.0)
        live_client._state = PlayerState.PAUSED  # noqa: SLF001
        live_client._elapsed_milliseconds = 10_400  # noqa: SLF001
        live_client._last_timestamp = 100.0  # noqa: SLF001
        await live_client._process_stat_stmr(b"")  # noqa: SLF001
        assert live_client.state == PlayerState.PLAYING
        assert live_client.elapsed_milliseconds == 10_400
        clock[0] = 110.0
        assert live_client.elapsed_milliseconds == 11_400

    @pytest.mark.asyncio
    async def test_the_confirmation_of_our_own_resume_does_not_move_the_clock(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """play() already restarted the clock, so the confirmation leaves it alone."""
        clock = self._clock(monkeypatch, 103.0)
        live_client._state = PlayerState.PAUSED  # noqa: SLF001
        live_client._elapsed_milliseconds = 10_700  # noqa: SLF001
        live_client._last_timestamp = 100.0  # noqa: SLF001
        await live_client.play()
        clock[0] = 103.5
        await live_client._process_stat_stmr(b"")  # noqa: SLF001
        assert live_client.elapsed_milliseconds == 11_200

    @pytest.mark.asyncio
    async def test_pausing_while_buffering_leaves_the_reported_position_alone(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is extrapolated while buffering, so there is nothing to hold."""
        self._clock(monkeypatch, 100.7)
        live_client._state = PlayerState.BUFFERING  # noqa: SLF001
        live_client._elapsed_milliseconds = 4_000  # noqa: SLF001
        await live_client.pause()
        assert live_client.state == PlayerState.PAUSED
        assert live_client.elapsed_milliseconds == 4_000


class TestSqueezePlayVolumeCurve:
    """A player gets the volume curve its own firmware applies."""

    @staticmethod
    def _helo(device_id: int) -> bytes:
        """Build a HELO for a player that reports `device_id`, as a real one does."""
        return bytes([device_id, 0]) + b"\xaa\xbb\xcc\xdd\xee\xff" + bytes(28)

    @staticmethod
    async def _curve(client: SlimClient, device_id: int) -> tuple[int, int, float]:
        """Process a HELO for `device_id`; return the volume parameters it produced."""
        client.send_frame = AsyncMock()
        await client._process_helo(TestSqueezePlayVolumeCurve._helo(device_id))  # noqa: SLF001
        control = client.volume_control
        client.disconnect()
        return control.total_volume_range, control.step_point, control.step_fraction

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("device_id", "what"),
        [
            (9, "SqueezePlay: Radio, Touch, Controller"),
            (10, "Boom"),
            (12, "SqueezePlay software"),
        ],
    )
    async def test_players_whose_firmware_uses_the_boom_table(
        self, client: SlimClient, device_id: int, what: str
    ) -> None:
        """SqueezePlay and Boom firmware map the server's gain via the Boom curve."""
        curve = await self._curve(client, device_id)

        assert curve == (-74, 25, 0.5), what

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("device_id", "what"),
        [
            (2, "Squeezebox"),
            (4, "Squeezebox2/Classic"),
            (5, "Transporter"),
            (7, "Receiver"),
        ],
    )
    async def test_players_with_the_squeezebox2_curve(
        self, client: SlimClient, device_id: int, what: str
    ) -> None:
        """Pre-SqueezePlay hardware keeps the Squeezebox2 curve."""
        curve = await self._curve(client, device_id)

        assert curve == (-50, -1, 1), what


# far from any real monotonic reading, so the default clock can't pass for it
_SERVER_CLOCK = 5_000_000.0
# every STAT status field distinct; bytes received spans both 32-bit words
_RAW_STATUS = PlayerStatus(
    received=_SERVER_CLOCK,
    jiffies=3_000_000_123,
    elapsed_seconds=42,
    elapsed_milliseconds=42_250,
    bytes_received=5 * 2**32 + 123_456,
    stream_buffer_size=2_097_152,
    stream_buffer_fullness=1_048_000,
    output_buffer_size=3_528_000,
    output_buffer_fullness=1_764_321,
    server_timestamp=0xDEADBEEF,
)


def _full_stmt_payload(status: PlayerStatus) -> bytes:
    """Build an STMt STAT payload that reports every field of status."""
    return b"STMt" + struct.pack(
        "!BBBLLLLHLLLLHLL",
        1,
        2,
        3,
        status.stream_buffer_size,
        status.stream_buffer_fullness,
        status.bytes_received >> 32,
        status.bytes_received & 0xFFFFFFFF,
        77,
        status.jiffies,
        status.output_buffer_size,
        status.output_buffer_fullness,
        status.elapsed_seconds,
        0x1234,
        status.elapsed_milliseconds,
        status.server_timestamp,
    )


class TestStatusRequest:
    """A status request is a strm t whose timestamp the player echoes back."""

    @pytest.mark.asyncio
    async def test_request_status_sends_the_timestamp(
        self, live_client: SlimClient
    ) -> None:
        """The timestamp goes out in the replay gain field of the strm t."""
        await live_client.request_status(0xDEADBEEF)

        live_client._send_strm.assert_awaited_once_with(  # noqa: SLF001
            b"t", autostart=b"0", flags=0, replay_gain=0xDEADBEEF
        )

    @pytest.mark.asyncio
    async def test_heartbeat_requests_the_status_with_its_id(
        self, live_client: SlimClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The periodic heartbeat is a status request stamped with its own id."""
        monkeypatch.setattr("aioslimproto.client.HEARTBEAT_INTERVAL", 0)
        live_client._connected = True  # noqa: SLF001
        live_client._last_heartbeat = 6  # noqa: SLF001
        live_client.display_control.disabled = True
        # one heartbeat, then the player is gone
        live_client._send_strm.side_effect = lambda *_args, **_kwargs: setattr(  # noqa: SLF001
            live_client, "_connected", False
        )

        await asyncio.wait_for(live_client._send_heartbeat(), 1)  # noqa: SLF001

        live_client._send_strm.assert_awaited_once_with(  # noqa: SLF001
            b"t", autostart=b"0", flags=0, replay_gain=7
        )


class TestHeartbeatStatus:
    """The heartbeat event carries the status exactly as the player reported it."""

    @pytest.mark.asyncio
    async def test_status_read_from_the_socket(self, writer: Mock) -> None:
        """Every STAT field arrives unchanged, stamped on the server clock."""
        reader = asyncio.StreamReader()
        handled = asyncio.Event()
        callback = Mock(side_effect=lambda *_args: handled.set())
        slim_client = SlimClient(reader, writer, callback, clock=lambda: _SERVER_CLOCK)

        reader.feed_data(_frame(b"STAT", _full_stmt_payload(_RAW_STATUS)))
        try:
            await asyncio.wait_for(handled.wait(), 1)
        finally:
            slim_client._reader_task.cancel()  # noqa: SLF001

        callback.assert_any_call(slim_client, EventType.PLAYER_HEARTBEAT, _RAW_STATUS)

    @pytest.mark.asyncio
    async def test_status_is_raw_while_awaiting_the_stream_start(
        self, live_client: SlimClient
    ) -> None:
        """A flushed stream's position is reported, but not taken as the position."""
        await live_client.play_url(url=_TRACK_URL, autostart=False, send_flush=True)

        await live_client._process_stat(  # noqa: SLF001
            _full_stmt_payload(_RAW_STATUS), received=_SERVER_CLOCK
        )

        live_client.callback.assert_any_call(
            live_client, EventType.PLAYER_HEARTBEAT, _RAW_STATUS
        )
        assert live_client._elapsed_milliseconds == 0  # noqa: SLF001
        assert live_client._elapsed_seconds == 0  # noqa: SLF001
        assert live_client._jiffies == _RAW_STATUS.jiffies  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_status_is_raw_while_paused(self, live_client: SlimClient) -> None:
        """A paused player's position is reported, but the paused one is kept."""
        live_client._state = PlayerState.PAUSED  # noqa: SLF001
        live_client._elapsed_milliseconds = 10_350  # noqa: SLF001
        live_client._elapsed_seconds = 10  # noqa: SLF001

        await live_client._process_stat(  # noqa: SLF001
            _full_stmt_payload(_RAW_STATUS), received=_SERVER_CLOCK
        )

        live_client.callback.assert_any_call(
            live_client, EventType.PLAYER_HEARTBEAT, _RAW_STATUS
        )
        assert live_client._elapsed_milliseconds == 10_350  # noqa: SLF001
        assert live_client._elapsed_seconds == 10  # noqa: SLF001
        assert live_client._jiffies == _RAW_STATUS.jiffies  # noqa: SLF001


class TestServerClock:
    """Every server time of a client is on the clock it was given."""

    @staticmethod
    def _client(writer: Mock, now: list[float]) -> SlimClient:
        """Create a client on a clock that reads now[0]."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        slim_client = SlimClient(reader, writer, Mock(), clock=lambda: now[0])
        slim_client.packet_latency = 0.0
        return slim_client

    @pytest.mark.asyncio
    async def test_server_time_follows_the_injected_clock(self, writer: Mock) -> None:
        """server_time() reads the injected clock."""
        now = [_SERVER_CLOCK]
        slim_client = self._client(writer, now)

        assert slim_client.server_time() == _SERVER_CLOCK
        now[0] += 1.5
        assert slim_client.server_time() == _SERVER_CLOCK + 1.5

    @pytest.mark.asyncio
    async def test_server_time_defaults_to_the_monotonic_clock(
        self, writer: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a clock, server time is time.monotonic()."""
        monkeypatch.setattr(
            "aioslimproto.client.time",
            SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 123.25),
        )
        reader = asyncio.StreamReader()
        reader.feed_eof()

        assert SlimClient(reader, writer, Mock()).server_time() == 123.25

    @pytest.mark.asyncio
    async def test_socket_read_maps_the_player_clock_onto_the_injected_clock(
        self, writer: Mock
    ) -> None:
        """A heartbeat read from the socket sets the epoch on the injected clock."""
        reader = asyncio.StreamReader()
        handled = asyncio.Event()
        slim_client = SlimClient(
            reader,
            writer,
            Mock(side_effect=lambda *_args: handled.set()),
            clock=lambda: _SERVER_CLOCK,
        )

        reader.feed_data(_frame(b"STAT", _stmt_payload(40_000)))
        try:
            await asyncio.wait_for(handled.wait(), 1)
        finally:
            slim_client._reader_task.cancel()  # noqa: SLF001

        assert slim_client.jiffies_epoch == _SERVER_CLOCK - 40

    @pytest.mark.asyncio
    async def test_song_position_extrapolates_on_the_injected_clock(
        self, writer: Mock
    ) -> None:
        """A playing player's song position counts on in injected-clock time."""
        now = [_SERVER_CLOCK]
        slim_client = self._client(writer, now)
        slim_client._state = PlayerState.PLAYING  # noqa: SLF001
        slim_client._jiffies_epoch = _SERVER_CLOCK - 10  # noqa: SLF001
        slim_client._jiffies = 10_000  # noqa: SLF001
        slim_client._elapsed_milliseconds = 42_000  # noqa: SLF001
        now[0] += 0.25

        assert slim_client.song_elapsed_seconds == pytest.approx(42.25)


class TestHeadlessSqueezeEsp32:
    """SqueezeESP32 reports device id 101 when it has no display."""

    @pytest.mark.asyncio
    async def test_device_id_101_is_a_squeezeesp32(self, client: SlimClient) -> None:
        """A headless SqueezeESP32 gets its device type and sync deadband."""
        client.send_frame = AsyncMock()

        await client._process_helo(  # noqa: SLF001
            bytes([101, 0]) + b"\xaa\xbb\xcc\xdd\xee\xff" + bytes(28)
        )
        client.disconnect()

        assert client.device_type == "squeezeesp32"
        assert client.min_sync_adjust == 10


class TestGainAfterPowerOn:
    """A player that powers on gets the level from the model, not the one it kept."""

    @pytest.mark.asyncio
    async def test_power_on_sends_the_gain(self, client: SlimClient) -> None:
        """A power-on sends aude and then the gain."""
        client.send_frame = AsyncMock()
        client.volume_control.volume = 30

        await client.power(powered=True)

        old_gain = client.volume_control.old_gain()
        new_gain = client.volume_control.new_gain()
        assert client.send_frame.await_args_list[0] == call(
            b"aude", struct.pack("2B", 1, 1)
        )
        assert (
            call(
                b"audg",
                struct.pack("!LLBBLL", old_gain, old_gain, 1, 255, new_gain, new_gain),
            )
            in client.send_frame.await_args_list
        )

    @pytest.mark.asyncio
    async def test_power_on_while_muted_sends_zero_gain(
        self, client: SlimClient
    ) -> None:
        """A muted player that powers on is given zero gain, not its stored level."""
        client.send_frame = AsyncMock()
        client.volume_control.volume = 12
        await client.mute(True)  # noqa: FBT003
        client._powered = False  # noqa: SLF001

        await client.power(powered=True)

        gains = [
            _gains(c.args[1])
            for c in client.send_frame.await_args_list
            if c.args[0] == b"audg"
        ]
        assert gains, "a power-on while muted sends no gain frame at all"
        assert gains[-1] == (0, 0)

    @pytest.mark.asyncio
    async def test_power_off_sends_no_gain(self, client: SlimClient) -> None:
        """Powering off stops the player and does not touch the gain."""
        client.send_frame = AsyncMock()
        client._powered = True  # noqa: SLF001

        await client.power(powered=False)

        assert b"aude" in [c.args[0] for c in client.send_frame.await_args_list]
        assert b"audg" not in [c.args[0] for c in client.send_frame.await_args_list]
