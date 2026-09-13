"""Tests for SlimClient's stream start and next-media promotion."""

import asyncio
import struct
from unittest.mock import AsyncMock, Mock, call

import pytest

from aioslimproto.client import SlimClient
from aioslimproto.models import MediaDetails, PlayerState

# a cont frame as LMS sends it: metaint (no ICY metadata), loop and guid count
_CONT_PAYLOAD = struct.pack("!IBH", 0, 0, 0)


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
        assert live_client.state == PlayerState.BUFFERING

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
