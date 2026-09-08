"""Tests for SlimClient's next-media promotion on STMd/STMu."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aioslimproto.client import SlimClient
from aioslimproto.models import MediaDetails, PlayerState


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
