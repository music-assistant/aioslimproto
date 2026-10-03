"""Tests for SlimClient's stream start and next-media promotion."""

import asyncio
import struct
from unittest.mock import AsyncMock, Mock, call

import pytest

from aioslimproto.client import SlimClient
from aioslimproto.models import EventType, MediaDetails, PlayerState

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
            stream_threshold=media.stream_threshold,
            output_threshold=media.output_threshold,
            autostart=autostart,
        )


def _stmt_payload(jiffies: int) -> bytes:
    """Build an STMt STAT payload with the given player clock."""
    return b"STMt" + struct.pack(
        "!BBBLLLLHLLLLHLL", 0, 0, 0, 0, 0, 0, 0, 0, jiffies, 0, 0, 0, 0, 0, 0
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
        await live_client._process_stat(_stmt_payload(40_000))  # noqa: SLF001
        now[0] += 0.5
        await live_client.play_url(url=_TRACK_URL, autostart=False, send_flush=True)
        now[0] += 0.25

        assert live_client.jiffies == 40_750

    @pytest.mark.asyncio
    async def test_stream_start_keeps_player_clock(
        self, live_client: SlimClient, now: list[float]
    ) -> None:
        """A stream start (STMs) doesn't move the player clock."""
        await live_client._process_stat(_stmt_payload(40_000))  # noqa: SLF001
        now[0] += 0.8
        await live_client._process_stat(b"STMs" + bytes(47))  # noqa: SLF001
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
