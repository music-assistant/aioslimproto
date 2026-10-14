"""Tests for the SlimProto CLI."""

import asyncio
from itertools import chain
import json
import logging
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import pytest

from aioslimproto.cli import (
    CometDClient,
    SlimProtoCLI,
    _drop_empty_quality_fields,
    playlist_item_from_media_details,
)
from aioslimproto.client import SlimClient
from aioslimproto.models import MediaDetails, PlayerState
from aioslimproto.server import SlimServer


@pytest.fixture
def writer() -> Mock:
    """Create a mocked CLI connection writer."""
    result = Mock()
    result.drain = AsyncMock()
    result.is_closing.return_value = False
    result.wait_closed = AsyncMock()
    return result


async def send_cli_command(
    cli: SlimProtoCLI,
    writer: Mock,
    command: str,
) -> bytes:
    """Send one complete command through the in-memory CLI connection."""
    reader = asyncio.StreamReader()
    reader.feed_data(f"{command}\n".encode())
    reader.feed_eof()

    await cli._handle_cli_client(reader, writer)  # noqa: SLF001 # Accessing a protected member for testing purposes

    assert writer.write.call_count == 1
    return writer.write.call_args.args[0]


def encode_response(*tags: str | dict[str, str | int]) -> bytes:
    """Encode CLI response tags as the legacy CLI does."""
    flat_tags = chain.from_iterable(
        (tag,)
        if isinstance(tag, str)
        else (f"{key}:{value}" for key, value in tag.items())
        for tag in tags
    )
    return f"{' '.join(quote(tag) for tag in flat_tags)}\n".encode("iso-8859-1")


@pytest.fixture
def dummy_player() -> SlimClient:
    """Create a dummy player for testing."""
    return cast(
        "SlimClient",
        SimpleNamespace(
            player_id="a5:41:d2:cd:cd:05",
            name="Kitchen",
            device_model="Squeezebox Touch",
            connected=True,
            state=PlayerState.STOPPED,
            powered=True,
            device_type="touch",
            extra_data={"uuid": "player-uuid-1", "seq_no": 1},
            device_address="127.0.0.1",
            current_media=None,
            next_media=None,
            presets=[],
            elapsed_seconds=0,
            volume_level=50,
            volume_set=AsyncMock(),
            muted=False,
            mute=AsyncMock(),
        ),
    )


@pytest.fixture
def dummy_server(dummy_player: SlimClient) -> SlimServer:
    """Create a dummy server for testing."""
    return cast(
        "SlimServer",
        SimpleNamespace(
            name="testserver",
            logger=logging.getLogger(),
            players=[dummy_player],
            get_player=lambda player_id: (
                dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
            ),
        ),
    )


class TestPlayersCommand:
    """Tests for the players command.

    Reference: https://lyrion.org/reference/cli/players/#players
    """

    @pytest.fixture
    def several_dummy_players(self, dummy_player: SlimClient) -> list[SlimClient]:
        """Create several dummy players for testing."""
        return [
            dummy_player,
            cast(
                "SlimClient",
                SimpleNamespace(
                    player_id="player-2",
                    name="Living Room",
                    device_model="Squeezebox Radio",
                    connected=True,
                    state=PlayerState.PLAYING,
                    powered=True,
                    device_type="radio",
                    extra_data={"uuid": "player-uuid-2", "seq_no": 2},
                    device_address="127.0.0.2",
                ),
            ),
            cast(
                "SlimClient",
                SimpleNamespace(
                    player_id="player-3",
                    name="Bedroom",
                    device_model="Squeezebox Controller",
                    connected=True,
                    state=PlayerState.PLAYING,
                    powered=True,
                    device_type="controller",
                    extra_data={"uuid": "player-uuid-3", "seq_no": 3},
                    device_address="127.0.0.3",
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_returns_all_registered_players(
        self, writer: Mock, several_dummy_players: list[SlimClient]
    ) -> None:
        """Should return all players when no limit is specified."""
        server = cast(
            "SlimServer",
            SimpleNamespace(logger=logging.getLogger(), players=several_dummy_players),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(cli, writer, "players 0")
        expected_response = encode_response(
            "players",
            "0",
            "count:3",
            {
                "playerindex": 0,
                "playerid": "a5:41:d2:cd:cd:05",
                "name": "Kitchen",
                "modelname": "Squeezebox Touch",
                "connected": 1,
                "isplaying": 0,
                "power": 1,
                "model": "touch",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-1",
                "seq_no": 1,
                "ip": "127.0.0.1",
            },
            {
                "playerindex": 1,
                "playerid": "player-2",
                "name": "Living Room",
                "modelname": "Squeezebox Radio",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "radio",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-2",
                "seq_no": 2,
                "ip": "127.0.0.2",
            },
            {
                "playerindex": 2,
                "playerid": "player-3",
                "name": "Bedroom",
                "modelname": "Squeezebox Controller",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "controller",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-3",
                "seq_no": 3,
                "ip": "127.0.0.3",
            },
        )

        assert response == expected_response

    @pytest.mark.asyncio
    async def test_returns_second_player(
        self, writer: Mock, several_dummy_players: list[SlimClient]
    ) -> None:
        """Should return the 2nd player when offset and limit are set to 1."""
        server = cast(
            "SlimServer",
            SimpleNamespace(logger=logging.getLogger(), players=several_dummy_players),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(cli, writer, "players 1 1")
        expected_response = encode_response(
            "players",
            "1",
            "1",
            "count:1",
            {
                "playerindex": 1,
                "playerid": "player-2",
                "name": "Living Room",
                "modelname": "Squeezebox Radio",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "radio",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-2",
                "seq_no": 2,
                "ip": "127.0.0.2",
            },
        )

        assert response == expected_response


class TestStatusCommand:
    """Tests for the status command.

    Reference: https://lyrion.org/reference/cli/compoundqueries/#status
    """

    @pytest.mark.asyncio
    async def test_echoes_without_player(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should echo the command when no player is specified."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "status 0 2 tags:")
        expected_response = encode_response("status", "0", "2", "tags:")

        assert response == expected_response

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Many values are hard-coded")
    async def test_simple_example(self, writer: Mock, dummy_server: SlimServer) -> None:
        """The "simple example" on the reference page should work."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 status 0 2 tags:"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05",
            "status",
            "0",
            "2",
            "tags:",
            {
                "player_name": "Kitchen",
                "player_connected": 1,
                "player_needs_upgrade": 0,
                "player_is_upgrading": 0,
                "power": 1,
                "signalstrength": 50,
                "waitingToPlay": 0,
                "mode": "stop",
                "remote": 1,
                "current_title": "testserver",
                "time": 0,
                "duration": 0,
                "mixer volume": 50,
                # Not documented, but sent by the server (https://github.com/LMS-Community/slimserver/tree/cf756254749c489a1ac859dd4aad139b513dc655/Slim/Control/Queries.pm#L4055)
                "player_ip": "127.0.0.1",
                "playlist_cur_index": 1,
                "playlist_tracks": 0,
                "uuid": "player-uuid-1",
                "seq_no": 1,
            },
        )

        assert response == expected_response


class TestStatusMenuItemLoop:
    """Tests for the item_loop of a status command with menu:menu."""

    @staticmethod
    async def _menu_status(player: SlimClient, server: SlimServer) -> dict:
        """Return the menu-mode status of the player."""
        cli = SlimProtoCLI(server)
        return cast(
            "dict",
            await cli._handle_status(  # noqa: SLF001 # Accessing a protected member for testing purposes
                player.player_id, "-", 2, menu="menu"
            ),
        )

    @pytest.mark.asyncio
    async def test_omitted_when_nothing_is_loaded(
        self, dummy_player: SlimClient, dummy_server: SlimServer
    ) -> None:
        """Clients read item_loop[1] when it exists, so an empty one is omitted."""
        status = await self._menu_status(dummy_player, dummy_server)

        assert "item_loop" not in status
        assert status["count"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("current", "following", "expected"),
        [
            ("Song", None, ["Song"]),
            ("Song", "Next", ["Song", "Next"]),
            (None, "Next", ["Next"]),
        ],
    )
    async def test_lists_the_current_and_next_track(
        self,
        dummy_player: SlimClient,
        dummy_server: SlimServer,
        current: str | None,
        following: str | None,
        expected: list[str],
    ) -> None:
        """The loaded track and the one queued after it are listed."""

        def media(title: str | None) -> MediaDetails | None:
            if title is None:
                return None
            return MediaDetails(
                url="http://host/stream.flac",
                metadata={"title": title},  # type: ignore[typeddict-item]
            )

        dummy_player.current_media = media(current)
        dummy_player.next_media = media(following)

        status = await self._menu_status(dummy_player, dummy_server)

        assert [item["text"] for item in status["item_loop"]] == expected


class TestMixerVolumeCommand:
    """Tests for the mixer volume command.

    Reference: https://lyrion.org/reference/cli/players/#mixer-volume
    """

    @pytest.mark.asyncio
    async def test_get_volume(self, writer: Mock, dummy_player: SlimClient) -> None:
        """Should return the volume when requested with a '?'."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume ?"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "50"
        )

        assert response == expected_response

    @pytest.mark.asyncio
    async def test_set_volume_absolute(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should set the volume to an absolute value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume 75"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "75"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(75)

    @pytest.mark.asyncio
    async def test_set_volume_relative(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should change the volume by a relative value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume +25"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "+25"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(75)

        cast("Mock", dummy_player.volume_set).reset_mock()
        writer.reset_mock()

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume -50"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "-50"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(0)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Fractional volume is not yet implemented")
    async def test_set_volume_fractional(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should set the volume to a fractional value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume 25.5"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "25.5"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(25.5)


class TestMixerMutingCommand:
    """Tests for the mixer muting command.

    Reference: https://lyrion.org/reference/cli/players/#mixer-muting
    """

    @pytest.mark.asyncio
    async def test_can_toggle_muting(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow toggling the muting."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting toggle"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "muting", "toggle"
        )

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_toggle_muting_without_keyword(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow toggling the muting without the 'toggle' keyword."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "a5:41:d2:cd:cd:05 mixer muting")
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_mute(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow muting regardless of the current state."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting 1"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "1")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_unmute(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow unmuting regardless of the current state."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting 0"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "0")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(False)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_query_muting(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should return the current state of muting."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting ?"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "0")
        assert response == expected_response


class TestCommandHandler:
    """Tests for application-provided Slim command handling."""

    @pytest.mark.asyncio
    async def test_replaces_each_query_marker_with_scalar_response(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should pass parsed commands to the handler and substitute query values."""
        handler = AsyncMock(return_value=[12, 34])
        cli = SlimProtoCLI(dummy_server, command_handler=handler)

        response = await send_cli_command(
            cli,
            writer,
            "a5:41:d2:cd:cd:05 info total genres ? ? tags:summary offset:2",
        )

        assert response == encode_response(
            "a5:41:d2:cd:cd:05",
            "info",
            "total",
            "genres",
            "12",
            "34",
            "tags:summary",
            "offset:2",
        )

    @pytest.mark.asyncio
    async def test_returns_complex_response_blocks(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should append each complex response block to the echoed query."""
        cli = SlimProtoCLI(
            dummy_server,
            command_handler=AsyncMock(
                return_value=[
                    {"id": 1, "name": "First"},
                    {"id": 2, "name": "Second"},
                ]
            ),
        )

        response = await send_cli_command(cli, writer, "library items")

        assert response == encode_response(
            "library",
            "items",
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second"},
        )

    @pytest.mark.asyncio
    async def test_none_response_echoes_the_request(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should echo a command handled without a response payload."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))

        response = await send_cli_command(cli, writer, "external command value")

        assert response == encode_response("external", "command", "value")

    @pytest.mark.asyncio
    async def test_not_implemented_falls_back_to_builtin_handler(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should use the built-in handler when the application declines a command."""
        cli = SlimProtoCLI(
            dummy_server, command_handler=AsyncMock(side_effect=NotImplementedError)
        )

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume ?"
        )

        assert response == encode_response("a5:41:d2:cd:cd:05", "mixer", "volume", "50")


class TestCometDLongPollWait:
    """Tests for to ensure handshake requests and no id slim don't queue."""

    _MAX_ACCEPTABLE_SECONDS = 10  # well under the real 30s wait - if either
    # fix regresses, this fails clearly with a real elapsed time, rather
    # than the earlier (broken) version silently taking 30x longer and
    # still passing.

    @pytest.mark.asyncio
    async def test_handshake_only_batch_responds_immediately(
        self, dummy_server: SlimServer
    ) -> None:
        """A handshake-only batch must not hit the long-poll wait."""
        cli = SlimProtoCLI(dummy_server)
        request = Mock(
            json=AsyncMock(return_value=[{"channel": "/meta/handshake", "id": "1"}])
        )

        start = time.monotonic()
        resp = await cli._handle_cometd_client(request)  # noqa: SLF001
        elapsed = time.monotonic() - start

        assert resp.status == 200
        assert elapsed < self._MAX_ACCEPTABLE_SECONDS, (
            f"took {elapsed:.1f}s - hit the 30s long-poll wait, had_handshake regressed"
        )

    @pytest.mark.asyncio
    async def test_id_less_slim_request_responds_immediately(
        self, dummy_server: SlimServer
    ) -> None:
        """An id-less /slim/request must not hit the long-poll wait either."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))

        handshake_request = Mock(
            json=AsyncMock(return_value=[{"channel": "/meta/handshake", "id": "1"}])
        )
        handshake_resp = await cli._handle_cometd_client(handshake_request)  # noqa: SLF001
        clientid = json.loads(handshake_resp.text)[0]["clientId"]

        request = Mock(
            json=AsyncMock(
                return_value=[
                    {
                        "channel": "/slim/request",
                        "data": {
                            "request": [
                                "",
                                ["artworkspec", "add", "225x225_m", "test"],
                            ],
                            "response": f"/{clientid}/slim/request",
                        },
                    }
                ]
            )
        )

        start = time.monotonic()
        resp = await cli._handle_cometd_client(request)  # noqa: SLF001
        elapsed = time.monotonic() - start

        assert resp.status == 200
        assert elapsed < self._MAX_ACCEPTABLE_SECONDS, (
            f"took {elapsed:.1f}s - hit the 30s long-poll wait."
        )


class TestDisplayName:
    """Tests for the display_name_lookup hook used for player names."""

    def test_uses_lookup_when_it_returns_a_name(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """A name from the lookup overrides the device-reported name."""
        cli = SlimProtoCLI(dummy_server)
        cli.display_name_lookup = lambda _player_id: "Renamed in app"

        assert cli._display_name(dummy_player) == "Renamed in app"  # noqa: SLF001

    def test_falls_back_when_lookup_returns_none(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Falls back to the device-reported name when the lookup has no answer."""
        cli = SlimProtoCLI(dummy_server)
        cli.display_name_lookup = lambda _player_id: None

        assert cli._display_name(dummy_player) == dummy_player.name  # noqa: SLF001

    def test_falls_back_without_lookup(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """With no lookup set, the device-reported name is used."""
        cli = SlimProtoCLI(dummy_server)

        assert cli._display_name(dummy_player) == dummy_player.name  # noqa: SLF001

    def test_players_list_uses_lookup(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """The players list reports the looked-up name."""
        cli = SlimProtoCLI(dummy_server)
        cli.display_name_lookup = lambda _player_id: "Renamed in app"

        response = cli._handle_players(dummy_player.player_id)  # noqa: SLF001

        assert response["players_loop"][0]["name"] == "Renamed in app"


class TestCometDStaleSession:
    """Tests for requests that name a client session the server has already dropped."""

    @staticmethod
    async def _post(cli: SlimProtoCLI, messages: list[dict]) -> list[dict]:
        request = Mock(json=AsyncMock(return_value=messages))
        resp = await cli._handle_cometd_client(request)  # noqa: SLF001
        return json.loads(resp.text)

    @classmethod
    async def _handshake(cls, cli: SlimProtoCLI) -> str:
        return (await cls._post(cli, [{"channel": "/meta/handshake", "id": "1"}]))[0][
            "clientId"
        ]

    @pytest.mark.asyncio
    async def test_unsubscribe_for_dropped_session_succeeds(
        self, dummy_server: SlimServer
    ) -> None:
        """A stale unsubscribe succeeds and does not take over the batch."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))
        old_id = await self._handshake(cli)
        # The server drops a session when its streaming connection ends or it goes idle.
        cli._cometd_clients.pop(old_id)  # noqa: SLF001
        new_id = await self._handshake(cli)

        # A reconnecting client unsubscribes the old session's channels and then
        # subscribes on the new session in the same request.
        response = await self._post(
            cli,
            [
                {
                    "id": 7,
                    "channel": "/slim/unsubscribe",
                    "data": {"unsubscribe": f"/{old_id}/slim/menustatus/aa:bb"},
                },
                {
                    "id": 8,
                    "channel": "/slim/subscribe",
                    "data": {
                        "request": ["", ["serverstatus", 0, 50, "subscribe:60"]],
                        "response": f"/{new_id}/slim/serverstatus",
                    },
                },
            ],
        )

        assert response[0]["channel"] == "/slim/unsubscribe"
        assert response[0]["successful"] is True
        assert "advice" not in response[0]
        # The subscribe was handled under the new session.
        assert any(msg["channel"] == "/slim/subscribe" for msg in response)
        assert (
            f"/{new_id}/slim/serverstatus"
            in cli._cometd_clients[new_id].slim_subscriptions  # noqa: SLF001
        )

    @pytest.mark.asyncio
    async def test_unsubscribe_alone_for_dropped_session_succeeds(
        self, dummy_server: SlimServer
    ) -> None:
        """A request with only a stale unsubscribe gets its reply straight away."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))
        old_id = await self._handshake(cli)
        cli._cometd_clients.pop(old_id)  # noqa: SLF001

        # Nothing follows the unsubscribe, so no session is left to drain a queue from.
        response = await self._post(
            cli,
            [
                {
                    "id": 7,
                    "channel": "/slim/unsubscribe",
                    "data": {"unsubscribe": f"/{old_id}/slim/menustatus/aa:bb"},
                }
            ],
        )

        assert len(response) == 1
        assert response[0]["id"] == 7
        assert response[0]["channel"] == "/slim/unsubscribe"
        assert response[0]["clientId"] == old_id
        assert response[0]["successful"] is True
        assert "advice" not in response[0]
        # The dropped session was not brought back, and no other one was made.
        assert not cli._cometd_clients  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_other_requests_for_dropped_session_still_need_handshake(
        self, dummy_server: SlimServer
    ) -> None:
        """Only unsubscribe is forgiving; other channels still get invalid clientId."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))
        old_id = await self._handshake(cli)
        cli._cometd_clients.pop(old_id)  # noqa: SLF001

        response = await self._post(
            cli,
            [
                {
                    "id": 9,
                    "channel": "/slim/subscribe",
                    "data": {
                        "request": ["", ["serverstatus", 0, 50, "subscribe:60"]],
                        "response": f"/{old_id}/slim/serverstatus",
                    },
                }
            ],
        )

        assert response[0]["successful"] is False
        assert response[0]["error"] == "invalid clientId"
        assert response[0]["advice"]["reconnect"] == "handshake"


class TestPlaylistItem:
    """Tests for the codec and quality fields of a playlist item."""

    @staticmethod
    def _media(**metadata: str) -> MediaDetails:
        return MediaDetails(
            url="http://host/stream.flac",
            metadata=metadata,  # type: ignore[arg-type]
        )

    def test_takes_the_codec_and_quality_from_the_metadata(self) -> None:
        """The type, bitrate, samplerate and samplesize come from the metadata."""
        item = playlist_item_from_media_details(
            0,
            self._media(
                type="flac",
                bitrate="912kbps",
                samplerate="44100",
                samplesize="16",
            ),
        )

        assert item["type"] == "flac"
        assert item["bitrate"] == "912kbps"
        assert item["samplerate"] == "44100"
        assert item["samplesize"] == "16"

    def test_defaults_to_empty_strings_without_the_metadata(self) -> None:
        """Metadata that omits the fields gives empty strings, not missing keys."""
        item = playlist_item_from_media_details(0, self._media())

        assert item["type"] == ""
        assert item["bitrate"] == ""
        assert item["samplerate"] == ""
        assert item["samplesize"] == ""

    @pytest.mark.asyncio
    async def test_status_playlist_loop_carries_the_fields(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """A loaded track's codec and quality reach the status playlist_loop."""
        dummy_player.current_media = self._media(
            type="flac", bitrate="912kbps", samplerate="44100", samplesize="16"
        )
        cli = SlimProtoCLI(dummy_server)

        status = await cli._handle_status(  # noqa: SLF001
            dummy_player.player_id, "-", 2
        )

        assert status["playlist_loop"][0]["type"] == "flac"
        assert status["playlist_loop"][0]["bitrate"] == "912kbps"


class TestPushPopups:
    """The popups pushed to a device's displaystatus subscription."""

    @staticmethod
    def _cli_with_subscription(
        dummy_server: SlimServer, *, subscribed: bool = True
    ) -> tuple[SlimProtoCLI, CometDClient]:
        cli = SlimProtoCLI(dummy_server)
        client = CometDClient(client_id="c1", player_id="a5:41:d2:cd:cd:05")
        if subscribed:
            client.slim_subscriptions["/c1/slim/displaystatus/a5:41:d2:cd:cd:05"] = {
                "id": 11,
                "data": {"response": "/c1/slim/displaystatus/a5:41:d2:cd:cd:05"},
            }  # type: ignore[assignment]
        cli._cometd_clients["c1"] = client  # noqa: SLF001
        return cli, client

    def test_a_mixed_popup_carries_the_text_style_and_icon(
        self, dummy_server: SlimServer
    ) -> None:
        """The confirmation popup has the label, the title, a badge and cover art."""
        cli, client = self._cli_with_subscription(dummy_server)
        cli.push_show_briefly(
            "a5:41:d2:cd:cd:05",
            ["Adding", "A Song"],
            style="add",
            icon_id="42",
            duration_ms=2500,
        )
        message = client.queue.get_nowait()
        assert message["channel"] == "/c1/slim/displaystatus/a5:41:d2:cd:cd:05"
        assert message["id"] == 11
        assert message["data"]["display"] == {
            "type": "mixed",
            "text": ["Adding", "A Song"],
            "duration": 2500,
            "style": "add",
            "icon-id": "42",
        }

    def test_the_optional_parts_are_left_out_when_not_given(
        self, dummy_server: SlimServer
    ) -> None:
        """No style and no icon id mean neither key is sent."""
        cli, client = self._cli_with_subscription(dummy_server)
        cli.push_show_briefly(
            "a5:41:d2:cd:cd:05", ["Now Playing", "A Song"], kind="song"
        )
        display = client.queue.get_nowait()["data"]["display"]
        assert display["type"] == "song"
        assert "style" not in display
        assert "icon-id" not in display

    def test_the_play_icon_popup_always_has_text(
        self, dummy_server: SlimServer
    ) -> None:
        """The client fails on a missing text even though it never shows it."""
        cli, client = self._cli_with_subscription(dummy_server)
        cli.push_play_icon("a5:41:d2:cd:cd:05", ["A Song"], icon_id="42")
        display = client.queue.get_nowait()["data"]["display"]
        assert display == {
            "type": "icon",
            "style": "play",
            "play-mode": "play",
            "is-remote": 0,
            "text": ["A Song"],
            "icon-id": "42",
        }

    def test_nothing_is_queued_without_a_session_or_a_subscription(
        self, dummy_server: SlimServer
    ) -> None:
        """A device that is not listening for popups is not an error."""
        cli, client = self._cli_with_subscription(dummy_server, subscribed=False)
        cli.push_show_briefly("a5:41:d2:cd:cd:05", ["Adding", "A Song"])
        assert client.queue.empty()
        cli.push_show_briefly("someone-else", ["Adding", "A Song"])
        assert client.queue.empty()

    def test_a_session_with_the_subscription_gets_the_popup_not_an_older_one(
        self, dummy_server: SlimServer
    ) -> None:
        """A reconnect can leave an older session listed first for the same player."""
        cli, newer = self._cli_with_subscription(dummy_server)
        older = CometDClient(client_id="c0", player_id="a5:41:d2:cd:cd:05")
        cli._cometd_clients = {"c0": older, **cli._cometd_clients}  # noqa: SLF001
        cli.push_show_briefly("a5:41:d2:cd:cd:05", ["Adding", "A Song"])
        assert older.queue.empty()
        assert newer.queue.get_nowait()["data"]["display"]["text"] == [
            "Adding",
            "A Song",
        ]

    def test_a_full_queue_drops_the_popup_without_failing(
        self, dummy_server: SlimServer
    ) -> None:
        """The popup is a courtesy, so a backlog on the connection never raises."""
        cli, client = self._cli_with_subscription(dummy_server)
        for _ in range(client.queue.maxsize):
            client.queue.put_nowait({"channel": "x", "id": 0, "data": {}})  # type: ignore[typeddict-item]
        cli.push_show_briefly("a5:41:d2:cd:cd:05", ["Adding", "A Song"])
        assert client.queue.full()


class TestStatusShape:
    """Tests for which fields the status reply carries, and when."""

    @staticmethod
    def _media(**metadata: str | int) -> MediaDetails:
        return MediaDetails(
            url="http://host/stream.flac",
            metadata=metadata,  # type: ignore[arg-type]
        )

    @staticmethod
    async def _status(server: SlimServer, player: SlimClient) -> dict:
        cli = SlimProtoCLI(server)
        return cast(
            "dict",
            await cli._handle_status(player.player_id, "-", 2),  # noqa: SLF001
        )

    @pytest.mark.asyncio
    async def test_waiting_to_play_is_never_sent_as_zero(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """The client treats a sent 0 as true, so the key is left out."""
        status = await self._status(dummy_server, dummy_player)

        assert "waitingToPlay" not in status

    @pytest.mark.asyncio
    async def test_mode_and_queue_are_sent_when_the_player_is_powered_off(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Power is one more reported field, so it does not hide the queue."""
        dummy_player.powered = False
        dummy_player.current_media = self._media(title="Song", duration=200)

        status = await self._status(dummy_server, dummy_player)

        assert status["power"] == 0
        assert status["mode"] == "stop"
        assert status["playlist_tracks"] == 1
        assert status["playlist_loop"][0]["title"] == "Song"

    @pytest.mark.asyncio
    async def test_nothing_loaded_sends_no_song_fields(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """With nothing in the queue there is no track, position or index."""
        status = await self._status(dummy_server, dummy_player)

        assert status["playlist_tracks"] == 0
        assert status["playlist_loop"] == []
        for key in ("time", "duration", "remote", "playlist_cur_index"):
            assert key not in status

    @pytest.mark.asyncio
    async def test_a_loaded_track_sends_its_position_and_duration(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """A loaded track has a position, a duration and an index."""
        dummy_player.current_media = self._media(title="Song", duration=200)
        dummy_player.elapsed_seconds = 12.7

        status = await self._status(dummy_server, dummy_player)

        assert status["time"] == 12
        assert status["duration"] == 200
        assert status["playlist_cur_index"] == 0

    @pytest.mark.asyncio
    async def test_empty_quality_fields_are_left_out_of_the_playlist_loop(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """An empty string is true to the client, so only filled quality keys go out."""
        dummy_player.current_media = self._media(title="Song", type="flac")

        status = await self._status(dummy_server, dummy_player)

        item = status["playlist_loop"][0]
        assert item["type"] == "flac"
        for key in ("bitrate", "samplerate", "samplesize"):
            assert key not in item

    @pytest.mark.asyncio
    async def test_extra_data_is_merged_into_the_reply(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """The host's extra_data keys reach the client whatever the power state."""
        dummy_player.powered = False

        status = await self._status(dummy_server, dummy_player)

        assert status["uuid"] == "player-uuid-1"
        assert status["seq_no"] == 1

    @pytest.mark.asyncio
    async def test_a_next_track_alone_still_has_a_queue(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """With only a next track, the queue lists it and the song fields follow it."""
        dummy_player.next_media = self._media(title="Next", duration=90)

        status = await self._status(dummy_server, dummy_player)

        assert status["playlist_tracks"] == 1
        assert status["playlist_loop"][0]["title"] == "Next"
        assert status["duration"] == 90
        assert status["playlist_cur_index"] == 0


class TestDropEmptyQualityFields:
    """The helper that trims a playlist item for the wire."""

    def test_only_empty_quality_fields_are_dropped(self) -> None:
        """Other empty fields stay; filled quality fields stay."""
        item = playlist_item_from_media_details(
            0,
            MediaDetails(
                url="http://host/stream.flac",
                metadata={"title": "Song", "type": "flac", "samplerate": "96000"},  # type: ignore[typeddict-item]
            ),
        )

        trimmed = _drop_empty_quality_fields(item)

        assert trimmed["type"] == "flac"
        assert trimmed["samplerate"] == "96000"
        assert "bitrate" not in trimmed
        assert "samplesize" not in trimmed
        assert trimmed["artist"] == ""
        assert trimmed["title"] == "Song"
