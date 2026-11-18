"""Tests for the SlimProto CLI."""

import asyncio
from itertools import chain
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import aiohttp
from aiohttp import web
import pytest

from aioslimproto.cli import (
    CometDClient,
    SlimProtoCLI,
    _drop_empty_quality_fields,
    playlist_item_from_media_details,
)
from aioslimproto.client import SlimClient
from aioslimproto.models import EventType, MediaDetails, PlayerState
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
            alarm_fields=lambda: {
                "alarm_state": "none",
                "alarm_snooze_seconds": 540,
                "alarm_timeout_seconds": 3600,
            },
            play=AsyncMock(),
            pause=AsyncMock(),
            toggle_pause=AsyncMock(),
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


class TestPauseCommand:
    """Tests for the pause command.

    Reference: https://lyrion.org/reference/cli/players/#pause
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", [PlayerState.PLAYING, PlayerState.PAUSED])
    @pytest.mark.parametrize(
        ("command", "expected_call"),
        [
            ("pause 1", "pause"),
            ("pause 0", "play"),
            ("pause", "toggle_pause"),
            ("pause 0 2", "play"),
            ("mode pause", "pause"),
            ("mode play", "play"),
        ],
    )
    async def test_pause_command(
        self,
        writer: Mock,
        dummy_server: SlimServer,
        dummy_player: SlimClient,
        command: str,
        expected_call: str,
        state: PlayerState,
    ) -> None:
        """Should pause, unpause or toggle as requested, whatever the player state."""
        dummy_player.state = state
        cli = SlimProtoCLI(dummy_server)

        await send_cli_command(cli, writer, f"a5:41:d2:cd:cd:05 {command}")

        for method in ("play", "pause", "toggle_pause"):
            mock = cast("AsyncMock", getattr(dummy_player, method))
            if method == expected_call:
                mock.assert_awaited_once_with()
            else:
                mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_command_over_jsonrpc(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should accept the value as a number in a JSON-RPC request."""
        cli = SlimProtoCLI(dummy_server)

        result = await cli._handle_command(("a5:41:d2:cd:cd:05", ["pause", 0]))  # noqa: SLF001 # Accessing a protected member for testing purposes

        assert result == {}
        cast("AsyncMock", dummy_player.play).assert_awaited_once_with()
        cast("AsyncMock", dummy_player.pause).assert_not_called()


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


async def send_jsonrpc_request(cli: SlimProtoCLI, request: list) -> dict:
    """Send one slim.request to the JSON-RPC endpoint and return the decoded answer."""
    message = {"id": 1, "method": "slim.request", "params": request}
    response = await cli._handle_jsonrpc_client(  # noqa: SLF001
        Mock(json=AsyncMock(return_value=message))
    )
    return json.loads(response.text)


class TestSingleValueQueries:
    """Tests for the key and value of a query answered with one value.

    Reference: Slim/Control/Queries.pm in LMS, which names the value after the word
    before the "?".
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_", "expected"),
        [
            (["mixer", "volume", "?"], {"_volume": "50"}),
            (["mixer", "muting", "?"], {"_muting": "0"}),
            (["time", "?"], {"_time": 0}),
            (["power", "?"], {"_power": 1}),
            (["mode", "?"], {"_mode": "stop"}),
            (["playlist", "repeat", "?"], {"_repeat": "0"}),
            (["playlist", "shuffle", "?"], {"_shuffle": "0"}),
        ],
    )
    async def test_builtin_query_over_jsonrpc(
        self, dummy_server: SlimServer, request_: list, expected: dict
    ) -> None:
        """Should answer under the key and with the value type LMS uses."""
        cli = SlimProtoCLI(dummy_server)

        answer = await send_jsonrpc_request(cli, ["a5:41:d2:cd:cd:05", request_])

        assert answer["result"] == expected

    @pytest.mark.asyncio
    async def test_wiim_queries_after_a_volume_change(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should answer the two queries a WiiM sends after each volume frame."""
        dummy_player.volume_level = 31
        cli = SlimProtoCLI(dummy_server)

        muting = await send_jsonrpc_request(
            cli, ["a5:41:d2:cd:cd:05", ["mixer", "muting", "?"]]
        )
        volume = await send_jsonrpc_request(
            cli, ["a5:41:d2:cd:cd:05", ["mixer", "volume", "?"]]
        )

        assert muting == {
            "id": 1,
            "method": "slim.request",
            "params": ["a5:41:d2:cd:cd:05", ["mixer", "muting", "?"]],
            "result": {"_muting": "0"},
        }
        assert volume == {
            "id": 1,
            "method": "slim.request",
            "params": ["a5:41:d2:cd:cd:05", ["mixer", "volume", "?"]],
            "result": {"_volume": "31"},
        }

    @pytest.mark.asyncio
    async def test_wiim_polls_answer_what_the_status_reports(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should answer the play state, repeat and shuffle polls as status does."""
        dummy_player.state = PlayerState.PAUSED
        dummy_player.extra_data["playlist repeat"] = 2
        dummy_player.extra_data["playlist shuffle"] = 1
        cli = SlimProtoCLI(dummy_server)

        status = await send_jsonrpc_request(
            cli, ["a5:41:d2:cd:cd:05", ["status", "-", 1, "tags:acdeIlKsT"]]
        )
        mode = await send_jsonrpc_request(cli, ["a5:41:d2:cd:cd:05", ["mode", "?"]])
        repeat = await send_jsonrpc_request(
            cli, ["a5:41:d2:cd:cd:05", ["playlist", "repeat", "?"]]
        )
        shuffle = await send_jsonrpc_request(
            cli, ["a5:41:d2:cd:cd:05", ["playlist", "shuffle", "?"]]
        )

        assert mode["result"] == {"_mode": "pause"}
        assert repeat["result"] == {"_repeat": "2"}
        assert shuffle["result"] == {"_shuffle": "1"}
        assert status["result"]["mode"] == "pause"
        assert status["result"]["playlist repeat"] == 2
        assert status["result"]["playlist shuffle"] == 1

    @pytest.mark.asyncio
    async def test_mode_query_over_telnet(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should fill in the play state where the text interface asked for it."""
        dummy_player.state = PlayerState.PLAYING
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "a5:41:d2:cd:cd:05 mode ?")

        assert response == encode_response("a5:41:d2:cd:cd:05", "mode", "play")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_", [["playlist", "repeat", 2], ["playlist", "shuffle"]]
    )
    async def test_repeat_and_shuffle_changes_go_to_the_host(
        self, dummy_server: SlimServer, dummy_player: SlimClient, request_: list
    ) -> None:
        """Should leave setting or toggling repeat and shuffle to the host."""
        callback = Mock()
        dummy_player.callback = callback
        cli = SlimProtoCLI(dummy_server)

        answer = await send_jsonrpc_request(cli, ["a5:41:d2:cd:cd:05", request_])

        assert answer["error"] == {"code": -1, "message": "Invalid command"}
        callback.assert_called_once_with(
            dummy_player,
            EventType.PLAYER_CLI_EVENT,
            " ".join(str(arg) for arg in request_),
        )

    @pytest.mark.asyncio
    async def test_builtin_query_over_cometd(self, dummy_server: SlimServer) -> None:
        """Should answer a CometD request under the key LMS uses."""
        cli = SlimProtoCLI(dummy_server)
        client = CometDClient(client_id="c1")

        cli._handle_cometd_client_request(  # noqa: SLF001
            client,
            {
                "id": "7",
                "data": {
                    "request": ["a5:41:d2:cd:cd:05", ["mixer", "volume", "?"]],
                    "response": "/c1/slim/request",
                },
            },
        )
        answer = await asyncio.wait_for(client.queue.get(), timeout=1)

        assert answer == {
            "channel": "/c1/slim/request",
            "id": "7",
            "data": {"_volume": "50"},
            "ext": {"priority": None},
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_", "value", "expected"),
        [
            (["info", "total", "genres", "?"], 12, {"_genres": 12}),
            (["playlist", "title", 3, "?"], "Intro", {"_title": "Intro"}),
            (["can", "login", "?"], 1, {"_can": 1}),
        ],
    )
    async def test_application_handler_value(
        self, dummy_server: SlimServer, request_: list, value: Any, expected: dict
    ) -> None:
        """Should name a single value from the application handler like LMS does."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=value))

        answer = await send_jsonrpc_request(cli, ["a5:41:d2:cd:cd:05", request_])

        assert answer["result"] == expected


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


class TestElapsedLookup:
    """Tests for the elapsed_lookup hook used for the position in the current track."""

    @staticmethod
    async def _answers(cli: SlimProtoCLI, player: SlimClient) -> tuple[int, int]:
        """Return the `time ?` answer and the status `time` for the player."""
        player.current_media = MediaDetails(
            url="http://host/stream.flac",
            metadata={"title": "Song", "duration": 200},  # type: ignore[typeddict-item]
        )
        answer = await send_jsonrpc_request(cli, [player.player_id, ["time", "?"]])
        status = await cli._handle_status(player.player_id, "-", 1)  # noqa: SLF001
        return answer["result"]["_time"], status["time"]

    @pytest.mark.asyncio
    async def test_uses_lookup_when_it_returns_a_position(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Both answers report the looked-up position instead of the stream's own."""
        dummy_player.elapsed_seconds = 305.2
        cli = SlimProtoCLI(dummy_server)
        cli.elapsed_lookup = lambda _player_id: 20.6

        assert await self._answers(cli, dummy_player) == (20, 20)

    @pytest.mark.asyncio
    async def test_a_position_of_zero_is_an_answer(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """The start of a track is reported as such, not as the stream's position."""
        dummy_player.elapsed_seconds = 305.2
        cli = SlimProtoCLI(dummy_server)
        cli.elapsed_lookup = lambda _player_id: 0

        assert await self._answers(cli, dummy_player) == (0, 0)

    @pytest.mark.asyncio
    async def test_lookup_is_asked_for_the_player(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """The lookup gets the id of the player the answer is for."""
        cli = SlimProtoCLI(dummy_server)
        lookup = Mock(return_value=20.6)
        cli.elapsed_lookup = lookup

        await self._answers(cli, dummy_player)

        assert {call.args for call in lookup.call_args_list} == {
            (dummy_player.player_id,)
        }

    @pytest.mark.asyncio
    async def test_falls_back_when_lookup_returns_none(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Both answers report the player's own position when the lookup has none."""
        dummy_player.elapsed_seconds = 305.2
        cli = SlimProtoCLI(dummy_server)
        cli.elapsed_lookup = lambda _player_id: None

        assert await self._answers(cli, dummy_player) == (305, 305)

    @pytest.mark.asyncio
    async def test_falls_back_without_lookup(
        self, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """With no lookup set, both answers report the player's own position."""
        dummy_player.elapsed_seconds = 305.2
        cli = SlimProtoCLI(dummy_server)

        assert await self._answers(cli, dummy_player) == (305, 305)


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


class TestDeviceAlarm:
    """The alarm the player keeps itself travels in the status document."""

    @staticmethod
    def _cli_with_alarm(client: SlimClient) -> SlimProtoCLI:
        """Return a CLI whose only player is the given client."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                name="testserver",
                logger=logging.getLogger(),
                players=[client],
                get_player=lambda player_id: (
                    client if player_id == client.player_id else None
                ),
            ),
        )
        return SlimProtoCLI(server)

    @staticmethod
    def _client() -> SlimClient:
        """Return a real SlimClient on a socket that reports EOF."""
        writer = Mock()
        writer.drain = AsyncMock()
        writer.is_closing.return_value = False
        writer.can_write_eof.return_value = False
        writer.wait_closed = AsyncMock()
        writer.get_extra_info.return_value = None
        reader = asyncio.StreamReader()
        reader.feed_eof()
        client = SlimClient(reader, writer, Mock())
        client._player_id = "a5:41:d2:cd:cd:05"  # noqa: SLF001
        return client

    @pytest.mark.asyncio
    async def test_no_alarm_is_reported_as_none(self) -> None:
        """Without an alarm both answers say so, and carry the snooze values."""
        client = self._client()
        cli = self._cli_with_alarm(client)

        status = await cli._handle_status(client.player_id, "-", 2)  # noqa: SLF001
        menu = await cli._handle_status(  # noqa: SLF001
            client.player_id, "-", 2, menu="menu"
        )

        for answer in (status, menu):
            assert answer["alarm_state"] == "none"
            assert answer["alarm_snooze_seconds"] == 540

    @pytest.mark.asyncio
    async def test_a_set_alarm_reaches_both_answers(self) -> None:
        """The menu answer must not overwrite the alarm with a hardcoded none."""
        client = self._client()
        next_at = int(time.time()) + 8 * 3600
        client.set_alarm(next_at=next_at)
        cli = self._cli_with_alarm(client)

        status = await cli._handle_status(client.player_id, "-", 2)  # noqa: SLF001
        menu = await cli._handle_status(  # noqa: SLF001
            client.player_id, "-", 2, menu="menu"
        )

        for answer in (status, menu):
            assert answer["alarm_state"] == "set"
            assert answer["alarm_next2"] == next_at
            assert answer["alarm_next"] == next_at
            assert answer["alarm_version"] == 2
            assert answer["alarm_repeat"] == 1
            assert answer["alarm_days"] == "1111111"

    @pytest.mark.asyncio
    async def test_clearing_reports_none_again(self) -> None:
        """Clearing is how a player that is already armed is disarmed."""
        client = self._client()
        client.set_alarm(next_at=int(time.time()) + 60)
        client.clear_alarm()
        cli = self._cli_with_alarm(client)

        status = await cli._handle_status(client.player_id, "-", 2)  # noqa: SLF001

        assert status["alarm_state"] == "none"

    @pytest.mark.asyncio
    async def test_a_single_day_mask_is_the_weekday_of_the_alarm(self) -> None:
        """A one-shot alarm sends one set day, Sunday first."""
        client = self._client()

        client.set_alarm(next_at=int(time.time()) + 60, repeat=False, days="0100000")

        assert client.alarm_fields()["alarm_days"] == "0100000"
        assert client.alarm_fields()["alarm_repeat"] == 0

    @pytest.mark.parametrize("days", ["111111", "11111111", "1111a11", ""])
    @pytest.mark.asyncio
    async def test_a_mask_that_is_not_seven_flags_is_refused(self, days: str) -> None:
        """A malformed mask is a caller error, not something to send to a player."""
        client = self._client()

        with pytest.raises(ValueError, match="seven characters"):
            client.set_alarm(next_at=int(time.time()) + 60, days=days)


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
    @pytest.mark.parametrize("state", [PlayerState.BUFFERING, PlayerState.BUFFER_READY])
    async def test_waiting_to_play_is_sent_while_buffering(
        self,
        dummy_server: SlimServer,
        dummy_player: SlimClient,
        state: PlayerState,
    ) -> None:
        """The position stands still while buffering, so the client must wait."""
        dummy_player.state = state

        status = await self._status(dummy_server, dummy_player)

        assert status["mode"] == "play"
        assert status["waitingToPlay"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state", [PlayerState.PLAYING, PlayerState.PAUSED, PlayerState.STOPPED]
    )
    async def test_waiting_to_play_is_left_out_otherwise(
        self,
        dummy_server: SlimServer,
        dummy_player: SlimClient,
        state: PlayerState,
    ) -> None:
        """The client treats a sent 0 as true, so the key is left out otherwise."""
        dummy_player.state = state

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


class TestExtraRoutes:
    """Routes the consuming application adds to the CLI's own web server."""

    @staticmethod
    async def _started(routes: dict) -> SlimServer:
        """Start a server whose CLI web server (only) runs on a free port."""
        server = SlimServer(
            ip_address="127.0.0.1",
            name="test",
            cli_port=None,
            cli_port_json=0,
            extra_routes=routes,
        )
        await server.cli.start()
        return server

    async def test_a_route_of_the_application_is_served(self) -> None:
        """A GET reaches the handler, and a HEAD gets the same answer without a body."""

        async def cover(request: web.Request) -> web.Response:
            return web.Response(text=f"cover {request.match_info['id']}")

        server = await self._started({"/music/{id}/cover": cover})
        try:
            base = f"http://127.0.0.1:{server.cli.cli_port_json}"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/music/7/cover") as response:
                    assert (response.status, await response.text()) == (
                        200,
                        "cover 7",
                    )
                async with session.head(f"{base}/music/7/cover") as response:
                    assert (response.status, await response.read()) == (200, b"")
        finally:
            await server.cli.stop()

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    async def test_a_route_of_the_application_only_answers_get_and_head(
        self, method: str
    ) -> None:
        """Other methods are refused before they reach the handler."""
        calls = []

        async def cover(_request: web.Request) -> web.Response:
            calls.append(1)
            return web.Response(text="cover")

        server = await self._started({"/cover": cover})
        try:
            base = f"http://127.0.0.1:{server.cli.cli_port_json}"
            async with (
                aiohttp.ClientSession() as session,
                session.request(method, f"{base}/cover") as response,
            ):
                assert response.status == 405
            assert not calls
        finally:
            await server.cli.stop()

    async def test_the_routes_are_matched_in_the_order_given(self) -> None:
        """Of two paths with variables that both match, the one listed first answers."""

        async def numbers(_request: web.Request) -> web.Response:
            return web.Response(text="numbers")

        async def catch_all(_request: web.Request) -> web.Response:
            return web.Response(text="catch-all")

        server = await self._started(
            {r"/files/{number:\d+}": numbers, "/files/{tail:.*}": catch_all}
        )
        try:
            base = f"http://127.0.0.1:{server.cli.cli_port_json}"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/files/12") as response:
                    assert await response.text() == "numbers"
                async with session.get(f"{base}/files/other") as response:
                    assert await response.text() == "catch-all"
        finally:
            await server.cli.stop()

    async def test_the_built_in_routes_still_work(self) -> None:
        """Adding routes leaves /jsonrpc.js and /cometd alone."""

        async def handler(_request: web.Request) -> web.Response:
            return web.Response(text="extra")

        server = await self._started({"/extra": handler})
        try:
            base = f"http://127.0.0.1:{server.cli.cli_port_json}"
            request = {
                "id": 1,
                "method": "slim.request",
                "params": ["-", ["players", 0, 9]],
            }
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"{base}/jsonrpc.js", json=request) as response,
            ):
                assert response.status == 200
                assert (await response.json())["result"]["count"] == 0
        finally:
            await server.cli.stop()

    @pytest.mark.parametrize("path", ["/jsonrpc.js", "/cometd"])
    def test_a_built_in_route_cannot_be_replaced(
        self, dummy_server: SlimServer, path: str
    ) -> None:
        """Asking for one of the CLI's own paths fails at once, not when it starts."""

        async def handler(_request: web.Request) -> web.Response:
            return web.Response(text="extra")

        with pytest.raises(ValueError, match="built-in route"):
            SlimProtoCLI(dummy_server, extra_routes={path: handler})
        with pytest.raises(ValueError, match="built-in route"):
            SlimServer(ip_address="127.0.0.1", name="t", extra_routes={path: handler})

    def test_without_routes_there_are_none(self, dummy_server: SlimServer) -> None:
        """The default is no extra routes."""
        assert SlimProtoCLI(dummy_server).extra_routes == {}
