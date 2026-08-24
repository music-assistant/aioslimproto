"""Tests for the SlimProto CLI."""

import asyncio
from itertools import chain
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import pytest

from aioslimproto.cli import SlimProtoCLI
from aioslimproto.client import SlimClient
from aioslimproto.models import PlayerState
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

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Fractional values are not supported")
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
