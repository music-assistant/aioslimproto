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
from aioslimproto.client import PlayerState, SlimClient
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
        ),
    )


@pytest.fixture
def dummy_server() -> SlimServer:
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
    @pytest.mark.xfail(
        reason="The pagination returns one player too many, "
        "and the player index is incorrect"
    )
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
    @pytest.mark.xfail(
        reason="The echoed command should have the ':' symbol URL-escaped"
    )
    async def test_echoes_without_player(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should echo the command when no player is specified."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "status 0 2 tags:")
        expected_response = encode_response("status", "0", "2", "tags:")

        assert response == expected_response

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="The echoed command should start with the player ID")
    async def test_simple_example(self, writer: Mock, dummy_server: SlimServer) -> None:
        """The "simple example" on the reference page should work."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 status 0 2 tags:"
        )
        expected_response = encode_response(
            "status",
            "0",
            "2",
            {"player_name": "Kitchen"},
        )

        assert response == expected_response
