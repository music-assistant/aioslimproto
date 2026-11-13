"""Tests for SlimServer's player registration."""

import asyncio
import sys
from unittest.mock import Mock, patch

import pytest

from aioslimproto.models import EventType
from aioslimproto.server import SlimServer

_PLAYER_ID = "aa:bb:cc:dd:ee:ff"


@pytest.mark.asyncio
async def test_unregistered_socket_disconnect_keeps_registered_player() -> None:
    """A second socket of the same player that drops must not unregister it."""
    server = SlimServer(cli_port=None, cli_port_json=None, ip_address="127.0.0.1")
    events = []
    server.subscribe(events.append)
    with patch("aioslimproto.server.SlimClient") as slim_client_cls:
        await server._create_client(Mock(), Mock())  # noqa: SLF001
    client_callback = slim_client_cls.call_args.args[2]
    registered = Mock(player_id=_PLAYER_ID, connected=True)
    client_callback(registered, EventType.PLAYER_CONNECTED)
    await asyncio.sleep(0)
    events.clear()

    client_callback(Mock(player_id=_PLAYER_ID), EventType.PLAYER_DISCONNECTED)
    await asyncio.sleep(0)

    assert server.get_player(_PLAYER_ID) is registered
    assert events == []


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info[:2] == (3, 12),
    reason="3.12 has no Server.close_clients(), stop() waits for the read timeout",
)
async def test_stop_closes_unregistered_connection() -> None:
    """Stop must not hang on a connection that never registered as a player."""
    server = SlimServer(cli_port=None, cli_port_json=None, ip_address="127.0.0.1")
    server._server = await asyncio.start_server(  # noqa: SLF001
        server._create_client,  # noqa: SLF001
        "127.0.0.1",
        0,
    )
    port = server._server.sockets[0].getsockname()[1]  # noqa: SLF001
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)
    await asyncio.sleep(0.1)
    try:
        async with asyncio.timeout(5):
            await server.stop()
    finally:
        writer.close()
