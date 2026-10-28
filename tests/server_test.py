"""Tests for SlimServer's player registration."""

import asyncio
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
