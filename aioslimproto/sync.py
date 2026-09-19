"""LMS-faithful synchronisation helpers for aioslimproto.

Ported from Lyrion Music Server:

- ``Slim::Player::StreamingController::_CheckSync`` -> :meth:`SyncGroup.check`
- ``Slim::Player::StreamingController::_syncStart`` -> :meth:`SyncGroup.start`

The per-player clock mapping and play points live on :class:`SlimClient`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .const import (
    CHECK_SYNC_INTERVAL,
    MAX_DEVIATION_ADJUST,
    MIN_DEVIATION_ADJUST,
    PLAYPOINT_RECENT_THRESHOLD,
    SYNC_START_DELAY,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .client import SlimClient

LOGGER = logging.getLogger(__name__)


class SyncGroup:
    """Holds sync state for one group of players.

    Mirrors the synchronisation duties of LMS' StreamingController, without the
    transport/state-machine parts.
    """

    def __init__(self) -> None:
        """Initialize the group."""
        self.next_check: float = 0.0

    async def start(
        self,
        clients: Iterable[SlimClient],
        now: float | None = None,
    ) -> None:
        """Start all players at a common server time (port of ``_syncStart``)."""
        clients = list(clients)
        if len(clients) < 2:
            return
        now = now if now is not None else time.time()

        player_start_delay = 0
        for client in clients:
            player_start_delay = max(
                player_start_delay, client.start_delay + client.play_delay
            )

        start_at = now + (player_start_delay + SYNC_START_DELAY) / 1000

        for client in clients:
            target = start_at - (client.start_delay + client.play_delay) / 1000
            LOGGER.debug(
                "%s startAt in %.1fms",
                client.player_id,
                (target - now) * 1000,
            )
            await client.start_at(target)

    async def check(
        self,
        clients: Iterable[SlimClient],
        now: float | None = None,
    ) -> None:
        """Correct drift between players (port of ``_CheckSync``)."""
        clients = [c for c in clients if c.maintain_sync]
        if len(clients) < 2:
            return

        now = now if now is not None else time.time()
        if now < self.next_check:
            return
        self.next_check = now + CHECK_SYNC_INTERVAL

        # need a recent play-point from all players, otherwise give up
        recent_threshold = now - PLAYPOINT_RECENT_THRESHOLD
        play_points: list[tuple[SlimClient, float]] = []
        for client in clients:
            play_point = client.play_point
            if play_point is None:
                LOGGER.debug("%s bailing as no playPoint", client.player_id)
                return
            if play_point[0] > recent_threshold:
                play_points.append((client, play_point[1] + client.play_delay / 1000))
            else:
                LOGGER.debug(
                    "%s bailing as playPoint too old: %.1fs",
                    client.player_id,
                    now - play_point[0],
                )
                return
        if not play_points:
            return

        if LOGGER.isEnabledFor(logging.DEBUG):
            first = play_points[0][1]
            LOGGER.debug(
                "playPoints: %s",
                ", ".join(
                    f"{c.player_id}: {p:.3f} ({(p - first) * 1000:+.0f}ms)"
                    for c, p in play_points
                ),
            )

        # sort by decreasing apparent-start-time
        play_points.sort(key=lambda item: item[1], reverse=True)

        # reference = most-behind player that cannot skip ahead
        reference = 0
        while (
            reference < len(play_points) - 1
            and play_points[reference][0].can_skip_ahead
        ):
            reference += 1
        reference_time = play_points[reference][1]

        for index, (client, apparent_start) in enumerate(play_points):
            if index == reference:
                continue
            delta = abs(apparent_start - reference_time)
            if (
                delta > MAX_DEVIATION_ADJUST
                or delta < MIN_DEVIATION_ADJUST
                or delta < client.min_sync_adjust / 1000
            ):
                continue
            if index < reference:
                LOGGER.info("%s resync: skipAhead %dms", client.player_id, delta * 1000)
                await client.skip_over(int(delta * 1000))
                self.next_check += 1
            elif client.can_pause_for:
                LOGGER.info("%s resync: pauseFor %dms", client.player_id, delta * 1000)
                await client.pause_for(int(delta * 1000))
                self.next_check += delta
