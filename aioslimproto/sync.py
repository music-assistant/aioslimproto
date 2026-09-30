"""LMS-faithful synchronisation helpers for aioslimproto.

Ported from Lyrion Music Server:

- ``Slim::Player::StreamingController::_CheckSync`` -> :meth:`SyncGroup.check`
- ``Slim::Player::StreamingController::_syncStart`` -> :meth:`SyncGroup.start`
- ``Slim::Player::StreamingController::_Resume`` -> :meth:`SyncGroup.resume`

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
    SYNC_RESUME_HOLDOFF,
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
        """Start all players at a common server time (port of ``_syncStart``).

        LMS ``_syncStart`` sets no ``nextCheckSyncTime`` holdoff (unlike
        ``_Resume``), so the first ``_CheckSync`` can run ~1s into play and issue
        an audible ``skipAhead``/``pauseFor`` correction for a member that landed
        beyond its ``min_sync_adjust`` (observed: a Squeezebox Radio starting
        ~276-441ms off a SqueezeLite). A per-player learned start offset is a
        possible future improvement.
        """
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

    async def resume(
        self,
        clients: Iterable[SlimClient],
        paused_at: float,
        now: float | None = None,
    ) -> None:
        """Resume a paused group in sync (port of ``_Resume``).

        Mirrors LMS ``StreamingController::_Resume``: a single player is resumed
        plainly, while a group is resumed at a common server instant with each
        player delayed by however far past the pause point its reported song
        position has moved. The first sync check is held off
        (``SYNC_RESUME_HOLDOFF``) so it does not compare play points taken while
        paused.

        Known limitation: a resumed member can be heard as a short timing gap. LMS
        hides this behind a fade-in, but that fade is a server-side *volume* ramp
        (``Slim::Player::Player::fade_volume`` steps the volume 0 -> stored over
        ~0.3s), not a stream transition, and MA owns the volume (group volume,
        limits, mute): ramping it here would fight that state and bounce the UI.
        ``strm u`` (unpause) carries no transition either, so there is nothing
        stream-level to reuse. Left as a known gap.

        :param clients: The clients in the group.
        :param paused_at: The song position (seconds) at which the group was paused.
        :param now: Optional server time override (for testing).
        """
        clients = list(clients)
        now = now if now is not None else time.time()

        if len(clients) < 2:
            for client in clients:
                await client.unpause_at(0)
            return

        start_at_base = now + SYNC_START_DELAY / 1000
        for client in clients:
            start_at = start_at_base
            play_point = client.play_point
            if play_point is not None and (delay := play_point[2] - paused_at) >= 0:
                start_at += delay
            LOGGER.debug(
                "%s resume startAt in %.1fms",
                client.player_id,
                (start_at - now) * 1000,
            )
            await client.start_at(start_at)

        self.next_check = start_at_base + SYNC_RESUME_HOLDOFF

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
                # Currently unreachable: every client reports can_skip_ahead, so the
                # reference above is always the most-behind player and only the
                # skipAhead branch runs. Becomes live once a device class reports
                # can_skip_ahead=False (LMS gates that on SB1/SliMP3). Kept for LMS
                # parity; the WiiM case is handled server-side with a pre-start pause.
                LOGGER.info("%s resync: pauseFor %dms", client.player_id, delta * 1000)
                await client.pause_for(int(delta * 1000))
                self.next_check += delta
