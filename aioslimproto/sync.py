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
    SYNC_START_HOLDOFF,
)
from .models import PlayerState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .client import SlimClient

LOGGER = logging.getLogger(__name__)


class SyncGroup:
    """
    Holds sync state for one group of players.

    Mirrors the synchronisation duties of LMS' StreamingController, without the
    transport/state-machine parts. All server times are ``time.monotonic()``
    seconds, the clock :class:`SlimClient` maps the player clocks onto.
    """

    def __init__(self) -> None:
        """Initialize the group."""
        self.next_check: float = 0.0
        # player_id -> (behind the reference, status time of its play point, status
        # time of the reference's play point), for each player outside the deadband
        # on the previous check
        self._deviating: dict[str, tuple[bool, float, float]] = {}
        # end of the holdoff after the last start or resume, until the first check
        # after it
        self._holdoff_until: float | None = None

    async def start(
        self,
        clients: Iterable[SlimClient],
        now: float | None = None,
    ) -> None:
        """
        Start all players at a common server time (port of ``_syncStart``).

        Unlike ``_syncStart``, the first sync check is held off
        (``SYNC_START_HOLDOFF``), as play points from the first second of playback
        can be far off. That check then corrects a member that landed beyond its
        ``min_sync_adjust`` (e.g. a Squeezebox Radio starting ~276-441ms off a
        SqueezeLite, or a player with a slow audio output) with an audible
        ``skipAhead``/``pauseFor``. A per-player learned start offset is a
        possible future improvement.
        """
        clients = list(clients)
        if len(clients) < 2:
            return
        now = now if now is not None else time.monotonic()

        player_start_delay = 0
        for client in clients:
            player_start_delay = max(
                player_start_delay, client.start_delay + client.play_delay
            )

        start_at = now + (player_start_delay + SYNC_START_DELAY) / 1000
        self._holdoff_until = start_at + SYNC_START_HOLDOFF
        self.next_check = self._holdoff_until

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
        position has moved, less its ``play_delay`` (as in :meth:`start`).
        The first sync check is held off
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
        now = now if now is not None else time.monotonic()

        if len(clients) < 2:
            for client in clients:
                await client.unpause_at(0)
            return

        start_at_base = now + SYNC_START_DELAY / 1000
        self._holdoff_until = start_at_base + SYNC_RESUME_HOLDOFF
        self.next_check = self._holdoff_until
        for client in clients:
            start_at = start_at_base - client.play_delay / 1000
            play_point = client.play_point
            if play_point is not None and (delay := play_point[2] - paused_at) >= 0:
                start_at += delay
            LOGGER.debug(
                "%s resume startAt in %.1fms",
                client.player_id,
                (start_at - now) * 1000,
            )
            await client.start_at(start_at)

    async def check(
        self,
        clients: Iterable[SlimClient],
        now: float | None = None,
    ) -> None:
        """
        Correct drift between players (port of ``_CheckSync``).

        A player is only corrected once two consecutive checks, on new play points
        of both the player and the reference, find it off on the same side. The
        first check after a start or resume corrects at once, on play points
        reported after its holdoff. Nothing is checked while a player is paused or
        stopped.
        """
        clients = [c for c in clients if c.maintain_sync]
        if len(clients) < 2:
            return
        # A paused or stopped player's play point pairs a fresh clock with a frozen
        # position, so its apparent start moves with time. A player that rebuffers
        # after an underrun stays BUFFERING while it plays on, so it is checked.
        if any(
            client.state not in (PlayerState.PLAYING, PlayerState.BUFFERING)
            for client in clients
        ):
            return

        now = now if now is not None else time.monotonic()
        if now < self.next_check:
            return
        self.next_check = now + CHECK_SYNC_INTERVAL
        previous, self._deviating = self._deviating, {}

        # need a recent play-point from all players, otherwise give up; the first
        # check after a start or resume only takes play points from after its holdoff
        recent_threshold = max(
            now - PLAYPOINT_RECENT_THRESHOLD, self._holdoff_until or 0.0
        )
        play_points: list[tuple[SlimClient, float]] = []
        status_times: dict[str, float] = {}
        for client in clients:
            play_point = client.play_point
            if play_point is None:
                LOGGER.debug("%s bailing as no playPoint", client.player_id)
                return
            if play_point[0] > recent_threshold:
                play_points.append((client, play_point[1] + client.play_delay / 1000))
                status_times[client.player_id] = play_point[0]
            else:
                LOGGER.debug(
                    "%s bailing as playPoint too old: %.1fs",
                    client.player_id,
                    now - play_point[0],
                )
                return
        if not play_points:
            return
        first_check = self._holdoff_until is not None
        self._holdoff_until = None

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
        reference_client, reference_time = play_points[reference]

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
            # Deliberate deviation from _CheckSync, which corrects on a single play
            # point: one noisy reading would trigger an audible skip, so wait until
            # the next check confirms the deviation on the same side, from new
            # readings of both this player and the reference. The first check after
            # a start or resume only takes readings from after its holdoff, so it
            # corrects a member that started off right away.
            behind = index < reference
            reading = (
                behind,
                status_times[client.player_id],
                status_times[reference_client.player_id],
            )
            last = previous.get(client.player_id)
            if not first_check and (
                last is None
                or last[0] != behind
                or last[1] == reading[1]
                or last[2] == reading[2]
            ):
                self._deviating[client.player_id] = reading
                continue
            if behind:
                LOGGER.info("%s resync: skipAhead %dms", client.player_id, delta * 1000)
                await client.skip_over(int(delta * 1000))
                self.next_check += 1
            elif client.can_pause_for:
                # Currently unreachable: every client reports can_skip_ahead, so the
                # reference above is always the most-ahead player and only the
                # skipAhead branch runs. Becomes live once a device class reports
                # can_skip_ahead=False (LMS gates that on SB1/SliMP3). Kept for LMS
                # parity; the WiiM case is handled server-side with a pre-start pause.
                LOGGER.info("%s resync: pauseFor %dms", client.player_id, delta * 1000)
                await client.pause_for(int(delta * 1000))
                self.next_check += delta
