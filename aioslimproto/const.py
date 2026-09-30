"""Constants for aioslimproto."""

from __future__ import annotations

SLIMPROTO_PORT = 3483
FALLBACK_CODECS = ["pcm", "mp3"]

FALLBACK_MODEL = "Squeezebox"
FALLLBACK_FIRMWARE = "Unknown"
FALLBACK_SAMPLE_RATE = 96000
HEARTBEAT_INTERVAL = 5

# ---------------------------------------------------------------------------
# Synchronisation constants, ported 1:1 from Lyrion Music Server
# (Slim::Player::StreamingController / Slim::Player::Player; the check-sync
# constants are also defined in Slim::Player::Sync)
# ---------------------------------------------------------------------------

# StreamingController::_CheckSync
CHECK_SYNC_INTERVAL = 0.950  # seconds between sync checks per group
MIN_DEVIATION_ADJUST = 0.010  # only correct deviations larger than this (s)
MAX_DEVIATION_ADJUST = 10.000  # ignore absurd deviations (s)
PLAYPOINT_RECENT_THRESHOLD = 3.0  # play point must be newer than this (s)

# Player::publishPlayPoint - weighted play points. These are only needed for the
# older players (SB1/SliMP3); kept here (unused for now) until that path is implemented.
# PLAY_POINT_LIST_SIZE = 8  # noqa: ERA001
# MAX_STARTTIME_VARIATION = 0.015  # s  # noqa: ERA001

# Player::trackJiffiesEpoch
JIFFIES_OFFSET_TRACKING_LIST_SIZE = 50
JIFFIES_OFFSET_TRACKING_LIST_MIN = 10
JIFFIES_EPOCH_MIN_ADJUST = 0.001
JIFFIES_EPOCH_MAX_ADJUST = 0.005

# per-player pref 'packetLatency' (Player.pm), applied in jiffiesToTimestamp
PACKET_LATENCY = 0.002  # s

# per-player sync defaults. Player.pm base (software players and unknown
# devices); Squeezebox2.pm uses the tighter value for Squeezebox2-class hardware.
DEFAULT_MIN_SYNC_ADJUST = 30  # ms
HARDWARE_MIN_SYNC_ADJUST = 10  # ms
DEFAULT_PLAY_DELAY = 0  # ms
DEFAULT_START_DELAY = 0  # ms

# StreamingController::_syncStart
SYNC_START_DELAY = 200  # ms

# StreamingController::_Resume: nextCheckSyncTime = startAtBase + SYNC_RESUME_HOLDOFF
SYNC_RESUME_HOLDOFF = 2.0  # s
