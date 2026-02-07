from __future__ import annotations

"""Telegram notification transport and command listener utilities."""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from telegram import Bot
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.request import HTTPXRequest

logger = logging.getLogger(__name__)


@dataclass
class TelegramEvent:
    """Normalized inbound Telegram event for commands and reactions."""

    text: str
    chat_id: str
    message_id: Optional[int] = None
    reply_to_message_id: Optional[int] = None
    reaction_message_id: Optional[int] = None
    reaction_emojis: tuple[str, ...] = ()


class TelegramNotifier:
    """Thread-safe Telegram integration for media alerts and bot commands."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        """Initialize bot client and background asyncio loop when configured."""
        self.enabled = bool(bot_token and chat_id)
        self.chat_id = chat_id
        self.bot = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._listener_thread: threading.Thread | None = None
        self._listener_stop = threading.Event()
        self._update_offset = 0
        self._send_lock = threading.Lock()

        if self.enabled:
            request = HTTPXRequest(
                connection_pool_size=20,
                pool_timeout=30.0,
                connect_timeout=10.0,
                read_timeout=30.0,
                write_timeout=30.0,
            )
            self.bot = Bot(token=bot_token, request=request)
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_loop, name="telegram-loop", daemon=True)
            self._loop_thread.start()

    def _run_loop(self) -> None:
        """Run dedicated asyncio event loop for Telegram API calls."""
        if self._loop is None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _send_snapshot_async(self, image_path: Path, caption: str) -> int:
        """Asynchronously send one image alert message."""
        if not self.enabled:
            logger.info("Telegram not configured; skipping alert.")
            return 0

        if self.bot is None:
            return 0

        with image_path.open("rb") as photo:
            message = await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo,
                caption=caption,
                pool_timeout=30.0,
                connect_timeout=10.0,
                read_timeout=30.0,
                write_timeout=30.0,
            )
        return int(getattr(message, "message_id", 0) or 0)

    async def _send_text_async(self, text: str) -> None:
        """Asynchronously send plain-text message."""
        if not self.enabled or self.bot is None:
            return
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            pool_timeout=30.0,
            connect_timeout=10.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

    async def _get_updates_async(self, offset: int):
        """Poll Telegram updates for command processing."""
        if not self.enabled or self.bot is None:
            return []
        return await self.bot.get_updates(
            offset=offset,
            timeout=25,
            allowed_updates=["message", "message_reaction"],
        )

    def send_snapshot(self, image_path: Path, caption: str) -> Optional[int]:
        """Synchronously send a snapshot with retry/backoff semantics."""
        if not self.enabled:
            logger.info("Telegram not configured; skipping alert.")
            return None
        if self._loop is None:
            return None

        with self._send_lock:
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_snapshot_async(image_path=image_path, caption=caption),
                        self._loop,
                    )
                    message_id = int(future.result(timeout=90) or 0)
                    return message_id if message_id > 0 else None
                except RetryAfter as exc:
                    delay = float(getattr(exc, "retry_after", 2))
                    logger.warning("Telegram rate-limited; retrying in %.1fs (attempt %d/%d)", delay, attempt, attempts)
                    time.sleep(delay)
                except (TimedOut, NetworkError) as exc:
                    delay = 1.5 * attempt
                    logger.warning(
                        "Telegram send failed (%s); retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt,
                        attempts,
                    )
                    time.sleep(delay)
                except Exception as exc:
                    logger.exception("Unexpected Telegram error: %s", exc)
                    return None

        logger.error("Telegram send failed after %d attempts", attempts)
        return None

    def send_text(self, text: str) -> bool:
        """Synchronously send text message with retry/backoff semantics."""
        if not self.enabled:
            logger.info("Telegram not configured; skipping message.")
            return False
        if self._loop is None:
            return False

        with self._send_lock:
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_text_async(text=text),
                        self._loop,
                    )
                    future.result(timeout=60)
                    return True
                except RetryAfter as exc:
                    delay = float(getattr(exc, "retry_after", 2))
                    logger.warning("Telegram rate-limited; retrying in %.1fs (attempt %d/%d)", delay, attempt, attempts)
                    time.sleep(delay)
                except (TimedOut, NetworkError) as exc:
                    delay = 1.5 * attempt
                    logger.warning(
                        "Telegram message failed (%s); retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt,
                        attempts,
                    )
                    time.sleep(delay)
                except Exception as exc:
                    logger.exception("Unexpected Telegram error: %s", exc)
                    return False

        logger.error("Telegram text send failed after %d attempts", attempts)
        return False

    def start_command_listener(self, command_handler: Callable[[TelegramEvent], Optional[str]]) -> None:
        """Start polling commands and replying via provided handler callback."""
        if not self.enabled or self._loop is None:
            return
        if self._listener_thread is not None:
            return

        def _loop() -> None:
            # Restrict command handling to the configured chat for safety.
            allowed_chat = str(self.chat_id).strip()
            while not self._listener_stop.is_set():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._get_updates_async(offset=self._update_offset),
                        self._loop,
                    )
                    updates = future.result(timeout=40)
                except Exception:
                    logger.exception("Telegram command poll failed")
                    time.sleep(2.0)
                    continue

                for update in updates:
                    self._update_offset = int(update.update_id) + 1
                    message = getattr(update, "message", None)
                    if message is not None:
                        text = (getattr(message, "text", None) or "").strip()
                        incoming_chat = str(getattr(message, "chat_id", "")).strip()
                        if text and (not allowed_chat or incoming_chat == allowed_chat):
                            reply_to = None
                            reply_to_msg = getattr(message, "reply_to_message", None)
                            if reply_to_msg is not None:
                                reply_to = int(getattr(reply_to_msg, "message_id", 0) or 0) or None
                            event = TelegramEvent(
                                text=text,
                                chat_id=incoming_chat,
                                message_id=int(getattr(message, "message_id", 0) or 0) or None,
                                reply_to_message_id=reply_to,
                            )
                            reply = command_handler(event)
                            if reply:
                                self.send_text(reply)
                        continue

                    message_reaction = getattr(update, "message_reaction", None)
                    if message_reaction is None:
                        continue

                    reaction_chat = getattr(message_reaction, "chat", None)
                    incoming_chat = str(getattr(reaction_chat, "id", "")).strip()
                    if not incoming_chat:
                        incoming_chat = str(getattr(message_reaction, "chat_id", "")).strip()
                    if allowed_chat and incoming_chat != allowed_chat:
                        continue

                    emojis: list[str] = []
                    for reaction in getattr(message_reaction, "new_reaction", []) or []:
                        emoji = getattr(reaction, "emoji", "")
                        if emoji:
                            emojis.append(str(emoji))
                    if not emojis:
                        continue

                    event = TelegramEvent(
                        text="",
                        chat_id=incoming_chat,
                        reaction_message_id=int(getattr(message_reaction, "message_id", 0) or 0) or None,
                        reaction_emojis=tuple(emojis),
                    )
                    reply = command_handler(event)
                    if reply:
                        self.send_text(reply)

        self._listener_thread = threading.Thread(target=_loop, name="telegram-command-listener", daemon=True)
        self._listener_thread.start()

    def close(self) -> None:
        """Stop listener/event loop threads and release resources."""
        self._listener_stop.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2)
