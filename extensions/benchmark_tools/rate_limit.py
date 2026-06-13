"""Explicit rate limiting and retry primitives for benchmark orchestration.

These helpers deliberately live above provider clients. Callers should keep
LiteLLM retries disabled and wrap each model request with :func:`retry_call`.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar


T = TypeVar("T")
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


@dataclass
class _Reservation:
    timestamp: float
    tokens: int


@dataclass(frozen=True)
class LimiterMetadata:
    """Scheduling data suitable for attaching to an attempt record."""

    wait_seconds: float
    estimated_tokens: int


class RateLimitLease:
    """An acquired in-flight slot and its rate-budget reservation.

    ``reconcile_tokens`` replaces the estimated TPM reservation with actual
    usage. A lower value refunds capacity immediately; a higher value records
    the additional usage at the original reservation timestamp. Reconciliation
    does not block a completed request, but excess actual usage delays future
    acquisitions until it leaves the rolling window.
    """

    def __init__(
        self,
        limiter: AsyncRateLimiter,
        reservation: _Reservation,
        metadata: LimiterMetadata,
    ) -> None:
        self._limiter = limiter
        self._reservation = reservation
        self.metadata = metadata
        self._released = False

    async def reconcile_tokens(self, actual_tokens: int) -> None:
        if actual_tokens < 0:
            raise ValueError("actual_tokens must be non-negative")
        async with self._limiter._rate_lock:
            self._reservation.tokens = actual_tokens

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._limiter._in_flight.release()

    async def __aenter__(self) -> RateLimitLease:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()


class AsyncRateLimiter:
    """Limit global in-flight requests and rolling one-minute budgets."""

    def __init__(
        self,
        *,
        max_in_flight: int,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if requests_per_minute is not None and requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        if tokens_per_minute is not None and tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")
        self.max_in_flight = max_in_flight
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._in_flight = asyncio.Semaphore(max_in_flight)
        self._rate_lock = asyncio.Lock()
        self._reservations: deque[_Reservation] = deque()
        self._cooldown_until = 0.0

    async def impose_cooldown(self, seconds: float) -> None:
        """Delay all future acquisitions after a provider-wide rate limit."""

        if seconds < 0:
            raise ValueError("cooldown seconds must be non-negative")
        async with self._rate_lock:
            self._cooldown_until = max(
                self._cooldown_until,
                self._clock() + seconds,
            )

    async def cap_tokens_per_minute(
        self,
        provider_limit: int,
        *,
        headroom_ratio: float = 0.9,
    ) -> int:
        """Lower the local TPM ceiling when the provider reports a smaller one."""

        if provider_limit < 1:
            raise ValueError("provider_limit must be positive")
        if not 0 < headroom_ratio <= 1:
            raise ValueError("headroom_ratio must be between 0 and 1")
        adjusted_limit = max(1, int(provider_limit * headroom_ratio))
        async with self._rate_lock:
            if (
                self.tokens_per_minute is None
                or adjusted_limit < self.tokens_per_minute
            ):
                self.tokens_per_minute = adjusted_limit
            return self.tokens_per_minute

    async def acquire(self, *, estimated_tokens: int = 0) -> RateLimitLease:
        """Reserve rate budget, then acquire a global in-flight slot.

        Cancellation while waiting removes the unused rate reservation and
        never leaks an in-flight permit.
        """

        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative")
        if (
            self.tokens_per_minute is not None
            and estimated_tokens > self.tokens_per_minute
        ):
            raise ValueError("estimated_tokens exceeds tokens_per_minute")

        started = self._clock()
        reservation = await self._reserve_rate_budget(estimated_tokens)
        try:
            await self._in_flight.acquire()
        except BaseException:
            async with self._rate_lock:
                try:
                    self._reservations.remove(reservation)
                except ValueError:
                    pass
            raise
        return RateLimitLease(
            self,
            reservation,
            LimiterMetadata(
                wait_seconds=max(0.0, self._clock() - started),
                estimated_tokens=estimated_tokens,
            ),
        )

    async def _reserve_rate_budget(self, estimated_tokens: int) -> _Reservation:
        while True:
            async with self._rate_lock:
                now = self._clock()
                self._prune(now)
                delay = max(
                    self._cooldown_until - now,
                    self._required_delay(now, estimated_tokens),
                )
                if delay <= 0:
                    reservation = _Reservation(now, estimated_tokens)
                    self._reservations.append(reservation)
                    return reservation
            await self._sleep(delay)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._reservations and self._reservations[0].timestamp <= cutoff:
            self._reservations.popleft()

    def _required_delay(self, now: float, estimated_tokens: int) -> float:
        delays = [0.0]
        if (
            self.requests_per_minute is not None
            and len(self._reservations) >= self.requests_per_minute
        ):
            index = len(self._reservations) - self.requests_per_minute
            delays.append(self._reservations[index].timestamp + 60.0 - now)

        if self.tokens_per_minute is not None:
            token_total = sum(item.tokens for item in self._reservations)
            excess = token_total + estimated_tokens - self.tokens_per_minute
            if excess > 0:
                released = 0
                for item in self._reservations:
                    released += item.tokens
                    if released >= excess:
                        delays.append(item.timestamp + 60.0 - now)
                        break
        return max(delays)


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    reason: str
    status_code: int | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class RetryEvent:
    """Metadata emitted before an orchestration-level retry."""

    delay_seconds: float
    reason: str
    retry_index: int
    attempt_number: int
    status_code: int | None = None
    retry_after_seconds: float | None = None
    limiter_wait_seconds: float = 0.0


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay(
        self,
        retry_index: int,
        *,
        retry_after_seconds: float | None = None,
        random_value: float | None = None,
    ) -> float:
        if retry_index < 1:
            raise ValueError("retry_index must be positive")
        base = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (retry_index - 1)),
        )
        value = random.random() if random_value is None else random_value
        if not 0 <= value <= 1:
            raise ValueError("random_value must be between 0 and 1")
        multiplier = 1 - self.jitter_ratio + (2 * self.jitter_ratio * value)
        backoff = min(self.max_delay_seconds, base * multiplier)
        return max(backoff, retry_after_seconds or 0.0)


class RetryableContractError(ValueError):
    """A model response violated a contract and may succeed on retry."""


def classify_retry(
    error: BaseException,
    *,
    wall_clock: Clock = time.time,
) -> RetryDecision:
    """Classify known transient failures without retrying arbitrary bugs."""

    if isinstance(error, asyncio.CancelledError):
        return RetryDecision(False, "cancelled")
    if isinstance(error, RetryableContractError):
        return RetryDecision(True, "invalid_model_output")
    if isinstance(
        error, (ValueError, TypeError, KeyError, AssertionError, AttributeError)
    ):
        return RetryDecision(False, "contract_or_programming_error")

    status_code = _status_code(error)
    retry_after = _retry_after(error, wall_clock=wall_clock)
    if status_code in {408, 409, 429}:
        return RetryDecision(
            True,
            f"http_{status_code}",
            status_code,
            retry_after,
        )
    if status_code is not None and 500 <= status_code <= 599:
        return RetryDecision(True, "http_5xx", status_code, retry_after)
    if isinstance(error, (TimeoutError, ConnectionError)):
        return RetryDecision(True, type(error).__name__.lower())
    return RetryDecision(False, "non_transient_error", status_code, retry_after)


def provider_tpm_limit(error: BaseException) -> int | None:
    """Read a provider-reported TPM ceiling from a normalized rate-limit error."""

    record = getattr(error, "record", None)
    record_error = getattr(record, "error", None)
    messages = [str(error), str(getattr(record_error, "message", "") or "")]
    for message in messages:
        match = re.search(
            r"tokens per min(?:ute)?\s*\(TPM\).*?Limit\s+([0-9][0-9,]*)",
            message,
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1).replace(",", ""))
    return None


async def retry_call(
    operation: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    sleep: Sleep = asyncio.sleep,
    random_source: Callable[[], float] = random.random,
    on_retry: Callable[[RetryEvent], Any] | None = None,
    limiter_wait: Callable[[], float] | None = None,
) -> T:
    """Run ``operation(attempt_number)`` with visible orchestration retries."""

    for attempt_number in range(1, policy.max_attempts + 1):
        try:
            return await operation(attempt_number)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            decision = classify_retry(error)
            if not decision.retryable or attempt_number >= policy.max_attempts:
                raise
            retry_index = attempt_number
            delay = policy.delay(
                retry_index,
                retry_after_seconds=decision.retry_after_seconds,
                random_value=random_source(),
            )
            event = RetryEvent(
                delay_seconds=delay,
                reason=decision.reason,
                retry_index=retry_index,
                attempt_number=attempt_number,
                status_code=decision.status_code,
                retry_after_seconds=decision.retry_after_seconds,
                limiter_wait_seconds=limiter_wait() if limiter_wait else 0.0,
            )
            if on_retry is not None:
                result = on_retry(event)
                if inspect.isawaitable(result):
                    await result
            await sleep(delay)
    raise AssertionError("unreachable")


def _status_code(error: BaseException) -> int | None:
    details = _error_details(error)
    candidates = (
        error,
        getattr(error, "__cause__", None),
        getattr(error, "response", None),
        details,
    )
    for candidate in candidates:
        value = (
            candidate.get("status_code")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "status_code", None)
        )
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _retry_after(error: BaseException, *, wall_clock: Clock) -> float | None:
    value = getattr(error, "retry_after", None)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    details = _error_details(error)
    if value is None and isinstance(details, Mapping):
        value = details.get("retry_after")
        headers = headers or details.get("headers")
    if value is None and isinstance(headers, Mapping):
        value = next(
            (item for key, item in headers.items() if key.lower() == "retry-after"),
            None,
        )
    if value is None:
        message = str(error)
        match = re.search(
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|seconds?)",
            message,
            re.IGNORECASE,
        )
        if match:
            parsed = float(match.group(1))
            return parsed / 1000.0 if match.group(2).lower().startswith("m") else parsed
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value)).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, parsed - wall_clock())


def _error_details(error: BaseException) -> Mapping[str, Any] | None:
    record = getattr(error, "record", None)
    record_error = getattr(record, "error", None)
    details = getattr(record_error, "details", None)
    return details if isinstance(details, Mapping) else None
