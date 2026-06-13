from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from extensions.benchmark_tools.rate_limit import (
    AsyncRateLimiter,
    RetryPolicy,
    classify_retry,
    provider_tpm_limit,
    retry_call,
)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


@pytest.mark.asyncio
async def test_global_concurrency_cap():
    limiter = AsyncRateLimiter(max_in_flight=2)
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def worker() -> None:
        nonlocal active, maximum
        async with await limiter.acquire():
            active += 1
            maximum = max(maximum, active)
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(4)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert maximum == 2
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_rpm_pacing_is_deterministic():
    fake = FakeTime()
    limiter = AsyncRateLimiter(
        max_in_flight=2,
        requests_per_minute=2,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    first = await limiter.acquire()
    first.release()
    second = await limiter.acquire()
    second.release()
    third = await limiter.acquire()
    third.release()

    assert fake.sleeps == [60.0]
    assert third.metadata.wait_seconds == 60.0


@pytest.mark.asyncio
async def test_token_estimate_can_be_reconciled_and_refunded():
    fake = FakeTime()
    limiter = AsyncRateLimiter(
        max_in_flight=1,
        tokens_per_minute=100,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    lease = await limiter.acquire(estimated_tokens=80)
    await lease.reconcile_tokens(20)
    lease.release()
    next_lease = await limiter.acquire(estimated_tokens=80)
    next_lease.release()

    assert fake.sleeps == []


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503, 599])
def test_retry_classification_accepts_transient_http_errors(status_code):
    error = RuntimeError("provider failure")
    error.status_code = status_code

    decision = classify_retry(error)

    assert decision.retryable
    assert decision.status_code == status_code


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad input"),
        TypeError("bad call"),
        KeyError("missing"),
        AssertionError("bug"),
        RuntimeError("unknown"),
    ],
)
def test_retry_classification_rejects_contract_and_programming_errors(error):
    assert not classify_retry(error).retryable


def test_retry_classification_reads_normalized_call_record_details():
    error = RuntimeError("normalized provider failure")
    error.record = SimpleNamespace(error=SimpleNamespace(details={"status_code": 429}))

    decision = classify_retry(error)

    assert decision.retryable
    assert decision.reason == "http_429"


def test_retry_after_is_parsed_from_provider_message():
    error = RuntimeError("Rate limited. Please try again in 360ms.")
    error.status_code = 429

    decision = classify_retry(error)

    assert decision.retry_after_seconds == pytest.approx(0.36)


@pytest.mark.asyncio
async def test_retry_after_overrides_backoff_and_emits_metadata():
    events = []
    sleeps = []
    attempts = 0

    async def operation(attempt_number: int) -> str:
        nonlocal attempts
        attempts = attempt_number
        if attempt_number == 1:
            error = RuntimeError("limited")
            error.status_code = 429
            error.response = SimpleNamespace(headers={"Retry-After": "7"})
            raise error
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = await retry_call(
        operation,
        policy=RetryPolicy(
            max_attempts=2,
            base_delay_seconds=1,
            jitter_ratio=0,
        ),
        sleep=sleep,
        on_retry=events.append,
        limiter_wait=lambda: 2.5,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [7.0]
    assert events[0].delay_seconds == 7.0
    assert events[0].reason == "http_429"
    assert events[0].retry_index == 1
    assert events[0].limiter_wait_seconds == 2.5


@pytest.mark.asyncio
async def test_retry_stops_at_max_attempts():
    attempts = 0
    sleeps = []

    async def operation(attempt_number: int) -> None:
        nonlocal attempts
        attempts = attempt_number
        raise ConnectionError("offline")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(ConnectionError, match="offline"):
        await retry_call(
            operation,
            policy=RetryPolicy(max_attempts=3, jitter_ratio=0),
            sleep=sleep,
        )

    assert attempts == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_cancelled_limiter_wait_does_not_leak_permit():
    limiter = AsyncRateLimiter(max_in_flight=1)
    first = await limiter.acquire()
    waiting = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    first.release()
    lease = await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    lease.release()


@pytest.mark.asyncio
async def test_global_cooldown_delays_future_acquisitions():
    fake = FakeTime()
    limiter = AsyncRateLimiter(
        max_in_flight=1,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    await limiter.impose_cooldown(2.5)
    lease = await limiter.acquire()
    lease.release()

    assert fake.sleeps == [2.5]


@pytest.mark.asyncio
async def test_provider_tpm_cap_only_reduces_the_configured_limit():
    limiter = AsyncRateLimiter(max_in_flight=1, tokens_per_minute=1_500_000)

    assert await limiter.cap_tokens_per_minute(200_000) == 180_000
    assert await limiter.cap_tokens_per_minute(2_000_000) == 180_000


def test_provider_tpm_limit_is_parsed_from_rate_limit_message():
    error = RuntimeError(
        "Rate limit reached on tokens per min (TPM): "
        "Limit 200,000, Used 200000."
    )

    assert provider_tpm_limit(error) == 200_000


@pytest.mark.asyncio
async def test_retry_cancellation_is_not_swallowed():
    attempts = 0

    async def operation(attempt_number: int) -> None:
        nonlocal attempts
        attempts = attempt_number
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_call(operation, policy=RetryPolicy(max_attempts=3))

    assert attempts == 1
