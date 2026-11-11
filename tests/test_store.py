from concurrent.futures import ThreadPoolExecutor
import pytest

from flux_image_server.errors import ServiceError
from flux_image_server.store import Store
from conftest import OWNER, OTHER


def test_concurrent_idempotency_never_starts_duplicate_work(store):
    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(lambda _: store.submit(OWNER, "same", "A boat", True), range(8)))
    assert len({job["id"] for job in jobs}) == 1
    assert store.snapshot()[0] == {"generating": 1}
    assert store.quota(OWNER) == 1
    assert store.submit(OWNER, "same", "A boat", False)["id"] == jobs[0]["id"]
    with pytest.raises(ServiceError, match="another prompt"):
        store.submit(OWNER, "same", "Different", True)


def test_busy_global_and_account_capacity_no_waiting_queue(store):
    job = store.submit(OWNER, "one", "A boat", True)
    for account, code in ((OWNER, "user_capacity"), (OTHER, "busy")):
        with pytest.raises(ServiceError) as result:
            store.submit(account, "two", "Another boat", True)
        assert result.value.code == code and result.value.status == 429
    assert store.snapshot()[0] == {"generating": 1}
    assert store.quota(OTHER) == 0
    assert store.claim()["id"] == job["id"]
    assert store.claim() is None


def test_ownership_on_inspection_and_cancellation(store):
    job = store.submit(OWNER, "one", "A boat", True)
    for action in (store.owned, store.cancel):
        with pytest.raises(ServiceError) as result:
            action(OTHER, job["id"])
        assert result.value.status == 404
    assert store.owned(OWNER, job["id"])["cancel"] is False


def test_twenty_admissions_and_utc_reset(config):
    now = [1794355200.0]
    store = Store(config, clock=lambda: now[0])
    for index in range(20):
        job = store.submit(OWNER, f"job{index}", "A boat", True)
        store.claim()
        store.finish(job["id"], "failed", "generation_error")
    with pytest.raises(ServiceError) as result:
        store.submit(OWNER, "blocked", "A boat", True)
    assert result.value.code == "daily_quota"
    now[0] += 86400
    store.submit(OWNER, "tomorrow", "A boat", True)
    assert store.quota(OWNER) == 1


def test_restart_loses_metadata_and_quota_without_reexecuting(config):
    original = Store(config)
    job = original.submit(OWNER, "active", "A boat", True)
    original.claim()
    reopened = Store(config)
    reopened.recover()
    assert reopened.claim() is None
    assert reopened.quota(OWNER) == 0
    with pytest.raises(ServiceError) as result:
        reopened.owned(OWNER, job["id"])
    assert result.value.status == 404


def test_expiration_removes_output_and_tombstone_blocks_replay(config):
    now = [1794355200.0]
    store = Store(config, clock=lambda: now[0])
    job = store.submit(OWNER, "one", "A boat", True)
    store.claim()
    path = store.image_path(job["id"])
    path.write_bytes(b"image")
    store.finish(job["id"], "completed")
    now[0] += 86401
    store.cleanup()
    assert not path.exists()
    with pytest.raises(ServiceError) as result:
        store.submit(OWNER, "one", "A boat", True)
    assert result.value.status == 410


def test_atomic_output_completion_wins_late_cancel(store):
    job = store.submit(OWNER, "one", "A boat", True)
    store.claim()
    assert store.cancel(OWNER, job["id"])["cancel"]
    store.image_path(job["id"]).write_bytes(b"produced")
    assert store.finish(job["id"], "completed") == "completed"
    assert store.image_path(job["id"]).exists()
