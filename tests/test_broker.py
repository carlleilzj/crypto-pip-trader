import os

import live.broker_binance as bb
from live.broker_binance import BinanceUSDMBroker, load_env_file, probe_testnet


def test_probe_returns_dict():
    out = probe_testnet(timeout=2.0)
    assert "reachable" in out
    assert "url" in out


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("BINANCE_TESTNET_API_KEY=fromfile\nOTHER_FLAG=1\n")
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "existing")
    monkeypatch.delenv("OTHER_FLAG", raising=False)
    load_env_file(env)
    assert os.environ["BINANCE_TESTNET_API_KEY"] == "existing"
    assert os.environ["OTHER_FLAG"] == "1"


def test_round_qty_without_network(monkeypatch):
    b = object.__new__(BinanceUSDMBroker)
    b._filters = {
        "BTCUSDT": {
            "quantityPrecision": 4,
            "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"}],
        }
    }
    assert b.round_qty("BTCUSDT", 0.00328634) == 0.0032
    assert b.round_qty("BTCUSDT", 0.00004) == 0.0


def test_is_transient_error():
    assert bb._is_transient_error(RuntimeError('binance 500: {"code":-1000,"msg":"internal error"}'))
    assert bb._is_transient_error(RuntimeError("binance 503: service unavailable"))
    assert not bb._is_transient_error(RuntimeError("binance 400: {\"code\":-2019,\"msg\":\"margin is insufficient\"}"))
    assert not bb._is_transient_error(RuntimeError("binance 401: bad key"))
    assert not bb._is_transient_error(ValueError("nope"))


def _mk_broker() -> BinanceUSDMBroker:
    b = object.__new__(BinanceUSDMBroker)
    b.api_key = "k"
    b.api_secret = "s"
    b.timeout = 0.1
    b.base = bb.TESTNET_FAPI
    return b


class _R:
    """Minimal response double; subclasses set status_code/text/headers."""

    status_code = 200
    text = ""
    headers: dict = {}

    def json(self):
        return {}

    def raise_for_status(self):
        return None


def test_public_keeps_body_and_retry_after(monkeypatch):
    """A 418 from klines must carry the body (ban deadline) + Retry-After.

    Regression for 2026-09-05 08:40 UTC: raise_for_status() dropped the
    response body, so the runloop never saw 'banned until <ms>' and fell
    back to a fixed 600s window.
    """
    b = _mk_broker()

    class R(_R):
        status_code = 418
        text = '{"code":-1003,"msg":"Way too many requests; IP banned until 1788598218000"}'
        headers = {"Retry-After": "119"}

    monkeypatch.setattr(bb.requests, "get", lambda *a, **k: R())
    try:
        b._public("/fapi/v1/klines", {"symbol": "BTCUSDT"})
    except RuntimeError as e:
        msg = str(e)
        assert "binance 418" in msg
        assert "banned until 1788598218000" in msg
        assert "retry-after 119" in msg
    else:
        raise AssertionError("expected RuntimeError for 418")


def test_public_ok_passes_through(monkeypatch):
    b = _mk_broker()

    class R(_R):
        status_code = 200
        text = '[]'

        def json(self):
            return []

    monkeypatch.setattr(bb.requests, "get", lambda *a, **k: R())
    assert b._public("/fapi/v1/klines", {}) == []


def test_signed_retries_transient_5xx(monkeypatch):
    b = _mk_broker()
    calls = {"n": 0}
    sleeps: list[float] = []

    class R:
        status_code = 500
        text = '{"code":-1000,"msg":"internal error"}'

    monkeypatch.setattr(bb.requests, "request", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or R())
    monkeypatch.setattr(bb.time, "sleep", lambda s: sleeps.append(s))
    try:
        b._signed("GET", "/fapi/v2/account", {})
    except RuntimeError as e:
        assert "binance 500" in str(e)
    assert calls["n"] == bb._MAX_RETRIES + 1
    assert sleeps == [bb._RETRY_SLEEP_S] * bb._MAX_RETRIES


def test_signed_no_retry_on_client_error(monkeypatch):
    b = _mk_broker()
    calls = {"n": 0}

    class R:
        status_code = 400
        text = '{"code":-2019,"msg":"margin is insufficient"}'

    monkeypatch.setattr(bb.requests, "request", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or R())
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    try:
        b._signed("POST", "/fapi/v1/order", {})
    except RuntimeError as e:
        assert "binance 400" in str(e)
    assert calls["n"] == 1


def test_signed_retry_recovers(monkeypatch):
    b = _mk_broker()
    calls = {"n": 0}

    class R500:
        status_code = 502
        text = "bad gateway"

    class ROk:
        status_code = 200
        text = '{"ok":true}'

        def json(self):
            return {"ok": True}

    def fake_request(*a, **k):
        calls["n"] += 1
        return R500() if calls["n"] == 1 else ROk()

    monkeypatch.setattr(bb.requests, "request", fake_request)
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    out = b._signed("GET", "/fapi/v2/account", {})
    assert out == {"ok": True}
    assert calls["n"] == 2
