import json

from live.notify import (
    TelegramNotifier,
    fit_label,
    format_fill,
    format_hourly,
    format_leg,
    format_offline,
    format_online,
)


def test_fit_label_cooling_and_warn():
    assert "冷却" in fit_label(None, 7, 24)
    assert "走样" in fit_label(-0.1, 40, 24)
    assert "接近走样" in fit_label(0.1, 40, 24)
    assert "完好" in fit_label(0.7, 40, 24)


def test_format_hourly_contains_pnl_and_legs():
    text = format_hourly(
        {
            "acct": {
                "equity": 5059.1,
                "wallet": 4995.9,
                "unrealized": 63.2,
                "peak_equity": 5059.1,
                "day_start_equity": 5057.7,
                "halted": False,
            },
            "legs": [
                {
                    "symbol": "BTCUSDT",
                    "pos": 1,
                    "qty": 0.0159,
                    "entry": 76918.0,
                    "mark": 78399.0,
                    "upnl": 23.5,
                    "fit": None,
                    "bars_in_trade": 7,
                    "hold_left": 713,
                }
            ],
            "last_bar": "2026-08-26 18:00:00+00:00",
            "ping_ok": True,
            "host": "test-host",
            "loop_min": 5,
        }
    )
    assert "5059.10" in text
    assert "BTCUSDT 多" in text
    assert "停机 否" in text
    assert "形态" in text or "冷却" in text


def test_format_online_and_flat_leg():
    assert "盯仓上线" in format_online({"acct": {"equity": 1}, "legs": []})
    assert "空仓" in format_leg({"symbol": "ETHUSDT", "pos": 0})


def test_fill_and_offline_are_chinese():
    text = format_fill(
        {
            "symbol": "BTCUSDT",
            "action": "OPEN",
            "side": "BUY",
            "qty": 0.0159,
            "avg_price": 76918.0,
            "bar": "2026-08-26 18:00:00+00:00",
            "status": "FILLED",
        }
    )
    assert "开仓" in text and "买入" in text and "已成交" in text
    assert "OPEN" not in text and "BUY" not in text
    assert "盈利" not in text and "亏损" not in text
    assert "服务停止" in format_offline("signal 15")
    assert "signal 15" not in format_offline("signal 15")


def test_close_fill_shows_pnl():
    win = format_fill(
        {
            "symbol": "BTCUSDT",
            "action": "CLOSE",
            "side": "SELL",
            "qty": 0.0159,
            "avg_price": 0,
            "bar": "2026-08-27 11:00:00+00:00",
            "status": "FILLED",
            "pnl": 23.5,
            "pnl_pct": 0.0192,
        }
    )
    assert "平仓" in win
    assert "盈利 +23.50（+1.92%）" in win

    loss = format_fill(
        {
            "symbol": "ETHUSDT",
            "action": "CLOSE",
            "side": "BUY",
            "qty": 0.2,
            "avg_price": 3010.5,
            "bar": "2026-08-27 11:00:00+00:00",
            "status": "FILLED",
            "pnl": -5.234,
        }
    )
    assert "亏损 -5.23" in loss
    assert "%" not in loss.split("亏损")[1].splitlines()[0]

    no_pnl = format_fill(
        {
            "symbol": "BTCUSDT",
            "action": "CLOSE",
            "side": "SELL",
            "qty": 0.0159,
            "avg_price": 78400.0,
            "bar": "2026-08-27 11:00:00+00:00",
            "status": "FILLED",
        }
    )
    assert "盈利" not in no_pnl and "亏损" not in no_pnl


def test_notifier_disabled_without_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    n = TelegramNotifier(tmp_path)
    n.enabled = False
    assert n.send("hello") is False


def test_cooldown_skips_second_send(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    n = TelegramNotifier(tmp_path)
    sent = []

    def fake_post(url, json=None, timeout=0):
        sent.append(json["text"])

        class R:
            status_code = 200
            text = "{}"

        return R()

    monkeypatch.setattr("live.notify.requests.post", fake_post)
    assert n.send("a", key="k", cooldown_s=60) is True
    assert n.send("b", key="k", cooldown_s=60) is False
    assert sent == ["a"]
    st = json.loads((tmp_path / "notify_state.json").read_text())
    assert "k" in st["last"]
