# crypto-pip-trader

BTCUSDT USDT-M 永续。原 PIP+K-Means 形态挖掘已证伪；通过门禁的是预先指定的 30 日时间序列动量，仓位由**训练窗回撤**标定。

## 结论

| 策略 | 门禁 |
|---|---|
| PIP 形态挖掘 | 失败（置换 p=0.33） |
| **30d 动量 + 训练窗标定仓位 0.245** | **通过** |
| 30d 多/空仓 + 同仓位 | 通过（并列） |

冻结规格：`configs/passed.yaml`、`models/passed.json`。禁止用 OOS 再调参。

**仍未接交易所。** 纸交易 ≥2 周后再写 `live/broker_binance.py`。

## 通过规格（冻结）

- 信号：过去 720 根（30 日）收益符号，持有 720 根
- 仓位：训练窗 2020–2021 回撤 −40.8% → `0.5 * 0.20/0.408 = 0.245`
- 成交：下一根 open，taker 4bps + 1bp 滑点 + funding
- OOS 2022-01→2026-08：**+30.8%**，Martin 4.32，回撤 **14.8%**，28 笔，费 6.2，块置换 **p=0.048**

并列通过：`long_flat`（下跌空仓）+ 同仓位，+29.3%，回撤 14.5%，p=0.048。

**PIP 形态走样提前平**（fit&lt;0，同一套 OOS + 块置换）：+56.3%，Martin 7.61，回撤 14.3%，58 笔，**p=0.048，门禁通过**。Testnet 出场已切到 `exit_mode: pip`。`python scripts/compare_pip_exit.py`

## 硬门槛

扣完成本后：净收益>0、Martin>0、交易≥5、跑赢买入持有、回撤<25%、块置换 p<0.05。

## 流程

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest tests -q
python scripts/run_baseline.py
python scripts/run_paper.py
```

纸交易回放已与冻结规格对齐：信号一致率 100%，收益 +30.8%，回撤 14.8%。

公开行情纸交易（**不需要 API Key，不下单**）：

```bash
python scripts/run_live_paper.py
```

当前本机期货 REST 被 451 拦截，脚本走 `data-api.binance.vision` 现货 1H 作为代理。状态写在 `reports/live_paper_state.json`，模拟订单写在 `reports/live_paper_orders.csv`（FILLED，含 clientOrderId / 滑点 / 手续费）。每根已收盘 1H 跑一次即可（可用 cron）。≥2 周后再考虑带 Key 的实盘。

币安期货 **Testnet**（本机当前连不上，超时）：

```bash
export BINANCE_TESTNET_API_KEY=...
export BINANCE_TESTNET_API_SECRET=...
python scripts/probe_testnet.py
python scripts/run_testnet.py
```

`run_testnet.py` 会在 Testnet 发真实市价单（单向持仓、isolated、杠杆≤2）。Key 只走环境变量，禁止写入仓库。主网下单未实现。

看日志 / 看板：

```bash
cat reports/live_paper_state.json
tail -20 reports/live_paper_log.csv
python scripts/serve_dashboard.py   # http://127.0.0.1:8765
```

## 已跑对照

| 检查 | 结果 |
|---|---|
| PIP 原参数 | −35%，p=0.33 |
| PIP + 折过滤 | +5.8%，p=0.33 |
| 7d 动量 | +22%，回撤 40%，p=0.38 |
| 30d 动量 0.5x | +60%，回撤 27.8%，p=0.048，回撤超线 |
| 30d + 15% 硬停机 | −12%，第 2 笔停机 |
| **30d 标定 0.245x** | **+30.8%，回撤 14.8%，p=0.048，通过** |
| 买入持有 | +10%，回撤 42% |
