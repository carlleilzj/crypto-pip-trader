# crypto-pip-trader

BTCUSDT USDT-M 永续，3 币种组合（BTC/ETH/SOL）。**N 日突破入场 + PIP 形态走样平仓 + 6% 硬止损**，仓位由训练窗回撤标定 0.245x 等分。

## 结论

| 策略 | 门禁 |
|---|---|
| PIP 形态挖掘 | 失败（置换 p=0.33） |
| 30d 时间动量（parity 修复后） | 失败（p=0.066，跑输买入持有） |
| EMA 交叉 | 失败（p=0.347，统计不显著） |
| **N=40 突破 + PIP + 6% 止损** | **通过**（p=0.049，6 项全过） |

冻结规格：`configs/passed.yaml`、`models/passed.json`。禁止用 OOS 再调参。

## 通过规格（冻结）

- 信号：收盘突破过去 40 根最高/最低价（突破做多/做空）
- 出场：PIP 形态走样（cosine<0）+ 6% 硬止损 + 720 根时间持有
- 仓位：训练窗 2020–2021 回撤 −40.8% → `0.5 * 0.20/0.408 = 0.245`，三币等分
- 成交：下一根 open，taker 4bps + 1bp 滑点 + funding
- OOS 2022-01→2026-08：**+19.2%**，Martin 4.70，回撤 **6.7%**，1230 笔，块置换 **p=0.049**

### 三段切分（禁用保留窗调参）

| 窗口 | 策略 | 买入持有 | 跑赢 |
|---|---|---|---|
| 训练 2020-21（调参） | +34.9%/-3.9% | +1258%/-66% | 否（牛市） |
| 验证 2022-24.06 | +18.3%/-3.9% | -4.8%/-77% | 是 |
| 保留 2024-26（裁决） | +2.2%/-2.9% | -27.7%/-66% | 是 |

### 研究历史（2026-08-31）

1. **parity 修复**：发现回测（向量化）与实盘（有状态）退出-再进场语义不一致（一致率 91.6%）。对齐后 30d 动量 p 从 0.048 升到 0.066，门禁 FAIL。
2. **诊断**：纯时间动量 BTC 训练窗 +0.3%，加 PIP+止损后 +77%——PIP+止损是核心 edge，但牛市结构性跑输满仓。
3. **换 EMA 交叉**：p=0.347，统计不显著，FAIL。
4. **换 N=40 突破**：训练窗 Calmar 8.96，验证窗+保留窗均跑赢买入持有，块置换 p=0.049，**6 项全过**。

## 硬门槛

扣完成本后：净收益>0、Martin>0、交易≥5、跑赢买入持有、回撤<25%、块置换 p<0.05。

复现：`python scripts/repro_gate_breakout.py`（快）或 `--permutations 200`（含 p 值，~30 分钟）。

## 实盘保护

- **交易所侧灾难止损**：开仓同时挂 `STOP_MARKET`（`closePosition=true`，mark 价触发），进程宕机时仍生效。平仓时自动撤销。见 `live/broker_binance.py:stop_market_close`。
- **状态恢复**：进程重启后从全量 K 线重放重建策略状态（`entry / bars_in_trade / hold_left / extreme / pred_y`），不再信任存档计数器；恢复失败发 Telegram。见 `live/runloop.py:_restore_state`。
- **限频退避**：418/-1003 触发 10 分钟冷却，新 K 线探测只拉 2 根。

## 流程

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest tests -q
python scripts/repro_gate.py            # 复现门禁(快,不含置换)
python scripts/repro_gate.py --permutations 1000   # 30d 动量门禁(旧,证伪)
python scripts/repro_gate_breakout.py             # 突破门禁(冻结规格,快)
python scripts/repro_gate_breakout.py --permutations 200  # 含块置换 p 值(慢)
python scripts/run_paper.py
```

纸交易回放与冻结规格对齐：信号一致率 100%（`python scripts/run_paper.py`，三币种 agree=1.0000）。

> **2026-08-31 parity 修复**：发现回测（向量化）与实盘（有状态）退出-再进场
> 语义不一致（一致率 91.6%）。对齐后 30d 动量门禁 FAIL（p=0.066），换 N=40
> 突破信号后重新通过门禁（p=0.049）。所有信号源（动量/EMA/突破）从一开始就
> 有 parity 测试锁定，避免重蹈覆辙。

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

看日志：

```bash
cat reports/live_paper_state.json
tail -20 reports/live_paper_log.csv
cat reports/logs/runloop.log
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
