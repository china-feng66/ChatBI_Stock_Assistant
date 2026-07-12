# QuantQuery-A：可审计的多 Agent A 股量化研究核心

QuantQuery-A 是一个从空目录独立实现的量化开发作品集项目。它受到课程《项目实战：ChatBI开发实战》的启发，但不复制课程的 Agent 主程序；课程复现仓库只作为学习记录。

项目目前专注四件可以被测试和面试追问的事情：

1. **安全取数**：对分析 SQL 实施代码级只读、单语句、行数和超时限制。
2. **可信回测时点**：T 日收盘确认信号，最早在 T+1 开盘成交，避免同收盘价生成信号并成交的前视问题。
3. **受限多 Agent 编排**：Data、Quant、Risk 三类角色通过 QueryFrame 协作，所有策略与回测结论必须通过 Risk Critic。
4. **可评测闭环**：提供隐私化 Trace、冻结合成案例和 `/eval/run`；默认总尝试最多 2 次，即最多重试 1 次。

它是研究和学习原型，不是实盘交易系统，不宣称发现 Alpha，也不构成投资建议。

## 已实现

- SQLite 只读 URI、`query_only`、单条 SELECT/CTE 策略。
- 查询超时、最大返回行数和截断标记。
- MACD 持续目标仓位生成。
- 单资产、日频、只做多回测。
- 整手、可配置佣金、最低佣金、卖出税费与滑点。
- 总收益、年化收益、波动率、Sharpe、最大回撤、基准、超额收益、换手和成本。
- 合成数据库与合成行情自动化测试；不读取真实股票数据库。
- 规则路由与可插拔的关键词/语义/LLM 三路评分接口。
- 快路径与 Data → Quant → Risk 计划路径。
- JSONL Trace 敏感字段脱敏、证据指纹和默认总尝试最多 2 次。
- FastAPI `/healthz`、`/analyze`、`/eval/run`。

## 结构

```text
src/quantquery_a/
├── agents/
│   ├── data_agent.py
│   ├── quant_agent.py
│   ├── risk_critic.py
│   └── supervisor.py
├── routing/        # 可校准的多路意图评分
├── evaluation/     # 冻结合成金标与确定性门禁
├── observability/  # 隐私化 JSONL Trace
├── api.py           # FastAPI 边界
├── query.py         # 只读 SQLite 查询
├── strategy.py      # MACD 目标仓位
└── backtest.py      # T+1 开盘执行回测

tests/               # 合成数据与闭环测试
examples/            # 不依赖真实行情的演示
docs/                # 来源边界与任务计划
```

## 验证

```bash
python -m pip install -e ".[dev,api]"
python -m pytest -q
python examples/synthetic_demo.py
uvicorn quantquery_a.api:app --host 127.0.0.1 --port 8000
```

## 多 Agent 闭环

复杂任务使用固定 DAG，而不是让 Agent 自由互聊：

```text
意图评分（默认规则，可插拔语义/LLM 两路）→ QueryFrame
  ├─ 简单行情：Data Agent 快路径
  └─ 复合研究：Data Agent → Quant Agent → Risk Critic
                                      ├─ 可重试故障：总尝试最多 2 次
                                      └─ 通过：证据门禁 → 报告
```

当前默认实现不依赖在线 LLM，方便离线复现和测试。语义向量分类、LLM 分类、Chroma 情景记忆、Redis 工作记忆和在线权重更新均为条件性后续项，不作为已完成功能。数据库路径只能由服务端 `QUANTQUERY_DB_PATH` 配置，请求不能指定本地文件路径。

任务计划与验收边界见 [docs/MULTI_AGENT_IMPLEMENTATION_PLAN.md](docs/MULTI_AGENT_IMPLEMENTATION_PLAN.md)。

## 回测假设

- 单资产、日频、只做多、全仓或空仓。
- T 日目标仓位只能在 T+1 开盘执行。
- 买入数量向下取整为配置的整手。
- 成本参数由调用方显式传入，不把某一费率硬编码为所有时期的事实。
- 最后一日持仓按收盘价估值，不强制卖出。
- 暂不处理涨跌停排队、停牌恢复、复权、公司行动、多资产组合和实盘撮合。

## 来源说明

学习起点和个人贡献边界见 [docs/ORIGIN.md](docs/ORIGIN.md)。简历中建议写成：

> 在复现 ChatBI 课程原型后，独立重构可审计量化研究核心：以 QueryFrame 约束 Data/Quant/Risk 三类 Agent，采用只读查询和 T 日收盘信号、T+1 开盘成交规则，通过证据指纹、Risk Critic、运行 Trace 与冻结合成案例验证交易时点、成本和工具调用边界。

## 下一步

1. 用标注意图集验证语义向量与 LLM 分类器是否真正提升路由。
2. 增加停牌、复权、无有效开盘价和多资产组合约束。
3. 在多进程需求出现后再引入 Redis；在情景案例规模足够后再引入 Chroma。
4. 增加 CI、类型检查、OpenTelemetry 导出和可复现演示报告。
