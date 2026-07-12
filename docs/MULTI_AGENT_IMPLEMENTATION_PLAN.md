# Multi-Agent Quant Research Loop — Implementation Plan

## Objective

将现有只读查询、MACD 信号和 T+1 开盘回测核心封装为一个可审计的多 Agent 量化研究闭环。LLM/Agent 只负责路由、编排、解释与批判；数值计算、SQL 策略和风险门禁由确定性代码执行。

## Scope

### MVP now

- 一个 Supervisor 控制面。
- Data Agent、Quant Agent、Risk Critic 三类受限角色。
- Pydantic QueryFrame 作为跨 Agent 唯一数据契约。
- 规则路由与可插拔的三路评分接口；本轮不接真实向量模型或在线 LLM。
- 快路径与计划型慢路径。
- JSONL Trace、证据门禁、默认总尝试最多两次（即最多重试一次）。
- `/analyze` 与 `/eval/run` HTTP 接口。
- 只使用合成行情、临时 SQLite 和冻结金标案例进行测试。

### Later, only when evidence justifies it

- 语义向量意图分类与 LLM 分类器。
- Chroma 情景记忆与 Redis 多进程工作记忆。
- OpenTelemetry 导出、仪表盘和多模型路由。
- 影子评测后的候选权重发布；不允许在线自修改安全门禁。

## Task plan

| ID | Task | Deliverable | Acceptance | Status |
|---|---|---|---|---|
| T0 | Baseline and branch | Clean branch and baseline result | Existing 6 tests pass | Done |
| T1 | Contracts and routing | QueryFrame, route scores, typed agent results | Invalid frames rejected; route confidence exposed | Done |
| T2 | Trace | JSONL run/span events with redaction boundary | Trace contains no raw credentials or database contents | Done |
| T3 | Agents | Data, Quant and Risk roles | Each role has a narrow deterministic interface | Done |
| T4 | Supervisor loop | Fast/slow path, evidence gate, bounded retry | Risk failure cannot be silently bypassed | Done |
| T5 | API | `/healthz`, `/analyze`, `/eval/run` | Typed responses and stable error mapping | Done |
| T6 | Evaluation | Frozen synthetic cases and deterministic checks | Routing/tool/timing gates are reproducible | Done |
| T7 | Documentation and checkpoint | README, security notes, Git commit | Full tests pass and worktree is clean after commit | Done |

## Definition of done

1. 原有核心测试继续通过。
2. 新增测试不读取 `stock_data.db` 或任何完整数据集。
3. 简单查询走快路径；复合分析进入三 Agent 闭环。
4. 每个交易的 `execution_date > signal_date`，整手与成本假设可审计。
5. 所有策略与回测结论经过 Risk Critic；只读行情快路径经过数据证据门禁。
6. `/eval/run` 输出逐案例结果和汇总指标，不以 LLM Judge 替代数值真值。
7. 项目不包含 API Key、真实数据库、生成图片或个人敏感信息。

## Checkpoints

- C1: Contracts + routing + trace.
- C2: Three agents + supervisor loop.
- C3: API + evaluation + docs + full verification.
