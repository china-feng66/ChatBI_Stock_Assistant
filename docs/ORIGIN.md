# Origin and Personal Contribution

## 学习起点

本项目受到课程讲义《项目实战：ChatBI开发实战》的启发。课程 baseline 演示了 Qwen-Agent、SQLite 股票查询、MACD/布林带、ARIMA/Prophet 和 WebUI。

旧的课程复现仓库保留为学习记录；它不是本项目的源码基线，也不作为个人原创算法申报。

## 本项目独立实现

QuantQuery-A 从空目录开始实现：

- SQLite 只读查询策略、只读连接、超时和结果上限。
- T 日收盘信号、T+1 开盘成交的日频回测内核。
- 整手、可配置成本、卖出税费和滑点。
- 收益、波动率、Sharpe、最大回撤、基准和换手指标。
- MACD 目标仓位生成。
- 完全基于合成数据的自动化测试。

项目不宣称课程中的指标或模型为个人原创，也不宣称已经发现可交易 Alpha。
