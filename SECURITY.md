# Security

- 不提交 API Key、数据库密码、Cookie、访问令牌或可运行默认凭据。
- 数据库查询必须经过 `ReadOnlyQueryService`。
- 只允许单条 SELECT 或 SELECT-based CTE。
- SQLite 使用只读 URI，并启用 `query_only`。
- 查询时间和返回行数有上限。
- 测试使用合成数据库，不读取真实股票数据库。

如果凭据进入公开 Git 历史，应先在服务提供方撤销/轮换，再处理 Git 历史；仅修改当前文件不足以消除泄露。
