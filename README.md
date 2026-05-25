# ChatBI 股票分析助手

基于 AI 大模型的智能股票分析助手，支持自然语言查询、技术指标分析、价格预测和可视化展示。

## 功能特性

| 功能 | 工具名称 | 说明 |
|------|---------|------|
| SQL 查询与可视化 | `ExcSql` | 执行 SQL 查询股票历史数据，自动生成走势图 |
| MACD 技术分析 | `macd_stock` | MACD 金叉/死叉买卖信号识别，回测收益率 |
| 布林带分析 | `boll_stock_main` | 布林带超买超卖点检测，信号收益回测 |
| ARIMA 价格预测 | `arima_stock` | ARIMA(5,1,5) 模型预测未来 N 天股价 |
| Prophet 周期性分析 | `prophet_analysis` | 趋势、周/年季节性成分分解与分析 |

## 支持的股票

| 股票代码 | 股票名称 |
|----------|---------|
| 600519.SH | 贵州茅台 |
| 000858.SZ | 五粮液 |
| 601211.SH | 国泰君安 |
| 688981.SH | 中芯国际 |

## 项目结构

```
ChatBI_Stock_Assistant/
├── stock_analysis_assistant-6.py  # 主程序入口
├── boll_detection.py             # 布林带分析模块
├── prophet_analysis.py           # Prophet 周期性分析模块
├── stock_data.db                 # SQLite 股票数据库
├── faq.txt                       # FAQ 知识库
├── requirements.txt              # Python 依赖
├── image_show/                   # 生成的图表（运行后自动创建）
└── README.md                     # 项目说明
```

## 环境要求

- **Python 3.8+**
- **DeepSeek API Key**（或兼容 OpenAI 接口的其他模型）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置环境变量：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key"

# Linux / Mac
export DEEPSEEK_API_KEY="your-api-key"
```

或在 `stock_analysis_assistant-6.py` 的 `main()` 函数中直接替换 `api_key` 参数。

默认使用 **DeepSeek** 模型，如需切换其他模型，修改 `llm_cfg` 配置：

```python
llm_cfg = {
    'model': 'deepseek-chat',          # 模型名称
    'model_server': 'https://api.deepseek.com/v1',  # API 地址
    'api_key': os.getenv('DEEPSEEK_API_KEY'),
}
```

### 3. 运行

```bash
python stock_analysis_assistant-6.py
```

启动后自动打开 Web 界面，在浏览器中与助手对话。

## 使用示例

- "查询贵州茅台最近一个月的股价走势"
- "对比2024年中芯国际和贵州茅台的涨跌幅"
- "使用 MACD 分析贵州茅台过去一年的买卖点"
- "使用布林带检测 600519.SH 股票的超买超卖点"
- "使用 ARIMA 模型预测贵州茅台未来 7 天的价格"
- "使用 Prophet 分析 600519.SH 股票的趋势和周期性"

## 数据库说明

`stock_data.db` 为 SQLite 数据库，包含 `stock_history` 表，结构如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | TEXT | 股票代码 |
| trade_date | TEXT | 交易日期 (YYYY-MM-DD) |
| open | REAL | 开盘价 |
| high | REAL | 最高价 |
| low | REAL | 最低价 |
| close | REAL | 收盘价 |
| pre_close | REAL | 前收盘价 |
| change | REAL | 涨跌额 |
| pct_chg | REAL | 涨跌幅(%) |
| vol | INTEGER | 成交量(手) |
| amount | REAL | 成交额(千元) |
| stock_name | TEXT | 股票名称 |

## 技术栈

- **AI 框架**: [qwen-agent](https://github.com/QwenLM/Qwen-Agent)（兼容 OpenAI API）
- **数据分析**: pandas, numpy
- **图表绘制**: matplotlib, plotly
- **统计分析**: statsmodels (ARIMA), Prophet
- **数据库**: SQLite

## 注意事项

- Prophet 分析功能需要额外安装 `prophet` 库（`pip install prophet`）
- 图表文件保存在 `image_show/` 目录下
- 本工具仅供学习和研究使用，不构成投资建议
