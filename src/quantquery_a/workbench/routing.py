"""LLM + TF-IDF + keyword intent fusion."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from quantquery_a.agents.contracts import (
    AnalyzeRequest,
    IntentSourceScore,
    RouteDecision,
    RoutePath,
    TaskType,
)
from quantquery_a.routing.ensemble import RuleIntentScorer

from .contracts import RouterWeights


INTENT_PROTOTYPES: tuple[tuple[str, TaskType], ...] = (
    ("查询股票最新收盘价和行情", TaskType.MARKET_QUERY),
    ("今天这只股票价格是多少", TaskType.MARKET_QUERY),
    ("拉取最近交易日的行情数据", TaskType.MARKET_QUERY),
    ("分析 MACD 策略信号", TaskType.STRATEGY_ANALYSIS),
    ("用布林带查看买卖区间", TaskType.STRATEGY_ANALYSIS),
    ("解释技术指标产生的信号", TaskType.STRATEGY_ANALYSIS),
    ("回测策略并计算收益", TaskType.BACKTEST),
    ("评估交易成本后的回测表现", TaskType.BACKTEST),
    ("做历史模拟并对比基准", TaskType.BACKTEST),
    ("检查最大回撤和前视偏差", TaskType.RISK_DIAGNOSIS),
    ("诊断策略风险和异常交易", TaskType.RISK_DIAGNOSIS),
    ("比较几只股票哪个表现更好", TaskType.STOCK_COMPARISON),
    ("对比多个标的并进行排名", TaskType.STOCK_COMPARISON),
    ("构建价格成交量因子组合", TaskType.STOCK_COMPARISON),
    ("什么是夏普比率", TaskType.KNOWLEDGE_EXPLAIN),
    ("解释最大回撤的含义", TaskType.KNOWLEDGE_EXPLAIN),
    ("说明量化研究方法原理", TaskType.KNOWLEDGE_EXPLAIN),
    ("替我下单买入股票", TaskType.UNSUPPORTED),
    ("连接实盘自动交易", TaskType.UNSUPPORTED),
    ("保证策略能够盈利", TaskType.UNSUPPORTED),
)


def _normalize(values: Mapping[TaskType, float]) -> dict[TaskType, float]:
    cleaned = {task: max(float(values.get(task, 0.0)), 0.0) for task in TaskType}
    total = sum(cleaned.values())
    if total <= 0:
        return {task: 1.0 / len(TaskType) for task in TaskType}
    return {task: value / total for task, value in cleaned.items()}


class TfidfIntentScorer:
    name = "tfidf"

    def __init__(self) -> None:
        self._texts = [text for text, _ in INTENT_PROTOTYPES]
        self._tasks = [task for _, task in INTENT_PROTOTYPES]
        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def score(self, query: str) -> dict[TaskType, float]:
        similarities = cosine_similarity(
            self._vectorizer.transform([query]), self._matrix
        )[0]
        values = {task: 0.001 for task in TaskType}
        for similarity, task in zip(similarities, self._tasks, strict=True):
            values[task] += float(similarity)
        return _normalize(values)


class HybridIntentRouter:
    def __init__(
        self,
        *,
        weights: RouterWeights | None = None,
        clarification_threshold: float = 0.55,
        margin_threshold: float = 0.15,
    ) -> None:
        self.weights = weights or RouterWeights(llm=0.45, tfidf=0.30, keyword=0.25)
        self.clarification_threshold = clarification_threshold
        self.margin_threshold = margin_threshold
        self.keyword = RuleIntentScorer()
        self.tfidf = TfidfIntentScorer()

    def route(
        self,
        request: AnalyzeRequest,
        llm_probabilities: Mapping[str, float],
    ) -> RouteDecision:
        if request.task_hint is not None:
            exact = {task: float(task is request.task_hint) for task in TaskType}
            sources = {"explicit_hint": exact}
            combined = exact
        else:
            llm = _normalize(
                {
                    task: float(llm_probabilities.get(task.value, 0.0))
                    for task in TaskType
                }
            )
            tfidf = self.tfidf.score(request.query)
            keyword = _normalize(self.keyword.score(request.query))
            sources = {"llm": llm, "tfidf": tfidf, "keyword": keyword}
            combined = {
                task: self.weights.llm * llm[task]
                + self.weights.tfidf * tfidf[task]
                + self.weights.keyword * keyword[task]
                for task in TaskType
            }
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        task, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        missing = []
        if task not in {TaskType.KNOWLEDGE_EXPLAIN, TaskType.UNSUPPORTED}:
            if not request.symbols:
                missing.append("symbols")
        if confidence < self.clarification_threshold or margin < self.margin_threshold:
            path = RoutePath.CLARIFICATION
        elif task in {TaskType.MARKET_QUERY, TaskType.KNOWLEDGE_EXPLAIN}:
            path = RoutePath.FAST
        else:
            path = RoutePath.PLANNED
        if missing:
            path = RoutePath.CLARIFICATION
        return RouteDecision(
            task_type=task,
            path=path,
            confidence=confidence,
            margin=margin,
            source_scores=[
                IntentSourceScore(
                    source=name,
                    probabilities={task.value: value for task, value in values.items()},
                )
                for name, values in sources.items()
            ],
            missing_fields=missing,
        )

