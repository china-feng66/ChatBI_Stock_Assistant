from quantquery_a.workbench.llm import FakeQwenClient
from quantquery_a.workbench.quality import QualityManager
from quantquery_a.workbench.service import WorkbenchService


def test_frozen_evaluation_and_manual_weight_promotion(tmp_path) -> None:
    service = WorkbenchService(
        runtime_dir=tmp_path / "quality-runtime",
        llm=FakeQwenClient(),
    )
    quality = QualityManager(service.database_path, service.runtime_dir)

    evaluation = quality.evaluate()
    assert evaluation.case_count == 8
    assert evaluation.passed_count == 8
    assert evaluation.route_accuracy >= 0.9
    assert evaluation.token_over_budget_count == 0
    assert evaluation.deterministic_gate_rate == 1.0

    candidate = quality.generate_candidate()
    assert candidate.promoted is False
    promoted = quality.promote(candidate.candidate_id, service)
    assert promoted.promoted is True
    assert quality.candidates()[0].promoted is True
