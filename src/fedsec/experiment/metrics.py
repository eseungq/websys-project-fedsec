"""클래스별·클라이언트별 성능 (명세 성능지표: "클래스별·클라이언트별 성능").

클라이언트별 성능은 분포 가중 정확도로 정의한다:
    클라이언트 i의 성능 = Σ_c p_i(c) × (클래스 c의 정확도)
p_i는 클라이언트 i의 학습 데이터 클래스 비율이다. 클라이언트마다 따로 떼어 둔 평가 데이터가 없으므로,
공통 평가 데이터(검증·테스트)의 클래스별 정확도를 각 클라이언트의 분포로 가중한다.
IID에서는 모두 전체 정확도와 비슷하고, 레이블 편향에서는 어느 클라이언트가 손해를 보는지 드러난다.
"""

from collections.abc import Sequence

import torch

from fedsec.experiment.evaluate import NUM_CLASSES, Evaluation


def class_accuracy(result: Evaluation) -> tuple[float | None, ...]:
    """클래스별 정확도. 그 클래스 이미지가 없으면 None."""
    return tuple(c / t if t else None for c, t in zip(result.class_correct, result.class_total))


def label_distribution(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> tuple[float, ...]:
    """레이블의 클래스 비율 (합 1)."""
    if len(labels) == 0:
        raise ValueError("레이블이 없는 데이터의 분포는 정의되지 않습니다.")
    counts = torch.bincount(labels.cpu(), minlength=num_classes).tolist()
    return tuple(count / len(labels) for count in counts)


def distribution_weighted_accuracy(class_acc: Sequence[float | None], distribution: Sequence[float]) -> float:
    """Σ_c p(c) × acc(c). 평가 데이터에 없는 클래스(acc None)는 빼고 남은 비율로 다시 나눈다."""
    if len(class_acc) != len(distribution):
        raise ValueError("클래스 정확도와 분포의 클래스 수가 다릅니다.")
    pairs = [(p, acc) for p, acc in zip(distribution, class_acc) if acc is not None and p > 0]
    weight = sum(p for p, _ in pairs)
    if weight == 0:
        raise ValueError("평가 데이터에 이 클라이언트의 클래스가 하나도 없습니다.")
    return sum(p * acc for p, acc in pairs) / weight


def class_rows(round_idx: int, split: str, result: Evaluation) -> list[dict]:
    """class_acc.csv 줄들: 라운드 × 클래스."""
    return [{"round": round_idx, "split": split, "class": c, "correct": correct, "total": total,
             "acc": round(correct / total, 4) if total else ""}
            for c, (correct, total) in enumerate(zip(result.class_correct, result.class_total))]


def client_acc_rows(round_idx: int, split: str, result: Evaluation,
                    distributions: Sequence[Sequence[float]]) -> list[dict]:
    """client_acc.csv 줄들: 라운드 × 클라이언트의 분포 가중 정확도."""
    accs = class_accuracy(result)
    return [{"round": round_idx, "split": split, "client": cid,
             "weighted_acc": round(distribution_weighted_accuracy(accs, dist), 4)}
            for cid, dist in enumerate(distributions)]
