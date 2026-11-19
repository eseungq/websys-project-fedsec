import pytest

from fedsec.experiment.config import load_config


BASE_TEXT = """
seed: 0
split: configs/splits/iid_seed0.json
data_root: data
rounds: 40
model: simple_cnn
train: {optimizer: sgd, lr: 0.01, momentum: 0.9, weight_decay: 5e-4, local_ep: 1, local_bs: 32}
eval_bs: 1000
note: 기록되지 않는 키
"""


def write_base(tmp_path, text=BASE_TEXT):
    path = tmp_path / "base.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_reads_exponent_notation_as_number(tmp_path):
    cfg = load_config(write_base(tmp_path))                    # YAML 1.1은 5e-4를 문자열로 읽음

    assert cfg.train.weight_decay == pytest.approx(0.0005)


def test_load_config_rejects_non_numeric_value(tmp_path):
    path = write_base(tmp_path, BASE_TEXT.replace("lr: 0.01", "lr: abc"))

    with pytest.raises(ValueError, match="항목이 잘못"):
        load_config(path)
