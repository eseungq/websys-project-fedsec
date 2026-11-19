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


@pytest.mark.parametrize("line, value", [("local_ep: 1", "local_ep: 1.9"), ("local_bs: 32", "local_bs: 32.5")])
def test_load_config_rejects_fractional_integer(tmp_path, line, value):
    path = write_base(tmp_path, BASE_TEXT.replace(line, value))

    with pytest.raises(ValueError, match="정수여야 합니다"):
        load_config(path)


def test_load_config_accepts_whole_number_float_and_string(tmp_path):
    path = write_base(tmp_path, BASE_TEXT.replace("local_ep: 1", "local_ep: 2.0").replace("local_bs: 32", "local_bs: '64'"))

    cfg = load_config(path)

    assert (cfg.train.local_ep, cfg.train.local_bs) == (2, 64)
    assert isinstance(cfg.train.local_ep, int)



@pytest.mark.parametrize("value", [9007199254740993, "9007199254740993"])
def test_load_config_keeps_large_seed_exact(tmp_path, value):
    # float를 거치면 2**53 + 1이 2**53으로 바뀌던 문제
    path = write_base(tmp_path, BASE_TEXT.replace("seed: 0", f"seed: {value!r}"))

    assert load_config(path).seed == 9007199254740993

@pytest.mark.parametrize("text", [
    BASE_TEXT + "rouns: 50\n",                              # 오타 키: 예전에는 조용히 무시되어 40라운드로 실행됨
    BASE_TEXT.replace("eval_bs: 1000\n", ""),               # 빠진 항목
    "",                                                     # 빈 파일
    BASE_TEXT.replace("lr: 0.01", "lr: .inf"),              # 무한대
    BASE_TEXT.replace("local_ep: 1", "local_ep: true"),     # true는 정수가 아님
])
def test_load_config_rejects_unknown_missing_or_bad_values(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(write_base(tmp_path, text))
