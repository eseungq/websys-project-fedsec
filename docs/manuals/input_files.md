# 설정·입력 파일 설명서

경로는 모두 레포 루트 기준입니다. 설정 파일을 바꾸거나 다른 파일을 `--config`, `--env`, `--attack-defense`로 주면 코드를 고치지 않고 다른 실험을 할 수 있습니다. 실제로 쓴 설정 전체는 실행마다 `config.yaml`로 남습니다.

## 학습 설정 `configs/base.yaml`
| 항목 | 값 | 뜻 |
|---|---|---|
| `seed` | 0 | 초기 모델·참여 일정·악성 ID·셔플 시드 (명령줄 `--seed`) |
| `split` | `configs/splits/iid_seed0.json` | 클라이언트 분할 파일 (환경 파일의 `split`이 우선) |
| `data_root` | `data` | CIFAR-10 위치 |
| `rounds` | 40 | 라운드 수 |
| `model` | `simple_cnn` | 학습 모델 |
| `train.lr`, `momentum`, `weight_decay` | 0.01, 0.9, 0.0005 | 로컬 SGD |
| `train.local_ep`, `local_bs` | 1, 32 | 로컬 에포크, 배치 크기 |
| `eval_bs` | 1000 | 평가 배치 크기 |

명령줄: `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs`. 정한 근거: [`docs/stages/1_학습설정-결과.md`](../stages/1_학습설정-결과.md)

## 운영 환경 `configs/env/*.yaml`
`baseline.yaml`, `delay.yaml`, `partial.yaml`, `label_skew.yaml`. 기준에서 한 항목씩만 다릅니다.

| 항목 | 뜻 | 기준 | 다른 값 |
|---|---|---|---|
| `split` | 분할 파일 | IID | 레이블 편향: `configs/splits/label_skew_a0.5_seed0.json` (클라이언트별 표본 수 유지) |
| `per_round` | 라운드 참여자 수 (`null`이면 전원) | `null` | 부분 참여: 5 |
| `malicious.count`, `ids` | 악성 수, 직접 지정 ID (`null`이면 시드로 선택) | 2, `null` | |
| `delayed.count`, `ids`, `avoid_malicious` | 지연 수, 직접 지정 ID, 악성과 겹치지 않게 뽑기 | 0 | 지연: 3, `null`, `true` |
| `virtual_time` | `base_time`(표본 4,500장 기준 완료 시각), `base_samples`, `delay`(지연 추가 시간), `deadline`(마감) | 1.0, 4500, 2.0, 1.5 | |
| `min_accepted` | 채택이 이보다 적으면 모델을 갱신하지 않음 | 3 | |

근거: [`docs/stages/4_운영환경.md`](../stages/4_운영환경.md)

## 공격·방어 `configs/attack_defense.yaml`
| 항목 | 값 | 뜻 |
|---|---|---|
| `attack.enabled` | `false` | 악성의 공모 공격 (명령줄 `--attack/--no-attack`) |
| `attack.scale` | −5.0 | 같은 라운드 악성들의 정상 업데이트 평균 × scale |
| `defense.aggregator` | `fedavg` | `fedavg`, `clip_fedavg`, `median` (`--aggregator`) |
| `defense.clip_norm` | 3.5 | 노름 클리핑 상한, 정상 업데이트 노름 분포로 정해 전체 실험에서 고정 (`--clip-norm`) |
| `defense.clip_before_median` | `false` | 중앙값 전에 클리핑 (`--clip-before-median`) |

근거: [`docs/stages/3_공격-방어.md`](../stages/3_공격-방어.md)

## 모델 역추적 `configs/dlg.yaml`
| 항목 | 값 | 뜻 |
|---|---|---|
| `split`, `client`, `images` | IID, 0, `[1, 27, 38, 52, 61]` | 클라이언트 0의 이미지 5장 (truck, dog, cat, horse, automobile) |
| `model_seed`, `dummy_seed`, `noise_seed` | 0, 0, 0 | 모델·복원 초기값·노이즈 시드를 따로 둠 (하나만 바꿔도 나머지는 같음) |
| `sigmas` | `[0, 1e-2, 1e-1]` | 노이즈 없음·작음·큼 (`--sigmas`) |
| `iterations` | `[300]` | 복원 반복 횟수 (`--iterations`) |
| `record_every` | 10 | 궤적 기록 간격 |

근거: [`docs/stages/3_DLG-결과.md`](../stages/3_DLG-결과.md)

## 분할 파일 `configs/splits/*.json`
| 키 | 뜻 |
|---|---|
| `num_images` | 학습 이미지 수 (50,000) |
| `val` | 검증 이미지 인덱스 5,000개 |
| `clients` | 클라이언트 10명의 학습 이미지 인덱스 (각 4,500개) |

`scripts/make_split.py`로 만들었고 커밋되어 있어 다시 만들 필요가 없습니다.

## 저장된 모델 `experiments/<실행>/models/round_XXX.pt`
- 라운드 XXX 집계 후 전역 모델 (= 라운드 XXX+1의 시작 모델). 파라미터 이름 → 텐서 사전을 `torch.save`한 파일
- 라운드 0, 5의 배수, 마지막 라운드(발산했으면 발산 직전)에 저장. `scripts/replay.py`의 출발점
- 읽을 때는 `torch.load(weights_only=True)`로만 읽고 키·shape·dtype·NaN을 검사합니다
