# references — 참고 자료

팀이 본 자료를 모아 둡니다. **PDF 파일은 올리지 않고 링크만 남깁니다.** 용량 문제가 있고, 레포를 공개로 바꿀 때 저작권 문제가 생길 수 있기 때문입니다.

자료를 추가할 때는 아래 표에 한 줄씩 넣습니다. "관련 단계"는 명세의 추진 단계 번호입니다.

---

## 개념 이해

| 자료 | 한 줄 요약 | 관련 단계 | 본 사람 |
|---|---|---|---|
| [Google 연합학습 만화](https://federated.withgoogle.com) | 연합학습 개념을 그림으로 설명. 데이터는 기기에 두고 업데이트만 보내는 방식 | 1 | |
| [Google Research 블로그: Federated Learning (2017)](https://research.google/blog/federated-learning-collaborative-machine-learning-without-centralized-training-data/) | 중앙에 데이터를 모으지 않는 협업 학습. 연합학습을 처음 소개한 글 | 1 | |
| [Google Research 블로그: Federated Analytics](https://research.google/blog/federated-analytics-collaborative-data-science-without-data-collection/) | 학습이 아니라 통계·분석을 데이터 수집 없이 하는 방법. 연합학습과 비교할 때 참고 | 배경 | |
| [PyTorch CIFAR-10 분류 튜토리얼](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) | 명세와 같은 데이터셋으로 이미지 분류 모델을 학습하는 기본 예제 | 1 | |

## 논문

| 자료 | 한 줄 요약 | 관련 단계 | 본 사람 |
|---|---|---|---|
| [FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (McMahan et al., AISTATS 2017) | FedAvg 원 논문. 클라이언트 업데이트를 표본 수로 가중 평균 | 1–2 | |
| [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) (Zhu et al., NeurIPS 2019) | DLG 원 논문. 그래디언트만으로 원본 이미지와 레이블을 복원, 노이즈 방어 실험 포함 | 3 | |
| [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/abs/2001.02610) (Zhao et al., 2020) | 마지막 층 그래디언트로 정답 레이블을 바로 알아내 DLG 복원을 안정화 | 3 (참고) | |
| [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) (Kairouz et al.) | 연합학습 전반을 정리한 서베이. 공격·방어, 데이터 이질성, 프라이버시 절을 참고 | 배경, 6 | |

## 공개 코드·프레임워크

| 자료 | 한 줄 요약 | 관련 단계 | 본 사람 |
|---|---|---|---|
| [mit-han-lab/dlg](https://github.com/mit-han-lab/dlg) | 명세의 "DLG 공개 구현". PyTorch | 3 | |
| [google-research/federated](https://github.com/google-research/federated) | Google Research의 연합학습 연구 코드 모음. TensorFlow Federated 기반 | 1–2 (참고) | |
| [google-parfait/tensorflow-federated](https://github.com/google-parfait/tensorflow-federated) | TensorFlow Federated(TFF) 프레임워크 | 1–2 (참고) | |

> ⚠️ 위의 Google 코드 두 개는 **TensorFlow** 기반이고, DLG 공개 구현은 **PyTorch** 기반입니다. 명세의 "FedAvg 공개 구현"으로 무엇을 쓸지, 프레임워크를 무엇으로 통일할지 정해서 `docs/decisions/`에 기록해야 합니다.

## 실행 환경

| 자료 | 한 줄 요약 | 관련 단계 | 본 사람 |
|---|---|---|---|
| [Kaggle](https://www.kaggle.com) | 무료 GPU 노트북. 본 실험 실행용 | 4–5 | |

## 확인이 필요한 자료

다른 분석에서 인용된 자료입니다. 보고서에 인용하기 전에 제목과 내용이 주장과 맞는지 확인합니다.

| 자료 | 인용된 맥락 | 확인 |
|---|---|---|
| [ICML 2023 (PMLR v202, liu23d)](https://proceedings.mlr.press/v202/liu23d/liu23d.pdf) | 데이터 이질성 환경에서 강건 집계의 한계 | [ ] |
| [NeurIPS 2023 (2082273791021571c410f41d565d0b45)](https://papers.neurips.cc/paper_files/paper/2023/hash/2082273791021571c410f41d565d0b45-Abstract-Conference.html) | 복원 이미지의 프라이버시 위험을 화질 지표로만 평가하는 문제 | [ ] |
