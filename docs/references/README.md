# references — 참고 자료

팀이 본 자료를 모아 둡니다. **PDF 파일은 올리지 않고 링크만 남깁니다.** 용량 문제가 있고, 레포를 공개로 바꿀 때 저작권 문제가 생길 수 있기 때문입니다.

자료를 추가할 때는 아래 표에 한 줄씩 넣습니다. "관련 단계"는 명세의 추진 단계 번호입니다.

---

## 개념 이해

| 자료 | 한 줄 요약 | 관련 단계 |
|---|---|---|
| [Google 연합학습 만화](https://federated.withgoogle.com) | 연합학습 개념을 그림으로 설명. 데이터는 기기에 두고 업데이트만 보내는 방식 | 1 |
| [Google Research 블로그: Federated Learning (2017)](https://research.google/blog/federated-learning-collaborative-machine-learning-without-centralized-training-data/) | 중앙에 데이터를 모으지 않는 협업 학습. 연합학습을 처음 소개한 글 | 1 |
| [Google Research 블로그: Federated Analytics](https://research.google/blog/federated-analytics-collaborative-data-science-without-data-collection/) | 학습이 아니라 통계·분석을 데이터 수집 없이 하는 방법. 연합학습과 비교할 때 참고 | 배경 |
| [PyTorch CIFAR-10 분류 튜토리얼](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) | 명세와 같은 데이터셋으로 이미지 분류 모델을 학습하는 기본 예제 | 1 |

## 팀 자료

| 자료 | 한 줄 요약 | 관련 단계 |
|---|---|---|
| [Kaggle 노트북: fedavg](https://www.kaggle.com/code/chojunhyeop/fedavg) | 공개 코드 `Federated-Learning-PyTorch`로 CIFAR-10 FedAvg를 Kaggle GPU에서 실행 (조준협) | 1 |

## 논문

| 자료 | 한 줄 요약 | 관련 단계 |
|---|---|---|
| [FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (McMahan et al., AISTATS 2017) | FedAvg 원 논문. 클라이언트 업데이트를 표본 수로 가중 평균 | 1–2 |
| [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) (Zhu et al., NeurIPS 2019) | DLG 원 논문. 그래디언트만으로 원본 이미지와 레이블을 복원, 노이즈 방어 실험 포함 | 3 |
| [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/abs/2001.02610) (Zhao et al., 2020) | 마지막 층 그래디언트로 정답 레이블을 바로 알아내 DLG 복원을 안정화 | 3 (참고) |
| [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) (Kairouz et al.) | 연합학습 전반을 정리한 서베이. 공격·방어, 데이터 이질성, 프라이버시 절을 참고 | 배경, 6 |
| [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/abs/1803.01498) (Yin et al., ICML 2018) | 좌표별 중앙값·절사 평균 집계로 악의적 참여자가 있어도 오차를 보장. 방어 ② 중앙값의 근거 | 3 |
| [Can You Really Backdoor Federated Learning?](https://arxiv.org/abs/1911.07963) (Sun et al., NeurIPS 2019 FL 워크숍) | 업데이트 노름 클리핑(과 약한 노이즈)으로 악성 업데이트 영향을 줄임. 방어 ① 클리핑의 근거 | 3 |
| [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459) (Bagdasaryan et al., AISTATS 2020) | 악성 업데이트를 크게 키워 평균 후 전역 모델을 바꾸는 공격. 업데이트에 배수를 곱하는 공격 방식 참고 | 3 |
| [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/abs/1911.11815) (Fang et al., USENIX Security 2020) | 중앙값 같은 강건 집계도 공모 공격으로 성능이 떨어질 수 있음. 방어의 한계 설명 참고 | 3, 6 |
| [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (Hsu et al., 2019) | Dirichlet(α)로 클라이언트별 레이블 분포를 치우치게 나누는 방법. 레이블 편향 환경 참고 | 4 |
| [Federated Optimization in Heterogeneous Networks (FedProx)](https://arxiv.org/abs/1812.06127) (Li et al., MLSys 2020) | 기기 성능 차이로 늦거나 일부만 학습한 클라이언트(straggler)를 다룸. 지연·부분 참여 환경 참고 | 4 (참고) |
| [Inverting Gradients](https://arxiv.org/abs/2003.14053) (Geiping et al., NeurIPS 2020) | DLG 후속. 코사인 유사도로 더 큰 이미지·배치에서도 복원 | 3 (참고) |

## 공개 코드·프레임워크

| 자료 | 한 줄 요약 | 관련 단계 |
|---|---|---|
| [mit-han-lab/dlg](https://github.com/mit-han-lab/dlg) | 명세의 "DLG 공개 구현". PyTorch | 3 |
| [google-research/federated](https://github.com/google-research/federated) | Google Research의 연합학습 연구 코드 모음. TensorFlow Federated 기반 | 1–2 (참고) |
| [google-parfait/tensorflow-federated](https://github.com/google-parfait/tensorflow-federated) | TensorFlow Federated(TFF) 프레임워크 | 1–2 (참고) |

## 실행 환경

| 자료 | 한 줄 요약 | 관련 단계 |
|---|---|---|
| [Kaggle](https://www.kaggle.com) | 무료 GPU 노트북. 1단계에서 T4 결과가 Mac과 같은지 확인하는 데만 씀 (본 실험은 Mac MPS) | 1 |
