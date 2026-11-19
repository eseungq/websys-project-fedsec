# 조준협

> 코드는 본인이 직접 이 폴더에 올릴 예정입니다. 지금은 작업 기록만 정리해 두었습니다.

## 작업 요약
- 공개 코드 `Federated-Learning-PyTorch`를 fork해서 Kaggle GPU에서 실행
- 1단계: 하이퍼파라미터 6단계 순차 탐색 (model → optimizer → lr → epochs → local_ep → local_bs)
- 2단계: Flask 서버 구현 (직렬화 모듈, 중앙 서버, 가상 클라이언트 실행기)

## Kaggle 노트북
- https://www.kaggle.com/code/chojunhyeop/fedavg

## 문서
| 파일 | 내용 |
|---|---|
| [`docs/01_하이퍼파라미터_탐색.md`](docs/01_하이퍼파라미터_탐색.md) | 공개 코드 파악, Kaggle 실행, 하이퍼파라미터 6단계 탐색 결과 |
| [`docs/02_Flask_서버.md`](docs/02_Flask_서버.md) | Flask 서버·클라이언트 실행기 구조, 실행 절차, 명세 대응 여부 |

## 구현 범위
- 명세 단계:

## 실행 명령
```bash
```

## 설정 값
| 항목 | 값 |
|---|---|
| 모델 | |
| 옵티마이저 | |
| 학습률 | |
| round | |
| local_ep | |
| local_bs | |

## 결과
| 설정 | 정확도 | 시간 | 시드 |
|---|---|---|---|
| | | | |

## 커밋 ID
-
