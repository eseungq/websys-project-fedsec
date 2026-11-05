# data/

CIFAR-10 원본 데이터가 저장되는 위치입니다. **git에 커밋하지 않습니다.**

- 처음 실행 시 `torchvision.datasets.CIFAR10(download=True)`로 이 폴더에 내려받습니다.
- 클라이언트별 분할 결과(인덱스 목록)는 `experiments/<실험ID>/`에 설정과 함께 저장합니다.
