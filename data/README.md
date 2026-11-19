# data/

CIFAR-10 원본 데이터가 저장되는 위치입니다. **git에 커밋하지 않습니다.**

- 처음 실행 시 `torchvision.datasets.CIFAR10(download=True)`로 이 폴더에 내려받습니다.
- 팀 공용 클라이언트별 분할 파일(인덱스 목록)은 `configs/splits/`에 있습니다 (예: `configs/splits/iid_seed0.json`).
