# Issac_project (delivery_project)

Isaac Gym 기반 물류/딜리버리 시뮬레이션 프로젝트. AMR·포크리프트·컨베이어·로봇팔(Franka, Indy7, Doosan A0509)을
이용한 창고 픽앤플레이스 / 이송 시나리오를 다룬다.

에셋(URDF·메쉬·텍스처)은 용량이 커서 **별도 저장소**로 분리되어 있다 → [Issac_asset](https://github.com/godhot1146/Issac_asset)

프로젝트 파일 구성과 모듈 동작 흐름은 [delivery_project/설명 문서/ARCHITECTURE.md](delivery_project/설명%20문서/ARCHITECTURE.md) 참고.

---

## 1. 요구 환경

- **Python 3.8** (검증: 3.8.20 / conda / Ubuntu / CUDA 12.1)
- **NVIDIA Isaac Gym** (PyPI에 없는 NVIDIA 전용 패키지 — 별도 설치 필요)

```bash
pip install -e ~/isaacgym/python
```

## 2. 의존 패키지 설치

```bash
pip install -r delivery_project/requirements_for_this_project.txt
```

핵심: `numpy==1.24.4`, `scipy==1.10.1`, `torch==2.4.1`
(`Old_files/`의 옛 스크립트를 돌릴 때만 `matplotlib`, `opencv-python`, `PyYAML` 주석 해제)

## 3. 에셋 저장소 연결 (⭐ 다른 컴퓨터에서 실행 시 필수)

코드는 에셋 위치를 환경변수 `ISAAC_ASSETS`로 읽는다. 에셋 repo를 클론한 뒤 경로를 지정한다.

```bash
# 에셋 저장소 클론 (원하는 위치)
git clone https://github.com/godhot1146/Issac_asset.git

# 클론한 에셋의 실제 루트를 환경변수로 지정
export ISAAC_ASSETS=/절대경로/Issac_asset/isaac_assets
```

> 환경변수를 지정하지 않으면 코드의 기본 경로가 사용되는데, 이는 원작자 PC 기준 경로이므로
> **다른 컴퓨터에서는 반드시 `ISAAC_ASSETS`를 설정**해야 한다.
> `~/.bashrc`에 `export` 줄을 추가해두면 매번 입력하지 않아도 된다.

## 4. 실행 예시

```bash
cd delivery_project
python run.py                    # 창고 통합 환경 (메인)
python demos/integration_test.py # 통합 테스트
python demos/carter_test.py      # 단일 에셋 로드 확인
```

> 제어 모듈은 `controllers/`, 개별 데모는 `demos/`에 있다. 각 스크립트가 `controllers/`를
> import 경로에 자동 추가하므로 위치만 옮겨도 그대로 실행된다.

---

## 저장소 구성

| 저장소 | 내용 | 용량 |
|---|---|---|
| **Issac_project** (이 repo) | 시뮬레이션 코드 (`.py`) | ~4MB |
| **Issac_asset** | URDF·메쉬·텍스처 에셋 | ~536MB |

## 제외된 파일 (`.gitignore`)

실행에 불필요하고 각 PC에서 자동 생성되는 것들만 제외 — 이식성엔 영향 없음:

- `__pycache__/`, `*.pyc` — 파이썬 바이트코드 캐시 (파이썬 버전별로 재생성)
- `.vscode/` — 편집기 설정 및 IntelliSense DB(`browse.vc.db`, 1GB+, 절대경로 포함)
