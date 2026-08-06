# Issac_project

Isaac Gym 기반 로봇 시뮬레이션 프로젝트. 두 개의 하위 프로젝트로 구성된다.

| 하위 프로젝트 | 내용 | 상태 |
|---|---|---|
| **`AJ_Project/`** | 두산 A0509 로봇팔 — JSC/TSC/OSC 제어, 키보드 조작, 선반·컴프레셔 셀 | ⭐ **현재 주력** |
| `delivery_project/` | 창고 물류 시뮬 (AMR·포크리프트·컨베이어·Indy7 픽앤플레이스) | 레거시 |

에셋(URDF·메쉬·텍스처)은 용량이 커서 **별도 저장소**로 분리 → [Issac_asset](https://github.com/godhot1146/Issac_asset)

---

## 1. 요구 환경

- **Python 3.8** (검증: 3.8.20 / conda(issac_env) / Ubuntu / CUDA 12.1)
- **NVIDIA Isaac Gym** (PyPI에 없는 NVIDIA 전용 패키지 — 별도 설치)

```bash
pip install -e ~/isaacgym/python
```

## 2. 에셋 저장소 연결 (⭐ 다른 컴퓨터에서 필수)

코드는 에셋 위치를 환경변수 `ISAAC_ASSETS`로 읽는다.

```bash
git clone https://github.com/godhot1146/Issac_asset.git
export ISAAC_ASSETS=/절대경로/Issac_asset/isaac_assets   # ~/.bashrc에 추가 권장
```

> 미설정 시 원작자 PC 기준 기본 경로가 쓰이므로, 다른 컴퓨터에서는 반드시 지정해야 한다.

---

## 3. AJ_Project — 두산 A0509 (현재 주력)

두산 A0509 6축 로봇팔을 선반 위에 장착하고, **세 가지 제어 방식**을 키보드로 직접 조작한다.

### 제어 방식
| 약어 | 방식 | 설명 |
|---|---|---|
| **JSC** | 관절공간 | 관절각 직접 지정 (위치제어) |
| **TSC** | 작업공간 | 좌표(x,y,z) → ikpy 역기구학 → 관절각 (위치제어) |
| **OSC** | 작업공간·동역학 | 자코비안·질량행렬로 토크 계산 + 중력보상 (토크제어) |

세 방식 모두 `controllers/doosan_controller.py`(통합)에 구현. 모드는 실행 중 키로 전환된다.

### 실행
```bash
conda activate issac_env        # OSC용 gymtorch → ninja 실행파일 PATH 필요
cd AJ_Project
python run.py                   # 키보드 다중모드 제어
```
```
[모드] 1:JSC  2:TSC  3:OSC
[좌표·TSC/OSC] W/S:X±  A/D:Y±  Q/E:Z±
[관절·JSC]     J/L:관절선택  U/O:각도±
[공통] R:홈자세   (창 닫기: 종료)
```

### 의존 패키지
```bash
pip install -r AJ_Project/설명\ 문서/requirements_for_this_project.txt
```
- 필수: `numpy`, `ikpy`
- OSC 사용 시: `torch` + isaacgym `gymtorch`(최초 로드에 `ninja` 필요)

### 구조
- `run.py` — 조립·실행 (씬 구성 + 키보드 루프)
- `controllers/doosan_controller.py` — JSC/TSC/OSC 통합 제어
- `test_scripts/` — 단일 에셋/기능 데모
- 상세: [AJ_Project/설명 문서/ARCHITECTURE.md](AJ_Project/설명%20문서/ARCHITECTURE.md)

---

## 4. delivery_project — 창고 물류 (레거시)

AMR·포크리프트·컨베이어·Indy7 로봇팔로 창고 픽앤플레이스/이송을 시뮬한다.

```bash
cd delivery_project
python run.py                    # 창고 통합 환경
python demos/integration_test.py # 통합 테스트
```
- 의존: `numpy`, `scipy`, `torch`
- 상세: [delivery_project/설명 문서/ARCHITECTURE.md](delivery_project/설명%20문서/ARCHITECTURE.md)

---

## 저장소 구성

| 저장소 | 내용 | 용량 |
|---|---|---|
| **Issac_project** (이 repo) | 시뮬레이션 코드 (`.py`) | ~4MB |
| **Issac_asset** | URDF·메쉬·텍스처 에셋 | ~536MB |

## 제외된 파일 (`.gitignore`)

각 PC에서 자동 생성되는 것만 제외 (이식성 영향 없음):
- `__pycache__/`, `*.pyc` — 파이썬 바이트코드 캐시
- `.vscode/` — 편집기 설정 및 IntelliSense DB
