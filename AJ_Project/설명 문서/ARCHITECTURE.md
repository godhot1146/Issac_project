# 프로젝트 구조 및 동작 설명

Isaac Gym 기반 창고 물류/딜리버리 시뮬레이션의 파일 구성과 모듈 간 동작 흐름을 정리한 문서.

---

## 📂 트리 구조 (역할별)

> 각 파일 앞의 `[번호]`는 계층/읽는 순서를 나타내는 표기일 뿐, 실제 파일 이름은 아니다.
> (파이썬 모듈명은 숫자로 시작할 수 없어 import가 깨지므로 파일명에는 붙이지 않는다.)

```
delivery_project/
│
├─ [0] 🎯 run.py   ★ 메인 통합 환경·실행 진입점 (root의 유일한 실행 파일)
│         └ controllers/ 모듈들을 모두 불러와 창고 시뮬레이션을 조립·실행
│
├─ 📁 controllers/   🧩 제어 모듈 (재사용 라이브러리 — run.py가 import하는 순서)
│   ├─ [1] indy7_controller_v1.py        Indy7 로봇팔 제어   (1353줄, 핵심)
│   │        └ [1a] indy7_box_controller.py   [1] 감싼 박스 픽앤플레이스 (198줄)
│   ├─ [2] low_amr_controller_v2.py      저상형 AMR 제어     (716줄)
│   ├─ [3] forklift_amr_controller_v1.py 포크리프트 AMR 제어  (569줄)
│   ├─ [4] conveyor_belt.py              컨베이어 벨트        (476줄)
│   └─ [5] cardboard_box_manager.py      박스 생성·관리       (568줄)
│
├─ 📁 demos/         🔬 단일 에셋 확인용 데모/테스트
│   ├─ [D1] carter_test.py               Carter AMR 로드 확인
│   ├─ [D2] franka_test.py               Franka 로드 확인
│   ├─ [D3] a0509_control_demo.py        Doosan A0509 데모
│   ├─ [D4] test_doosan_a0509.py         A0509 로드/파싱 점검
│   ├─ [D5] amr_moving_test.py           forklift + low_amr 이동 테스트
│   └─ [D6] integration_test.py          통합 테스트
│
├─ 📁 tests/                         모듈 격리 실행 스크립트 (test_common의 최소 환경에서
│                                    conveyor·forklift·lowamr·indyarm·box 단독 수동 점검)
├─ 📁 usability_improvement_tests/   raycast 마우스 피킹 실험
├─ 📁 map_for_robot_navigation/      생성된 occupancy map (png+yaml, 로봇 네비게이션용)
├─ 📁 debug/                         디버그 출력
├─ 📁 설명 문서/                     프로젝트 문서 (이 ARCHITECTURE.md + 개요.html + 각 handoff)
├─ 📁 Old_files/                     옛 버전 아카이브 (a_star 경로계획, main.py 등)
│
├─ requirements_for_this_project.txt  의존 패키지
└─ .gitignore
```

**번호 표기 규칙**
- `[0]` = 실행 진입점 (여기서 시작)
- `[1]~[5]` = `controllers/`의 핵심 제어 모듈 (`run.py`가 불러오는 순서)
- `[1a]` = `[1]`을 감싼 파생 모듈
- `[D1]~[D6]` = `demos/`의 개별 장비 확인용 데모/테스트 (Demo)

> **import 경로:** 제어 모듈이 `controllers/` 하위로 분리됐으므로, 각 실행 스크립트는
> `sys.path`에 `controllers/`를 추가한다 (`run.py`·`demos/`·`tests/`·`usability_.../` 모두 처리됨).
> 파일명·import 문은 그대로이고 경로만 추가되어 `from conveyor_belt import ...` 형태는 유지된다.

---

## ⚙️ 모듈 의존 관계 (누가 누구를 import 하나)

실제 import 관계를 분석하면 명확한 계층 구조를 이룬다.

```
                   [0] run.py   ← 최상위 오케스트레이터
                        │ (import)
        ┌───────────┬───┴────┬──────────────┬──────────────┐
        ▼           ▼        ▼              ▼              ▼
 [5]cardboard   [4]conveyor [1]indy7_     [2]low_amr_   [3]forklift_amr_
   _box_manager   _belt     controller_v1  controller_v2  controller_v1
   (박스생성)     (이송)     (로봇팔)        (AMR)          (포크리프트)
                            ▲
                            │ (import)
                  [1a]indy7_box_controller   [D5]amr_moving_test
                     (픽앤플레이스 래퍼)        → [3]forklift + [2]low_amr 사용
```

| 파일 | import 하는 모듈 |
|---|---|
| `run.py` | cardboard_box_manager, conveyor_belt, indy7_controller_v1, low_amr_controller_v2, forklift_amr_controller_v1 |
| `indy7_box_controller.py` | indy7_controller_v1 |
| `amr_moving_test.py` | forklift_amr_controller_v1, low_amr_controller_v2 |

---

## 🔄 런타임 동작 흐름

1. `run.py` 실행 → Isaac Gym `sim` 생성
2. 환경변수 `ISAAC_ASSETS` 경로에서 각 모듈이 담당 **URDF를 로드** (Issac_asset 저장소의 에셋)
3. **`cardboard_box_manager`** 가 박스를 스폰 → **`conveyor_belt`** 이 이송
4. **`indy7_controller_v1`** (로봇팔)이 박스를 집어 **AMR**(`low_amr` / `forklift_amr`)에 적재
5. 매 프레임 각 컨트롤러의 제어 루프가 관절/바퀴 목표값을 갱신하며 시뮬레이션 진행

---

## 🧭 한눈 요약

- **`run.py`** = 지휘자 (전체 조립·실행)
- **`*_controller` / `*_manager`** = 각 장비 담당 단원 (재사용 모듈)
- 루트의 **`*_test.py` / `*_demo.py`** = 개별 장비를 따로 확인하는 리허설
- **`tests/`** = 자동 단위 검증

---

## 🌳 `run.py` 내부 구조 (메인 실행 파일, 820줄)

`run.py`는 `[SECTION 1]~[SECTION 12]`로 구획되어 있다.

```
run.py
│
├─ 📦 import — isaacgym + 제어 모듈 5종
│
├─ ⚙️ [1] 시뮬레이션/물리 엔진 초기화 (SimParams·PhysX GPU·add_ground)
├─ 🏭 [2] 공장 레이아웃 — 방 10.0×8.5, 벽 4면, 코너 기둥 4개
├─ 📚 [3] 창고 구조물 스폰 — 랙·팔레트·패키지·컨베이어·공정설비·게스트공간
├─ 🤖 [4] 로봇 인스턴스 스폰 — IndyArm ×3, LowAMR, ForkliftAMR, ConveyorBelt
├─ 📐 [5] 박스 접기 단계 레지스트리 — unfolded_flat→fold_sides→…→close_top
├─ 📦 [6] 카드보드 박스 스폰 8개 + CardboardBoxManager
├─ 🧮 [7] 텐서 준비 (gym.prepare_sim 이후 setup_tensors)
├─ 🎛️ [8] 로봇 프리셋 등록 — register_attachable / register_joint_pose
├─ 🔧 [9] 스텝 콜백 함수 — _attach/_detach, _box_*_fold, _mark_* 등
│
├─ 🎬 [10] 자율 시퀀스 러너 등록  ("무엇을 언제" 선언만)
│   ├─ 10-1 LowAMR:   MOVE_TO_CARGO → LIFT_UP → RETREAT → MOVE_TO_DEST
│   ├─ 10-2 Forklift: INITIAL → 선반밑 진입+리프트 → 선반 이동 → 재진입
│   ├─ 10-3 Indy7 팔:  runner1(컨베이어 픽업→팔레트) /
│   │                  runner2(박스 접기 FOLD1~14) / runner3(패키지 적재)
│   └─ 10-4 Conveyor: FIRST_MOVE → SECOND_MOVE
│
├─ ⌨️ [11] 키 입력 등록 — SPACE(시작)·R(리셋)·WASDQE(박스수동)·9/0(팔선택) 등
│
└─ 🔁 [12] 메인 시뮬레이션 루프 (while 뷰어 열림)
    ├─ 12-1 입력 이벤트 처리
    ├─ 12-2 물리 스텝 (simulate→fetch→draw→refresh 텐서)
    ├─ 12-3 팔 3대 처리 (키입력→step→흡착추종→마커)
    ├─ 12-4 자율 러너 6개 update(frame_count)   ← 시퀀스 실제 실행
    ├─ 12-5 flush_dof_targets (관절 명령 일괄 반영)
    └─ 12-6 박스 매니저 후처리 (관절/플랫폼락/자식갱신)
```

### 핵심 패턴 — "선언"과 "실행"의 분리

- **[1]~[8] = 준비(setup)**: 씬·로봇·박스를 만들고 텐서·프리셋을 한 번만 등록
- **[9]~[10] = 시나리오 선언**: "언제 무엇을" 러너로 정의 (아직 실행 안 함)
- **[12] 루프 = 실행 엔진**: 매 프레임 물리 1스텝을 돌리고, 등록된 **러너 6개**(LowAMR·Forklift·팔3·컨베이어)의 `update()`를 호출해 시퀀스를 전진

새 동작을 추가하려면 **SECTION 10에 러너 스텝만 추가**하면 된다. 러너끼리는 `wait_for` / `on_complete` 콜백으로 신호를 주고받으며 연결된다
(예: 팔이 박스를 팔레트에 놓으면 `_mark_forklift_ready` 콜백으로 포크리프트가 출발).

---

> 실행 방법·의존성·에셋 연결(`ISAAC_ASSETS`)은 저장소 최상위 [README](../../README.md) 참고.
