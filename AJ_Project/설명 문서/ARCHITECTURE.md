# AJ_Project 구조 및 동작 설명

Isaac Gym 기반 **두산 A0509 로봇팔** 시뮬레이션. 좌표(x,y,z) 웨이포인트를 역기구학(IK)으로
풀어 손끝(link_6)이 그 좌표를 따라가게 하고, 로봇은 선반(스탠드) 위에 장착되며 선반 안에는
에어 컴프레셔가 놓인다.

> ⚠️ 이 프로젝트는 원래 창고 물류(AMR·컨베이어·Indy7) 시뮬이었으나 **두산 A0509 중심으로 재구성**되었다.
> 옛 창고 컨트롤러·테스트는 아직 파일로 남아 있지만 **현재 `run.py`는 사용하지 않는다**(레거시). → [레거시 섹션](#-레거시-미사용) 참고.

---

## 📂 현재 트리 구조 (역할별)

```
AJ_Project/
│
├─ 🎯 run.py                      ★ 메인 실행 진입점 (조립 전용, 약 123줄)
│      └ 무대(sim·env) 준비 → 선반+로봇 통합 배치 → 컴프레셔 배치 →
│        좌표 시퀀스 선언 → 루프 (제어 로직은 controllers/에 위임)
│
├─ 📁 controllers/
│   └─ 🤖 doosan_a0509_controller.py   두산 A0509 제어 모듈 (약 128줄, 현재 유일 사용)
│           에셋 로드·스폰·DOF 위치제어·ikpy 역기구학을 캡슐화
│   └─ (indy7_*, low_amr_*, forklift_*, conveyor_belt, cardboard_box_manager
│       → 옛 창고용, 현재 run.py 미사용 = 레거시)
│
├─ 📁 test_scripts/
│   ├─ a0509_ik_demo.py         ★ 좌표 추종(IK) 단독 데모 — run.py 이전의 검증본
│   ├─ a0509_control_demo.py    관절각 직접 제어 데모 (IK 없이)
│   ├─ test_doosan_a0509.py     A0509 로드/파싱 점검
│   └─ (carter/franka/amr/conveyor/indyarm/... → 레거시 창고 테스트)
│
├─ 📁 설명 문서/                  이 ARCHITECTURE.md + 개요.html + 각종 handoff(레거시)
├─ 📁 map_for_robot_navigation/   옛 네비게이션 맵 (레거시)
├─ 📁 debug/                      디버그 출력
└─ run.py.bak                     창고 버전 run.py 백업(920줄) — 복원용
```

---

## 🧩 사용하는 에셋 (Issac_asset 저장소)

`ISAAC_ASSETS` 경로 하위의 URDF. 이번 재구성에서 새로 만든 것 3개:

| 에셋 | 경로 | 내용 |
|---|---|---|
| **통합(선반+로봇)** | `urdf/a0509_on_stand/a0509_on_stand.urdf` | A0509 + 선반을 fixed joint로 결합. `run.py`가 스폰하는 실체 |
| **선반 단독** | `urdf/robot_stand/robot_stand.urdf` | 60×60×80cm 스탠드, Ø30 기둥 4개(개방 프레임). 편집용 소스 |
| **에어 컴프레셔** | `urdf/air_compressor/air_compressor.urdf` | STL→변환 단일 바디(약 46×31×65cm). 시각=메쉬 / 충돌=박스 |
| (원본 로봇팔) | `urdf/doosan_a0509/a0509.urdf` | 순수 A0509 6축. **IK 계산용**으로 참조 |

> **통합 에셋 생성 방식**: `a0509.urdf`(로봇) 내용을 그대로 두고 앞에 `stand_base_link`(선반)와
> `stand_to_robot`(fixed joint, z=0.8)만 끼워 넣은 파일. 로봇 메쉬는 `../doosan_a0509/meshes` 참조.

---

## 🔄 run.py 런타임 동작 흐름

```
[1] 시뮬 초기화      sim 생성(PhysX·중력·60Hz) + 바닥면(sim에 부착, 전체 공유)
[2] 무대 + 배치      env(빈 방) 생성
      ├ 선반+로봇 통합  DoosanA0509Controller (fix_base=False → 바닥에 얹힘)
      │                 · urdf   = a0509_on_stand (액터)
      │                 · ik_urdf= a0509 (순수 로봇팔, IK 체인용)
      └ 컴프레셔         선반 바닥판 위에 얹음 (fix_base=False)
[3] 좌표 시퀀스 선언  waypoints 리스트 → arm.plan_path()로 관절각 사전 IK
[4] 뷰어 생성
[5] 메인 루프        웨이포인트 순회: go_joints() → simulate → fetch → draw
                     (HOLD_SEC마다 다음 좌표로 전환, 실제 도달 오차 출력)
```

### 핵심 좌표계
- **선반/로봇 통합 원점** = 선반 바닥면 중심(z=0). 스폰 Vec3가 이 점을 배치.
- 로봇 base_link는 통합 에셋 내부 fixed joint로 **z=0.8**에 자동 장착.
- **컴프레셔 원점** = 바닥면 중심(z=0). 스폰 Vec3의 z가 곧 바닥 높이.
- **IK 좌표(waypoints)** 는 로봇 base_link 기준 → 로봇을 선반 위로 올려도 그대로 유효.

---

## 🤖 DoosanA0509Controller (controllers/doosan_a0509_controller.py)

두산 관련 저수준 처리를 캡슐화 → `run.py`는 좌표만 넘긴다.

| 메서드 | 역할 |
|---|---|
| `__init__(gym, sim, env, asset_root, urdf, ik_urdf, fix_base, spawn_transform, ...)` | 에셋 로드 + 액터 스폰 + DOF 위치제어(PD) 설정 + ikpy 체인 구성 |
| `solve_ik(xyz, seed)` | 좌표 → 6관절 각도(rad), 도달좌표·오차(mm) 반환 |
| `go_cartesian(xyz)` | 좌표 목표 설정 (IK 변환, 직전 해를 seed로 연속성) |
| `go_joints(q6)` | 6관절 각도 직접 설정 (IK 없이) |
| `plan_path(waypoints)` | 좌표 리스트 → 관절각 리스트 사전 IK (seed 이어가기) |
| `current_tcp()` / `current_joints()` | 현재 손끝 좌표 / 현재 관절각 (FK) |

- **제어 방식**: `DOF_MODE_POS`(위치제어) + PD 게인(stiffness 600 / damping 50)
- **IK 라이브러리**: ikpy (base_link→joint_1 체인, 6축)

---

## 🧭 한눈 요약

- **`run.py`** = 지휘자 (무대 준비·배치·시퀀스 선언·루프)
- **`doosan_a0509_controller.py`** = 두산 팔 전담 (로드·제어·IK)
- **에셋 3종**(통합·선반·컴프레셔)이 씬을 구성
- 좌표만 바꾸면(=`run.py`의 `waypoints`) 동작이 바뀜

---

## 🗄 레거시 (미사용)

아래는 **옛 창고 물류 버전**의 잔재로, 현재 `run.py`가 import/실행하지 않는다.
필요 없으면 삭제해도 현재 시뮬에 영향 없다.

- `controllers/` : `indy7_controller_v1`, `indy7_box_controller`, `low_amr_controller_v2`,
  `forklift_amr_controller_v1`, `conveyor_belt`, `cardboard_box_manager`
- `test_scripts/` : `carter_test`, `franka_test`, `amr_moving_test`, `integration_test`,
  `test_conveyor`, `test_forklift`, `test_lowamr`, `test_indyarm*`, `test_cardboardbox`, `test_common`
- `설명 문서/` : `ConveyorBelt_handoff.html`, `LowAMR_handoff.html`, `ForkliftAMR_handoff.html`,
  `CardboardBoxManager_handoff.html`, `IndyArmController_handoff.html`, `개요.html` (창고 기준)
- `map_for_robot_navigation/` : 옛 네비게이션 맵
- `run.py.bak` : 창고 버전 run.py 백업(920줄)

---

## ▶ 실행

```bash
conda activate issac_env
cd ~/Desktop/Issac_project/AJ_Project
python run.py
```
의존: `numpy`, `ikpy`, NVIDIA `isaacgym`. (에셋 경로는 `ISAAC_ASSETS` 환경변수)
