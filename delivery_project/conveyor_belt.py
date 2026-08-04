import os
import torch
from isaacgym import gymapi, gymtorch
from scipy.spatial.transform import Rotation as R

class ConveyorMotionStep:
    """
    컨베이어 한 구간 이동을 기술하는 스텝.
      wait_for: 이 스텝을 '시작'하기 전 만족해야 할 조건 (예: 특정 팔 러너의 완료)
      on_enter: 이동 시작 직전 실행할 콜백들 (예: 박스 접기 마무리 작업)
      on_complete: 이동이 끝나고 hold까지 마친 뒤 실행할 콜백들 (예: 완료 플래그 세팅)
    """
    def __init__(self, name,
                 distance=None,
                 direction='FORWARD',
                 hold_seconds=0.0,
                 wait_for=None,
                 on_enter=None,
                 on_complete=None):
        self.name = name
        self.distance = distance

        direction = str(direction).strip().upper()
        if direction not in ('FORWARD', 'BACKWARD'):
            raise ValueError(f"direction은 'FORWARD' 또는 'BACKWARD'만 가능: {direction}")
        self.direction = direction

        self.hold_seconds = max(0.0, hold_seconds)
        self.wait_for = wait_for
        self.on_enter = on_enter or []
        self.on_complete = on_complete or []


class ConveyorStepRunner:
    """
    ConveyorMotionStep 리스트를 순차 실행.
      0: 이동 중 (컨베이어가 멈출 때까지, 즉 conveyor.running이 False가 될 때까지 대기)
      1: 정지 유지(hold) — hold_seconds만큼 대기 후 on_complete 실행 및 다음 스텝 진행
    """
    SUB_MOVING = 0
    SUB_HOLD = 1

    def __init__(self, conveyor: "ConveyorBelt", steps, context=None):
        self.conveyor = conveyor
        self.steps = steps
        self.context = context
        self.index = 0
        self.sub_stage = self.SUB_MOVING
        self._entered_current_step = False
        self._hold_start_frame = None
        self.done = False

    def _current_step(self):
        return self.steps[self.index]

    def _advance_to_next_step(self, step):
        for cb in step.on_complete:
            cb(self.context)
        print(f"[ConveyorStepRunner] 스텝 완료: {step.name}")

        self.index += 1
        self.sub_stage = self.SUB_MOVING
        self._entered_current_step = False
        self._hold_start_frame = None

        if self.index >= len(self.steps):
            self.done = True

    def update(self, frame_count=0):
        self.conveyor.update(frame_count)

        if self.done:
            return

        step = self._current_step()

        if step.wait_for is not None and not self._entered_current_step:
            if not step.wait_for(self.context):
                return

        if not self._entered_current_step:
            for cb in step.on_enter:
                cb(self.context)
            self._entered_current_step = True
            print(f"[ConveyorStepRunner] 스텝 진입: {step.name}")

            if step.direction == 'FORWARD':
                self.conveyor.move_forward(distance=step.distance)
            else:
                self.conveyor.move_backward(distance=step.distance)
            print(f"[ConveyorStepRunner] 컨베이어 이동 시작: {step.name} (distance={step.distance})")

        if self.sub_stage == self.SUB_MOVING:
            if not self.conveyor.running:
                self._enter_hold(step, frame_count)

        elif self.sub_stage == self.SUB_HOLD:
            hold_frames = self.conveyor.seconds_to_frames(step.hold_seconds)
            if (frame_count - self._hold_start_frame) >= hold_frames:
                self._advance_to_next_step(step)

    def _enter_hold(self, step, frame_count):
        if step.hold_seconds > 0.0:
            self.sub_stage = self.SUB_HOLD
            self._hold_start_frame = frame_count
            print(f"[ConveyorStepRunner] {step.hold_seconds:.1f}초 대기 시작: {step.name}")
        else:
            self._advance_to_next_step(step)

# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def _world_to_local(origin_pos: gymapi.Vec3, origin_rot: gymapi.Quat, world_pos: gymapi.Vec3) -> gymapi.Vec3:
    """world_pos를 origin(위치/회전) 기준 로컬 좌표로 변환."""
    delta = gymapi.Vec3(world_pos.x - origin_pos.x,
                         world_pos.y - origin_pos.y,
                         world_pos.z - origin_pos.z)
    return origin_rot.inverse().rotate(delta)


def _flatten_keep_yaw(rot: gymapi.Quat) -> gymapi.Quat:
    """roll/pitch만 0으로 눌러 수평 정렬하고, yaw(z축 회전)는 현재 값 그대로 유지."""
    q = R.from_quat([rot.x, rot.y, rot.z, rot.w])
    _, _, yaw = q.as_euler('xyz', degrees=False)
    flat = R.from_euler('z', yaw)
    fx, fy, fz, fw = flat.as_quat()
    return gymapi.Quat(fx, fy, fz, fw)


class ConveyorBelt:
    """벨트 위 물체에 속도를 주입해 밀어내고, 텍스처 스크롤로 이동감을 표현하는 컨베이어 벨트."""

    # [FIX 5] 안착 판정용 매직넘버를 클래스 상수로 승격.
    # 두 값 모두 dt=1/60, 표준 박스 크기(약 0.1~0.3m급)를 기준으로 튜닝된 값.
    # dt를 바꾸거나(예: 30Hz 시뮬레이션) 훨씬 작거나/큰 물체를 다룬다면 함께 재튜닝 필요.
    LANDING_Z_STABLE_EPS = 0.0001   # 이 값 이하로 z가 변하면 "더 이상 낙하하지 않는다"로 판정
    LANDING_MIN_HEIGHT = 0.05       # 낙하 초기 오탐(아직 공중인데 안착으로 오판) 방지용 최소 높이(m)

    # ==========================================================================
    # 초기화
    # ==========================================================================
    def __init__(self, gym, sim, env, conveyor_handle,
                 belt_link_name="belt_link",
                 texture_frame_paths=None,
                 speed=0.5,
                 dt=1.0 / 60.0):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = conveyor_handle
        self.belt_link_name = belt_link_name
        self.speed = speed
        self.dt = dt

        # 텍스처(로컬 인덱스)는 setup_tensors 없이도 바로 조회 가능
        self.belt_rb_local_idx = gym.find_actor_rigid_body_index(
            env, conveyor_handle, belt_link_name, gymapi.DOMAIN_ACTOR)
        belt_link_found = self.belt_rb_local_idx != -1
        if not belt_link_found:
            print(f"[ConveyorBelt] 경고: '{belt_link_name}' 링크를 찾지 못했습니다. "
                  f"텍스처 스크롤이 비활성화됩니다.")

        # [FIX 6] 텍스처 애니메이션 프레임 로드 전, 경로 존재 여부를 먼저 검증한다.
        # 존재하지 않는 경로는 건너뛰고 경고만 남긴다(예외로 전체 초기화가 죽지 않도록).
        self.texture_handles = []
        if texture_frame_paths:
            valid_paths = []
            for path in texture_frame_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"[ConveyorBelt] 경고: 텍스처 프레임 경로를 찾을 수 없어 건너뜁니다: '{path}'")
            if not valid_paths and texture_frame_paths:
                print(f"[ConveyorBelt] 경고: texture_frame_paths에 유효한 경로가 하나도 없습니다. "
                      f"텍스처 스크롤이 비활성화됩니다.")
            for path in valid_paths:
                self.texture_handles.append(gym.create_texture_from_file(sim, path))
        self.num_frames = len(self.texture_handles)
        self.texture_pixel_pos = 0.0

        # [FIX 7] "조용한 비활성화" 대신, 호출부가 코드로 바로 확인할 수 있도록
        # 텍스처 스크롤 가능 여부를 명시적인 상태 플래그로 노출한다.
        # belt_link_name을 찾았고(belt_link_found) 유효한 텍스처 프레임이 1개 이상 로드된
        # 경우에만 True. 로그를 놓치더라도 `if not conveyor.is_texture_enabled: ...`처럼
        # 분기할 수 있다.
        self.is_texture_enabled = belt_link_found and self.num_frames > 0

        # [FIX 1] 방향 부호가 빠진 "크기(magnitude)"만 저장. 실제 스크롤 시 self.direction을 곱한다.
        # direction=+1(정방향) 기준의 프레임당 픽셀 이동량. setup_tensors()/set_speed()에서 계산됨.
        self._texture_pixels_per_frame_base = 0.0
        self.belt_length = None  # setup_tensors에서 설정

        # 동작 상태
        self.running = False
        self.direction = +1              # +1: 정방향(로컬 +Y), -1: 역방향(로컬 -Y)
        self.target_distance = None      # None이면 stop() 호출 전까지 무한 이동
        self._moved_distance = 0.0
        self._distance_ref_item = None   # 이동 거리 측정 기준으로 삼을 아이템 handle

        # 벨트 위 물체 관리: handle -> 상태 dict
        self.items = {}

        # prepare_sim 이후에 채워짐
        self.root_states = None
        self.rb_states = None
        self.belt_rb_sim_idx = None

    def setup_tensors(self, belt_length=3.5):
        """gym.prepare_sim(sim) 호출 '이후'에 반드시 실행."""
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor).view(-1, 13)

        self.belt_rb_sim_idx = self.gym.find_actor_rigid_body_index(
            self.env, self.handle, self.belt_link_name, gymapi.DOMAIN_SIM)
        if self.belt_rb_sim_idx == -1:
            # 링크 이름을 못 찾으면 액터의 0번 리지드바디로 폴백
            self.belt_rb_sim_idx = self.gym.get_actor_rigid_body_index(
                self.env, self.handle, 0, gymapi.DOMAIN_SIM)
            print(f"[ConveyorBelt] 경고: belt_rb_sim_idx를 0번 링크로 폴백했습니다. "
                  f"실제 벨트 링크 이름을 확인하세요.")

        self.belt_length = belt_length
        self._recompute_texture_scroll_rate()

    # ==========================================================================
    # 속도 제어
    # ==========================================================================
    def set_speed(self, new_speed: float):
        """
        [FIX 2] 런타임에 벨트 속도를 변경한다.
        아이템에 주입되는 속도뿐 아니라 텍스처 스크롤 속도도 함께 재계산되므로,
        speed를 직접 self.speed에 대입하지 말고 항상 이 메서드를 사용할 것.
        """
        self.speed = new_speed
        self._recompute_texture_scroll_rate()

    def _recompute_texture_scroll_rate(self):
        """belt_length/speed/num_frames를 바탕으로 텍스처 스크롤 기본 속도를 재계산."""
        if self.num_frames > 0 and self.belt_length:
            self._texture_pixels_per_frame_base = \
                -(self.speed * self.dt * self.num_frames) / self.belt_length
        else:
            self._texture_pixels_per_frame_base = 0.0

    # ==========================================================================
    # 물체(아이템) 등록 / 해제
    # ==========================================================================
    def register_item(self, actor_handle, root_link_index=0) -> bool:
        """컨베이어가 관리할 물체를 등록. 여러 개 등록 가능."""
        actor_sim_idx = self.gym.get_actor_index(self.env, actor_handle, gymapi.DOMAIN_SIM)
        rb_sim_idx = self.gym.get_actor_rigid_body_index(
            self.env, actor_handle, root_link_index, gymapi.DOMAIN_SIM)

        if actor_sim_idx == -1 or rb_sim_idx == -1:
            print(f"[ConveyorBelt] register_item 실패: 유효하지 않은 액터/링크 인덱스")
            return False

        self.items[actor_handle] = {
            "actor_sim_index": actor_sim_idx,
            "rb_sim_index": rb_sim_idx,
            "is_landed": False,
            "prev_z": 999.0,
            "start_local_y": None,   # 현재 run(move_forward/backward) 시작 시점의 로컬 y
            "last_local_y": None,    # [FIX 3] 매 프레임 갱신되는 최신 로컬 y (기준 아이템 교체용)
            "grabbed": False,
        }
        # [FIX 8] 여기 있던 `if actor_handle not in self.items: return False`는
        # 바로 위에서 딕셔너리에 넣은 직후라 항상 거짓인 죽은 코드였으므로 제거함.
        # 새로 등록된 아이템이 없어서 기준 아이템이 비어있었다면 즉시 채워준다.
        if self._distance_ref_item is None:
            self._distance_ref_item = actor_handle
        return True

    def unregister_item(self, actor_handle):
        """더 이상 컨베이어가 관리하지 않도록 제거 (예: 벨트 끝에서 다른 공정으로 넘어갈 때)."""
        self.items.pop(actor_handle, None)
        # [FIX 3] 참고: 여기서 _distance_ref_item을 즉시 교체하지 않아도,
        # update()의 _update_distance_tracking()이 다음 프레임에 자동으로
        # 다른 landed 아이템으로 기준을 교체한다. (프레임 한 번 지연될 수 있음)

    def set_item_grabbed(self, actor_handle, grabbed: bool):
        """로봇 등이 물체를 붙잡고 있는 동안에는 컨베이어 힘을 가하지 않도록 표시."""
        if actor_handle in self.items:
            self.items[actor_handle]["grabbed"] = grabbed

    def reset_item_landing(self, actor_handle):
        """물체를 다시 벨트 위에 새로 올려놓았을 때(예: 재사용) 안착 판정을 리셋."""
        if actor_handle in self.items:
            self.items[actor_handle]["is_landed"] = False
            self.items[actor_handle]["prev_z"] = 999.0
            self.items[actor_handle]["start_local_y"] = None
            self.items[actor_handle]["last_local_y"] = None

    # ==========================================================================
    # 동작 제어 API
    # ==========================================================================
    def move_forward(self, distance=None):
        """정방향(로컬 +Y)으로 이동 시작. distance(m)를 주면 그만큼만 이동 후 자동 정지."""
        self._start_run(direction=+1, distance=distance)

    def move_backward(self, distance=None):
        """역방향(로컬 -Y)으로 이동 시작. distance(m)를 주면 그만큼만 이동 후 자동 정지."""
        self._start_run(direction=-1, distance=distance)

    def stop(self):
        """즉시 정지. 물체들의 속도도 0으로 리셋됨(다음 update()에서)."""
        self.running = False
        self.target_distance = None
        self._moved_distance = 0.0

    @property
    def moved_distance(self) -> float:
        return self._moved_distance

    def _start_run(self, direction: int, distance):
        self.running = True
        self.direction = direction
        self.target_distance = distance
        self._moved_distance = 0.0
        # 거리 측정 기준: 현재 등록된 아이템 중 가장 먼저 등록된 것.
        # (없으면 None -> update()에서 landed된 아이템이 생기는 대로 자동 채워짐)
        self._distance_ref_item = next(iter(self.items), None)
        for handle, st in self.items.items():
            st["start_local_y"] = None  # update()에서 현재 위치로 새로 기록
            st["last_local_y"] = None

    # ==========================================================================
    # 매 프레임 업데이트 (메인 루프에서 호출)
    # ==========================================================================
    def update(self, frame_count=0):
        if self.root_states is None:
            raise RuntimeError("ConveyorBelt.setup_tensors()를 먼저 호출해야 합니다.")

        belt_p, belt_r = self._get_belt_pose()
        updated_indices = []

        for handle, st in self.items.items():
            changed = self._update_item(st, belt_p, belt_r)
            if changed:
                updated_indices.append(st["actor_sim_index"])

        if updated_indices:
            self._push_root_states(updated_indices)

        # [FIX 3] 거리 기준 이동 중이면, 기준 아이템 유효성 확인 및 거리 갱신/자동 정지 판단.
        self._update_distance_tracking()

        # 텍스처 스크롤 (가동 중일 때만)
        if self.running and self.num_frames > 0:
            self._scroll_texture()

    # --- update() 내부 헬퍼 -----------------------------------------------

    def _get_belt_pose(self):
        """현재 벨트 리지드바디의 월드 위치/회전을 반환."""
        belt_pos_t = self.rb_states[self.belt_rb_sim_idx, 0:3]
        belt_rot_t = self.rb_states[self.belt_rb_sim_idx, 3:7]
        belt_r = gymapi.Quat(belt_rot_t[0].item(), belt_rot_t[1].item(),
                              belt_rot_t[2].item(), belt_rot_t[3].item())
        belt_p = gymapi.Vec3(belt_pos_t[0].item(), belt_pos_t[1].item(), belt_pos_t[2].item())
        return belt_p, belt_r

    def _update_item(self, st: dict, belt_p: gymapi.Vec3, belt_r: gymapi.Quat) -> bool:
        """개별 아이템 하나의 안착 판정 / 속도 주입 / 로컬 위치 기록을 수행. root_states가 변경되면 True 반환."""
        rb_idx = st["rb_sim_index"]
        actor_idx = st["actor_sim_index"]

        pos_t = self.rb_states[rb_idx, 0:3]
        cur_z = pos_t[2].item()
        z_diff = abs(cur_z - st["prev_z"])

        # 낙하 이후 z가 안정화되면 "안착"으로 판정 (기준: LANDING_Z_STABLE_EPS / LANDING_MIN_HEIGHT)
        if not st["is_landed"] and z_diff < self.LANDING_Z_STABLE_EPS and cur_z > self.LANDING_MIN_HEIGHT:
            st["is_landed"] = True

        new_state = self.root_states[actor_idx].clone()
        changed = False

        if self.running and st["is_landed"] and not st["grabbed"]:
            local_vel = gymapi.Vec3(0.0, self.direction * self.speed, 0.0)
            world_vel = belt_r.rotate(local_vel)

            rot_t = self.rb_states[rb_idx, 3:7]
            cur_q = gymapi.Quat(rot_t[0].item(), rot_t[1].item(), rot_t[2].item(), rot_t[3].item())
            flat_q = _flatten_keep_yaw(cur_q)

            new_state[3], new_state[4], new_state[5], new_state[6] = \
                flat_q.x, flat_q.y, flat_q.z, flat_q.w
            new_state[7], new_state[8], new_state[9] = world_vel.x, world_vel.y, world_vel.z
            new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
            changed = True

        elif st["is_landed"] and (not self.running or st["grabbed"]):
            # 정지 상태이거나 로봇이 붙잡고 있으면 컨베이어發 속도는 0으로
            new_state[7], new_state[8], new_state[9] = 0.0, 0.0, 0.0
            new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
            changed = True

        st["prev_z"] = cur_z

        if changed:
            self.root_states[actor_idx] = new_state

        # [FIX 3] 거리 측정을 위해 "기준 아이템 여부와 무관하게" 모든 landed 아이템의
        # 로컬 y 위치를 매 프레임 기록해둔다. 이렇게 해야 기준 아이템이 도중에
        # unregister되어도 다른 landed 아이템으로 즉시(끊김 없이) 대체할 수 있다.
        if self.running and self.target_distance is not None and st["is_landed"]:
            world_pos = gymapi.Vec3(pos_t[0].item(), pos_t[1].item(), cur_z)
            local_pos = _world_to_local(belt_p, belt_r, world_pos)
            if st["start_local_y"] is None:
                st["start_local_y"] = local_pos.y
            st["last_local_y"] = local_pos.y

        return changed

    def _update_distance_tracking(self):
        """
        [FIX 3] 거리 기준 이동 중일 때, 기준 아이템(_distance_ref_item)의 유효성을
        확인하고 이동 거리를 갱신한다. 기준 아이템이 unregister되었거나 아직
        안착하지 않았다면, landed & start_local_y가 기록된 다른 아이템으로 자동 교체한다.
        """
        if not (self.running and self.target_distance is not None):
            return

        ref_handle = self._distance_ref_item
        ref_st = self.items.get(ref_handle)

        # 기준 아이템이 없거나(제거됨) 아직 위치 기록이 없다면 대체 아이템을 찾는다.
        if ref_st is None or ref_st.get("start_local_y") is None or ref_st.get("last_local_y") is None:
            candidate_handle = next(
                (h for h, s in self.items.items()
                 if s.get("start_local_y") is not None and s.get("last_local_y") is not None),
                None
            )
            if candidate_handle is None:
                return  # 아직 측정 가능한 아이템이 하나도 없음 (예: 전부 낙하 중)
            self._distance_ref_item = candidate_handle
            ref_st = self.items[candidate_handle]

        self._moved_distance = abs(ref_st["last_local_y"] - ref_st["start_local_y"])
        if self._moved_distance >= self.target_distance:
            self.stop()

    def _push_root_states(self, updated_indices: list):
        """변경된 root_states만 골라 물리 엔진에 반영."""
        idx_t = torch.tensor(updated_indices, dtype=torch.int32, device=self.root_states.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(idx_t),
            len(updated_indices)
        )

    def _scroll_texture(self):
        """벨트 표면 텍스처를 다음 프레임으로 스크롤."""
        # [FIX 4 + FIX 7] 링크 이름을 못 찾았거나 유효한 텍스처 프레임이 없으면
        # (self.is_texture_enabled == False) set_rigid_body_texture 호출 시
        # 런타임 에러가 나므로 조용히 skip한다. 플래그 하나로 판단을 통일해
        # belt_rb_local_idx/num_frames를 따로 체크하던 중복 로직을 제거했다.
        if not self.is_texture_enabled:
            return

        # [FIX 1] 방향(self.direction)을 매 프레임 곱해서 정방향/역방향 이동 시
        # 텍스처 스크롤 방향도 함께 반전되도록 한다.
        self.texture_pixel_pos += self._texture_pixels_per_frame_base * self.direction
        tex_idx = int(self.texture_pixel_pos) % self.num_frames
        self.gym.set_rigid_body_texture(
            self.env, self.handle, self.belt_rb_local_idx, gymapi.MESH_VISUAL,
            self.texture_handles[tex_idx]
        )

    def seconds_to_frames(self, seconds):
        """self.dt 기준으로 초를 프레임 수로 환산. ConveyorStepRunner의 hold_seconds 대기 판정에 사용."""
        return max(0, int(round(seconds / self.dt)))