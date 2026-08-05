import numpy as np
from isaacgym import gymapi

class StepContext:
    def __init__(self, **extra):
        for k, v in extra.items():
            setattr(self, k, v)

class LowAmrMotionStep:
    def __init__(self, name,
                 target_x, target_y,
                 axis_order='x',
                 initial_facing=None,
                 direction1='FORWARD',
                 mid_facing=None,
                 direction2='FORWARD',
                 final_facing='+x',
                 lift_height=None,
                 lift_threshold=0.01,
                 reset_lift_plate=False,
                 lift_rotate_angle=None,      # 🆕 rad. None이면 회전 없음(이전 각도 유지)
                 lift_rotate_threshold=0.02,  # 🆕 회전 도달 판정 허용오차(rad, 약 1.15도)
                 lift_yaw_lock=None,          # True: 이 스텝 끝(회전 완료 후)의 현재 각도를 기준으로 잠금
                                               # False: 잠금 해제
                                               # None: 현재 상태 유지
                 hold_seconds=0.1,
                 wait_for=None,
                 on_enter=None,
                 on_complete=None):
        self.name = name
        self.target_x = target_x
        self.target_y = target_y

        axis_order = str(axis_order).strip().lower()
        if axis_order not in ('x', 'y'):
            raise ValueError(f"axis_order는 'x' 또는 'y'만 가능: {axis_order}")
        self.axis_order = axis_order

        self.initial_facing = initial_facing
        self.direction1 = direction1
        self.mid_facing = mid_facing
        self.direction2 = direction2
        self.final_facing = final_facing

        self.lift_height = lift_height
        self.lift_threshold = lift_threshold
        self.reset_lift_plate = reset_lift_plate

        self.lift_rotate_angle = lift_rotate_angle
        self.lift_rotate_threshold = lift_rotate_threshold

        self.lift_yaw_lock = lift_yaw_lock
        self.hold_seconds = max(0.0, hold_seconds)

        self.wait_for = wait_for
        self.on_enter = on_enter or []
        self.on_complete = on_complete or []

class LowAmrStepRunner:
    """
    LowAmrMotionStep 리스트를 순차 실행.
      0: initial_facing 정렬
      1: 1구간 이동
      2: mid_facing 정렬
      3: 2구간 이동                      <- XY 좌표 도달
      4: final_facing 정렬
      5: 리프트 높이 변화 (XY/방향 도달 후에만 시작, 실제 물리 도달까지 대기)
      6: 리프트 판 회전 (높이 변화 완료 후에만 시작, lift_rotate_angle이 None이면 즉시 통과)
      7: 정지 유지(hold) — 회전까지 끝난 뒤 hold_seconds만큼 대기.
         이 구간이 끝나는 시점에 lift_yaw_lock 요청을 실제로 반영한다.
    """
    SUB_INITIAL_FACING = 0
    SUB_SEGMENT1 = 1
    SUB_MID_FACING = 2
    SUB_SEGMENT2 = 3
    SUB_FINAL_FACING = 4
    SUB_LIFT = 5
    SUB_LIFT_ROTATE = 6
    SUB_HOLD = 7

    def __init__(self, low_amr: "LowAMR", steps, context=None):
        self.low_amr = low_amr
        self.steps = steps
        self.context = context
        self.index = 0
        self.sub_stage = self.SUB_INITIAL_FACING
        self._entered_current_step = False
        self._hold_start_frame = None
        self.done = False

    def _current_step(self):
        return self.steps[self.index]

    def _resolve_facing(self, explicit_facing, axis, target_coord, direction):
        if explicit_facing is not None:
            return LowAMR_direction_to_yaw(explicit_facing)

        curr_p, _ = self.low_amr.get_pose()
        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        err = target_coord - current_val
        is_backward = str(direction).strip().upper() == 'BACKWARD'

        if axis.upper() == 'X':
            base_yaw = 0.0 if err > 0 else np.pi
        else:
            base_yaw = np.pi / 2.0 if err > 0 else -np.pi / 2.0

        if is_backward:
            yaw = (base_yaw + np.pi) % (2.0 * np.pi)
            if yaw > np.pi:
                yaw -= 2.0 * np.pi
            return yaw
        return base_yaw

    def _advance_to_next_step(self, step):
        for cb in step.on_complete:
            cb(self.context)
        print(f"[LowAmrStepRunner] 스텝 완료: {step.name}")

        self.index += 1
        self.sub_stage = self.SUB_INITIAL_FACING
        self._entered_current_step = False
        self._hold_start_frame = None

        if self.index >= len(self.steps):
            self.done = True

    def _apply_lift_yaw_lock_request(self, step):
        """스텝에 지정된 lift_yaw_lock 요청을 현재 상태와 비교해 반영."""
        if step.lift_yaw_lock is True:
            if not self.low_amr.lift_yaw_lock_active:
                # target_local_angle=None -> "지금 이 순간(회전까지 끝난 뒤)의 회전판 각도"를
                # 그대로 기준으로 잠금. 회전 단계(SUB_LIFT_ROTATE)가 먼저 끝나 있으므로,
                # lift_rotate_angle을 지정했다면 그 각도가 잠금 기준이 된다.
                self.low_amr.start_lift_yaw_lock(target_local_angle=None)
                print(f"[LowAmrStepRunner] 리프트 Yaw-Lock 활성화 (현재 각도 기준): {step.name}")
        elif step.lift_yaw_lock is False:
            if self.low_amr.lift_yaw_lock_active:
                self.low_amr.stop_lift_yaw_lock()
                print(f"[LowAmrStepRunner] 리프트 Yaw-Lock 해제: {step.name}")

    def update(self, frame_count=0):
        if self.done:
            return

        step = self._current_step()

        if step.wait_for is not None and not self._entered_current_step:
            if not step.wait_for(self.context):
                return

        if not self._entered_current_step:
            if step.reset_lift_plate:
                if self.low_amr.lift_yaw_lock_active:
                    self.low_amr.stop_lift_yaw_lock()
                self.low_amr.set_lift_rotation(0.0)
                print(f"[LowAmrStepRunner] 리프트 회전판 즉시 리셋: {step.name}")

            for cb in step.on_enter:
                cb(self.context)
            self._entered_current_step = True
            print(f"[LowAmrStepRunner] 스텝 진입: {step.name}")

        first_axis, second_axis = ('X', 'Y') if step.axis_order == 'x' else ('Y', 'X')
        first_target = step.target_x if first_axis == 'X' else step.target_y
        second_target = step.target_y if second_axis == 'Y' else step.target_x

        if self.sub_stage == self.SUB_INITIAL_FACING:
            yaw = self._resolve_facing(step.initial_facing, first_axis, first_target, step.direction1)
            if self.low_amr.turn_to_yaw(yaw):
                self.low_amr.target_yaw_fixed = yaw
                self.sub_stage = self.SUB_SEGMENT1

        elif self.sub_stage == self.SUB_SEGMENT1:
            arrived = self.low_amr.drive_along_axis(first_axis, first_target, direction=step.direction1)
            if arrived:
                self.low_amr.target_yaw_fixed = None
                self.sub_stage = self.SUB_MID_FACING

        elif self.sub_stage == self.SUB_MID_FACING:
            yaw = self._resolve_facing(step.mid_facing, second_axis, second_target, step.direction2)
            if self.low_amr.turn_to_yaw(yaw):
                self.low_amr.target_yaw_fixed = yaw
                self.sub_stage = self.SUB_SEGMENT2

        elif self.sub_stage == self.SUB_SEGMENT2:
            arrived = self.low_amr.drive_along_axis(second_axis, second_target, direction=step.direction2)
            if arrived:
                self.low_amr.target_yaw_fixed = None
                self.sub_stage = self.SUB_FINAL_FACING

        elif self.sub_stage == self.SUB_FINAL_FACING:
            final_yaw = LowAMR_direction_to_yaw(step.final_facing)
            if self.low_amr.turn_to_yaw(final_yaw):
                self.low_amr.set_twist_velocity(0.0, 0.0)
                self.sub_stage = self.SUB_LIFT
                print(f"[LowAmrStepRunner] XY/방향 도달 완료, 리프트 변화 시작: {step.name}")

        elif self.sub_stage == self.SUB_LIFT:
            self.low_amr.set_twist_velocity(0.0, 0.0)
            if step.lift_height is None:
                self.sub_stage = self.SUB_LIFT_ROTATE
            else:
                self.low_amr.set_lift_height(step.lift_height)
                current_lift = self.low_amr.get_lift_height()
                if frame_count % 120 == 0:
                    print(f"[LowAMR 물리 피드백] 리프트 현재 높이: {current_lift:.4f}m / 목표: {step.lift_height}m")
                if abs(current_lift - step.lift_height) < step.lift_threshold:
                    print(f"[LowAmrStepRunner] 리프트 높이 목표 도달 ({current_lift:.4f}m): {step.name}")
                    self.sub_stage = self.SUB_LIFT_ROTATE

        # 🆕 리프트 판 회전 (높이 변화가 끝난 뒤에만 진입)
        elif self.sub_stage == self.SUB_LIFT_ROTATE:
            self.low_amr.set_twist_velocity(0.0, 0.0)
            if step.lift_rotate_angle is None:
                self._enter_hold(step, frame_count)
            else:
                # Yaw-Lock이 걸려 있는 상태에서 회전판을 별도로 직접 조작하면 두 로직이
                # 충돌하므로, 회전을 시작하기 전에 잠금을 명시적으로 해제한다.
                if self.low_amr.lift_yaw_lock_active:
                    self.low_amr.stop_lift_yaw_lock()
                    print(f"[LowAmrStepRunner] 회전판 직접 조작을 위해 Yaw-Lock 임시 해제: {step.name}")

                arrived = self.low_amr.rotate_lift_to_angle(
                    step.lift_rotate_angle, threshold=step.lift_rotate_threshold
                )
                if arrived:
                    print(f"[LowAmrStepRunner] 리프트 판 회전 목표 도달 "
                          f"({np.degrees(step.lift_rotate_angle):.1f}°): {step.name}")
                    self._enter_hold(step, frame_count)

        elif self.sub_stage == self.SUB_HOLD:
            self.low_amr.set_twist_velocity(0.0, 0.0)
            hold_frames = self.low_amr.seconds_to_frames(step.hold_seconds)
            if (frame_count - self._hold_start_frame) >= hold_frames:
                # hold(안정화 대기)가 끝나는 이 시점에 lift_yaw_lock 요청을 반영한다.
                # (회전판 회전이 있었다면, 그 회전 후 각도가 그대로 잠금 기준이 됨)
                self._apply_lift_yaw_lock_request(step)
                self._advance_to_next_step(step)

        if frame_count % 120 == 0:
            curr_p, curr_r = self.low_amr.get_pose()
            curr_yaw_deg = np.degrees(self.low_amr._get_current_yaw(curr_r))
            tgt_yaw_deg = np.degrees(self.low_amr.target_yaw_fixed) if self.low_amr.target_yaw_fixed is not None else None
            print(f"[DEBUG] step={step.name} sub_stage={self.sub_stage} "
                f"pos=({curr_p['x']:.3f},{curr_p['y']:.3f}) yaw={curr_yaw_deg:.1f}° "
                f"target_yaw_fixed={tgt_yaw_deg}")

        # 매 프레임 카운터 회전(Yaw-Lock) 갱신 — 비활성 시 내부에서 즉시 리턴되어 안전
        self.low_amr._update_lift_yaw_lock()

        self.low_amr.apply_actuator_commands()

    def _enter_hold(self, step, frame_count):
        if step.hold_seconds > 0.0:
            self.sub_stage = self.SUB_HOLD
            self._hold_start_frame = frame_count
            print(f"[LowAmrStepRunner] {step.hold_seconds:.1f}초 대기 시작: {step.name}")
        else:
            self._apply_lift_yaw_lock_request(step)
            self._advance_to_next_step(step)

# ==========================================================================
# 방향 문자열 -> yaw 변환 (ForkliftAMR.direction_to_yaw와 동일 로직, LowAMR엔 없으므로 여기 정의)
# ==========================================================================
_LOW_AMR_DIRECTION_TO_YAW = {
    '+x': 0.0,
    '+y': np.pi / 2.0,
    '-x': np.pi,
    '-y': -np.pi / 2.0,
}


def LowAMR_direction_to_yaw(direction_str):
    clean = str(direction_str).strip().lower()
    if clean not in _LOW_AMR_DIRECTION_TO_YAW:
        raise ValueError(f"알 수 없는 방향 문자열: '{direction_str}' (허용: +x,-x,+y,-y)")
    return _LOW_AMR_DIRECTION_TO_YAW[clean]

class LowAMR:
    """
    저상형(low-profile) AMR 제어 클래스. ForkliftAMR과 유사하지만 캐스터 대신
    승강(lift) + 회전(turntable) 조인트로 선반을 들어올리며, 리프트가 상승한
    상태에서는 화물의 월드 방향을 고정하는 카운터 회전(Yaw-Lock) 기능을 지원한다.
    """

    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # ==========================================================================
        # 1. 하드웨어 제원 및 제어 파라미터 (상수 설정)
        # ==========================================================================
        self.TRACK_WIDTH = 0.60      # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.15     # 구동 바퀴 반지름 (m)

        # 주행 속도 및 가속도 한계
        self.MAX_LINEAR_VEL = 1.0    # 최대 선속도 (m/s)
        self.MAX_ANGULAR_VEL = 2.0   # 최대 각속도 (rad/s)
        self.MIN_LINEAR_VEL = 0.25
        self.MIN_ANGULAR_VEL = 0.5
        self.LIFT_MIN_LINEAR_VEL = 0.25   # 화물을 든 상태에서의 최소 선속도
        self.LIFT_MIN_ANGULAR_VEL = 0.5  # 화물을 든 상태에서의 최소 각속도
        self.MAX_LINEAR_ACCEL = 0.04  # 프레임당 최대 선속도 변화량
        self.MAX_ANGULAR_ACCEL = 0.4  # 프레임당 최대 각속도 변화량

        self.Kp_LIN = 2.5             # 선속도 제어 비례 게인
        self.Kp_ANG = 6.0             # 조향 각속도 제어 비례 게인

        # 리프트(승강) 조인트 설정 - URDF: lift_down_prismatic_joint (mast_link -> lift_down_virtual_link)
        self.LIFT_JOINT_NAME = "lift_down_prismatic_joint"
        self.LIFT_MIN_HEIGHT = 0.0
        self.LIFT_MAX_HEIGHT = 0.2   # URDF <limit lower="0.0" upper="0.2">와 동일

        # 부드러운 승강을 위한 슬루 레이트(Slew-Rate): 목표 높이가 순간적으로 바뀌어도
        # 실제 명령값은 이 속도로만 천천히 따라가도록 하여 선반이 "퉁" 튕기지 않게 함
        self.LIFT_RAMP_RATE_MPS = 0.1     # 리프트 최대 승강 속도 (m/s) - 값을 낮출수록 더 천천히, 부드럽게 움직입니다.
        self.LIFT_FRAME_DT = 1.0 / 60.0   # 시뮬레이션 프레임 주기 (60Hz 가정)
        self.MAX_LIFT_STEP = self.LIFT_RAMP_RATE_MPS * self.LIFT_FRAME_DT  # 프레임당 허용되는 최대 높이 변화량

        # 리프트 회전(턴테이블) 조인트 설정 - URDF: lift_down_revolute_joint (lift_down_virtual_link -> lift_down_link)
        self.LIFT_ROTATE_JOINT_NAME = "lift_down_revolute_joint"
        self.LIFT_ROTATE_MIN = -6.28  # URDF <limit lower="-6.28" upper="6.28">와 동일
        self.LIFT_ROTATE_MAX = 6.28

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "cast_wheel_link3", "cast_wheel_link4"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]
        self.LIFT_LINK_NAMES = ["lift_down_link", "lift_up_link"]

        # ==========================================================================
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # ==========================================================================
        self.current_v_lin = 0.0
        self.current_v_ang = 0.0

        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.lift_rotate_joint_idx = -1
        self.dof_count = 0

        self.target_yaw_fixed = None  # 회전 시작 시 고정할 목표 각도
        self.locked_turn_direction = None

        # 부드러운 승강을 위한 램프(slew-rate) 상태 변수
        self.lift_height_target_desired = 0.0  # 최종적으로 도달하고자 하는 목표 높이
        self.lift_height_ramped = 0.0          # 실제로 액추에이터에 주입되는, 서서히 목표를 따라가는 명령값

        # 리프트 회전 제어용 상태 변수
        self.locked_lift_rotate_direction = None

        # 리프트 카운터 회전(Yaw-Lock) 상태 변수 - 차체가 회전해도 화물의 월드 방향을 고정시키는 기능
        self.lift_yaw_lock_active = False           # 카운터 회전 기능 활성화 여부
        self.lift_yaw_lock_prev_base_yaw = None      # 직전 프레임의 차체 Yaw (델타 계산용)
        self.lift_yaw_lock_cum_delta = 0.0           # 잠금 시작 이후 누적된 차체 Yaw 변화량 (언랩 상태로 누적)
        self.lift_yaw_lock_base_local_angle = 0.0    # 잠금을 시작한 시점의 리프트 회전 조인트 로컬 각도(기준점)
        self._lift_yaw_lock_offset_ramped = 0.0      # 리드 보정량의 슬루 레이트 적용 버퍼

        # 조인트 인덱스 캐싱 및 물리 튜닝 일괄 적용
        self._cache_joint_indices()
        self.configure_actuators(damping=800.0, max_effort=100000.0)
        self.configure_contact_surface(caster_friction=0.0, drive_friction=2.5)
        self._configure_lift()

        # 전역 공유 제어 버퍼
        self.dof_position_targets = np.zeros(self.dof_count, dtype=np.float32)
        self.dof_velocity_targets = np.zeros(self.dof_count, dtype=np.float32)

    # ==========================================================================
    # 초기 설정 / 튜닝
    # ==========================================================================
    def _cache_joint_indices(self):
        """조인트 인덱스 조회를 초기 1회만 실행하여 런타임 오버헤드를 방지합니다."""
        self.dof_count = self.gym.get_actor_dof_count(self.env, self.handle)
        self.left_wheel_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LEFT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.right_wheel_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.RIGHT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.lift_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LIFT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.lift_rotate_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LIFT_ROTATE_JOINT_NAME, gymapi.DOMAIN_ACTOR)

        if self.left_wheel_idx == -1 or self.right_wheel_idx == -1:
            print("[경고] URDF 내 구동륜 조인트 이름을 찾을 수 없습니다.")
        if self.lift_joint_idx == -1:
            print("[경고] URDF 내 리프트(승강) 조인트 이름을 찾을 수 없습니다.")
        if self.lift_rotate_joint_idx == -1:
            print("[경고] URDF 내 리프트 회전(lift_down_revolute_joint) 조인트 이름을 찾을 수 없습니다.")

    def configure_actuators(self, damping=800.0, max_effort=100000.0):
        """모터의 구동 모드(벨로시티/포지션) 및 강성을 세팅합니다."""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        # 기본 속도 제어 모드로 초기화
        dof_props['driveMode'].fill(gymapi.DOF_MODE_VEL)
        dof_props['stiffness'].fill(0.0)
        dof_props['damping'].fill(damping)

        # 구동륜 모터 파워 한계 강화
        for idx in [self.left_wheel_idx, self.right_wheel_idx]:
            if idx != -1:
                dof_props['effort'][idx] = max_effort
                dof_props['velocity'][idx] = self.MAX_ANGULAR_VEL / self.WHEEL_RADIUS * 2.0

        # 🔧 리프트 승강/회전 조인트의 POS 모드 및 게인 세팅은 _configure_lift()가 전담한다.
        # (과거 이 메서드에서도 리프트 회전 조인트를 별도 값으로 세팅했으나, __init__에서
        #  _configure_lift()가 바로 뒤이어 호출되며 그 값을 덮어썼기 때문에 실질적으로는
        #  죽은 세팅이었다. 튜닝 값이 두 곳에 나뉘어 혼동을 주던 부분을 제거하고
        #  _configure_lift() 한 곳으로 일원화했다.)

        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)

    def configure_contact_surface(self, caster_friction=0.0, drive_friction=2.5):
        """캐스터와 구동륜 마찰력을 분리 제어하여 주행 슬립을 방지합니다."""
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        num_bodies = self.gym.get_actor_rigid_body_count(self.env, self.handle)
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.env, self.handle)

        for body_idx in range(num_bodies):
            body_name = rigid_body_names[body_idx]
            shape_range = self.gym.get_actor_rigid_body_shape_indices(self.env, self.handle)[body_idx]

            for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
                if body_name in self.CASTER_LINK_NAMES:
                    shape_props[shape_idx].friction = caster_friction
                    shape_props[shape_idx].rolling_friction = 0.0
                elif body_name in self.DRIVE_LINK_NAMES:
                    shape_props[shape_idx].friction = drive_friction
                    shape_props[shape_idx].rolling_friction = 0.1
                elif body_name in self.LIFT_LINK_NAMES:
                    shape_props[shape_idx].friction = 10.0          # 정적/동적 마찰 계수를 10.0으로 (강력 홀딩)
                    shape_props[shape_idx].rolling_friction = 0.5   # 굴러 떨어짐 방지 마찰 주입
                    shape_props[shape_idx].restitution = 0.0        # 반발 탄성 0 (결착 시 튀어오름/채터링 제거)

        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

    def _configure_lift(self):
        """리프트(승강/회전) 조인트의 강성/감쇠/속도/토크 한계를 튜닝합니다."""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        if self.lift_joint_idx != -1:
            dof_props["driveMode"][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            # set_lift_height()의 슬루 레이트가 목표를 서서히 이동시켜주므로,
            # 강성은 그 램프된 목표를 오차 없이 따라갈 정도면 충분합니다.
            dof_props["stiffness"][self.lift_joint_idx] = 500000.0
            # 진동/오버슈트 억제용 감쇠력
            dof_props["damping"][self.lift_joint_idx] = 20000.0
            # 슬루 레이트(LIFT_RAMP_RATE_MPS)와 같은 값으로 맞춰, 물리 엔진이 이중으로
            # 속도를 제한하며 서로 다른 한계끼리 충돌하지 않도록 함
            dof_props["velocity"][self.lift_joint_idx] = self.LIFT_RAMP_RATE_MPS
            # 무거운 화물을 들 수 있을 만큼은 강하게, 다만 "무제한"은 아니도록 유한한 상한 설정
            dof_props["effort"][self.lift_joint_idx] = 500000.0

        if self.lift_rotate_joint_idx != -1:
            dof_props["driveMode"][self.lift_rotate_joint_idx] = gymapi.DOF_MODE_POS
            # 회전 조인트는 승강 조인트만큼 극단적인 강성이 필요 없으므로 URDF 한계치 내에서 안정적으로 세팅
            dof_props["stiffness"][self.lift_rotate_joint_idx] = 8000.0
            dof_props["damping"][self.lift_rotate_joint_idx] = 400.0
            dof_props["velocity"][self.lift_rotate_joint_idx] = 10.0
            dof_props["effort"][self.lift_rotate_joint_idx] = 30000.0

        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)

    def apply_actuator_commands(self):
        """매 스텝 축적된 포지션/벨로시티 공유 버퍼 명령을 Isaac Gym 제어기에 주입합니다."""
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.dof_position_targets)
        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, self.dof_velocity_targets)

    def seconds_to_frames(self, seconds):
        """LIFT_FRAME_DT 기준으로 초를 프레임 수로 환산."""
        return max(0, int(round(seconds / self.LIFT_FRAME_DT)))

    # ==========================================================================
    # 3. LIFT SYSTEM (승강/회전 제어)
    # ==========================================================================
    def set_lift_height(self, target_height):
        """
        리프팅 목표 높이(0.0m ~ 0.2m)를 공유 버퍼에 안전하게 업데이트합니다.

        목표값을 그대로 액추에이터에 꽂으면 큰 강성(stiffness)/최대 출력(effort) 때문에
        선반이 순간적으로 "퉁" 튕기듯 상승합니다. 이를 막기 위해 실제로 명령되는 값
        (self.lift_height_ramped)은 목표값을 향해 프레임당 MAX_LIFT_STEP만큼만
        서서히 이동하도록 슬루 레이트를 적용합니다.
        """
        if self.lift_joint_idx == -1:
            return

        safe_height = np.clip(target_height, self.LIFT_MIN_HEIGHT, self.LIFT_MAX_HEIGHT)
        self.lift_height_target_desired = safe_height

        height_err = safe_height - self.lift_height_ramped
        step = np.clip(height_err, -self.MAX_LIFT_STEP, self.MAX_LIFT_STEP)
        self.lift_height_ramped += step

        self.dof_position_targets[self.lift_joint_idx] = self.lift_height_ramped

    def get_lift_height(self):
        """현재 리프팅 조인트의 실제 포지션(높이, m)을 반환합니다."""
        if self.lift_joint_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.lift_joint_idx]

    def set_lift_rotation(self, target_rad):
        """리프트 회전(턴테이블) 조인트의 목표 각도(rad)를 공유 버퍼에 안전하게 업데이트합니다."""
        if self.lift_rotate_joint_idx != -1:
            safe_rad = np.clip(target_rad, self.LIFT_ROTATE_MIN, self.LIFT_ROTATE_MAX)
            self.dof_position_targets[self.lift_rotate_joint_idx] = safe_rad

    def get_lift_rotation(self):
        """현재 리프트 회전 조인트의 실제 각도(rad)를 반환합니다."""
        if self.lift_rotate_joint_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.lift_rotate_joint_idx]

    def rotate_lift_to_angle(self, target_rad, threshold=0.01):
        """
        리프트 회전 조인트를 목표 각도(rad)로 회전시키고 도달 여부를 반환합니다.
        POS 모드로 구동되므로 목표값을 공유 버퍼에 계속 주입하며, 최단 각도 오차가
        threshold 이내로 좁혀지면 도달(True)로 판단합니다.

        :param target_rad: 목표 회전 각도 (라디안). LIFT_ROTATE_MIN ~ LIFT_ROTATE_MAX 범위로 클램프됩니다.
        :param threshold: 도달 인정 각도 공차 (라디안)
        :return: 목표 각도 도달 시 True, 그 외 False
        """
        if self.lift_rotate_joint_idx == -1:
            return False

        current_rad = self.get_lift_rotation()
        self.set_lift_rotation(target_rad)

        # 각도 오차를 -pi ~ pi 범위로 정규화하여 최단 경로 기준으로 도달 여부 판단
        angle_error = np.arctan2(np.sin(target_rad - current_rad), np.cos(target_rad - current_rad))

        if abs(angle_error) < threshold:
            self.locked_lift_rotate_direction = None
            return True

        self.locked_lift_rotate_direction = 1.0 if angle_error > 0 else -1.0
        return False

    # ==========================================================================
    # 3-1. LIFT COUNTER-ROTATION (YAW-LOCK)
    #      차체가 회전(turn_to_yaw)하는 동안 리프트 회전 조인트를 반대 방향으로
    #      같은 각도만큼 돌려서, 리프트 위에 실린 화물(선반)이 월드 좌표계 기준으로는
    #      회전하지 않고 그대로 있는 것처럼 보이게 하는 기능입니다.
    # ==========================================================================
    def start_lift_yaw_lock(self, target_local_angle=None):
        """
        리프트 카운터 회전 잠금을 시작합니다.
        호출 시점의 차체 Yaw를 기준(0)으로 잡고, 이후 차체가 회전한 만큼(delta)을
        계속 추적하여 리프트 회전 조인트에 -delta 를 더해 상쇄시킵니다.

        :param target_local_angle: 잠금 기준이 되는 리프트 로컬 각도(rad).
                                    None이면 현재 리프트 회전 조인트 각도를 그대로 기준으로 사용합니다.
        """
        _, curr_r = self.get_pose()
        self.lift_yaw_lock_active = True
        self.lift_yaw_lock_prev_base_yaw = self._get_current_yaw(curr_r)
        self.lift_yaw_lock_cum_delta = 0.0
        self.lift_yaw_lock_base_local_angle = (
            self.get_lift_rotation() if target_local_angle is None else target_local_angle
        )
        self._lift_yaw_lock_offset_ramped = 0.0

    def stop_lift_yaw_lock(self):
        """리프트 카운터 회전 잠금을 해제합니다. (마지막으로 유지되던 각도는 그대로 남습니다)"""
        self.lift_yaw_lock_active = False
        self.lift_yaw_lock_prev_base_yaw = None
        self.lift_yaw_lock_cum_delta = 0.0

    def _update_lift_yaw_lock(self):
        """
        매 프레임 호출되어야 하는 카운터 회전 갱신 함수입니다.

        리드(lead) 보정 방식: 차체 각속도를 추정해 "곧 도달해야 할 위치"를 미리 목표로
        던져주되, 그 리드 보정량 자체도 슬루 레이트를 적용해 프레임당 서서히 반영합니다.
        (velocity target을 직접 주입하는 feedforward 방식은 Isaac Gym POS 드라이브 모드에서
        오히려 추종을 악화시키는 것으로 확인되어 제외했습니다.)
        """
        if not self.lift_yaw_lock_active or self.lift_rotate_joint_idx == -1:
            return

        _, curr_r = self.get_pose()
        curr_base_yaw = self._get_current_yaw(curr_r)

        step_delta = np.arctan2(
            np.sin(curr_base_yaw - self.lift_yaw_lock_prev_base_yaw),
            np.cos(curr_base_yaw - self.lift_yaw_lock_prev_base_yaw)
        )
        self.lift_yaw_lock_cum_delta += step_delta
        self.lift_yaw_lock_prev_base_yaw = curr_base_yaw

        # 순수 카운터 목표 (리드 보정 전)
        raw_counter_target = self.lift_yaw_lock_base_local_angle - self.lift_yaw_lock_cum_delta

        body_ang_vel = step_delta / self.LIFT_FRAME_DT
        LEAD_TIME = 0.05  # 초 단위, 실험적으로 조정 (너무 크면 오버슈트/진동 발생)
        desired_lead_offset = -(body_ang_vel * LEAD_TIME)

        # 리드 보정량을 즉시 반영하지 않고, MAX_LEAD_STEP만큼만 프레임당 서서히 따라가도록
        # 슬루 레이트를 적용합니다. (리프트 높이 램프에 쓰는 MAX_LIFT_STEP과 동일한 방식)
        LEAD_RAMP_RATE_DPS = 120.0  # 리드 보정량이 초당 최대 몇 도(deg)만큼 변할 수 있는지
        MAX_LEAD_STEP = np.radians(LEAD_RAMP_RATE_DPS) * self.LIFT_FRAME_DT

        offset_err = desired_lead_offset - self._lift_yaw_lock_offset_ramped
        offset_step = np.clip(offset_err, -MAX_LEAD_STEP, MAX_LEAD_STEP)
        self._lift_yaw_lock_offset_ramped += offset_step

        lead_counter_target = raw_counter_target + self._lift_yaw_lock_offset_ramped
        lead_counter_target = np.clip(lead_counter_target, self.LIFT_ROTATE_MIN, self.LIFT_ROTATE_MAX)
        self.set_lift_rotation(lead_counter_target)

    # ==========================================================================
    # 4. MOBILE BASE MOBILITY (주행 및 조향 제어)
    # ==========================================================================
    def get_pose(self):
        """로봇의 전역 위치 좌표(p)와 회전 쿼터니언(r)을 반환합니다."""
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        return body_states['pose']['p'][0], body_states['pose']['r'][0]

    def _get_current_yaw(self, q):
        """쿼터니언으로부터 라디안 단위의 Yaw 회전각을 계산합니다."""
        siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        cosy_cosp = 1.0 - 2.0 * (q['y'] ** 2 + q['z'] ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def set_twist_velocity(self, linear_x, angular_z):
        """ROS Twist 스타일 명령을 받아 공유 벨로시티 버퍼에 속도 대상을 업데이트합니다."""
        target_v_lin = np.clip(linear_x, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        target_v_ang = np.clip(angular_z, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

        # 가속도 제한(슬루 레이트) 필터링
        v_lin_err = target_v_lin - self.current_v_lin
        self.current_v_lin += np.clip(v_lin_err, -self.MAX_LINEAR_ACCEL, self.MAX_LINEAR_ACCEL)

        v_ang_err = target_v_ang - self.current_v_ang
        self.current_v_ang += np.clip(v_ang_err, -self.MAX_ANGULAR_ACCEL, self.MAX_ANGULAR_ACCEL)

        # 차분 구동형 휠 기구학 연산
        v_left_linear = self.current_v_lin - self.current_v_ang * (self.TRACK_WIDTH / 2.0)
        v_right_linear = self.current_v_lin + self.current_v_ang * (self.TRACK_WIDTH / 2.0)

        rad_sec_left = v_left_linear / self.WHEEL_RADIUS
        rad_sec_right = v_right_linear / self.WHEEL_RADIUS

        # 공유 벨로시티 버퍼 슬롯에 주입
        self.dof_velocity_targets.fill(0.0)
        if self.left_wheel_idx != -1:
            self.dof_velocity_targets[self.left_wheel_idx] = rad_sec_left
        if self.right_wheel_idx != -1:
            self.dof_velocity_targets[self.right_wheel_idx] = rad_sec_right

    def turn_to_yaw(self, target_yaw):
        """선회 시작 시 방향을 잠금(Lock)하여 차체 떨림을 차단하는 정밀 회전 함수입니다."""
        _, curr_r = self.get_pose()
        curr_yaw = self._get_current_yaw(curr_r)

        ANGLE_ARRIVE_THRESHOLD = 0.01
        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        if abs(yaw_error) < ANGLE_ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            self.locked_turn_direction = None
            return True

        if self.locked_turn_direction is None:
            # 🔧 'yaw_error < 0' -> 'yaw_error >= 0'으로 변경 (ForkliftAMR과 동일한 부호 규칙)
            self.locked_turn_direction = 1.0 if yaw_error < 0 else -1.0

        p_control_speed = self.Kp_ANG * abs(yaw_error)

        if self.get_lift_height() > 0.1:
            scaled_turn_speed = np.clip(p_control_speed, self.LIFT_MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        else:
            scaled_turn_speed = np.clip(p_control_speed, self.MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

        angular_cmd = self.locked_turn_direction * scaled_turn_speed
        self.set_twist_velocity(linear_x=0.0, angular_z=angular_cmd)
        return False

    def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
        """지정된 축을 따라 직진 주행하며, 목표 Yaw를 유지하도록 조향을 보정합니다."""
        curr_p, curr_r = self.get_pose()   # 🔧 curr_r도 필요해짐 (기존엔 _만 받음)
        DISTANCE_THRESHOLD = 0.01

        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        axis_error = target_coordinate - current_val

        if abs(axis_error) < DISTANCE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            return True

        if self.get_lift_height() > 0.1:
            base_speed = np.clip(self.Kp_LIN * abs(axis_error), self.LIFT_MIN_LINEAR_VEL, self.MAX_LINEAR_VEL)
        else:
            base_speed = np.clip(self.Kp_LIN * abs(axis_error), self.MIN_LINEAR_VEL, self.MAX_LINEAR_VEL)

        linear_cmd = base_speed if direction.upper() == "FORWARD" else -base_speed

        # 🆕 ForkliftAMR과 동일한 방식의 직진 유지 보정 추가 (BACKWARD 반전 버그는 재현하지 않음)
        if self.target_yaw_fixed is None:
            if axis.upper() == 'X':
                self.target_yaw_fixed = 0.0 if linear_cmd > 0 else np.pi
            else:
                self.target_yaw_fixed = np.pi / 2.0 if linear_cmd > 0 else -np.pi / 2.0

        curr_yaw = self._get_current_yaw(curr_r)
        yaw_error = np.arctan2(np.sin(self.target_yaw_fixed - curr_yaw), np.cos(self.target_yaw_fixed - curr_yaw))
        Kp_STRAIGHT_ANG = 4.5
        angular_correction = -Kp_STRAIGHT_ANG * yaw_error

        self.set_twist_velocity(linear_x=linear_cmd, angular_z=angular_correction)
        return False