import numpy as np
from isaacgym import gymapi

class StepContext:
    def __init__(self, **extra):
        for k, v in extra.items():
            setattr(self, k, v)

class ForkliftMotionStep:
    def __init__(self, name,
                 target_x, target_y,
                 axis_order='x',
                 initial_facing=None,
                 direction1='FORWARD',
                 mid_facing=None,
                 direction2='FORWARD',
                 final_facing='+x',
                 lift_height=None,
                 lift_threshold=0.002,      # 🆕 리프트 도달 판정 허용오차(m)
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
        self.hold_seconds = max(0.0, hold_seconds)

        self.wait_for = wait_for
        self.on_enter = on_enter or []
        self.on_complete = on_complete or []

class ForkliftStepRunner:
    """
    ForkliftMotionStep 리스트를 순차 실행.
    각 스텝은 내부적으로 sub_stage 0~6을 거쳐 완료된다.
      0: initial_facing 정렬
      1: 1구간(axis_order 첫 축) 이동
      2: mid_facing 정렬
      3: 2구간(나머지 축) 이동           <- 여기서 XY 좌표 도달
      4: final_facing 정렬
      5: 리프트 변화 (XY/방향 도달 후에만 시작, 실제 물리 도달까지 대기)
      6: 정지 유지(hold) — 리프트 도달 후 hold_seconds만큼 대기
    """
    SUB_INITIAL_FACING = 0
    SUB_SEGMENT1 = 1
    SUB_MID_FACING = 2
    SUB_SEGMENT2 = 3
    SUB_FINAL_FACING = 4
    SUB_LIFT = 5
    SUB_HOLD = 6

    def __init__(self, forklift: "ForkliftAMR", steps, context=None):
        self.forklift = forklift
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
        """explicit_facing이 있으면 그걸 yaw로, 없으면 기존 방식대로 오차 부호로 자동 계산."""
        if explicit_facing is not None:
            return ForkliftAMR.direction_to_yaw(explicit_facing)

        curr_p, _ = self.forklift.get_pose()
        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        err = target_coord - current_val
        return self.forklift.compute_axis_yaw(axis, err, direction)

    def _advance_to_next_step(self, step):
        """현재 스텝을 완전히 종료 처리하고 다음 스텝(또는 done)으로 전환."""
        for cb in step.on_complete:
            cb(self.context)
        print(f"[ForkliftStepRunner] 스텝 완료: {step.name}")

        self.index += 1
        self.sub_stage = self.SUB_INITIAL_FACING
        self._entered_current_step = False
        self._hold_start_frame = None

        if self.index >= len(self.steps):
            self.done = True

    def update(self, frame_count=0):
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
            print(f"[ForkliftStepRunner] 스텝 진입: {step.name}")

        first_axis, second_axis = ('X', 'Y') if step.axis_order == 'x' else ('Y', 'X')
        first_target = step.target_x if first_axis == 'X' else step.target_y
        second_target = step.target_y if second_axis == 'Y' else step.target_x

        if self.sub_stage == self.SUB_INITIAL_FACING:
            yaw = self._resolve_facing(step.initial_facing, first_axis, first_target, step.direction1)
            if self.forklift.turn_to_yaw(yaw):
                self.forklift.target_yaw_fixed = yaw   # ✅ 정렬 직후 명시적으로 고정
                self.sub_stage = self.SUB_SEGMENT1

        elif self.sub_stage == self.SUB_SEGMENT1:
            arrived = self.forklift.drive_along_axis(first_axis, first_target, direction=step.direction1)
            if arrived:
                self.forklift.hard_stop()
                self.forklift.target_yaw_fixed = None   # ✅ 다음 구간(다른 축) 위해 반드시 리셋
                self.sub_stage = self.SUB_MID_FACING

        elif self.sub_stage == self.SUB_MID_FACING:
            yaw = self._resolve_facing(step.mid_facing, second_axis, second_target, step.direction2)
            if self.forklift.turn_to_yaw(yaw):
                self.forklift.target_yaw_fixed = yaw   # ✅ 명시적으로 고정
                self.sub_stage = self.SUB_SEGMENT2

        elif self.sub_stage == self.SUB_SEGMENT2:
            arrived = self.forklift.drive_along_axis(second_axis, second_target, direction=step.direction2)
            if arrived:
                self.forklift.target_yaw_fixed = None   # ✅ 다음 스텝의 SUB_SEGMENT1을 위해 반드시 리셋
                self.sub_stage = self.SUB_FINAL_FACING

        elif self.sub_stage == self.SUB_FINAL_FACING:
            final_yaw = ForkliftAMR.direction_to_yaw(step.final_facing)
            if self.forklift.turn_to_yaw(final_yaw):
                self.forklift.hard_stop()
                self.sub_stage = self.SUB_LIFT
                print(f"[ForkliftStepRunner] XY/방향 도달 완료, 리프트 변화 시작: {step.name}")

        elif self.sub_stage == self.SUB_LIFT:
            self.forklift.hard_stop()
            if step.lift_height is None:
                self._enter_hold(step, frame_count)
            else:
                self.forklift.set_lift_height(step.lift_height)
                current_lift = self.forklift.get_lift_height()
                if frame_count % 120 == 0:
                    print(f"[Forklift 물리 피드백] 리프트 현재 높이: {current_lift:.4f}m / 목표: {step.lift_height}m")
                if abs(current_lift - step.lift_height) < step.lift_threshold:
                    print(f"[ForkliftStepRunner] 리프트 목표 도달 ({current_lift:.4f}m): {step.name}")
                    self._enter_hold(step, frame_count)

        elif self.sub_stage == self.SUB_HOLD:
            self.forklift.hard_stop()
            hold_frames = self.forklift.seconds_to_frames(step.hold_seconds)
            if (frame_count - self._hold_start_frame) >= hold_frames:
                self._advance_to_next_step(step)

        self.forklift.apply_actuator_commands()

    def _enter_hold(self, step, frame_count):
        """리프트 변화(또는 생략)가 끝난 시점에 호출. hold_seconds가 0이면 즉시 다음 스텝으로."""
        if step.hold_seconds > 0.0:
            self.sub_stage = self.SUB_HOLD
            self._hold_start_frame = frame_count
            print(f"[ForkliftStepRunner] {step.hold_seconds:.1f}초 대기 시작: {step.name}")
        else:
            self._advance_to_next_step(step)

class ForkliftAMR:
    """
    포크리프트형 AMR(자율주행 이동로봇) 제어 클래스.
    차동 구동(differential drive) 주행 + 조향 캐스터 + 리프트(승강) 축을 통합 제어하며,
    ForkliftMotionStep/ForkliftStepRunner 기반의 순차 이동·정렬·리프트 시퀀스를 실행하기 위한
    저수준 구동/센싱 API(주행, 회전, 리프트 제어 등)를 제공한다.
    """

    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # ==========================================================================
        # 1. 하드웨어 제원 및 제어 파라미터 (상수 설정)
        # ==========================================================================
        self.TRACK_WIDTH = 0.70      # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.075    # 구동 바퀴 반지름 (m)

        # 주행 속도 및 가속도 한계
        self.MAX_LINEAR_VEL = 1.0    # 최대 선속도 (m/s)
        self.MIN_LINEAR_VEL = 0.6
        self.MAX_ANGULAR_VEL = 1.0   # 최대 각속도 (rad/s)
        self.MIN_ANGULAR_VEL = 0.75
        self.MAX_LINEAR_ACCEL = 0.04 # 프레임당 최대 선속도 변화량
        self.MAX_ANGULAR_ACCEL = 0.4 # 프레임당 최대 각속도 변화량

        self.Kp_LIN = 2.0            # 선속도 제어 비례 게인
        self.Kp_ANG = 2.0            # 조향 각속도 제어 비례 게인

        # 도달 판정 공차 (drive_along_axis / turn_to_yaw 가 기준이 되는 단일 소스)
        self.ARRIVE_THRESHOLD = 0.01        # 위치(XY) 도달 인정 공차 (m)
        self.ANGLE_ARRIVE_THRESHOLD = 0.01  # 각도(yaw) 도달 인정 공차 (rad)

        # 리프트 조인트 설정
        self.LIFT_JOINT_NAME = "lift_z_joint"
        self.LIFT_MIN_HEIGHT = -0.1
        self.LIFT_MAX_HEIGHT = 0.2

        self.LEFT_CASTER_JOINT_NAME = "cast_wheel_joint2"   # xyz 상 좌측 (+y)
        self.RIGHT_CASTER_JOINT_NAME = "cast_wheel_joint1"  # xyz 상 우측 (-y)

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "middle_wheel_link"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]
        self.LIFT_LINK_NAMES = ["lift_shuttle_link"]

        # ==========================================================================
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # ==========================================================================
        self.lift_height_target_desired = 0.0
        self.lift_height_ramped = 0.0

        self.LIFT_RAMP_RATE_MPS = 0.1    # 리프트 최대 승강 속도 (m/s) - 값을 낮출수록 더 천천히, 부드럽게 움직입니다.
        self.LIFT_FRAME_DT = 1.0 / 60.0  # 시뮬레이션 프레임 주기 (60Hz 가정)
        self.MAX_LIFT_STEP = self.LIFT_RAMP_RATE_MPS * self.LIFT_FRAME_DT

        self.current_v_lin = 0.0
        self.current_v_ang = 0.0

        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.dof_count = 0

        self.target_yaw_fixed = None  # 회전/직진 중 유지할 목표 각도
        self.locked_turn_direction = None

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

        self.left_caster_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LEFT_CASTER_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.right_caster_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.RIGHT_CASTER_JOINT_NAME, gymapi.DOMAIN_ACTOR)

        if self.left_wheel_idx == -1 or self.right_wheel_idx == -1:
            print("[경고] URDF 내 구동륜 조인트 이름을 찾을 수 없습니다.")
        if self.lift_joint_idx == -1:
            print("[경고] URDF 내 리프트 조인트 이름을 찾을 수 없습니다.")
        if self.left_caster_idx == -1 or self.right_caster_idx == -1:
            print("[참고] URDF 내 캐스터 조인트 이름을 찾을 수 없거나 고정 상태입니다.")

    def configure_actuators(self, damping=1000.0, max_effort=10000.0):
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

        # 조향 축이 흔들리지 않고 목표 각도를 단단하게 버티도록 stiffness를 높게 잡아줍니다.
        for idx in [self.left_caster_idx, self.right_caster_idx]:
            if idx != -1:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][idx] = 1000.0  # 지면 리프팅 저항을 견딜 수 있도록 강성 대폭 확보
                dof_props['damping'][idx] = 1.0       # 진동 떨림 제거용 감쇠
                dof_props['effort'][idx] = 100.0      # 바닥면을 찍어누를 수 있는 충분한 토크 확보

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
        """리프트 조인트의 강성/감쇠/속도/토크 한계를 튜닝합니다."""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        if self.lift_joint_idx != -1:
            dof_props["driveMode"][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][self.lift_joint_idx] = 100000.0
            dof_props["damping"][self.lift_joint_idx] = 1000.0
            dof_props["velocity"][self.lift_joint_idx] = 0.3
            dof_props["effort"][self.lift_joint_idx] = 1000000.0

        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)

    def apply_actuator_commands(self):
        """매 스텝 축적된 포지션/벨로시티 공유 버퍼 명령을 Isaac Gym 제어기에 주입합니다."""
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.dof_position_targets)
        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, self.dof_velocity_targets)

    # ==========================================================================
    # 3. LIFT SYSTEM (리프트 제어)
    # ==========================================================================
    def get_caster_angle(self):
        """현재 캐스터 텐덤 링크 조인트의 실제 포지션(Y축 회전각, rad)을 반환합니다."""
        if self.left_caster_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.left_caster_idx]

    def set_caster_angle(self, target_angle_rad):
        """
        오프셋 조향 축(캐스터 조인트)의 목표 각도를 단독 지정합니다.
        URDF에 정의된 revolute 조인트 limit (0.0 ~ 1.5708 rad) 범위를 준수합니다.
        """
        MIN_LIMIT = 0.0
        MAX_LIMIT = 1.5708

        safe_angle = np.clip(target_angle_rad, MIN_LIMIT, MAX_LIMIT)

        if self.left_caster_idx != -1:
            self.dof_position_targets[self.left_caster_idx] = safe_angle
        if self.right_caster_idx != -1:
            self.dof_position_targets[self.right_caster_idx] = safe_angle

    def set_lift_height(self, target_height):
        """
        리프팅 목표 높이(0.0m ~ 0.2m)를 공유 버퍼에 안전하게 업데이트하며,
        동시에 리프트 고도별 캐스터 각도 테이블을 기반으로 캐스터 각도를 자동 연동 주입합니다.

        [매핑 스펙]
        - 0.00m (0cm)  -> 0도
        - 0.05m (5cm)  -> 20도
        - 0.10m (10cm) -> 30도
        - 0.15m (15cm) -> 40도
        - 0.20m (20cm) -> 60도
        """
        if self.lift_joint_idx == -1:
            return

        # 1. 리프트 높이 안전 범위 클리핑 및 램프(속도 제한) 적용
        safe_height = np.clip(target_height, self.LIFT_MIN_HEIGHT, self.LIFT_MAX_HEIGHT)
        self.lift_height_target_desired = safe_height

        height_err = safe_height - self.lift_height_ramped
        step = np.clip(height_err, -self.MAX_LIFT_STEP, self.MAX_LIFT_STEP)
        self.lift_height_ramped += step

        self.dof_position_targets[self.lift_joint_idx] = self.lift_height_ramped

        # 2. 현재 리프트 높이에 따른 캐스터 각도 선형 보간
        lift_nodes = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        caster_deg_nodes = np.array([0.0, 20.0, 30.0, 40.0, 60.0])
        caster_rad_nodes = np.radians(caster_deg_nodes)
        target_caster_angle = np.interp(self.lift_height_ramped, lift_nodes, caster_rad_nodes)

        # 3. 계산된 연동 각도를 캐스터 조인트 버퍼에 일괄 주입
        if self.left_caster_idx != -1:
            self.dof_position_targets[self.left_caster_idx] = target_caster_angle
        if self.right_caster_idx != -1:
            self.dof_position_targets[self.right_caster_idx] = target_caster_angle

    def get_lift_height(self):
        """현재 리프팅 조인트의 실제 포지션(높이, m)을 반환합니다."""
        if self.lift_joint_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.lift_joint_idx]

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

        # 공유 벨로시티 버퍼는 전역 fill(0.0) 없이, 구동 바퀴 조인트 인덱스에만 정확히 주입
        # (다른 조인트, 예: 리프트의 벨로시티 명령 슬롯을 침범하지 않도록)
        if self.left_wheel_idx != -1:
            self.dof_velocity_targets[self.left_wheel_idx] = rad_sec_left
        if self.right_wheel_idx != -1:
            self.dof_velocity_targets[self.right_wheel_idx] = rad_sec_right

    _DIRECTION_TO_YAW = {
        '+x': 0.0,
        '+y': np.pi / 2.0,
        '-x': np.pi,
        '-y': -np.pi / 2.0,
    }

    @staticmethod
    def direction_to_yaw(direction_str):
        """'+x'/'-x'/'+y'/'-y' 문자열을 목표 yaw(rad)로 변환."""
        clean = str(direction_str).strip().lower()
        if clean not in ForkliftAMR._DIRECTION_TO_YAW:
            raise ValueError(f"알 수 없는 방향 문자열: '{direction_str}' (허용: +x,-x,+y,-y)")
        return ForkliftAMR._DIRECTION_TO_YAW[clean]

    @staticmethod
    def compute_axis_yaw(axis, axis_error, direction="FORWARD"):
        """
        축(X/Y)과 목표까지의 부호 있는 오차(axis_error), 진행 방향(FORWARD/BACKWARD)으로부터
        차체가 향해야 할 목표 yaw(rad)를 계산합니다.

        - _resolve_facing()(스텝 진입 시 자동 정렬 각도 산출)과
          drive_along_axis()(주행 중 유지할 target_yaw_fixed 산출)가 공통으로 사용하는
          단일 소스이며, 두 곳에서 각각 구현되어 있던 동일 로직을 통합한 것입니다.
        """
        is_backward = str(direction).strip().upper() == 'BACKWARD'

        if axis.upper() == 'X':
            base_yaw = 0.0 if axis_error > 0 else np.pi
        else:
            base_yaw = np.pi / 2.0 if axis_error > 0 else -np.pi / 2.0

        if is_backward:
            yaw = (base_yaw + np.pi) % (2.0 * np.pi)
            if yaw > np.pi:
                yaw -= 2.0 * np.pi
            return yaw
        return base_yaw

    def turn_to_yaw(self, target_yaw):
        """선회 시작 시 방향을 잠금(Lock)하여 차체 떨림을 차단하는 정밀 회전 함수입니다."""
        _, curr_r = self.get_pose()
        curr_yaw = self._get_current_yaw(curr_r)

        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        if abs(yaw_error) < self.ANGLE_ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            self.locked_turn_direction = None
            return True

        if self.locked_turn_direction is None:
            self.locked_turn_direction = 1.0 if yaw_error >= 0 else -1.0

        p_control_speed = self.Kp_ANG * abs(yaw_error)
        scaled_turn_speed = np.clip(p_control_speed, self.MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        angular_cmd = self.locked_turn_direction * scaled_turn_speed

        self.set_twist_velocity(linear_x=0.0, angular_z=angular_cmd)
        return False

    def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
        curr_p, curr_r = self.get_pose()

        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        axis_error = target_coordinate - current_val

        if abs(axis_error) < self.ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            return True

        base_speed = np.clip(self.Kp_LIN * abs(axis_error), self.MIN_LINEAR_VEL, self.MAX_LINEAR_VEL)
        linear_cmd = base_speed if direction.upper() == "FORWARD" else -base_speed

        if self.target_yaw_fixed is None:
            self.target_yaw_fixed = self.compute_axis_yaw(axis, axis_error, direction)

        curr_yaw = self._get_current_yaw(curr_r)
        yaw_error = np.arctan2(np.sin(self.target_yaw_fixed - curr_yaw), np.cos(self.target_yaw_fixed - curr_yaw))

        Kp_STRAIGHT_ANG = 4.5
        angular_correction = Kp_STRAIGHT_ANG * yaw_error
        # BACKWARD 반전 없음: yaw 보정은 차동구동 특성상 전진/후진과 무관하게
        # "현재 orientation을 target_yaw_fixed로 수렴시키는 방향"으로 항상 동일해야 함.
        # (turn_to_yaw()도 이런 반전 없이 동작하며, 두 함수가 같은 target_yaw_fixed를
        #  공유하므로 부호 규칙도 일치해야 함)

        self.set_twist_velocity(linear_x=linear_cmd, angular_z=angular_correction)
        return False

    # ==========================================================================
    # 정지 / 시간 유틸리티
    # ==========================================================================
    def hard_stop(self):
        """
        가속도 제한(슬루 레이트) 필터를 우회하고 즉시 속도를 0으로 강제 고정.
        set_twist_velocity(0,0)은 점진적으로만 감속되므로,
        "이 프레임에 확실히 멈춰야 한다"는 지점(스텝 완료 등)에서 사용한다.
        """
        self.current_v_lin = 0.0
        self.current_v_ang = 0.0
        if self.left_wheel_idx != -1:
            self.dof_velocity_targets[self.left_wheel_idx] = 0.0
        if self.right_wheel_idx != -1:
            self.dof_velocity_targets[self.right_wheel_idx] = 0.0

    def seconds_to_frames(self, seconds):
        """LIFT_FRAME_DT(시뮬레이션 프레임 주기) 기준으로 초를 프레임 수로 환산."""
        return max(0, int(round(seconds / self.LIFT_FRAME_DT)))