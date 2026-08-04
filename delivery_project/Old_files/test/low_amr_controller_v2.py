import numpy as np
from isaacgym import gymapi

class LowAMR:
    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # -----------------------------------------------------------------
        # 1. 하드웨어 제원 및 제어 파라미터 (상수 설정)
        # -----------------------------------------------------------------
        self.TRACK_WIDTH = 0.60     # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.15    # 구동 바퀴 반지름 (m)
        
        # 주행 속도 및 가속도 한계 세팅
        self.MAX_LINEAR_VEL = 1.0   # 최대 선속도 (m/s)
        self.MAX_ANGULAR_VEL = 2.0  # 최대 각속도 (rad/s)
        self.MIN_LINEAR_VEL = 0.5
        self.MIN_ANGULAR_VEL = 1.0
        self.LIFT_MIN_LINEAR_VEL = 0.5
        self.LIFT_MIN_ANGULAR_VEL = 1.0
        self.MAX_LINEAR_ACCEL = 0.04 # 프레임당 최대 선속도 변화량
        self.MAX_ANGULAR_ACCEL = 0.4 # 프레임당 최대 각속도 변화량

        self.Kp_LIN = 2.5           # 선속도 제어 비례 게인
        self.Kp_ANG = 6.0           # 조향 각속도 제어 비례 게인
        self.ARRIVE_THRESHOLD = 0.01 # 목적지 도달 인정 공차 (5cm)

        # 리프트(승강) 조인트 설정 - URDF: lift_down_prismatic_joint (mast_link -> lift_down_virtual_link)
        self.LIFT_JOINT_NAME = "lift_down_prismatic_joint"
        self.LIFT_MIN_HEIGHT = 0.0
        self.LIFT_MAX_HEIGHT = 0.2   # URDF <limit lower="0.0" upper="0.2">와 동일

        # 🎯 [부드러운 승강] 목표 높이가 순간적으로 바뀌어도 실제 명령값은 이 속도로만 천천히 따라가도록
        #    슬루 레이트(Slew-Rate)를 걸어, 선반이 "퉁" 하고 튕기지 않고 부드럽게 상승/하강하도록 합니다.
        self.LIFT_RAMP_RATE_MPS = 0.1      # 리프트 최대 승강 속도 (m/s) - 값을 낮출수록 더 천천히, 부드럽게 움직입니다.
        self.LIFT_FRAME_DT = 1.0 / 60.0     # 시뮬레이션 프레임 주기 (60Hz 가정)
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
        # URDF 상 실제 리프트를 구성하는 링크 (기존 "lift_shuttle_link"는 URDF에 존재하지 않아 수정)
        self.LIFT_LINK_NAMES = ["lift_down_link", "lift_up_link"]

        # -----------------------------------------------------------------
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # -----------------------------------------------------------------
        self.current_v_lin = 0.0
        self.current_v_ang = 0.0
        
        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.lift_rotate_joint_idx = -1
        self.dof_count = 0

        self.grid_move_stage = 0     # 0: X축 맞추기 단계, 1: Y축 맞추기 단계
        self.target_yaw_fixed = None # 회전 시작 시 고정할 목표 각도
        self.turn_complete = False
        self.locked_turn_direction = None

        # 🎯 부드러운 승강을 위한 램프(slew-rate) 상태 변수
        self.lift_height_target_desired = 0.0  # 최종적으로 도달하고자 하는 목표 높이
        self.lift_height_ramped = 0.0          # 실제로 액추에이터에 주입되는, 서서히 목표를 따라가는 명령값

        # 리프트 회전 제어용 상태 변수
        self.locked_lift_rotate_direction = None

        # 🎯 리프트 카운터 회전(Yaw-Lock) 상태 변수 - 차체가 회전해도 화물의 월드 방향을 고정시키는 기능
        self.lift_yaw_lock_active = False          # 카운터 회전 기능 활성화 여부
        self.lift_yaw_lock_prev_base_yaw = None     # 직전 프레임의 차체 Yaw (델타 계산용)
        self.lift_yaw_lock_cum_delta = 0.0          # 잠금 시작 이후 누적된 차체 Yaw 변화량 (랩어라운드 방지를 위해 언랩 상태로 누적)
        self.lift_yaw_lock_base_local_angle = 0.0   # 잠금을 시작한 시점의 리프트 회전 조인트 로컬 각도(기준점)

        self.STATE_IDLE = -1

        self.STATE_MOVE_TO_SHELF    = 0
        self.STATE_ALIGN_SHELF      = 1
        self.STATE_LIFT_UP          = 2
        self.STATE_MOVE_TO_DEST     = 3
        self.STATE_TASK_COMPLETE    = 4

        self.current_state = self.STATE_IDLE
        self.state_start_frame = None
        self.manual_steering_start_frame = None

        # 조인트 인덱스 캐싱 및 물리 튜닝 일괄 적용
        self._cache_joint_indices()
        self.configure_actuators(damping=800.0, max_effort=100000.0)
        self.configure_contact_surface(caster_friction=0.0, drive_friction=2.5)
        self._configure_lift()

        # 🎯 전역 공유 제어 버퍼 세팅
        self.dof_position_targets = np.zeros(self.dof_count, dtype=np.float32)
        self.dof_velocity_targets = np.zeros(self.dof_count, dtype=np.float32)

    def apply_actuator_commands(self):
        """매 스텝 축적된 포지션/벨로시티 공유 버퍼 명령을 Isaac Gym 제어기에 주입합니다."""
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.dof_position_targets)
        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, self.dof_velocity_targets)

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

    def configure_actuators(self, damping=800.0, max_effort=10000.0):
        """모터의 구동 모드(벨로시티/포지션) 및 강성을 세팅합니다."""
        dof_props = self.gym.get_actor_of_properties(self.env, self.handle) if hasattr(self.gym, 'get_actor_of_properties') else self.gym.get_actor_dof_properties(self.env, self.handle)
        
        # 기본 속도 제어 모드로 초기화
        dof_props['driveMode'].fill(gymapi.DOF_MODE_VEL)
        dof_props['stiffness'].fill(0.0)
        dof_props['damping'].fill(damping)

        # 구동륜 모터 파워 한계 강화
        for idx in [self.left_wheel_idx, self.right_wheel_idx]:
            if idx != -1:
                dof_props['effort'][idx] = max_effort
                dof_props['velocity'][idx] = self.MAX_ANGULAR_VEL / self.WHEEL_RADIUS * 2.0

        # 리프트(승강) 조인트 POS 제어 모드로 분리 셋업
        # 🎯 실제 부드러운 속도 제어는 set_lift_height()의 슬루 레이트가 담당하므로,
        #    여기서는 목표(램프된) 위치를 오차 없이 잘 "따라가기" 위한 정도의 강성만 사용합니다.
        if self.lift_joint_idx != -1:
            dof_props['driveMode'][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.lift_joint_idx] = 40000.0  
            dof_props['damping'][self.lift_joint_idx] = 4000.0    
            dof_props['effort'][self.lift_joint_idx] = 50000.0       

        # 리프트 회전 조인트 POS 제어 모드로 분리 셋업 (URDF: effort=10000, velocity=1.0)
        if self.lift_rotate_joint_idx != -1:
            dof_props['driveMode'][self.lift_rotate_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.lift_rotate_joint_idx] = 8000.0
            dof_props['damping'][self.lift_rotate_joint_idx] = 400.0
            dof_props['effort'][self.lift_rotate_joint_idx] = 10000.0
            dof_props['velocity'][self.lift_rotate_joint_idx] = 1.0

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
                    shape_props[shape_idx].friction = 10.0           # 정적/동적 마찰 계수를 10.0으로 버스트 (강력 홀딩)
                    shape_props[shape_idx].rolling_friction = 0.5    # 굴러 떨어짐 방지 마찰 주입
                    shape_props[shape_idx].restitution = 0.0         # 반발 탄성을 0으로 묶어 결착 시 튀어오름(채터링) 제거

        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

    def _configure_lift(self):
        dof_props = self.gym.get_actor_dof_properties(
            self.env,
            self.handle
        )

        if self.lift_joint_idx != -1:
            dof_props["driveMode"][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            # 🎯 [완화 조정] 예전에는 강성/Effort를 극단적으로 키워서 목표 높이로 순간 이동하려다
            # 보니 선반이 "퉁" 하고 튕겼습니다. 지금은 set_lift_height()의 슬루 레이트가 목표를
            # 서서히 이동시켜주므로, 강성은 그 램프된 목표를 오차 없이 따라갈 정도면 충분합니다.
            dof_props["stiffness"][self.lift_joint_idx] = 200000.0
            # 진동/오버슈트 억제용 감쇠력
            dof_props["damping"][self.lift_joint_idx] = 20000.0
            # 슬루 레이트(LIFT_RAMP_RATE_MPS)와 같은 값으로 맞춰, 물리 엔진이 이중으로
            # 속도를 제한하며 서로 다른 한계끼리 충돌하지 않도록 함
            dof_props["velocity"][self.lift_joint_idx] = self.LIFT_RAMP_RATE_MPS
            # 💡 무거운 화물을 들 수 있을 만큼은 강하게, 다만 "무제한"은 아니도록 유한한 상한 설정
            dof_props["effort"][self.lift_joint_idx] = 200000.0

        if self.lift_rotate_joint_idx != -1:
            dof_props["driveMode"][self.lift_rotate_joint_idx] = gymapi.DOF_MODE_POS
            # 회전 조인트는 승강 조인트만큼 극단적인 강성이 필요 없으므로 URDF 한계치 내에서 안정적으로 세팅
            dof_props["stiffness"][self.lift_rotate_joint_idx] = 8000.0
            dof_props["damping"][self.lift_rotate_joint_idx] = 400.0
            dof_props["velocity"][self.lift_rotate_joint_idx] = 10.0
            dof_props["effort"][self.lift_rotate_joint_idx] = 30000.0

        self.gym.set_actor_dof_properties(
            self.env,
            self.handle,
            dof_props
        )

    def set_state(self, state):
        self.current_state = state

    # -----------------------------------------------------------------
    # 3. LIFT SYSTEM FUNCTIONS (리프트 제어 파트)
    # -----------------------------------------------------------------
    def set_lift_height(self, target_height):
        """
        리프팅 목표 높이(0.0m ~ 0.2m)를 공유 버퍼에 안전하게 업데이트합니다.

        🎯 [부드러운 승강] 목표값을 그대로 액추에이터에 꽂으면 큰 강성(stiffness)/최대
        출력(effort) 때문에 선반이 순간적으로 "퉁" 튕기듯 상승합니다. 이를 막기 위해
        실제로 명령되는 값(self.lift_height_ramped)은 목표값을 향해 프레임당
        MAX_LIFT_STEP(LIFT_RAMP_RATE_MPS로 결정)만큼만 서서히 이동하도록 슬루 레이트를 적용합니다.
        """
        if self.lift_joint_idx != -1:
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

    def stop_lift_rotation_at_current(self):
        """현재 각도에서 리프트 회전을 고정(홀드)합니다."""
        if self.lift_rotate_joint_idx != -1:
            current_rad = self.get_lift_rotation()
            self.set_lift_rotation(current_rad)
            self.locked_lift_rotate_direction = None

    # -----------------------------------------------------------------
    # 3-1. LIFT COUNTER-ROTATION (YAW-LOCK) FUNCTIONS
    #      차체가 회전(turn_to_yaw)하는 동안 리프트 회전 조인트를 반대 방향으로
    #      같은 각도만큼 돌려서, 리프트 위에 실린 화물(선반)이 월드 좌표계 기준으로는
    #      회전하지 않고 그대로 있는 것처럼 보이게 하는 기능입니다.
    # -----------------------------------------------------------------
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

    # def _update_lift_yaw_lock(self):
    #     """
    #     매 프레임 호출되어야 하는 카운터 회전 갱신 함수입니다.
    #     🎯 [리드(lead) 보정 방식으로 변경] velocity target을 dof_velocity_targets에
    #     직접 주입하는 feedforward는 Isaac Gym의 POS 드라이브 모드에서 오히려
    #     추종을 악화시키는 것으로 확인되어 제거했습니다. 대신 "포지션 목표 자체를
    #     예상 각속도만큼 앞서가게" 만드는 리드 보정을 적용합니다.
    #     """
    #     if not self.lift_yaw_lock_active or self.lift_rotate_joint_idx == -1:
    #         return

    #     _, curr_r = self.get_pose()
    #     curr_base_yaw = self._get_current_yaw(curr_r)

    #     step_delta = np.arctan2(
    #         np.sin(curr_base_yaw - self.lift_yaw_lock_prev_base_yaw),
    #         np.cos(curr_base_yaw - self.lift_yaw_lock_prev_base_yaw)
    #     )
    #     self.lift_yaw_lock_cum_delta += step_delta
    #     self.lift_yaw_lock_prev_base_yaw = curr_base_yaw

    #     # 순수 카운터 목표 (지연 보정 전)
    #     raw_counter_target = self.lift_yaw_lock_base_local_angle - self.lift_yaw_lock_cum_delta

    #     # 🎯 [리드 보정] 차체 각속도를 추정해 "곧 도달해야 할 위치"를 미리 목표로 던져줍니다.
    #     #    LEAD_TIME은 서보의 응답 지연 시간과 비슷한 값으로 튜닝합니다.
    #     #    (로그 상 정상상태 오차 ≈ 2° @ 약 100°/s 이므로 LEAD_TIME ≈ 0.02~0.05초부터 튜닝 권장)
    #     body_ang_vel = step_delta / self.LIFT_FRAME_DT
    #     LEAD_TIME = 0.0625  # 초 단위, 실험적으로 조정 (너무 크면 오버슈트/진동 발생)
    #     lead_counter_target = raw_counter_target - (body_ang_vel * LEAD_TIME)

    #     lead_counter_target = np.clip(lead_counter_target, self.LIFT_ROTATE_MIN, self.LIFT_ROTATE_MAX)
    #     self.set_lift_rotation(lead_counter_target)

    def _update_lift_yaw_lock(self):
        """
        매 프레임 호출되어야 하는 카운터 회전 갱신 함수입니다.
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

        # 순수 카운터 목표 (지연 보정 전)
        raw_counter_target = self.lift_yaw_lock_base_local_angle - self.lift_yaw_lock_cum_delta

        body_ang_vel = step_delta / self.LIFT_FRAME_DT
        LEAD_TIME = 0.062

        # 이번 프레임에 "이상적으로" 적용되어야 할 리드 보정량 (목표값)
        desired_lead_offset = -(body_ang_vel * LEAD_TIME)

        # 🎯 [핵심 수정] 리드 보정량을 즉시 반영하지 않고, MAX_LEAD_STEP만큼만
        # 프레임당 서서히 따라가도록 슬루 레이트를 적용합니다.
        # (리프트 높이 램프에 쓰는 MAX_LIFT_STEP과 동일한 방식)
        if not hasattr(self, '_lift_yaw_lock_offset_ramped'):
            self._lift_yaw_lock_offset_ramped = 0.0

        LEAD_RAMP_RATE_DPS = 120.0  # 리드 보정량이 초당 최대 몇 도(deg)만큼 변할 수 있는지 (값을 낮출수록 더 천천히, 부드럽게)
        MAX_LEAD_STEP = np.radians(LEAD_RAMP_RATE_DPS) * self.LIFT_FRAME_DT

        offset_err = desired_lead_offset - self._lift_yaw_lock_offset_ramped
        offset_step = np.clip(offset_err, -MAX_LEAD_STEP, MAX_LEAD_STEP)
        self._lift_yaw_lock_offset_ramped += offset_step

        lead_counter_target = raw_counter_target + self._lift_yaw_lock_offset_ramped

        lead_counter_target = np.clip(lead_counter_target, self.LIFT_ROTATE_MIN, self.LIFT_ROTATE_MAX)
        self.set_lift_rotation(lead_counter_target)

    # -----------------------------------------------------------------
    # 4. MOBILE BASE MOBILITY FUNCTIONS (주행 및 직교 분리형 제어 파트)
    # -----------------------------------------------------------------
    def get_pose(self):
        """로봇의 전역 위치 좌표(p)와 회전 쿼터니언(r)을 반환합니다."""
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        return body_states['pose']['p'][0], body_states['pose']['r'][0]

    def _get_current_yaw(self, q):
        """쿼터니언으로부터 라디안 단위의 Yaw 회전각을 계산합니다."""
        siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        cosy_cosp = 1.0 - 2.0 * (q['y']**2 + q['z']**2)
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

        # 🎯 공유 벨로시티 버퍼 슬롯에 주입
        self.dof_velocity_targets.fill(0.0)
        if self.left_wheel_idx != -1:   self.dof_velocity_targets[self.left_wheel_idx] = rad_sec_left
        if self.right_wheel_idx != -1:  self.dof_velocity_targets[self.right_wheel_idx] = rad_sec_right

    def turn_to_yaw(self, target_yaw):
        """선회 시작 시 방향을 잠금(Lock)하여 차체 떨림을 차단하는 정밀 회전 함수입니다."""
        _, curr_r = self.get_pose()
        curr_yaw = self._get_current_yaw(curr_r)

        # print("turn")

        ANGLE_ARRIVE_THRESHOLD = 0.01

        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        if abs(yaw_error) < ANGLE_ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            self.locked_turn_direction = None
            return True

        if self.locked_turn_direction is None:
            self.locked_turn_direction = 1.0 if yaw_error < 0 else -1.0
        
        p_control_speed = self.Kp_ANG * abs(yaw_error)
        value = 0.75
        if self.get_lift_height()>0.1:
            value = 0.85

        if self.get_lift_height()>0.1:
            scaled_turn_speed = np.clip(p_control_speed, self.LIFT_MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        else:
            scaled_turn_speed = np.clip(p_control_speed, self.MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        angular_cmd = self.locked_turn_direction * scaled_turn_speed
        
        self.set_twist_velocity(linear_x=0.0, angular_z=angular_cmd)
        return False

    def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
        """지정된 축을 따라 정선 주행하는 독립 제어 함수입니다."""
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.01

        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        axis_error = target_coordinate - current_val

        if abs(axis_error) < DISTANCE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            return True

        if self.get_lift_height()>0.1:
            base_speed = np.clip(self.Kp_LIN * abs(axis_error), self.LIFT_MIN_LINEAR_VEL, self.MAX_LINEAR_VEL)
        else:
            base_speed = np.clip(self.Kp_LIN * abs(axis_error), self.MIN_LINEAR_VEL, self.MAX_LINEAR_VEL)
        
        # base_speed = self.MAX_LINEAR_VEL
        linear_cmd = base_speed if direction.upper() == "FORWARD" else -base_speed

        self.set_twist_velocity(linear_x=linear_cmd, angular_z=0.0)
        return False

    def move_to_point(self, target_x, target_y, final_direction='+x'):
        """
        [방향 문자열 적용판]
        Stage 0: X축 정렬 및 주행
        Stage 1: Y축 정렬 및 주행
        Stage 2: 주행 종료 후 final_direction('+x', '-x', '+y', '-y') 방향으로 최종 제자리 정렬
        
        :param final_direction: '+x', '-x', '+y', '-y' (대소문자 무관)
        """
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.05

        # [Stage 0: X축 정렬 및 주행]
        if self.grid_move_stage == 0:
            err_x = target_x - curr_p['x']
            if abs(err_x) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None 
                self.turn_complete = False  
                self.grid_move_stage = 1     
                return False

            if self.target_yaw_fixed is None:
                self.target_yaw_fixed = 0.0 if err_x > 0 else np.pi
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='X', target_coordinate=target_x, direction="FORWARD")

        # [Stage 1: Y축 정렬 및 주행]
        elif self.grid_move_stage == 1:
            err_y = target_y - curr_p['y']
            if abs(err_y) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 2 
                return False

            if self.target_yaw_fixed is None:
                self.target_yaw_fixed = np.pi / 2.0 if err_y > 0 else -np.pi / 2.0
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='Y', target_coordinate=target_y, direction="FORWARD")

        # [Stage 2: 지정된 방향 문자열을 기반으로 최종 Yaw 제정렬]
        elif self.grid_move_stage == 2:
            # 💡 방향 문자열 매핑 딕셔너리 정의
            direction_map = {
                '+x': 0.0,
                '+y': np.pi / 2.0,
                '-x': np.pi,
                '-y': -np.pi / 2.0
            }
            
            # 입력받은 문자열 정제 및 예외 처리 (정의되지 않은 값일 경우 기본값 +x 축 정렬)
            clean_dir = str(final_direction).strip().lower()
            calculated_final_yaw = direction_map.get(clean_dir, 0.0)
            
            # 계산된 라디안 각도로 제자리 회전 지시
            if self.turn_to_yaw(calculated_final_yaw):
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 0 
                print(f"[주행 완료] 최종 지정 방향 '{clean_dir}' ({calculated_final_yaw} rad) 정렬 완료.")
                return True 

        return False
    
    def move_to_point_v2(self, target_x, target_y, final_direction='+x', direction='FORWARD'):
        """
        [방향 문자열 및 전/후진 적용판]
        Stage 0: X축 정렬 및 주행 (FORWARD / BACKWARD 반영)
        Stage 1: Y축 정렬 및 주행 (FORWARD / BACKWARD 반영)
        Stage 2: 주행 종료 후 final_direction('+x', '-x', '+y', '-y') 방향으로 최종 제자리 정렬
        
        :param final_direction: '+x', '-x', '+y', '-y' (대소문자 무관)
        :param direction: 'FORWARD' 또는 'BACKWARD' (대소문자 무관)
        """
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.05
        
        # 전/후진 파라미터 정제
        drive_dir = str(direction).strip().upper()
        is_backward = (drive_dir == 'BACKWARD')

        # [Stage 0: X축 정렬 및 주행]
        if self.grid_move_stage == 0:
            err_x = target_x - curr_p['x']
            if abs(err_x) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None 
                self.turn_complete = False  
                self.grid_move_stage = 1     
                return False

            if self.target_yaw_fixed is None:
                # 기본 전진 방향 설정
                base_yaw = 0.0 if err_x > 0 else np.pi
                # 후진(BACKWARD)일 경우, 로봇이 목표를 등져야 하므로 반대 방향(+pi)을 바라봄
                self.target_yaw_fixed = (base_yaw + np.pi) % (2.0 * np.pi) if is_backward else base_yaw
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='X', target_coordinate=target_x, direction=drive_dir)

        # [Stage 1: Y축 정렬 및 주행]
        elif self.grid_move_stage == 1:
            err_y = target_y - curr_p['y']
            if abs(err_y) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 2 
                return False

            if self.target_yaw_fixed is None:
                # 기본 전진 방향 설정
                base_yaw = np.pi / 2.0 if err_y > 0 else -np.pi / 2.0
                # 후진(BACKWARD)일 경우, 반대 방향(+pi)을 바라봄
                self.target_yaw_fixed = (base_yaw + np.pi)
                # -pi ~ pi 또는 0 ~ 2pi 범위 정규화 (필요시 turn_to_yaw 내부 로직에 맞춰 처리됨)
                if self.target_yaw_fixed > np.pi:
                    self.target_yaw_fixed -= 2.0 * np.pi
                    
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='Y', target_coordinate=target_y, direction=drive_dir)

        # [Stage 2: 지정된 방향 문자열을 기반으로 최종 Yaw 제정렬]
        elif self.grid_move_stage == 2:
            direction_map = {
                '+x': 0.0,
                '+y': np.pi / 2.0,
                '-x': np.pi,
                '-y': -np.pi / 2.0
            }
            
            clean_dir = str(final_direction).strip().lower()
            calculated_final_yaw = direction_map.get(clean_dir, 0.0)
            
            if self.turn_to_yaw(calculated_final_yaw):
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 0 
                print(f"[주행 완료] 최종 지정 방향 '{clean_dir}' ({calculated_final_yaw} rad) 정렬 완료.")
                return True 

        return False
    
    def lift_and_locate_fsm(self, frame_count, target_shelf, delivery_pos, manual_angular_cmd=0.0, use_manual_steering=False, lift_target_pos=0.16, lift_threshold=0.005):
        """
        [LowAMR FSM 마스터 제어 함수 - 수동 조향 옵션 인젝션판]
        
        :param frame_count: 메인 루프의 현재 프레임 카운트
        :param target_shelf: 선반 진입 타겟 좌표 (x, y)
        :param delivery_pos: 최종 목적지 좌표 (x, y)
        :param manual_angular_cmd: 수동 회전 각속도 (라디안)
        :param use_manual_steering: True면 키보드 수동 조향 활성화, False면 자동 정지 대기 💡
        :return: (tote_trigger_signal) True일 때 LiftingAMR 탈출 시동 트리거 반환
        """

        # 내부 단계 제어용 서브 스텝 변수가 없다면 초기화
        if not hasattr(self, 'sub_step'):
            self.sub_step = 0

        current_lift_pos = self.get_lift_height()
        tote_trigger_signal = False

        # [Stage 0: 선반 밑으로 주행]
        if self.current_state == self.STATE_MOVE_TO_SHELF:
            self.set_lift_height(0.0)
            
            # [Stage 0 - 1단계] 선반 전 경유지로 주행
            if self.sub_step == 0:
                arrived = self.move_to_point(target_shelf[0], target_shelf[1] - 1.3, '+y')
                if arrived:
                    print(f"[LowAMR FSM] 1단계 경유지 안착 완료 ➡️ 2단계 실제 선반 이동")
                    self.sub_step = 1
            
            # [Stage 0 - 2단계] 실제 선반 밑으로 주행
            elif self.sub_step == 1:
                arrived = self.move_to_point(target_shelf[0], target_shelf[1], '+y')
                if arrived:
                    print("[LowAMR FSM] 2단계 선반 밑 정위치 안착 완료 ➡️ STATE_ALIGN_SHELF 단계 진입")
                    self.current_state = self.STATE_ALIGN_SHELF
                    self.sub_step = 0  # 다음 서브 스텝을 위해 초기화

        # [Stage 1: 도킹 후 1초 대기]
        elif self.current_state == self.STATE_ALIGN_SHELF:
            self.set_twist_velocity(0.0, 0.0)
            if self.state_start_frame is None:
                self.state_start_frame = frame_count
                print("[LowAMR FSM] 도킹 정렬 완료. 물리 안정을 위해 1초간 대기합니다...")

            if (frame_count - self.state_start_frame) >= 60:
                print("[LowAMR FSM] 완전 정적 도킹 해제 완료 ➡️ STATE_LIFT_UP 단계 진입 (수직 상승)")
                self.current_state = self.STATE_LIFT_UP
                self.state_start_frame = frame_count

        # [Stage 2: 리프트 상승 및 도달 후 1초 대기]
        elif self.current_state == self.STATE_LIFT_UP:
            
            # [Stage 2 - 1단계] 리프트 상승 수행 및 안정화 대기
            if self.sub_step == 0:
                self.set_lift_height(lift_target_pos)

                if frame_count % 120 == 0:
                    print(f"[LowAMR 물리 피드백] 리프트 현재 높이: {current_lift_pos:.4f}m / 목표: {lift_target_pos}m")
                    print("frame_count - ", frame_count, " state_start_frame - ", self.state_start_frame)

                if abs(current_lift_pos - lift_target_pos) < lift_threshold:
                    if self.state_start_frame is None:
                        self.state_start_frame = frame_count
                    
                    if (frame_count - self.state_start_frame) >= 60:
                        print(f"[LowAMR FSM] 리프트 고도 {current_lift_pos:.3f}m 안정화 완료 ➡️ 2단계 경유지 후진 시작")
                        self.sub_step = 1
                        self.state_start_frame = None

            # [Stage 2 - 2단계] 선반을 든 상태로 drive_along_axis 기능을 이용해 Y축 정선 후진 💡
            elif self.sub_step == 1:
                self.set_lift_height(lift_target_pos)  # 리프트 유지
                
                # 목적지 Y값 계산
                backward_target_y = target_shelf[1] - 1.3
                
                # Y축을 타겟으로 잡고 뒤로(BACKWARD) 주행 지시
                arrived = self.drive_along_axis(axis='Y', target_coordinate=backward_target_y, direction="BACKWARD")
                
                if not arrived:
                    if frame_count % 30 == 0:
                        current_pos, _ = self.get_pose()
                        print(f"[LowAMR 후진 중] ", end='\r')
                else:
                    self.set_twist_velocity(0.0, 0.0)
                    print(f"\n[LowAMR FSM] 후진 완수 ➡️ STATE_MOVE_TO_DEST 단계 진입")
                    
                    # 🎯 [핵심 수정 포인트] 다음 주행(move_to_point)을 위한 주행 그리드 초기화
                    self.grid_move_stage = 0     # 0번(X축 정렬)부터 다시 시작하도록 리셋
                    self.target_yaw_fixed = None # 고정 각도 해제
                    self.turn_complete = False   # 회전 완료 플래그 리셋
                    
                    self.current_state = self.STATE_MOVE_TO_DEST
                    self.sub_step = 0  # 서브 스텝 초기화

        # [Stage 3: 목적지로 운송 주행]
        elif self.current_state == self.STATE_MOVE_TO_DEST:
            self.set_lift_height(lift_target_pos)
            arrived = self.move_to_point(delivery_pos[0], delivery_pos[1], '+y')
            print(f"목적지로 운송 주행 중", end='\r')
            if arrived:
                print("[LowAMR FSM] 물류 운송 시퀀스 최종 완수 ➡️ STATE_TASK_COMPLETE 단계 진입")
                self.current_state = self.STATE_TASK_COMPLETE
                tote_trigger_signal = True 

        # [Stage 4: 작업 완료 후 대기 또는 수동 조향]
        elif self.current_state == self.STATE_TASK_COMPLETE:
            self.set_lift_height(lift_target_pos)
            
            # 수동 조향 옵션이 켜져 있고 키 입력이 들어왔을 때
            if use_manual_steering and abs(manual_angular_cmd) > 0.0:
                
                # 1. 수동 조작이 시작된 최초 프레임을 기록 (타이머 시작)
                if self.manual_steering_start_frame is None:
                    self.manual_steering_start_frame = frame_count
                    print(f"\n[LowAMR] 수동 조향이 시작되었습니다. (2초간 유효)")

                # 2. 경과 시간 계산 (현재 프레임 - 시작 프레임)
                elapsed_frames = frame_count - self.manual_steering_start_frame

                # 3. 2초(120프레임) 이내일 때만 모터 속도 입력 허용
                if elapsed_frames < 120:
                    self.set_twist_velocity(linear_x=0.0, angular_z=manual_angular_cmd)
                    
                    if frame_count % 60 == 0:
                        low_p, low_r = self.get_pose()
                        low_yaw = self._get_current_yaw(low_r)
                        remaining_time = (120 - elapsed_frames) / 60.0
                        print(f"[LowAMR 수동 조향 중] 각도: {np.degrees(low_yaw):.2f}° | 남은 시간: {remaining_time:.1f}초", end='\r')
                else:
                    self.current_state = self.STATE_IDLE 
                    # 2초가 지나면 키를 누르고 있어도 강제로 멈춤
                    if frame_count % 60 == 0:
                        print("[LowAMR] 수동 조향 제한 시간(2초)이 만료되어 제어권을 회수합니다.", end='\r')
                    self.set_twist_velocity(0.0, 0.0)
            else:
                # 옵션이 꺼져있거나, 키 입력이 아예 없거나, 손을 뗐을 때 완전 정지 락
                self.set_twist_velocity(0.0, 0.0)
                # 키에서 손을 떼면 다음 조작 시 다시 2초를 쓸 수 있도록 타이머 초기화 (선택 사항)
                # 만약 단 한 번만 2초간 조작 가능하게 하고 싶다면 아래 라인을 주석 처리하세요.
                self.manual_steering_start_frame = None

        # 액추에이터 버퍼 명령 최종 주입
        self.apply_actuator_commands()
        
        return tote_trigger_signal
    

    def lift_and_locate_fsm_v2(self, frame_count, waypoint_list,  manual_angular_cmd=0.0, use_manual_steering=False, lift_target_pos=0.16, lift_threshold=0.005):
        """
        [LowAMR FSM 마스터 제어 함수 - 수동 조향 옵션 인젝션판]
        
        :param frame_count: 메인 루프의 현재 프레임 카운트
        :param waypoint_list: 이동경로 4개, (x좌료, y좌표, 최종 정렬 방향, 앞뒤 방향)
        :param manual_angular_cmd: 수동 회전 각속도 (라디안)
        :param use_manual_steering: True면 키보드 수동 조향 활성화, False면 자동 정지 대기 💡
        :return: (tote_trigger_signal) True일 때 LiftingAMR 탈출 시동 트리거 반환
        """

        # 내부 단계 제어용 서브 스텝 변수가 없다면 초기화
        if not hasattr(self, 'sub_step'):
            self.sub_step = 0

        current_lift_pos = self.get_lift_height()
        tote_trigger_signal = False

        # [Stage 0: 선반 밑으로 주행]
        if self.current_state == self.STATE_MOVE_TO_SHELF:
            self.set_lift_height(0.0)
            
            # [Stage 0 - 1단계] 선반 전 경유지로 주행
            if self.sub_step == 0:
                arrived = self.move_to_point_v2(waypoint_list[0][0], waypoint_list[0][1], waypoint_list[0][2], waypoint_list[0][3])
                if arrived:
                    print(f"[LowAMR FSM] 1단계 경유지 안착 완료 ➡️ 2단계 실제 선반 이동")
                    self.sub_step = 1
            
            # [Stage 0 - 2단계] 실제 선반 밑으로 주행
            elif self.sub_step == 1:
                arrived = self.move_to_point_v2(waypoint_list[1][0], waypoint_list[1][1], waypoint_list[1][2], waypoint_list[1][3])
                if arrived:
                    print("[LowAMR FSM] 2단계 선반 밑 정위치 안착 완료 ➡️ STATE_ALIGN_SHELF 단계 진입")
                    self.current_state = self.STATE_ALIGN_SHELF
                    self.sub_step = 0  # 다음 서브 스텝을 위해 초기화

        # [Stage 1: 도킹 후 1초 대기]
        elif self.current_state == self.STATE_ALIGN_SHELF:
            self.set_twist_velocity(0.0, 0.0)
            if self.state_start_frame is None:
                self.state_start_frame = frame_count
                print("[LowAMR FSM] 도킹 정렬 완료. 물리 안정을 위해 1초간 대기합니다...")

            if (frame_count - self.state_start_frame) >= 60:
                print("[LowAMR FSM] 완전 정적 도킹 해제 완료 ➡️ STATE_LIFT_UP 단계 진입 (수직 상승)")
                self.current_state = self.STATE_LIFT_UP
                self.state_start_frame = frame_count

        # [Stage 2: 리프트 상승 및 도달 후 1초 대기]
        elif self.current_state == self.STATE_LIFT_UP:
            
            # [Stage 2 - 1단계] 리프트 상승 수행 및 안정화 대기
            if self.sub_step == 0:
                self.set_lift_height(lift_target_pos)

                if frame_count % 120 == 0:
                    print(f"[LowAMR 물리 피드백] 리프트 현재 높이: {current_lift_pos:.4f}m / 목표: {lift_target_pos}m")

                if abs(current_lift_pos - lift_target_pos) < lift_threshold:
                    if self.state_start_frame is None:
                        self.state_start_frame = frame_count
                    
                    if (frame_count - self.state_start_frame) >= 60:
                        print(f"[LowAMR FSM] 리프트 고도 {current_lift_pos:.3f}m 안정화 완료 ➡️ 2단계 경유지 후진 시작")
                        self.sub_step = 1
                        self.state_start_frame = None

            # [Stage 2 - 2단계] 선반을 든 상태로 drive_along_axis 기능을 이용해 Y축 정선 후진 💡
            elif self.sub_step == 1:
                self.set_lift_height(lift_target_pos)  # 리프트 유지
                
                arrived = self.move_to_point_v2(waypoint_list[2][0], waypoint_list[2][1], waypoint_list[2][2], waypoint_list[2][3])
                if not arrived:
                    if frame_count % 30 == 0:
                        print(f"[LowAMR 후진 중] ", end='\r')
                else:
                    self.set_twist_velocity(0.0, 0.0)
                    print(f"\n[LowAMR FSM] 후진 완수 ➡️ STATE_MOVE_TO_DEST 단계 진입")
                    
                    # 다음 주행(move_to_point)을 위한 주행 그리드 변수 초기화
                    self.grid_move_stage = 0     
                    self.target_yaw_fixed = None 
                    self.turn_complete = False   
                    
                    self.current_state = self.STATE_MOVE_TO_DEST
                    self.sub_step = 0  # 초기화

        # [Stage 3: 목적지로 운송 주행]
        elif self.current_state == self.STATE_MOVE_TO_DEST:
            self.set_lift_height(lift_target_pos)
            arrived = self.move_to_point_v2(waypoint_list[3][0], waypoint_list[3][1], waypoint_list[3][2], waypoint_list[3][3])
            print(f"목적지로 운송 주행 중", end='\r')
            if arrived:
                print("[LowAMR FSM] 물류 운송 시퀀스 최종 완수 ➡️ STATE_TASK_COMPLETE 단계 진입")
                self.current_state = self.STATE_TASK_COMPLETE
                tote_trigger_signal = True 

        # [Stage 4: 작업 완료 후 대기 또는 수동 조향]
        elif self.current_state == self.STATE_TASK_COMPLETE:
            self.set_lift_height(lift_target_pos)
            
            # 수동 조향 옵션이 켜져 있고 키 입력이 들어왔을 때
            if use_manual_steering and abs(manual_angular_cmd) > 0.0:
                
                # 1. 수동 조작이 시작된 최초 프레임을 기록 (타이머 시작)
                if self.manual_steering_start_frame is None:
                    self.manual_steering_start_frame = frame_count
                    print(f"\n[LowAMR] 수동 조향이 시작되었습니다. (2초간 유효)")

                # 2. 경과 시간 계산 (현재 프레임 - 시작 프레임)
                elapsed_frames = frame_count - self.manual_steering_start_frame

                # 3. 2초(120프레임) 이내일 때만 모터 속도 입력 허용
                if elapsed_frames < 120:
                    self.set_twist_velocity(linear_x=0.0, angular_z=manual_angular_cmd)
                    
                    if frame_count % 60 == 0:
                        low_p, low_r = self.get_pose()
                        low_yaw = self._get_current_yaw(low_r)
                        remaining_time = (120 - elapsed_frames) / 60.0
                        print(f"[LowAMR 수동 조향 중] 각도: {np.degrees(low_yaw):.2f}° | 남은 시간: {remaining_time:.1f}초", end='\r')
                else:
                    self.current_state = self.STATE_IDLE 
                    # 2초가 지나면 키를 누르고 있어도 강제로 멈춤
                    if frame_count % 60 == 0:
                        print("[LowAMR] 수동 조향 제한 시간(2초)이 만료되어 제어권을 회수합니다.", end='\r')
                    self.set_twist_velocity(0.0, 0.0)
            else:
                # 옵션이 꺼져있거나, 키 입력이 아예 없거나, 손을 뗐을 때 완전 정지 락
                self.set_twist_velocity(0.0, 0.0)
                # 키에서 손을 떼면 다음 조작 시 다시 2초를 쓸 수 있도록 타이머 초기화 (선택 사항)
                # 만약 단 한 번만 2초간 조작 가능하게 하고 싶다면 아래 라인을 주석 처리하세요.
                self.manual_steering_start_frame = None

        # 액추에이터 버퍼 명령 최종 주입
        self.apply_actuator_commands()
        
        return tote_trigger_signal

    def lift_and_locate_fsm_v3(self, frame_count, waypoint_list, manual_angular_cmd=0.0, use_manual_steering=False, lift_target_pos=0.16, lift_threshold=0.007):
        """
        [LowAMR FSM 마스터 제어 함수 - 리프트 카운터 회전(Yaw-Lock) 적용판]

        v2와 흐름은 동일하지만, 화물을 들어올린 순간부터 lift_down_revolute_joint를
        차체 Yaw 변화량만큼 반대로 돌려서(카운터 회전) 화물이 월드 좌표계 기준으로는
        회전하지 않는 것처럼 보이도록 합니다. (차체가 90도 돌면 리프트 판은 -90도 돌아
        결과적으로 화물의 절대 방향은 그대로 유지됩니다.)

        :param frame_count: 메인 루프의 현재 프레임 카운트
        :param waypoint_list: 이동경로 4개, (x좌표, y좌표, 최종 정렬 방향, 앞뒤 방향)
        :param manual_angular_cmd: 수동 회전 각속도 (라디안)
        :param use_manual_steering: True면 키보드 수동 조향 활성화, False면 자동 정지 대기 💡
        :return: (tote_trigger_signal) True일 때 LiftingAMR 탈출 시동 트리거 반환
        """

        # 내부 단계 제어용 서브 스텝 변수가 없다면 초기화
        if not hasattr(self, 'sub_step'):
            self.sub_step = 0

        current_lift_pos = self.get_lift_height()
        tote_trigger_signal = False

        # [Stage 0: 선반 밑으로 주행]
        if self.current_state == self.STATE_MOVE_TO_SHELF:
            self.set_lift_height(0.0)

            # 새로운 픽업 사이클 시작 시점이므로 이전 사이클의 카운터 회전 잠금은 해제하고
            # 리프트 회전 조인트를 기준 각도(0)로 되돌려 다음 잠금을 위한 상태를 정리합니다.
            if self.lift_yaw_lock_active:
                self.stop_lift_yaw_lock()
            self.set_lift_rotation(0.0)

            # [Stage 0 - 1단계] 선반 전 경유지로 주행
            if self.sub_step == 0:
                arrived = self.move_to_point_v2(waypoint_list[0][0], waypoint_list[0][1], waypoint_list[0][2], waypoint_list[0][3])
                if arrived:
                    print(f"[LowAMR FSM] 1단계 경유지 안착 완료 ➡️ 2단계 실제 선반 이동")
                    self.sub_step = 1

            # [Stage 0 - 2단계] 실제 선반 밑으로 주행
            elif self.sub_step == 1:
                arrived = self.move_to_point_v2(waypoint_list[1][0], waypoint_list[1][1], waypoint_list[1][2], waypoint_list[1][3])
                if arrived:
                    print("[LowAMR FSM] 2단계 선반 밑 정위치 안착 완료 ➡️ STATE_ALIGN_SHELF 단계 진입")
                    self.current_state = self.STATE_ALIGN_SHELF
                    self.sub_step = 0  # 다음 서브 스텝을 위해 초기화

        # [Stage 1: 도킹 후 1초 대기]
        elif self.current_state == self.STATE_ALIGN_SHELF:
            self.set_twist_velocity(0.0, 0.0)
            if self.state_start_frame is None:
                self.state_start_frame = frame_count
                print("[LowAMR FSM] 도킹 정렬 완료. 물리 안정을 위해 1초간 대기합니다...")

            if (frame_count - self.state_start_frame) >= 60:
                print("[LowAMR FSM] 완전 정적 도킹 해제 완료 ➡️ STATE_LIFT_UP 단계 진입 (수직 상승)")
                self.current_state = self.STATE_LIFT_UP
                self.state_start_frame = frame_count

        # [Stage 2: 리프트 상승 및 도달 후 1초 대기]
        elif self.current_state == self.STATE_LIFT_UP:

            # [Stage 2 - 1단계] 리프트 상승 수행 및 안정화 대기
            if self.sub_step == 0:
                self.set_lift_height(lift_target_pos)

                if frame_count % 120 == 0:
                    print(f"[LowAMR 물리 피드백] 리프트 현재 높이: {current_lift_pos:.4f}m / 목표: {lift_target_pos}m")

                if abs(current_lift_pos - lift_target_pos) < lift_threshold:
                    if self.state_start_frame is None:
                        self.state_start_frame = frame_count

                    if (frame_count - self.state_start_frame) >= 60:
                        print(f"[LowAMR FSM] 리프트 고도 {current_lift_pos:.3f}m 안정화 완료 ➡️ 2단계 경유지 후진 시작")

                        # 🎯 화물을 완전히 들어올려 안정화된 이 시점부터 카운터 회전 잠금을 시작합니다.
                        #    이후 차체가 어떻게 회전하든 화물의 월드 방향은 지금 각도(보통 0)로 고정됩니다.
                        self.start_lift_yaw_lock(target_local_angle=0.0)
                        print("[LowAMR FSM] 리프트 카운터 회전(Yaw-Lock) 활성화 ➡️ 화물 방향 고정 시작")

                        self.sub_step = 1
                        self.state_start_frame = None

            # [Stage 2 - 2단계] 선반을 든 상태로 drive_along_axis 기능을 이용해 Y축 정선 후진 💡
            elif self.sub_step == 1:
                self.set_lift_height(lift_target_pos)  # 리프트 유지

                arrived = self.move_to_point_v2(waypoint_list[2][0], waypoint_list[2][1], waypoint_list[2][2], waypoint_list[2][3])
                if not arrived:
                    if frame_count % 30 == 0:
                        print(f"[LowAMR 후진 중] ", end='\r')
                else:
                    self.set_twist_velocity(0.0, 0.0)
                    print(f"\n[LowAMR FSM] 후진 완수 ➡️ STATE_MOVE_TO_DEST 단계 진입")

                    # 다음 주행(move_to_point)을 위한 주행 그리드 변수 초기화
                    self.grid_move_stage = 0
                    self.target_yaw_fixed = None
                    self.turn_complete = False

                    self.current_state = self.STATE_MOVE_TO_DEST
                    self.sub_step = 0  # 초기화

        # [Stage 3: 목적지로 운송 주행]
        elif self.current_state == self.STATE_MOVE_TO_DEST:
            self.set_lift_height(lift_target_pos)
            arrived = self.move_to_point_v2(waypoint_list[3][0], waypoint_list[3][1], waypoint_list[3][2], waypoint_list[3][3])
            print(f"목적지로 운송 주행 중 (화물 방향 고정 유지 중)", end='\r')
            if arrived:
                print("[LowAMR FSM] 물류 운송 시퀀스 최종 완수 ➡️ STATE_TASK_COMPLETE 단계 진입")
                self.current_state = self.STATE_TASK_COMPLETE
                tote_trigger_signal = True

        # [Stage 4: 작업 완료 후 대기 또는 수동 조향]
        elif self.current_state == self.STATE_TASK_COMPLETE:
            self.set_lift_height(lift_target_pos)

            # 수동 조향 옵션이 켜져 있고 키 입력이 들어왔을 때
            if use_manual_steering and abs(manual_angular_cmd) > 0.0:

                # 1. 수동 조작이 시작된 최초 프레임을 기록 (타이머 시작)
                if self.manual_steering_start_frame is None:
                    self.manual_steering_start_frame = frame_count
                    print(f"\n[LowAMR] 수동 조향이 시작되었습니다. (2초간 유효, 화물 방향 고정 유지)")

                # 2. 경과 시간 계산 (현재 프레임 - 시작 프레임)
                elapsed_frames = frame_count - self.manual_steering_start_frame

                # 3. 2초(120프레임) 이내일 때만 모터 속도 입력 허용
                if elapsed_frames < 120:
                    self.set_twist_velocity(linear_x=0.0, angular_z=manual_angular_cmd)

                    if frame_count % 60 == 0:
                        low_p, low_r = self.get_pose()
                        low_yaw = self._get_current_yaw(low_r)
                        remaining_time = (120 - elapsed_frames) / 60.0
                        print(f"[LowAMR 수동 조향 중] 각도: {np.degrees(low_yaw):.2f}° | 남은 시간: {remaining_time:.1f}초", end='\r')
                else:
                    self.current_state = self.STATE_IDLE
                    # 2초가 지나면 키를 누르고 있어도 강제로 멈춤
                    if frame_count % 60 == 0:
                        print("[LowAMR] 수동 조향 제한 시간(2초)이 만료되어 제어권을 회수합니다.", end='\r')
                    self.set_twist_velocity(0.0, 0.0)
            else:
                # 옵션이 꺼져있거나, 키 입력이 아예 없거나, 손을 뗐을 때 완전 정지 락
                self.set_twist_velocity(0.0, 0.0)
                # 키에서 손을 떼면 다음 조작 시 다시 2초를 쓸 수 있도록 타이머 초기화 (선택 사항)
                # 만약 단 한 번만 2초간 조작 가능하게 하고 싶다면 아래 라인을 주석 처리하세요.
                self.manual_steering_start_frame = None

        # 🎯 매 프레임 카운터 회전 갱신 (잠금이 비활성이면 내부에서 즉시 리턴되어 아무 효과 없음)
        self._update_lift_yaw_lock()
        if self.lift_yaw_lock_active:
            # 🎯 [로깅 1] 차체 회전량(프레임당 Yaw 변화량) 로그 출력
            _log_p, _log_r = self.get_pose()
            _log_curr_yaw = self._get_current_yaw(_log_r)
            if not hasattr(self, '_prev_body_yaw_log') or self._prev_body_yaw_log is None:
                self._prev_body_yaw_log = _log_curr_yaw

            _frame_yaw_delta = np.arctan2(
                np.sin(_log_curr_yaw - self._prev_body_yaw_log),
                np.cos(_log_curr_yaw - self._prev_body_yaw_log)
            )

            # 🎯 [로깅 2] 리프트 회전 조인트의 프레임당 "목표 회전량" vs "실제 회전량" 비교 로그
            #    - 목표값: 이번 프레임에 dof_position_targets에 주입된 카운터 회전 목표각
            #    - 실제값: 물리 시뮬레이션 결과 실제로 관측되는 현재 회전 조인트 각도
            _lift_rotate_target_now = (
                float(self.dof_position_targets[self.lift_rotate_joint_idx])
                if self.lift_rotate_joint_idx != -1 else 0.0
            )
            _lift_rotate_actual_now = self.get_lift_rotation()

            if not hasattr(self, '_prev_lift_rotate_target_log') or self._prev_lift_rotate_target_log is None:
                self._prev_lift_rotate_target_log = _lift_rotate_target_now
            if not hasattr(self, '_prev_lift_rotate_actual_log') or self._prev_lift_rotate_actual_log is None:
                self._prev_lift_rotate_actual_log = _lift_rotate_actual_now

            _lift_rotate_target_delta = np.arctan2(
                np.sin(_lift_rotate_target_now - self._prev_lift_rotate_target_log),
                np.cos(_lift_rotate_target_now - self._prev_lift_rotate_target_log)
            )
            _lift_rotate_actual_delta = np.arctan2(
                np.sin(_lift_rotate_actual_now - self._prev_lift_rotate_actual_log),
                np.cos(_lift_rotate_actual_now - self._prev_lift_rotate_actual_log)
            )

            # 실제로 무언가 움직인 프레임만 출력 (부동소수점 noise 필터링)
            if abs(_frame_yaw_delta) > 0.01 or abs(_lift_rotate_target_delta) > 0.01 or abs(_lift_rotate_actual_delta) > 0.01:
                print(
                    f"Frame {frame_count} | "
                    f"차체 회전량: {np.degrees(_frame_yaw_delta):.4f}° (현재 Yaw: {np.degrees(_log_curr_yaw):.2f}°) | "
                    f"리프트 목표 회전량: {np.degrees(_lift_rotate_target_delta):.4f}° (목표각: {np.degrees(_lift_rotate_target_now):.2f}°) | "
                    f"리프트 실제 회전량: {np.degrees(_lift_rotate_actual_delta):.4f}° (실제각: {np.degrees(_lift_rotate_actual_now):.2f}°) | "
                    f"추종오차: {np.degrees(_lift_rotate_target_now - _lift_rotate_actual_now):.4f}°"
                )

            self._prev_body_yaw_log = _log_curr_yaw
            self._prev_lift_rotate_target_log = _lift_rotate_target_now
            self._prev_lift_rotate_actual_log = _lift_rotate_actual_now

        # 액추에이터 버퍼 명령 최종 주입
        self.apply_actuator_commands()

        return tote_trigger_signal