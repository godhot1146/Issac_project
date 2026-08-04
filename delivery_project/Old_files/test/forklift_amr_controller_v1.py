import numpy as np
from isaacgym import gymapi

class ForkliftAMR:
    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # -----------------------------------------------------------------
        # 1. 하드웨어 제원 및 제어 파라미터 (상수 설정)
        # -----------------------------------------------------------------
        self.TRACK_WIDTH = 0.70     # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.075    # 구동 바퀴 반지름 (m)
        
        # 주행 속도 및 가속도 한계 세팅
        self.MAX_LINEAR_VEL = 1.0   # 최대 선속도 (m/s)
        self.MAX_ANGULAR_VEL = 1.0  # 최대 각속도 (rad/s)
        self.MAX_LINEAR_ACCEL = 0.04 # 프레임당 최대 선속도 변화량
        self.MAX_ANGULAR_ACCEL = 0.4 # 프레임당 최대 각속도 변화량

        self.Kp_LIN = 2.0           # 선속도 제어 비례 게인
        self.Kp_ANG = 2.0           # 조향 각속도 제어 비례 게인
        self.ARRIVE_THRESHOLD = 0.01 # 목적지 도달 인정 공차 (5cm)

        # 리프트 조인트 설정
        self.LIFT_JOINT_NAME = "lift_z_joint"
        self.LIFT_MIN_HEIGHT = -0.1
        self.LIFT_MAX_HEIGHT = 0.2

        self.LEFT_CASTER_JOINT_NAME = "cast_wheel_joint2"   # xyz 상 좌측 (+y)
        self.RIGHT_CASTER_JOINT_NAME = "cast_wheel_joint1" # xyz 상 우측 (-y)

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "middle_wheel_link"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]
        self.LIFT_LINK_NAMES = ["lift_shuttle_link"]

        # -----------------------------------------------------------------
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # -----------------------------------------------------------------

        self.lift_height_target_desired = 0.0 
        self.lift_height_ramped = 0.0

        self.LIFT_RAMP_RATE_MPS = 0.1      # 리프트 최대 승강 속도 (m/s) - 값을 낮출수록 더 천천히, 부드럽게 움직입니다.
        self.LIFT_FRAME_DT = 1.0 / 60.0     # 시뮬레이션 프레임 주기 (60Hz 가정)
        self.MAX_LIFT_STEP = self.LIFT_RAMP_RATE_MPS * self.LIFT_FRAME_DT

        self.current_v_lin = 0.0
        self.current_v_ang = 0.0
        
        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.dof_count = 0

        self.grid_move_stage = 0     # 0: X축 맞추기 단계, 1: Y축 맞추기 단계
        self.target_yaw_fixed = None # 회전 시작 시 고정할 목표 각도
        self.turn_complete = False
        self.locked_turn_direction = None

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

        # 리프트 조인트 POS 제어 모드로 분리 셋업
        if self.lift_joint_idx != -1:
            dof_props['driveMode'][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.lift_joint_idx] = 30000.0  
            dof_props['damping'][self.lift_joint_idx] = 20.0    
            dof_props['effort'][self.lift_joint_idx] = 30000.0       

        # 조향 축이 흔들리지 않고 목표 각도를 단단하게 버티도록 stiffness를 높게 잡아줍니다.
        for idx in [self.left_caster_idx, self.right_caster_idx]:
            if idx != -1:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][idx] = 1000.0  # 지면 리프팅 저항을 견딜 수 있도록 강성 대폭 확보
                dof_props['damping'][idx] = 1.0     # 진동 떨림 제거용 감쇠
                dof_props['effort'][idx] = 100.0    # 바닥면을 찍어누를 수 있는 충분한 토크 확보

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
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        if self.lift_joint_idx != -1:
            dof_props["driveMode"][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            
            # 💡 [튜닝 1] 강성(Stiffness)을 150만에서 5만~10만 수준으로 낮춥니다.
            # 25kg의 상판과 10kg 상자를 들기에는 50000.0 정도로도 충분히 단단합니다.
            dof_props["stiffness"][self.lift_joint_idx] = 10000.0
            
            # 💡 [튜닝 2] 진동을 억제하기 위해 감쇠력(Damping)을 강성에 맞춰 조절합니다.
            dof_props["damping"][self.lift_joint_idx] = 100.0
            
            # 💡 [튜닝 3] 리프트 이동 속도를 안정적인 수준으로 제한합니다. (기존 0.04에서 조금 상향 가능)
            dof_props["velocity"][self.lift_joint_idx] = 0.3
            
            # 💡 [튜닝 4] Effort(최대 토크) 한계를 99999999에서 현실적인 대형 모터 수준인 20000으로 잠급니다.
            # 이렇게 해야 한계벽에 부딪혔을 때 시뮬레이션이 터지지 않고 힘의 평형을 유지합니다.
            dof_props["effort"][self.lift_joint_idx] = 20000.0

        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)

    def set_state(self, state):
        self.current_state = state

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
        
        # 안전 범위로 클리핑
        safe_angle = np.clip(target_angle_rad, MIN_LIMIT, MAX_LIMIT)
        
        # 좌우 조향 축 조인트 인덱스가 유효하다면 각각 포지션 타겟 버퍼에 주입
        if self.left_caster_idx != -1:
            self.dof_position_targets[self.left_caster_idx] = safe_angle
            
        if self.right_caster_idx != -1:
            self.dof_position_targets[self.right_caster_idx] = safe_angle

    # -----------------------------------------------------------------
    # 3. LIFT SYSTEM FUNCTIONS (리프트 제어 파트)
    # -----------------------------------------------------------------
    def set_lift_height(self, target_height):
        """
        리프팅 목표 높이(0.0m ~ 0.2m)를 공유 버퍼에 안전하게 업데이트하며,
        동시에 리프트 고도별 캐스터 각도 테이블을 기반으로 캐스터 각도를 자동 연동 주입합니다.
        
        [매핑 스펙]
        - 0.00m (0cm)  -> 0도
        - 0.05m (5cm)  -> 15도
        - 0.10m (10cm) -> 30도
        - 0.15m (15cm) -> 45도
        - 0.20m (20cm) -> 70도
        """
        if self.lift_joint_idx != -1:
            # 1. 리프트 높이 안전 범위 클리핑 및 주입
            safe_height = np.clip(target_height, self.LIFT_MIN_HEIGHT, self.LIFT_MAX_HEIGHT)

            self.lift_height_target_desired = safe_height

            height_err = safe_height - self.lift_height_ramped
            step = np.clip(height_err, -self.MAX_LIFT_STEP, self.MAX_LIFT_STEP)
            self.lift_height_ramped += step

            self.dof_position_targets[self.lift_joint_idx] = self.lift_height_ramped

            # 2. 입력된 목표 높이에 따른 캐스터 각도 비선형 보간 연산 (미터 단위 노드 세팅)
            lift_nodes = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
            caster_deg_nodes = np.array([0.0, 10.0, 20.0, 30.0, 60.0])
            caster_rad_nodes = np.radians(caster_deg_nodes)

            # 1차 선형 보간으로 정확한 라디안 각도 계산
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

        # 공유 벨로시티 버퍼 슬롯에 주입
        # 💡 [핵심 수정] 타겟 벨로시티 버퍼의 전역 fill(0.0)을 제거합니다!
        # 다른 조인트(리프트 등)의 벨로시티 명령 슬롯을 침범하지 않도록 
        # 오직 구동 바퀴 조인트 인덱스에만 정확하게 값을 타겟팅하여 인젝션합니다.
        if self.left_wheel_idx != -1:   
            self.dof_velocity_targets[self.left_wheel_idx] = rad_sec_left
        if self.right_wheel_idx != -1:  
            self.dof_velocity_targets[self.right_wheel_idx] = rad_sec_right

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
            self.locked_turn_direction = 1.0 if yaw_error >= 0 else -1.0
        
        p_control_speed = self.Kp_ANG * abs(yaw_error)
        value = 0.75
        if self.get_lift_height()>0.1:
            value = 0.85
        scaled_turn_speed = np.clip(p_control_speed, 0.75, self.MAX_ANGULAR_VEL)
        angular_cmd = self.locked_turn_direction * scaled_turn_speed
        
        self.set_twist_velocity(linear_x=0.0, angular_z=angular_cmd)
        return False

    # def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
    #     """지정된 축을 따라 정선 주행하는 독립 제어 함수입니다."""
    #     curr_p, curr_r = self.get_pose()
    #     DISTANCE_THRESHOLD = 0.01

    #     current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
    #     axis_error = target_coordinate - current_val

    #     if abs(axis_error) < DISTANCE_THRESHOLD:
    #         self.set_twist_velocity(0.0, 0.0)
    #         return True

    #     base_speed = np.clip(self.Kp_LIN * abs(axis_error), 0.6, self.MAX_LINEAR_VEL)
    #     # base_speed = self.MAX_LINEAR_VEL
    #     linear_cmd = base_speed if direction.upper() == "FORWARD" else -base_speed

    #     # 🎯 [직진 중 흔들림 포착 레이더]
    #     # 직진 주행 중에 원래 유지해야 하는 고정 각도(self.target_yaw_fixed)와 
    #     # 현재 실제 각도(curr_yaw)의 차이를 추적합니다.
    #     if self.target_yaw_fixed is None:
    #          # 만약 고정 각도가 비어있다면 현재 축 방향을 기준으로 역산 (0, pi, pi/2, -pi/2 중 하나)
    #          curr_yaw = self._get_current_yaw(curr_r)
    #          self.target_yaw_fixed = curr_yaw

    #     curr_yaw = self._get_current_yaw(curr_r)
    #     yaw_error_during_straight = np.arctan2(np.sin(self.target_yaw_fixed - curr_yaw), np.cos(self.target_yaw_fixed - curr_yaw))

    #     # 각도 오차가 1.5도(약 0.025 rad) 이상 틀어지는 순간을 포착
    #     if abs(yaw_error_during_straight) > 0.025:
    #         print(f"⚠️ [직진 흔들림 포착] 차체 방향 유실! "
    #               f"유지해야할 각도: {np.degrees(self.target_yaw_fixed):.1f}° | "
    #               f"현재 각도: {np.degrees(curr_yaw):.1f}° | "
    #               f"틀어진 각도: {np.degrees(yaw_error_during_straight):.2f}°")

    #     self.set_twist_velocity(linear_x=linear_cmd, angular_z=0.0)
    #     return False

    def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
        curr_p, curr_r = self.get_pose()
        DISTANCE_THRESHOLD = 0.01

        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        axis_error = target_coordinate - current_val

        if abs(axis_error) < DISTANCE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            return True

        base_speed = np.clip(self.Kp_LIN * abs(axis_error), 0.6, self.MAX_LINEAR_VEL)
        linear_cmd = base_speed if direction.upper() == "FORWARD" else -base_speed

        # 1. 주행 축 방향 수립 및 방어적 타겟 셋팅
        if self.target_yaw_fixed is None:
            curr_yaw = self._get_current_yaw(curr_r)
            if axis.upper() == 'X':
                self.target_yaw_fixed = 0.0 if linear_cmd > 0 else np.pi
            else:
                self.target_yaw_fixed = np.pi / 2.0 if linear_cmd > 0 else -np.pi / 2.0

        # 2. 현재 실제 Yaw 각도 계산 및 오차 산출
        curr_yaw = self._get_current_yaw(curr_r)
        yaw_error = np.arctan2(np.sin(self.target_yaw_fixed - curr_yaw), np.cos(self.target_yaw_fixed - curr_yaw))

        # 💡 [핵심 수정 포인트] 
        # 로봇이 0도, 180도, 90도 어디를 바라보든 상관없이 '내가 가고자 하는 전진 방향'을 기준으로 
        # 오차가 양수인지 음수인지 판단하여 조향 타겟 부호를 완벽히 정렬합니다.
        Kp_STRAIGHT_ANG = 4.5 # 직진 중 뒤틀림을 꽉 잡아줄 조향 비례 게인
        
        # 기본 에러 보정값 계산
        angular_correction = Kp_STRAIGHT_ANG * yaw_error
        
        # 실제 주행 메커니즘 상 후진(BACKWARD)일 때만 부호를 한번 더 반전시킵니다.
        if direction.upper() == "BACKWARD":
            angular_correction = -angular_correction

        # 3. 데이터가 안정 범위로 들어오면 로그 출력을 생략하고, 크게 튈 때만 디버깅 모니터링
        if abs(np.degrees(yaw_error)) > 1.5:
            pass
            print(f"🛠️ [직진 조향 보정 가동] 목표: {np.degrees(self.target_yaw_fixed):.1f}° | 현재: {np.degrees(curr_yaw):.1f}° | 보정치: {angular_correction:.3f}")

        # 4. 최종 Twist 명령 하달
        self.set_twist_velocity(linear_x=linear_cmd, angular_z=angular_correction)
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
    
    def lift_and_locate_fsm(self, frame_count, waypoint_list,  manual_angular_cmd=0.0, use_manual_steering=False, lift_target_pos=0.16, lift_threshold=0.005):
        """
        :param frame_count: 메인 루프의 현재 프레임 카운트
        :param waypoint_list: 이동경로 4개, (x좌료, y좌표, 최종 정렬 방향, 앞뒤 방향)
        :param manual_angular_cmd: 수동 회전 각속도 (라디안)
        :param use_manual_steering: True면 키보드 수동 조향 활성화, False면 자동 정지 대기 
        :return: (tote_trigger_signal) True일 때 LiftingAMR 탈출 시동 트리거 반환
        """

        # 내부 단계 제어용 서브 스텝 변수가 없다면 초기화
        if not hasattr(self, 'sub_step'):
            self.sub_step = 0

        current_lift_pos = self.get_lift_height()
        tote_trigger_signal = False

        # [Stage 0: 선반 밑으로 주행]
        if self.current_state == self.STATE_MOVE_TO_SHELF:
            self.set_lift_height(-0.02)
            
            # [Stage 0 - 1단계] 선반 전 경유지로 주행
            if self.sub_step == 0:
                arrived = self.move_to_point_v2(waypoint_list[0][0], waypoint_list[0][1], waypoint_list[0][2], waypoint_list[0][3])
                if arrived:
                    print(f"[FSM] 1단계 경유지 안착 완료")
                    self.sub_step = 1
            
            # [Stage 0 - 2단계] 실제 선반 밑으로 주행
            elif self.sub_step == 1:
                arrived = self.move_to_point_v2(waypoint_list[1][0], waypoint_list[1][1], waypoint_list[1][2], waypoint_list[1][3])
                if arrived:
                    print("[FSM] 2단계 선반 밑 정위치 안착 완료")
                    self.current_state = self.STATE_ALIGN_SHELF
                    self.sub_step = 0  

        # [Stage 1: 도킹 후 1초 대기]
        elif self.current_state == self.STATE_ALIGN_SHELF:
            self.set_twist_velocity(0.0, 0.0)
            if self.state_start_frame is None:
                self.state_start_frame = frame_count
                print("[FSM] 도킹 정렬 완료. 물리 안정을 위해 1초간 대기합니다...")

            if (frame_count - self.state_start_frame) >= 60:
                print("[FSM] 완전 정적 도킹 해제 완료 ➡️ STATE_LIFT_UP 단계 진입 (수직 상승)")
                self.current_state = self.STATE_LIFT_UP
                self.state_start_frame = frame_count

        # [Stage 2: 리프트 상승 및 도달 후 1초 대기]
        elif self.current_state == self.STATE_LIFT_UP:
            
            # [Stage 2 - 1단계] 리프트 상승 수행 및 안정화 대기
            if self.sub_step == 0:
                self.set_lift_height(lift_target_pos)

                if frame_count % 120 == 0:
                    print(f"[물리 피드백] 리프트 현재 높이: {current_lift_pos:.4f}m / 목표: {lift_target_pos}m")

                if abs(current_lift_pos - lift_target_pos) < lift_threshold:
                    if self.state_start_frame is None:
                        self.state_start_frame = frame_count
                    
                    if (frame_count - self.state_start_frame) >= 60:
                        print(f"[FSM] 리프트 고도 {current_lift_pos:.3f}m 안정화 완료 ➡️ 2단계 경유지 후진 시작")
                        self.sub_step = 1
                        self.state_start_frame = None

            # [Stage 2 - 2단계] 선반을 든 상태로 drive_along_axis 기능을 이용해 Y축 정선 후진 💡
            elif self.sub_step == 1:
                self.set_lift_height(lift_target_pos)  # 리프트 유지
                
                arrived = self.move_to_point_v2(waypoint_list[2][0], waypoint_list[2][1], waypoint_list[2][2], waypoint_list[2][3])
                if not arrived:
                    if frame_count % 30 == 0:
                        print(f"[후진 중] ", end='\r')
                else:
                    self.set_twist_velocity(0.0, 0.0)
                    print(f"\n[FSM] 후진 완수 ➡️ STATE_MOVE_TO_DEST 단계 진입")
                    
                    # 다음 주행(move_to_point)을 위한 주행 그리드 변수 초기화
                    self.grid_move_stage = 0     
                    self.target_yaw_fixed = None 
                    self.turn_complete = False   
                    
                    self.current_state = self.STATE_MOVE_TO_DEST
                    self.sub_step = 0  # 초기화
                    curr_p, curr_r = self.get_pose()
                    print("Stage 2 - 2단계 완료 ", curr_p)

        # [Stage 3: 목적지로 운송 주행]
        elif self.current_state == self.STATE_MOVE_TO_DEST:
            self.set_lift_height(lift_target_pos)
            arrived = self.move_to_point_v2(waypoint_list[3][0], waypoint_list[3][1], waypoint_list[3][2], waypoint_list[3][3])
            print(f"목적지로 운송 주행 중", end='\r')
            if arrived:
                print("[FSM] 물류 운송 시퀀스 최종 완수 ➡️ STATE_TASK_COMPLETE 단계 진입")
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
                    print(f"\n[] 수동 조향이 시작되었습니다. (2초간 유효)")

                # 2. 경과 시간 계산 (현재 프레임 - 시작 프레임)
                elapsed_frames = frame_count - self.manual_steering_start_frame

                # 3. 2초(120프레임) 이내일 때만 모터 속도 입력 허용
                if elapsed_frames < 120:
                    self.set_twist_velocity(linear_x=0.0, angular_z=manual_angular_cmd)
                    
                    if frame_count % 60 == 0:
                        low_p, low_r = self.get_pose()
                        low_yaw = self._get_current_yaw(low_r)
                        remaining_time = (120 - elapsed_frames) / 60.0
                        print(f"[ 수동 조향 중] 각도: {np.degrees(low_yaw):.2f}° | 남은 시간: {remaining_time:.1f}초", end='\r')
                else:
                    self.current_state = self.STATE_IDLE 
                    # 2초가 지나면 키를 누르고 있어도 강제로 멈춤
                    if frame_count % 60 == 0:
                        print("[] 수동 조향 제한 시간(2초)이 만료되어 제어권을 회수합니다.", end='\r')
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