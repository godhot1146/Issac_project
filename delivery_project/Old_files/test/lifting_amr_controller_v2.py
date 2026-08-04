import numpy as np
from isaacgym import gymapi

class LiftingAMR:
    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # -----------------------------------------------------------------
        # 1. 하드웨어 제원 및 제어 파라미터 (상수 설정)
        # -----------------------------------------------------------------
        self.TRACK_WIDTH = 0.80     # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.15    # 구동 바퀴 반지름 (m)
        
        # 주행 속도 및 가속도 한계 세팅
        self.MAX_LINEAR_VEL = 1.5   # 최대 선속도 (m/s)
        self.MAX_ANGULAR_VEL = 3.5  # 최대 각속도 (rad/s)
        self.MAX_LINEAR_ACCEL = 0.04 # 프레임당 최대 선속도 변화량
        self.MAX_ANGULAR_ACCEL = 0.4 # 프레임당 최대 각속도 변화량

        self.Kp_LIN = 2.5           # 선속도 제어 비례 게인
        self.Kp_ANG = 6.0           # 조향 각속도 제어 비례 게인
        self.ARRIVE_THRESHOLD = 0.01 # 목적지 도달 인정 공차 (5cm)

        # 리프트 조인트 설정
        self.LIFT_JOINT_NAME = "lift_z_joint"
        self.LIFT_MIN_HEIGHT = 0.0
        self.LIFT_MAX_HEIGHT = 2.5

        self.ROTATION_JOINT_NAME = "lift_rotation_joint"

        self.FORK_JOINT_NAME = "fork_extension_joint"
        self.FORK_MIN_LIMIT = 0.0
        self.FORK_MAX_LIMIT = 1.2  # URDF 상 upper="0.7" 반영

        self.LEFT_CLAW_JOINT_NAME = "left_claw_joint"
        self.RIGHT_CLAW_JOINT_NAME = "right_claw_joint"
        self.CLAW_MIN_LIMIT = -0.8  # URDF lower
        self.CLAW_MAX_LIMIT = 0.8   # URDF upper

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "cast_wheel_link3", "cast_wheel_link4"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]

        # -----------------------------------------------------------------
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # -----------------------------------------------------------------
        self.current_v_lin = 0.0
        self.current_v_ang = 0.0
        
        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.rot_joint_idx = -1
        self.dof_count = 0

        self.grid_move_stage = 0     # 0: X축 맞추기 단계, 1: Y축 맞추기 단계
        self.target_yaw_fixed = None # 회전 시작 시 고정할 목표 각도
        self.turn_complete = False
        self.locked_turn_direction = None

        # 조인트 인덱스 캐싱 및 물리 튜닝 일괄 적용
        self._cache_joint_indices()
        self.configure_actuators(damping=800.0, max_effort=100000.0)
        self.configure_contact_surface(caster_friction=0.0, drive_friction=2.5)

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
        self.rot_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.ROTATION_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.fork_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.FORK_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.left_claw_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LEFT_CLAW_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.right_claw_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.RIGHT_CLAW_JOINT_NAME, gymapi.DOMAIN_ACTOR)

        if self.left_wheel_idx == -1 or self.right_wheel_idx == -1:
            print("[경고] URDF 내 구동륜 조인트 이름을 찾을 수 없습니다.")

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

        if self.lift_joint_idx != -1:
            dof_props['driveMode'][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            # 고중량 셔틀이 처지지 않게 강한 강성(stiffness)과 댐핑을 부여합니다.
            dof_props['stiffness'][self.lift_joint_idx] = 5000.0  
            dof_props['damping'][self.lift_joint_idx] = 1200.0    
            dof_props['effort'][self.lift_joint_idx] = 5000.0       # URDF 내의 effort="500" 매핑

        if self.rot_joint_idx != -1:
            dof_props['driveMode'][self.rot_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.rot_joint_idx] = 8000.0   # 회전 모터 고유 강성
            dof_props['damping'][self.rot_joint_idx] = 400.0      # 떨림 방지 감쇠
            dof_props['effort'][self.rot_joint_idx] = 100.0       # URDF 내의 effort="100"

        if self.fork_joint_idx != -1:
            dof_props['driveMode'][self.fork_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.fork_joint_idx] = 10000.0  # 전진/후진 시 처짐 및 밀림 방지 강성
            dof_props['damping'][self.fork_joint_idx] = 500.0      # 오버슈트 방지 댐핑
            dof_props['effort'][self.fork_joint_idx] = 3000.0       # URDF 내의 effort="150" 매핑

        for idx in [self.left_claw_idx, self.right_claw_idx]:
            if idx != -1:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][idx] = 5000.0   # 물체를 단단히 고정하기 위한 강성
                dof_props['damping'][idx] = 200.0     # 채터링(떨림) 방지 댐핑
                dof_props['effort'][idx] = 50.0       # URDF 내의 effort="50" 매핑

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

        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

    def set_state(self, state):
        self.current_state = state

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

        ANGLE_ARRIVE_THRESHOLD = 0.01

        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        if abs(yaw_error) < ANGLE_ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            self.locked_turn_direction = None
            return True

        if self.locked_turn_direction is None:
            self.locked_turn_direction = 1.0 if yaw_error < 0 else -1.0
        
        p_control_speed = self.Kp_ANG * abs(yaw_error)
        scaled_turn_speed = np.clip(p_control_speed, 0.6, self.MAX_ANGULAR_VEL)
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

        base_speed = np.clip(self.Kp_LIN * abs(axis_error), 0.6, self.MAX_LINEAR_VEL)
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