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
        self.LIFT_MAX_HEIGHT = 0.2

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "cast_wheel_link3", "cast_wheel_link4"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]
        self.LIFT_LINK_NAMES = ["lift_shuttle_link"]

        # -----------------------------------------------------------------
        # 2. 내부 상태 변수 및 공유 버퍼 초기화
        # -----------------------------------------------------------------
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

        if self.left_wheel_idx == -1 or self.right_wheel_idx == -1:
            print("[경고] URDF 내 구동륜 조인트 이름을 찾을 수 없습니다.")
        if self.lift_joint_idx == -1:
            print("[경고] URDF 내 리프트 조인트 이름을 찾을 수 없습니다.")

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

        # 리프트 조인트 POS 제어 모드로 분리 셋업
        if self.lift_joint_idx != -1:
            dof_props['driveMode'][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.lift_joint_idx] = 40000.0  
            dof_props['damping'][self.lift_joint_idx] = 20.0    
            dof_props['effort'][self.lift_joint_idx] = 50000.0       

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
            # 💡 [상향 보정] 고중량에 눌려 멈추지 않도록 비례 복원 강성(Stiffness)을 5배 상향
            dof_props["stiffness"][self.lift_joint_idx] = 1500000.0
            # 진동 차단용 감쇠력 상향 동기화
            dof_props["damping"][self.lift_joint_idx] = 50000.0
            dof_props["velocity"][self.lift_joint_idx] = 0.04
            # 💡 [하드웨어 한계 개방] 중력 저항을 무시하도록 최대 Effort 제한을 극대화
            dof_props["effort"][self.lift_joint_idx] = 99999999.0

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
        """리프팅 목표 높이(0.0m ~ 2.5m)를 공유 버퍼에 안전하게 업데이트합니다."""
        if self.lift_joint_idx != -1:
            safe_height = np.clip(target_height, self.LIFT_MIN_HEIGHT, self.LIFT_MAX_HEIGHT)
            self.dof_position_targets[self.lift_joint_idx] = safe_height

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
        scaled_turn_speed = np.clip(p_control_speed, 0.75, self.MAX_ANGULAR_VEL)
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
        # if self.current_state == self.STATE_MOVE_TO_SHELF:
        #     self.set_lift_height(0.0)
        #     arrived = self.move_to_point(target_shelf[0], target_shelf[1], '+y')
        #     if arrived:
        #         print("[LowAMR FSM] 선반 밑 정위치 안착 완료 ➡️ STATE_ALIGN_SHELF 단계 진입")
        #         self.current_state = self.STATE_ALIGN_SHELF
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
        # elif self.current_state == self.STATE_LIFT_UP:
        #     self.set_lift_height(lift_target_pos)

        #     if frame_count % 120 == 0:
        #         print(f"[LowAMR 물리 피드백] 리프트 현재 높이: {current_lift_pos:.4f}m / 목표: {lift_target_pos}m")
        #         print("frame_count - ", frame_count, " state_start_frame - ", self.state_start_frame)

        #     if abs(current_lift_pos - lift_target_pos) < lift_threshold:
        #         if self.state_start_frame is None:
        #             self.state_start_frame = frame_count
                
        #         if (frame_count - self.state_start_frame) >= 60:
        #             print(f"[LowAMR FSM] 리프트 고도 {current_lift_pos:.3f}m 안정화 완료 ➡️ STATE_MOVE_TO_DEST 단계 진입")
        #             self.current_state = self.STATE_MOVE_TO_DEST
        #             self.state_start_frame = None
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