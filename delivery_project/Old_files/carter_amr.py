import numpy as np
from isaacgym import gymapi
import math

class CarterAMR:
    def __init__(self, gym, env, handle, tray_handle, planner=None):
        self.gym = gym
        self.env = env
        self.handle = handle
        self.tray_handle = tray_handle
        self.planner = planner 
        
        # 주행 속도 및 게인 
        self.kp_linear = 2.0
        self.kp_angular = 4.0 
        self.max_lin_vel = 3.33
        self.max_ang_vel = 2.0
        
        # Look-ahead 파라미터
        self.lookahead_dist = 1.2  
        self.min_dist = 0.4        

        self.current_path = []
        self.current_wp_idx = 0

        # ★ [안정화 1] 가속도 제한용 변수 및 로봇 하드웨어 치수 세팅
        self.prev_v_lin = 0.0
        self.prev_v_ang = 0.0
        self.wheel_radius = 0.14   # 11인치 바퀴 반지름
        self.track_width = 0.58    # 좌우 바퀴 간격 (0.29 * 2)

        self._setup_physics()

    def _setup_physics(self):
        s_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        for s in s_props:
            s.friction = 2.0
            s.rolling_friction = 0.1
        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, s_props)

        d_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        d_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        d_props["stiffness"].fill(0.0)
        # ★ [안정화 2] 50kg 쇳덩이를 밀어내기 위해 모터 댐핑(제동/토크 유지력)을 10배로 강화
        d_props["damping"].fill(1000.0)
        d_props["armature"].fill(0.05) # 모터 관성 약간 추가
        self.gym.set_actor_dof_properties(self.env, self.handle, d_props)

    def sync_tray(self):
        c_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        curr_p, curr_r = c_states['pose']['p'][0], c_states['pose']['r'][0]
        
        t_states = self.gym.get_actor_rigid_body_states(self.env, self.tray_handle, gymapi.STATE_ALL)
        for i in range(len(t_states)):
            t_states['pose']['p'][i]['x'] = curr_p['x']
            t_states['pose']['p'][i]['y'] = curr_p['y']
            t_states['pose']['p'][i]['z'] = curr_p['z'] + 0.07
            t_states['pose']['r'][i] = curr_r
            t_states['vel']['linear'][i] = (0, 0, 0)
            t_states['vel']['angular'][i] = (0, 0, 0)
            
        self.gym.set_actor_rigid_body_states(self.env, self.tray_handle, t_states, gymapi.STATE_ALL)
        return curr_p, curr_r

    def set_target(self, curr_pos, target_pos):
        if self.planner:
            self.current_path = self.planner.plan_path(curr_pos, target_pos)
            self.current_wp_idx = 0
            if self.current_path:
                print(f"경로 생성 완료: 총 {len(self.current_path)} 개의 웨이포인트")
        else:
            self.current_path = [target_pos]
            self.current_wp_idx = 0

    def drive_to_target(self, curr_p, curr_r, is_loaded):
        if not is_loaded or not self.current_path:
            self.gym.set_actor_dof_velocity_targets(self.env, self.handle, np.array([0, 0, 0, 0], dtype=np.float32))
            return

        final_wp = self.current_path[-1]
        final_dist = np.hypot(final_wp[0] - curr_p['x'], final_wp[1] - curr_p['y'])
        
        if final_dist < self.min_dist:
            # 도착 시 급정거 방지를 위해 속도를 0으로 부드럽게 초기화
            self.prev_v_lin = 0.0
            self.prev_v_ang = 0.0
            self.gym.set_actor_dof_velocity_targets(self.env, self.handle, np.array([0, 0, 0, 0], dtype=np.float32))
            self.current_path = []
            return

        target_wp = final_wp  
        
        for i in range(self.current_wp_idx, len(self.current_path)):
            wp = self.current_path[i]
            dist_to_wp = np.hypot(wp[0] - curr_p['x'], wp[1] - curr_p['y'])
            
            if dist_to_wp >= self.lookahead_dist:
                target_wp = wp
                self.current_wp_idx = i 
                break

        curr_yaw = np.arctan2(2.0 * (curr_r['w'] * curr_r['z'] + curr_r['x'] * curr_r['y']), 
                              1.0 - 2.0 * (curr_r['y']**2 + curr_r['z']**2))

        err_x = target_wp[0] - curr_p['x']
        err_y = target_wp[1] - curr_p['y']
        
        target_yaw = np.arctan2(err_y, err_x)
        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        dist_error = np.hypot(err_x, err_y)
        v_lin_target = np.clip(self.kp_linear * dist_error * np.cos(yaw_error), -self.max_lin_vel, self.max_lin_vel)
        
        if abs(yaw_error) > np.pi / 3:
            v_lin_target = 0.0 

        v_ang_target = np.clip(self.kp_angular * yaw_error, -self.max_ang_vel, self.max_ang_vel)

        # ★ [안정화 3] 가속도 제한(Slew Rate Filter) 적용
        # 로봇이 급발진하지 않고 스무스하게 가속/감속하도록 만듭니다.
        MAX_ACCEL = 0.04  
        MAX_ALPHA = 0.06  

        v_lin = self.prev_v_lin + np.clip(v_lin_target - self.prev_v_lin, -MAX_ACCEL, MAX_ACCEL)
        v_ang = self.prev_v_ang + np.clip(v_ang_target - self.prev_v_ang, -MAX_ALPHA, MAX_ALPHA)
        self.prev_v_lin = v_lin
        self.prev_v_ang = v_ang

        # ★ [안정화 4] 차분 구동(Differential Drive) 기구학 완벽 수식 적용
        # 1) 좌우 바퀴의 실제 선속도(m/s) 계산
        v_left_linear = v_lin - v_ang * (self.track_width / 2.0)
        v_right_linear = v_lin + v_ang * (self.track_width / 2.0)

        # 2) 선속도(m/s)를 물리 엔진 조인트 회전속도(rad/s)로 변환
        ls = v_left_linear / self.wheel_radius
        rs = v_right_linear / self.wheel_radius

        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, np.array([ls, rs, ls, rs], dtype=np.float32))