import os
import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from vacuum_controller_v2 import FrankaController
from scipy.spatial.transform import Rotation as R

# ============================================================
# 순수 수학 함수들 (로봇/씬과 무관, 어디서든 재사용 가능)
# ============================================================

def quat_to_rpy_deg(qx, qy, qz, qw):
    """
    쿼터니언(x,y,z,w) → Roll, Pitch, Yaw (degree, 내재적 xyz 순서)
    반환: (roll, pitch, yaw) 튜플, 단위 degree
    """
    rot = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def slerp_quat(q1, q2, alpha):
    """gymapi.Quat 두 개를 구면선형보간. alpha=0이면 q1, alpha=1이면 q2 반환."""
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w

    dot = x1*x2 + y1*y2 + z1*z2 + w1*w2
    if dot < 0.0:
        x2, y2, z2, w2 = -x2, -y2, -z2, -w2
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        x = x1 + alpha*(x2-x1); y = y1 + alpha*(y2-y1)
        z = z1 + alpha*(z2-z1); w = w1 + alpha*(w2-w1)
    else:
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta, sin_theta_0 = np.sin(theta), np.sin(theta_0)
        s1 = np.cos(theta) - dot*sin_theta/sin_theta_0
        s2 = sin_theta/sin_theta_0
        x = s1*x1+s2*x2; y = s1*y1+s2*y2; z = s1*z1+s2*z2; w = s1*w1+s2*w2

    norm = np.sqrt(x*x+y*y+z*z+w*w)
    return gymapi.Quat(x/norm, y/norm, z/norm, w/norm)


def scale_to_joint_limits(d_theta, dt, max_joint_vel):
    """관절 간 비율(=직선 방향)을 유지한 채, 최대속도 초과분만큼 균일하게 축소."""
    required_vel = torch.abs(d_theta.squeeze(-1)) / dt
    ratio = required_vel / max_joint_vel
    max_ratio = torch.max(ratio)
    if max_ratio > 1.0:
        return d_theta * (1.0 / max_ratio)
    return d_theta

def priority_ik(j_eef, dpose, lock_mask, damping_primary=0.05, damping_secondary=0.1):
    """
    엄격한 태스크 우선순위 기반 IK (Siciliano et al. 방식).
    lock_mask: bool tensor, shape=(6,) — True인 축이 '1순위(잠금, 엄밀히 고정)'
    dpose: (6,1) 오차 벡터 (pos_err 3 + rot_err 3)
    """
    device = j_eef.device
    n_dof = j_eef.shape[1]

    lock_idx = torch.nonzero(lock_mask).squeeze(-1)
    free_idx = torch.nonzero(~lock_mask).squeeze(-1)

    # 모든 축이 잠겨있거나 모두 자유로우면 단순 DLS로 처리
    if lock_idx.numel() == 0:
        J2 = j_eef[free_idx, :]
        e2 = dpose[free_idx, :]
        lmbda = torch.eye(n_dof, device=device) * (damping_secondary ** 2)
        return torch.inverse(J2.T @ J2 + lmbda) @ (J2.T @ e2)
    if free_idx.numel() == 0:
        J1 = j_eef[lock_idx, :]
        e1 = dpose[lock_idx, :]
        lmbda = torch.eye(n_dof, device=device) * (damping_primary ** 2)
        return torch.inverse(J1.T @ J1 + lmbda) @ (J1.T @ e1)

    # --- 1순위: 잠긴 축 ---
    J1 = j_eef[lock_idx, :]          # (m1, dof)
    e1 = dpose[lock_idx, :]          # (m1, 1)

    lmbda1 = torch.eye(n_dof, device=device) * (damping_primary ** 2)
    J1_pinv = torch.inverse(J1.T @ J1 + lmbda1) @ J1.T   # damped pinv, (dof, m1)

    dq1 = J1_pinv @ e1               # (dof, 1)

    # --- nullspace 투영 행렬 ---
    I_dof = torch.eye(n_dof, device=device)
    N1 = I_dof - J1_pinv @ J1        # (dof, dof)

    # --- 2순위: 자유 축, N1 안에서만 ---
    J2 = j_eef[free_idx, :]          # (m2, dof)
    e2 = dpose[free_idx, :]          # (m2, 1)

    J2N1 = J2 @ N1                   # (m2, dof)
    m2 = J2N1.shape[0]
    lmbda2 = torch.eye(m2, device=device) * (damping_secondary ** 2)
    J2N1_pinv = N1 @ J2N1.T @ torch.inverse(J2N1 @ J2N1.T + lmbda2)  # (dof, m2)

    residual = e2 - J2 @ dq1
    dq2 = J2N1_pinv @ residual       # 이미 N1 range 안에 있음

    return dq1 + dq2


def weighted_dls(j_eef, dpose, weights, damping):
    """가중치 기반 Damped Least Squares IK. weights가 클수록 그 축을 강하게 고정."""
    W = torch.diag(weights)
    JT_W = j_eef.transpose(0, 1) @ W
    lmbda = torch.eye(j_eef.shape[1], device=j_eef.device) * (damping ** 2)
    return torch.inverse(JT_W @ j_eef + lmbda) @ (JT_W @ dpose)


# ============================================================
# 직선 경로 궤적 생성기 (기존 그대로, 독립 클래스)
# ============================================================

class VelocityProfile:
    """
    진행률(alpha, 0~1)에 따른 속도 배율(0~1)을 여러 키포인트로 지정하고,
    그 사이를 보간하는 속도 프로파일.
    실제 속도 = linear_speed(최댓값) x speed_fraction(alpha) 이므로,
    키포인트의 speed_fraction이 1.0을 넘지 않는 한 linear_speed가 절대 최댓값이 된다.
    """

    def __init__(self, keypoints=None, interp="smooth"):
        """
        keypoints: [(alpha, speed_fraction), ...]
                   alpha: 0~1 (경로상의 진행률, 오름차순)
                   speed_fraction: 0~1 (1.0 = linear_speed 그대로, 즉 최댓값)
                   예) [(0.0, 0.1), (0.3, 1.0), (0.7, 1.0), (1.0, 0.1)]
                       -> 시작 10% 속도로 출발, 30% 지점까지 가속해서 최댓값 도달,
                          70% 지점까지 최댓값 유지, 이후 끝까지 10% 속도로 감속
        interp: "linear" -> 키포인트 사이를 직선으로 보간 (사다리꼴형, 가속도가 꺾임)
                "smooth" -> 각 구간을 smoothstep으로 보간 (부드러운 가감속, 급격한 가속변화 없음)
        """
        if keypoints is None:
            keypoints = [(0.0, 0.1), (0.3, 1.0), (0.7, 1.0), (1.0, 0.1)]
        self.keypoints = sorted(keypoints, key=lambda kp: kp[0])
        self.interp = interp

    def speed_fraction(self, alpha):
        alpha = min(1.0, max(0.0, alpha))
        kps = self.keypoints

        if alpha <= kps[0][0]:
            return kps[0][1]
        if alpha >= kps[-1][0]:
            return kps[-1][1]

        for i in range(len(kps) - 1):
            a0, v0 = kps[i]
            a1, v1 = kps[i + 1]
            if a0 <= alpha <= a1:
                local_t = (alpha - a0) / (a1 - a0) if a1 > a0 else 0.0
                if self.interp == "smooth":
                    local_t = 3.0 * local_t**2 - 2.0 * local_t**3  # smoothstep, 구간 내 단조 증가/감소
                return v0 + (v1 - v0) * local_t

        return kps[-1][1]


class LinearMotionGenerator:
    """
    직선 경로를 따라 목표를 매 프레임 조금씩 이동시켜주는 궤적 생성기.
    linear_speed는 이제 '평균 속도'가 아니라 '최댓값'이며,
    실제 순간 속도는 velocity_profile이 정한 배율(0~1)만큼만 낸다.
    """

    def __init__(self, linear_speed=0.15, angular_speed=1.0, dt=1.0/60.0,
                 velocity_profile=None):
        self.linear_speed = linear_speed          # 이제 '최댓값'
        self.angular_speed = angular_speed
        self.dt = dt
        self.velocity_profile = velocity_profile if velocity_profile is not None else VelocityProfile()
        self.active = False

    def start(self, start_pos, start_quat, goal_pos, goal_quat):
        self.start_pos = start_pos.clone()
        self.goal_pos = goal_pos.clone()
        self.start_quat = gymapi.Quat(*start_quat.tolist())
        self.goal_quat = gymapi.Quat(*goal_quat.tolist())

        self.total_dist = (goal_pos - start_pos).norm().item()
        self.total_ang = self._quat_angle(self.start_quat, self.goal_quat)
        self.traveled_dist = 0.0
        self.alpha = 0.0
        self.active = True

    def _quat_angle(self, q1, q2):
        dot = abs(q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w)
        return 2 * np.arccos(min(1.0, dot))

    def step(self):
        if not self.active:
            return None, None

        if self.total_dist < 1e-9:
            self.active = False
            way_quat_t = torch.tensor(
                [self.goal_quat.x, self.goal_quat.y, self.goal_quat.z, self.goal_quat.w],
                device=self.start_pos.device
            )
            return self.goal_pos.clone(), way_quat_t

        # 🆕 진행률에 따른 속도 배율 -> 실제 속도(m/s, linear_speed 이하로 항상 제한)
        #    -> 이번 프레임에 전진할 거리를 직접 누적 (시간표가 아니라 속도로 위치를 적분)
        speed_frac = self.velocity_profile.speed_fraction(self.alpha)
        speed = self.linear_speed * speed_frac
        self.traveled_dist = min(self.total_dist, self.traveled_dist + speed * self.dt)
        self.alpha = self.traveled_dist / self.total_dist

        way_pos = self.start_pos + (self.goal_pos - self.start_pos) * self.alpha
        way_quat = slerp_quat(self.start_quat, self.goal_quat, self.alpha)
        way_quat_t = torch.tensor(
            [way_quat.x, way_quat.y, way_quat.z, way_quat.w],
            device=self.start_pos.device
        )

        if self.alpha >= 1.0:
            self.active = False

        return way_pos, way_quat_t
    

def scurve_alpha(alpha):
    """
    5차 다항식(Minimum Jerk) 기반 S자 진행률 변환.
    입력 alpha(시간 비례 0~1)를 받아, 시작/끝에서 속도가 0에 가깝고
    중간 구간에서 속도가 최대가 되는 '느리게-빠르게-느리게' 진행률을 반환한다.
    quintic_interp와 동일한 수식이므로 결과가 이미 검증된 JointMotionGenerator와 일관성 있다.
    """
    a = min(1.0, max(0.0, alpha))
    return 10.0 * a**3 - 15.0 * a**4 + 6.0 * a**5


def quintic_interp(q_start, q_target, alpha):
    """
    5차 다항식(Minimum Jerk) 보간. alpha=0~1.
    시작/끝에서 속도·가속도가 0이 되어 부드럽게 정지/출발한다.
    q_start, q_target: torch.Tensor, shape=(dof,)
    """
    s = scurve_alpha(alpha)
    return q_start + (q_target - q_start) * s


class JointMotionGenerator:
    def __init__(self, max_joint_vel, dt=1.0/60.0):
        self.max_joint_vel = max_joint_vel
        self.dt = dt
        self.active = False

    def start(self, start_q, target_q, synchronized=True, min_duration=0.3):
        """
        synchronized=True  (기존 동작): 가장 느린 관절 기준으로 duration을 통일해 동시 도착.
        synchronized=False: 관절별 max_joint_vel로 독립적으로 이동, 도착 시점도 서로 다름.
        """
        self.start_q = start_q.clone()
        self.target_q = target_q.clone()
        self.synchronized = synchronized

        delta = torch.abs(self.target_q - self.start_q)
        per_joint_time = torch.clamp(delta / self.max_joint_vel, min=min_duration)

        if synchronized:
            self.duration = per_joint_time.max().item()
            self.per_joint_duration = None
        else:
            self.per_joint_duration = per_joint_time          # 관절별 duration 그대로 보관
            self.duration = per_joint_time.max().item()        # active 종료 판정용

        self.elapsed = 0.0
        self.active = True

    def step(self):
        if not self.active:
            return None
        self.elapsed += self.dt

        if self.synchronized:
            alpha = min(1.0, self.elapsed / self.duration)
            way_q = quintic_interp(self.start_q, self.target_q, alpha)
        else:
            # 관절마다 각자의 duration 기준 alpha (비동기)
            alpha_vec = torch.clamp(self.elapsed / self.per_joint_duration, max=1.0)
            s = 10.0 * alpha_vec**3 - 15.0 * alpha_vec**4 + 6.0 * alpha_vec**5
            way_q = self.start_q + (self.target_q - self.start_q) * s
            alpha = alpha_vec.max().item()   # 가장 늦게 끝나는 관절 기준으로 전체 종료 판단

        if alpha >= 1.0:
            self.active = False
        return way_q
    
class JointPoseSequencer:
    """등록된 관절 자세들을 순서대로 재생. 각 자세 도달 후 hold_time만큼 대기."""
    def __init__(self, arm: "IndyArmController"):
        self.arm = arm
        self.queue = []       # [(pose_name_or_tensor, hold_time), ...]
        self.hold_timer = 0.0
        self.waiting = False

    def load(self, sequence):
        """sequence: [(pose_name, hold_time_sec), ...]"""
        self.queue = list(sequence)
        self._advance()

    def _advance(self):
        if not self.queue:
            self.waiting = False
            return
        pose, hold = self.queue.pop(0)
        self.arm.move_to_joint_pose(pose)
        self._pending_hold = hold

    def update(self, dt):
        if self.arm.joint_motion_gen.active:
            return  # 아직 이동 중

        if not self.waiting:
            self.hold_timer = 0.0
            self.waiting = True

        self.hold_timer += dt
        if self.hold_timer >= self._pending_hold:
            self.waiting = False
            self._advance()


def flush_dof_targets(gym, sim, controllers):
    """
    여러 로봇 컨트롤러(IndyArmController 등)의 pending DOF 목표를
    단 한 번의 set_dof_position_target_tensor_indexed 호출로 물리 엔진에 반영한다.

    controllers: pending_dof_pos, dof_indices, actor_sim_index, dof_states_full
                 속성을 가진 컨트롤러 객체 리스트 (인터페이스만 맞으면 IndyArm 외의
                 다른 로봇 컨트롤러와 섞어서 넣어도 된다)
    """
    valid = [c for c in controllers if c.pending_dof_pos is not None]
    if not valid:
        return

    ref = valid[0]
    full_targets = ref.dof_states_full[:, 0].clone()

    index_list = []
    for c in valid:
        full_targets[c.dof_indices] = c.pending_dof_pos
        index_list.append(c.actor_sim_index)

    combined_indices = torch.cat(index_list)

    gym.set_dof_position_target_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(full_targets),
        gymtorch.unwrap_tensor(combined_indices),
        combined_indices.shape[0]
    )

class StepContext:
    def __init__(self, arm, **extra):
        self.arm = arm
        for k, v in extra.items():
            setattr(self, k, v)

class ArmMotionStep:
    def __init__(self, name, kind, target=None,
                 delta_axis=None, delta_value=None,
                 linear_speed=None, angular_speed=None,
                 on_start=None, on_complete=None,   # 단일 콜백 또는 콜백 리스트 둘 다 허용
                 stable_frames=10, stable_threshold=0.0005,
                 wait_for=None):
        self.name = name
        self.kind = kind
        self.target = target
        self.delta_axis = delta_axis
        self.delta_value = delta_value
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        # 리스트가 아니면 리스트로 감싸서 통일
        self.on_start = on_start if isinstance(on_start, (list, tuple)) else ([on_start] if on_start else [])
        self.on_complete = on_complete if isinstance(on_complete, (list, tuple)) else ([on_complete] if on_complete else [])
        self.stable_frames = stable_frames
        self.stable_threshold = stable_threshold
        self.wait_for = wait_for  # callable(ctx) -> bool. True가 될 때까지 스텝 시작 보류


class ArmStepRunner:
    def __init__(self, arm, steps, check_ee_stable_fn, context=None):
        self.arm = arm
        self.steps = steps
        self.check_ee_stable = check_ee_stable_fn
        self.idx = 0
        self.entered = False
        self.stable_counter = 0
        self.prev_ee_pos = None
        self.done = False
        self.ctx = context or StepContext(arm)
        self._speed_log_prev_pos = None
        self._speed_log_frame = 0

    def _is_active(self, step):
        return self.arm.joint_motion_gen.active if step.kind == "joint" else self.arm.motion_gen.active

    def _start(self, step):
        # on_start 콜백 전부 실행 (예: attach, detach, 상태 초기화 등)
        for cb in step.on_start:
            cb(self.ctx) 

        if step.linear_speed is not None:
            self.arm.motion_gen.linear_speed = step.linear_speed
        if step.angular_speed is not None:
            self.arm.motion_gen.angular_speed = step.angular_speed

        if step.kind == "joint":
            self.arm.move_to_joint_pose(step.target)

        elif step.kind == "cartesian_abs":
            pos, rot = step.target
            self.arm.move_to_cartesian(pos, rot)

        elif step.kind == "cartesian_delta":
            self.arm.gym.refresh_rigid_body_state_tensor(self.arm.sim)
            cur_pos = self.arm.rb_states[self.arm.ee_index_global, 0:3].clone()
            cur_rot = self.arm.rb_states[self.arm.ee_index_global, 3:7].clone()
            new_pos = cur_pos.clone()
            new_pos[step.delta_axis] += step.delta_value
            self.arm.move_to_cartesian(new_pos, cur_rot)

    def update(self):
        """매 프레임 호출. 전체 시퀀스가 끝나면 True."""
        if self.done:
            return True

        step = self.steps[self.idx]

        if not self.entered:
            # 🆕 wait_for 조건이 있으면 True가 될 때까지 시작하지 않음
            if step.wait_for is not None and not step.wait_for(self.ctx):
                return False
            self._start(step)
            self.entered = True
            self.stable_counter = 0
            self.prev_ee_pos = None
            self._speed_log_prev_pos = None
            return False

        if self._is_active(step):
            # 🆕 cartesian_delta 실시간 속도 로그
            if step.kind == "cartesian_delta":
                self.arm.gym.refresh_rigid_body_state_tensor(self.arm.sim)
                cur_pos = self.arm.rb_states[self.arm.ee_index_global, 0:3].clone()
                if self._speed_log_prev_pos is not None:
                    speed = (cur_pos - self._speed_log_prev_pos).norm().item() / self.arm.dt
                    if self._speed_log_frame % 10 == 0:  # 10프레임마다 출력 (너무 많으면 조절)
                        print(f"[{step.name}] EE speed={speed:.4f} m/s "
                              f"(target linear_speed={self.arm.motion_gen.linear_speed:.3f})")
                self._speed_log_prev_pos = cur_pos
                self._speed_log_frame += 1
            return False

        self.prev_ee_pos, is_stable = self.check_ee_stable(
            self.arm, self.prev_ee_pos, threshold=step.stable_threshold
        )
        self.stable_counter = self.stable_counter + 1 if is_stable else 0

        if self.stable_counter >= step.stable_frames:
            print(f"[{step.name}] 완료")

            # on_complete 콜백 전부 실행 (예: attach + speed 변경 둘 다)
            for cb in step.on_complete:
                cb(self.ctx) 

            self.idx += 1
            self.entered = False
            if self.idx >= len(self.steps):
                self.done = True
                return True

        return False

# ============================================================
# 핵심: 로봇팔 컨트롤러 클래스
# ============================================================

class IndyArmController:
    """
    Indy7 로봇팔의 로딩, IK, 키보드 조작을 한 번에 캡슐화한 클래스.
    - 씬(바닥/벽/기둥 등)에는 관여하지 않는다.
    - gym, sim, env가 이미 만들어진 상태에서 생성자에 넘겨받는다.
    """

    AXIS_MAP = {
        "move_x_pos": 0, "move_x_neg": 0,
        "move_y_pos": 1, "move_y_neg": 1,
        "move_z_pos": 2, "move_z_neg": 2,
        "rot_roll_pos": 3, "rot_roll_neg": 3,
        "rot_pitch_pos": 4, "rot_pitch_neg": 4,
        "rot_yaw_pos": 5, "rot_yaw_neg": 5,
    }

    def __init__(self, gym, sim, env, viewer, asset_root, urdf_path,
                 spawn_transform, actor_name="indy7_asset",
                 stiffness=50000.0, damping_gain=1000.0,
                 ik_damping=0.1,
                 move_speed=0.01, rot_speed=0.05,
                 linear_speed=5.0, angular_speed=1.0, max_joint_vel=None,
                 lock_weight=3000.0, free_weight=1.0,
                 dt=1.0/60.0):

        self.gym, self.sim, self.env, self.viewer = gym, sim, env, viewer
        self.dt = dt
        self.move_speed = move_speed
        self.rot_speed = rot_speed
        self.ik_damping = ik_damping
        self.lock_weight = lock_weight
        self.free_weight = free_weight
        self.actor_name = actor_name

        # --- 1. asset 로드 및 actor 생성 ---
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.density = 100.0
        self.asset = gym.load_asset(sim, asset_root, urdf_path, opts)
        self.handle = gym.create_actor(
            env, self.asset,
            spawn_transform,
            self.actor_name, -1, 0
        )

        # --- 2. DOF 구동 설정 ---
        self.dof_count = gym.get_asset_dof_count(self.asset)
        dof_props = gym.get_actor_dof_properties(env, self.handle)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)

        dof_props["stiffness"].fill(stiffness)
        dof_props["damping"].fill(damping_gain)
        
        dof_props["effort"][0] = 10000.0
        dof_props["effort"][1] = 10000.0
        dof_props["effort"][2] = 10000.0
        dof_props["effort"][3] = 10000.0
        dof_props["effort"][4] = 10000.0
        dof_props["effort"][5] = 10000.0

        gym.set_actor_dof_properties(env, self.handle, dof_props)

        # --- 3. end-effector 링크 인덱스 확보 ---
        body_names = gym.get_asset_rigid_body_names(self.asset)
        self.ee_name = body_names[-1]
        print("=== 링크 목록 ===", body_names)

        # --- 4. 키보드 바인딩 ---
        self._bind_keyboard_events()

        # --- 5. 텐서 및 목표 상태는 prepare_sim 이후에 초기화 (외부에서 setup_tensors 호출) ---
        self.jacobian = None
        self.rb_states = None
        self.dof_states_full = None      # ← self.dof_states 에서 이름 변경
        self.dof_indices = None          # ← 새로 추가 (setup_tensors에서 채워짐)
        self.dof_start_idx = None        # ← 새로 추가
        self.ee_index_actor = None
        self.ee_index_global = None
        self.target_pos = None
        self.target_rot = None
        self.active_axis_weights = None

        self.motion_gen = LinearMotionGenerator(
            linear_speed=linear_speed, angular_speed=angular_speed, dt=dt
        )

        # 관절별 최대속도 (기존 scale_to_joint_limits와 동일한 값 재사용 가능)
        if max_joint_vel is None:
            max_joint_vel = torch.tensor([2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159])
        self._max_joint_vel_cpu = max_joint_vel  # device는 setup_tensors 이후 맞춤

        self.joint_motion_gen = None  # setup_tensors에서 device 확정 후 생성

        # 제어 모드: "IK" (기존 좌표 기반) 또는 "JOINT" (신규 각도 기반)
        self.control_mode = "IK"

        # 이름 붙은 관절 자세 프리셋 (필요시 외부에서 register_joint_pose로 추가)
        self.joint_poses = {}

        # --- 흡착(attach/detach) 관련 상태 ---
        self.is_attached = False
        self.attached_handle = None
        self.attached_sim_index = None
        self.attached_env_index = None
        self.snap_local_pos_offset = None   # gymapi.Vec3, ee 좌표계 기준
        self.snap_local_rot_offset = None   # gymapi.Quat, ee 좌표계 기준
        self.snap_ee_local_offset = gymapi.Vec3(0.0, 0.0, 0.0)  # 흡착점이 tcp에서 얼마나 떨어져 있는지
        self.rb_states_writable = None
        self.root_states = None
        self.attached_actor_index = None

        # 흡착 후보로 등록해둔 액터 핸들 목록 (attach_nearest용)
        self.attachable_handles = []

        self.selected_joint_idx = 0
        self.manual_joint_step = np.radians(1.0)   # 한 번 입력당 변화량 (기본 1도)
        self.manual_joint_targets = None            # setup_tensors에서 device 확정 후 초기화
        self.actor_sim_index = None
        self.pending_dof_pos  = None

        self.manual_control_mode = "IK"

    def toggle_manual_control_mode(self):
        if self.manual_control_mode == "IK":
            self.manual_control_mode = "JOINT_MANUAL"
            self.enter_joint_manual_mode()   # control_mode = "JOINT_MANUAL"로 전환, 현재 관절각을 목표로 고정
            print("[IndyArm] === 수동 조작 모드: 관절 각도 조작(JOINT) ===")
        else:
            self.manual_control_mode = "IK"
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.target_pos = self.rb_states[self.ee_index_global, 0:3].clone()
            self.target_rot = self.rb_states[self.ee_index_global, 3:7].clone()
            self.control_mode = "IK"
            self.active_axis_weights = torch.full(
                (6,), self.lock_weight, device=self.dof_states_full.device
            )
            print("[IndyArm] === 수동 조작 모드: IK 좌표 이동(CARTESIAN) ===")

    def register_attachable(self, actor_handle):
        """흡착 가능한 대상으로 등록. 여러 번 호출해 여러 물체를 후보로 넣을 수 있다."""
        if actor_handle not in self.attachable_handles:
            self.attachable_handles.append(actor_handle)

    def register_joint_pose(self, name, angles_rad):
        """반복 작업에 쓸 관절각 프리셋을 이름으로 등록."""
        self.joint_poses[name] = torch.tensor(
            angles_rad, dtype=torch.float32, device=self.dof_states_full.device
        )

    def _bind_keyboard_events(self):
        g, v = self.gym, self.viewer
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_UP, "move_x_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_DOWN, "move_x_neg")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_LEFT, "move_y_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_RIGHT, "move_y_neg")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_Z, "move_z_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_X, "move_z_neg")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_T, "rot_roll_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_Y, "rot_roll_neg")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_U, "rot_pitch_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_I, "rot_pitch_neg")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_O, "rot_yaw_pos")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_P, "rot_yaw_neg")

    def setup_tensors(self):
        """gym.prepare_sim(sim) 호출 '이후'에 반드시 실행해야 함."""
        g, sim, env = self.gym, self.sim, self.env

        jac_tensor = g.acquire_jacobian_tensor(sim, self.actor_name)
        self.jacobian = gymtorch.wrap_tensor(jac_tensor)

        rb_tensor = g.acquire_rigid_body_state_tensor(sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor).view(-1, 13)

        dof_tensor = g.acquire_dof_state_tensor(sim)
        self.dof_states_full = gymtorch.wrap_tensor(dof_tensor).view(-1, 2)

        self.dof_start_idx = g.get_actor_dof_index(env, self.handle, 0, gymapi.DOMAIN_SIM)
        self.dof_indices = torch.arange(
            self.dof_start_idx, self.dof_start_idx + self.dof_count,
            device=self.dof_states_full.device
        )

        self.ee_index_actor = g.find_actor_rigid_body_index(
            env, self.handle, self.ee_name, gymapi.DOMAIN_ACTOR
        )
        self.ee_index_global = g.find_actor_rigid_body_index(
            env, self.handle, self.ee_name, gymapi.DOMAIN_SIM
        )

        # --- 신규: 베이스 링크(첫 번째 링크) 인덱스 확보 ---
        body_names = self.gym.get_asset_rigid_body_names(self.asset)
        self.base_name = body_names[0]
        self.base_index_global = g.find_actor_rigid_body_index(
            env, self.handle, self.base_name, gymapi.DOMAIN_SIM
        )

        # --- 신규: DOF 이름 목록 확보 (리포트 출력용) ---
        self.dof_names = self.gym.get_asset_dof_names(self.asset)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.manual_joint_targets = self.dof_states_full[self.dof_indices, 0].clone()

        g.refresh_rigid_body_state_tensor(sim)
        self.target_pos = self.rb_states[self.ee_index_global, 0:3].clone()
        self.target_rot = self.rb_states[self.ee_index_global, 3:7].clone()
        self.active_axis_weights = torch.full(
            (6,), self.lock_weight, device=self.dof_states_full.device
        )

        self.joint_motion_gen = JointMotionGenerator(
            max_joint_vel=self._max_joint_vel_cpu.to(self.dof_states_full.device),
            dt=self.dt
        )

        self.rb_states_writable = self.rb_states

        self.actor_sim_index = torch.tensor(
            [self.gym.get_actor_index(self.env, self.handle, gymapi.DOMAIN_SIM)],
            dtype=torch.int32, device=self.dof_states_full.device
        )

        # --- 신규: root state 텐서 (액터 단위, attach 대상 이동용) ---
        root_tensor = g.acquire_actor_root_state_tensor(sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)

        dof_tensor = g.acquire_dof_state_tensor(sim)
        self.dof_states_full = gymtorch.wrap_tensor(dof_tensor).view(-1, 2)

        print("jacobian shape:", self.jacobian.shape)
        print("dof_start_idx:", self.dof_start_idx, "dof_count:", self.dof_count)
        print("ee_index_actor:", self.ee_index_actor,
            "ee_index_global(sim):", self.ee_index_global)
        print("base_index_global(sim):", self.base_index_global)

    def select_prev_joint(self):
        self.selected_joint_idx = (self.selected_joint_idx - 1) % self.dof_count
        print(f"[IndyArm] 선택된 관절: [{self.selected_joint_idx}] {self.dof_names[self.selected_joint_idx]}")

    def select_next_joint(self):
        self.selected_joint_idx = (self.selected_joint_idx + 1) % self.dof_count
        print(f"[IndyArm] 선택된 관절: [{self.selected_joint_idx}] {self.dof_names[self.selected_joint_idx]}")

    def enter_joint_manual_mode(self):
        """수동 관절 조작 모드로 전환. 현재 실제 관절각을 그대로 목표로 삼아 시작."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.manual_joint_targets = self.dof_states_full[self.dof_indices, 0].clone()
        self.control_mode = "JOINT_MANUAL"
        self.manual_control_mode = "JOINT_MANUAL"
        print(f"[IndyArm] 수동 관절 조작 모드 진입 (선택된 관절: [{self.selected_joint_idx}] "
            f"{self.dof_names[self.selected_joint_idx]})")

    def adjust_selected_joint(self, direction):
        """direction: +1 또는 -1"""
        if self.control_mode != "JOINT_MANUAL":
            self.enter_joint_manual_mode()

        idx = self.selected_joint_idx
        self.manual_joint_targets[idx] += direction * self.manual_joint_step
        deg = np.degrees(self.manual_joint_targets[idx].item())
        print(f"[IndyArm] [{idx}] {self.dof_names[idx]} → {self.manual_joint_targets[idx].item():.4f} rad ({deg:.2f}°)")
        
    def move_to_joint_pose(self, name_or_tensor, max_joint_vel=None, synchronized=True):
        target_q = (self.joint_poses[name_or_tensor]
                    if isinstance(name_or_tensor, str) else name_or_tensor)
        self.gym.refresh_dof_state_tensor(self.sim)
        cur_q = self.dof_states_full[self.dof_indices, 0].clone()

        if max_joint_vel is not None:
            self.joint_motion_gen.max_joint_vel = max_joint_vel.to(cur_q.device)

        self.joint_motion_gen.start(cur_q, target_q, synchronized=synchronized)
        self.control_mode = "JOINT"

    def move_to_cartesian(self, pos, quat):
        """기존 IK 기반 좌표 이동으로 전환."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        cur_ee_pos = self.rb_states[self.ee_index_global, 0:3].clone()
        cur_ee_rot = self.rb_states[self.ee_index_global, 3:7].clone()
        self.motion_gen.start(cur_ee_pos, cur_ee_rot, pos, quat)
        self.target_pos, self.target_rot = pos, quat
        self.control_mode = "IK"

    def compute_target_quat_from_relative_rpy(self, roll_deg, pitch_deg, yaw_deg):
        """베이스 기준 목표 RPY(deg) → 월드 기준 목표 쿼터니언(gymapi.Quat)"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        base_rot = self.rb_states[self.base_index_global, 3:7]
        base_q = gymapi.Quat(base_rot[0].item(), base_rot[1].item(),
                            base_rot[2].item(), base_rot[3].item())

        rel_rot = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
        rx, ry, rz, rw = rel_rot.as_quat()
        rel_q = gymapi.Quat(rx, ry, rz, rw)

        return base_q * rel_q

    def move_to_relative_orientation(self, roll_deg, pitch_deg, yaw_deg, keep_position=True):
        target_q = self.compute_target_quat_from_relative_rpy(roll_deg, pitch_deg, yaw_deg)
        target_q_t = torch.tensor(
            [target_q.x, target_q.y, target_q.z, target_q.w],
            device=self.target_rot.device
        )

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        target_pos_t = (self.rb_states[self.ee_index_global, 0:3].clone()
                        if keep_position else self.target_pos.clone())

        # 회전(3,4,5)을 1순위(lock)로, 위치(0,1,2)를 2순위(free)로
        self.active_axis_weights = torch.tensor(
            [self.free_weight, self.free_weight, self.free_weight,
            self.lock_weight, self.lock_weight, self.lock_weight],
            device=self.dof_states_full.device
        )

        self.move_to_cartesian(target_pos_t, target_q_t)

    def process_keyboard_input(self, events):
        goal_changed = False
        pressed_axes = set()

        for evt in events:
            if evt.value <= 0:
                continue

            action = evt.action

            if action == "start_process":
                self.print_status_report()
                continue

            # --- 신규: 수동 조작 모드 토글 ---
            if action == "toggle_manual_mode":
                self.toggle_manual_control_mode()
                continue

            # --- 관절 선택/조작: JOINT_MANUAL 모드에서만 동작 ---
            if action == "select_prev_joint":
                if self.manual_control_mode == "JOINT_MANUAL":
                    self.select_prev_joint()
                continue
            elif action == "select_next_joint":
                if self.manual_control_mode == "JOINT_MANUAL":
                    self.select_next_joint()
                continue
            elif action == "increase_joint_angle":
                if self.manual_control_mode == "JOINT_MANUAL":
                    self.adjust_selected_joint(+1)
                continue
            elif action == "decrease_joint_angle":
                if self.manual_control_mode == "JOINT_MANUAL":
                    self.adjust_selected_joint(-1)
                continue

            # --- 좌표/회전 조작: IK 모드에서만 동작 ---
            if self.manual_control_mode != "IK":
                continue

            if action == "move_x_pos": self.target_pos[0] += self.move_speed; goal_changed = True
            elif action == "move_x_neg": self.target_pos[0] -= self.move_speed; goal_changed = True
            elif action == "move_y_pos": self.target_pos[1] += self.move_speed; goal_changed = True
            elif action == "move_y_neg": self.target_pos[1] -= self.move_speed; goal_changed = True
            elif action == "move_z_pos": self.target_pos[2] += self.move_speed; goal_changed = True
            elif action == "move_z_neg": self.target_pos[2] -= self.move_speed; goal_changed = True
            elif action in ("rot_roll_pos","rot_roll_neg","rot_pitch_pos",
                            "rot_pitch_neg","rot_yaw_pos","rot_yaw_neg"):
                axis_vec = {
                    "rot_roll_pos":[1,0,0], "rot_roll_neg":[-1,0,0],
                    "rot_pitch_pos":[0,1,0], "rot_pitch_neg":[0,-1,0],
                    "rot_yaw_pos":[0,0,1], "rot_yaw_neg":[0,0,-1],
                }[action]
                dq = gymapi.Quat.from_axis_angle(gymapi.Vec3(*axis_vec), self.rot_speed)

                self.gym.refresh_rigid_body_state_tensor(self.sim)
                actual_rot = self.rb_states[self.ee_index_global, 3:7]
                cur_q = gymapi.Quat(actual_rot[0].item(), actual_rot[1].item(),
                                    actual_rot[2].item(), actual_rot[3].item())

                new_q = dq * cur_q
                self.target_rot = torch.tensor(
                    [new_q.x, new_q.y, new_q.z, new_q.w], device=self.target_rot.device
                )
                goal_changed = True

            if action in self.AXIS_MAP:
                pressed_axes.add(self.AXIS_MAP[action])

        if goal_changed:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            cur_ee_pos = self.rb_states[self.ee_index_global, 0:3].clone()
            cur_ee_rot = self.rb_states[self.ee_index_global, 3:7].clone()
            self.motion_gen.start(cur_ee_pos, cur_ee_rot,
                                self.target_pos.clone(), self.target_rot.clone())

            self.active_axis_weights = torch.full(
                (6,), self.free_weight, device=self.dof_states_full.device
            )
            for ax in pressed_axes:
                self.active_axis_weights[ax] = self.lock_weight

        if not self.motion_gen.active and self.control_mode == "IK":
            self.active_axis_weights = torch.full(
                (6,), self.lock_weight, device=self.dof_states_full.device
            )

    # def step(self, max_joint_vel=None):
    #     self.gym.refresh_jacobian_tensors(self.sim)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     self.gym.refresh_dof_state_tensor(self.sim)

    #     if self.control_mode == "JOINT":
    #         self._step_joint_mode()
    #     elif self.control_mode == "JOINT_MANUAL":
    #         self._step_joint_manual_mode()
    #     else:
    #         self._step_ik_mode(max_joint_vel)

    def step(self, max_joint_vel=None):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        if self.control_mode == "JOINT":
            self._compute_joint_mode()
        elif self.control_mode == "JOINT_MANUAL":
            self._compute_joint_manual_mode()
        else:
            self._compute_ik_mode(max_joint_vel)

    def _compute_joint_manual_mode(self):
        self.pending_dof_pos = self.manual_joint_targets.clone()

    def _compute_joint_mode(self):
        way_q = self.joint_motion_gen.step()
        if way_q is None:
            self.pending_dof_pos = None   # 변화 없음, 이전 목표 유지
            return
        self.pending_dof_pos = way_q

    def _compute_ik_mode(self, max_joint_vel):
        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]

        way_pos, way_quat = self.motion_gen.step()
        if way_pos is None:
            way_pos, way_quat = self.target_pos, self.target_rot

        pos_err = way_pos - ee_pos
        cur_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
        goal_q = gymapi.Quat(way_quat[0].item(), way_quat[1].item(), way_quat[2].item(), way_quat[3].item())
        err_q = goal_q * cur_q.inverse()
        if err_q.w < 0:
            err_q.x, err_q.y, err_q.z, err_q.w = -err_q.x, -err_q.y, -err_q.z, -err_q.w
        rot_err = torch.tensor([err_q.x, err_q.y, err_q.z], device=pos_err.device) * 2.0
        dpose = torch.cat([pos_err, rot_err]).unsqueeze(-1)

        j_eef = self.jacobian[0, self.ee_index_actor - 1, :, :self.dof_count]
        lock_mask = self.active_axis_weights > (self.free_weight * 10)

        d_theta = priority_ik(j_eef, dpose, lock_mask,
                            damping_primary=0.02, damping_secondary=0.1)

        if max_joint_vel is not None:
            d_theta = scale_to_joint_limits(d_theta, self.dt, max_joint_vel)

        cur_dof = self.dof_states_full[self.dof_indices, 0]
        self.pending_dof_pos = cur_dof + d_theta.squeeze(-1)

        # if self.actor_name == "indy7_arm_2":
        #     pos_err_norm = pos_err.norm().item()
        #     print(f"pos_err={pos_err_norm:.4f}  joint1={cur_dof[1].item():.4f}rad  "
        #         f"d_theta_norm={d_theta.norm().item():.4f}")

    def _step_joint_manual_mode(self):
        """수동으로 선택한 관절만 목표각으로 이동. 나머지 관절은 현재 목표값 유지."""
        full_dof_pos_targets = self.dof_states_full[:, 0].clone()
        full_dof_pos_targets[self.dof_indices] = self.manual_joint_targets
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(full_dof_pos_targets),
            gymtorch.unwrap_tensor(self.actor_sim_index),
            1
        )

    def _step_joint_mode(self):
        """관절각 보간만 수행 — 자코비안/IK 계산 전혀 필요 없음."""
        way_q = self.joint_motion_gen.step()
        if way_q is None:
            return  # 이동 끝나면 마지막 목표각에서 그대로 유지 (이미 target으로 세팅되어 있음)

        full_dof_pos_targets = self.dof_states_full[:, 0].clone()
        full_dof_pos_targets[self.dof_indices] = way_q
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(full_dof_pos_targets),
            gymtorch.unwrap_tensor(self.actor_sim_index),
            1
        )

    def _step_ik_mode(self, max_joint_vel):
        """기존 IK 로직 그대로."""
        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]

        way_pos, way_quat = self.motion_gen.step()
        if way_pos is None:
            way_pos, way_quat = self.target_pos, self.target_rot

        pos_err = way_pos - ee_pos
        cur_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
        goal_q = gymapi.Quat(way_quat[0].item(), way_quat[1].item(), way_quat[2].item(), way_quat[3].item())
        err_q = goal_q * cur_q.inverse()
        if err_q.w < 0:
            err_q.x, err_q.y, err_q.z, err_q.w = -err_q.x, -err_q.y, -err_q.z, -err_q.w
        rot_err = torch.tensor([err_q.x, err_q.y, err_q.z], device=pos_err.device) * 2.0
        dpose = torch.cat([pos_err, rot_err]).unsqueeze(-1)

        j_eef = self.jacobian[0, self.ee_index_actor - 1, :, :self.dof_count]

        #d_theta = weighted_dls(j_eef, dpose, self.active_axis_weights, self.ik_damping)
        # active_axis_weights가 lock_weight면 '잠금'으로 간주
        lock_mask = self.active_axis_weights > (self.free_weight * 10)  # 임계값은 상황에 맞게

        d_theta = priority_ik(
            j_eef, dpose, lock_mask,
            damping_primary=0.02,   # 잠긴 축은 낮게 (정확도 우선)
            damping_secondary=0.1,  # 자유 축은 기존 ik_damping 정도로
        )

        if max_joint_vel is not None:
            d_theta = scale_to_joint_limits(d_theta, self.dt, max_joint_vel)

        cur_dof = self.dof_states_full[self.dof_indices, 0]
        new_dof_pos = cur_dof + d_theta.squeeze(-1)
        full_dof_pos_targets = self.dof_states_full[:, 0].clone()
        full_dof_pos_targets[self.dof_indices] = new_dof_pos
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(full_dof_pos_targets),
            gymtorch.unwrap_tensor(self.actor_sim_index),
            1
        )

    def draw_target_marker(self):
        origin = gymapi.Vec3(self.target_pos[0], self.target_pos[1], self.target_pos[2])
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.05*1.4, 10, 10, gymapi.Transform(p=origin), (0.0, 0.0, 1.0)
        )
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, gymapi.Transform())

    def draw_ee_marker(self, radius=0.04, color=(1.0, 0.0, 0.0)):
        """현재 end-effector(tcp)의 실제 월드 위치에 구체를 그린다. (기본: 빨간색)"""
        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        origin = gymapi.Vec3(ee_pos[0].item(), ee_pos[1].item(), ee_pos[2].item())
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius, 10, 10, gymapi.Transform(p=origin), color
        )
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, gymapi.Transform())

    def print_status_report(self):
        """
        스페이스바 등으로 호출: end-effector의 월드좌표/베이스 상대좌표/회전(쿼터니언+RPY),
        그리고 모든 관절의 현재 각도를 콘솔에 출력한다.
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]  # (x,y,z,w)

        base_pos = self.rb_states[self.base_index_global, 0:3]
        base_rot = self.rb_states[self.base_index_global, 3:7]

        # --- 베이스 기준 상대 위치/회전 계산 ---
        base_q = gymapi.Quat(base_rot[0].item(), base_rot[1].item(), base_rot[2].item(), base_rot[3].item())
        ee_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())

        delta_world = gymapi.Vec3(
            (ee_pos[0] - base_pos[0]).item(),
            (ee_pos[1] - base_pos[1]).item(),
            (ee_pos[2] - base_pos[2]).item(),
        )
        base_q_inv = base_q.inverse()
        relative_pos = base_q_inv.rotate(delta_world)
        relative_rot = base_q_inv * ee_q

        # --- 신규: 월드/상대 회전 각각 RPY로 변환 ---
        world_roll, world_pitch, world_yaw = quat_to_rpy_deg(
            ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item()
        )
        rel_roll, rel_pitch, rel_yaw = quat_to_rpy_deg(
            relative_rot.x, relative_rot.y, relative_rot.z, relative_rot.w
        )

        print("\n" + "="*60)
        print("[Indy7 상태 리포트]")
        print("-"*60)
        print(f"  월드 좌표 (Position, m):")
        print(f"    x={ee_pos[0].item():.4f}, y={ee_pos[1].item():.4f}, z={ee_pos[2].item():.4f}")
        print(f"  월드 회전 (Quaternion, x,y,z,w):")
        print(f"    x={ee_rot[0].item():.4f}, y={ee_rot[1].item():.4f}, "
            f"z={ee_rot[2].item():.4f}, w={ee_rot[3].item():.4f}")
        print(f"  월드 회전 (Roll, Pitch, Yaw, deg):")
        print(f"    roll={world_roll:8.3f}, pitch={world_pitch:8.3f}, yaw={world_yaw:8.3f}")
        print(f"  베이스 기준 상대 좌표 (m):")
        print(f"    x={relative_pos.x:.4f}, y={relative_pos.y:.4f}, z={relative_pos.z:.4f}")
        print(f"  베이스 기준 상대 회전 (Quaternion, x,y,z,w):")
        print(f"    x={relative_rot.x:.4f}, y={relative_rot.y:.4f}, "
            f"z={relative_rot.z:.4f}, w={relative_rot.w:.4f}")
        print(f"  베이스 기준 상대 회전 (Roll, Pitch, Yaw, deg):")
        print(f"    roll={rel_roll:8.3f}, pitch={rel_pitch:8.3f}, yaw={rel_yaw:8.3f}")
        print("-"*60)
        print(f"  관절 각도 ({self.dof_count}개):")
        cur_dof = self.dof_states_full[self.dof_indices, 0]
        for i, name in enumerate(self.dof_names):
            rad = cur_dof[i].item()
            deg = np.degrees(rad)
            print(f"    [{i}] {name:20s}: {rad:8.4f} rad  ({deg:7.2f}°)")
        for i, name in enumerate(self.dof_names):
            rad = cur_dof[i].item()
            deg = np.degrees(rad)
            print(f"{rad:.4f}, ",end="")

        print("\n" + "="*60 + "\n")

    def _distance_to_ee(self, target_sim_idx, ee_offset_world):
        target_pos = self.rb_states[target_sim_idx, 0:3]
        dx = target_pos[0].item() - ee_offset_world.x
        dy = target_pos[1].item() - ee_offset_world.y
        dz = target_pos[2].item() - ee_offset_world.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5
    
    def _get_root_indices(self, actor_handle):
        """대상 액터의 root state 텐서 상 인덱스(액터 기준, SIM 전체)를 반환."""
        actor_sim_idx = self.gym.get_actor_index(self.env, actor_handle, gymapi.DOMAIN_SIM)
        rb_sim_idx = self.gym.get_actor_rigid_body_index(self.env, actor_handle, 0, gymapi.DOMAIN_SIM)
        return actor_sim_idx, rb_sim_idx

    def attach(self, target_handle, distance_threshold=0.15, ee_local_offset=(0.0, 0.0, 0.0)):
        if self.is_attached:
            print("[IndyArm] 이미 흡착 중입니다. 먼저 detach() 하세요.")
            return False

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)   # ← 신규: root state도 갱신

        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]
        ee_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())

        local_offset = gymapi.Vec3(*ee_local_offset)
        world_offset = ee_q.rotate(local_offset)
        attach_world_pos = gymapi.Vec3(
            ee_pos[0].item() + world_offset.x,
            ee_pos[1].item() + world_offset.y,
            ee_pos[2].item() + world_offset.z,
        )

        actor_sim_idx, rb_sim_idx = self._get_root_indices(target_handle)
        if actor_sim_idx == -1 or rb_sim_idx == -1:
            print("[IndyArm] attach 실패: 대상 인덱스를 찾을 수 없습니다.")
            return False

        dist = self._distance_to_ee(rb_sim_idx, attach_world_pos)
        if dist > distance_threshold:
            print(f"[IndyArm] attach 실패: 거리 초과 ({dist:.4f}m > {distance_threshold}m)")
            return False

        target_pos = self.rb_states[rb_sim_idx, 0:3]
        target_rot = self.rb_states[rb_sim_idx, 3:7]
        target_q = gymapi.Quat(target_rot[0].item(), target_rot[1].item(), target_rot[2].item(), target_rot[3].item())

        ee_q_inv = ee_q.inverse()
        delta_world = gymapi.Vec3(
            target_pos[0].item() - attach_world_pos.x,
            target_pos[1].item() - attach_world_pos.y,
            target_pos[2].item() - attach_world_pos.z,
        )

        self.snap_local_pos_offset = ee_q_inv.rotate(delta_world)
        self.snap_local_rot_offset = ee_q_inv * target_q
        self.snap_ee_local_offset = local_offset

        self.is_attached = True
        self.attached_handle = target_handle
        self.attached_actor_index = actor_sim_idx   # ← root_states 인덱싱용 (신규)
        self.attached_sim_index = rb_sim_idx        # ← rb_states 읽기용 (기존 유지)

        print(f"[IndyArm] 흡착 성공 (거리 {dist:.4f}m)")
        return True

    def attach_nearest(self, distance_threshold=0.15, ee_local_offset=(0.0, 0.0, 0.0)):
        """register_attachable로 등록된 후보 중 가장 가까운 물체를 자동으로 흡착 시도."""
        if not self.attachable_handles:
            print("[IndyArm] 등록된 흡착 후보가 없습니다. register_attachable()로 먼저 등록하세요.")
            return False
 
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]
        ee_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
        world_offset = ee_q.rotate(gymapi.Vec3(*ee_local_offset))
        attach_world_pos = gymapi.Vec3(
            ee_pos[0].item() + world_offset.x,
            ee_pos[1].item() + world_offset.y,
            ee_pos[2].item() + world_offset.z,
        )
 
        best_handle, best_dist = None, float("inf")
        for handle in self.attachable_handles:
            _, rb_sim_idx = self._get_root_indices(handle)
            if rb_sim_idx == -1:
                continue
            dist = self._distance_to_ee(rb_sim_idx, attach_world_pos)
            if dist < best_dist:
                best_dist, best_handle = dist, handle
 
        if best_handle is None or best_dist > distance_threshold:
            print(f"[IndyArm] attach_nearest 실패: 근처에 흡착 가능한 물체 없음 (최소거리 {best_dist:.4f}m)")
            return False
 
        return self.attach(best_handle, distance_threshold=distance_threshold, ee_local_offset=ee_local_offset)

    def detach(self):
        if not self.is_attached:
            return False
        self.is_attached = False
        self.attached_handle = None
        self.attached_sim_index = None
        self.attached_env_index = None
        self.attached_actor_index = None
        self.snap_local_pos_offset = None
        self.snap_local_rot_offset = None
        print("[IndyArm] 탈착 완료")
        return True
    
    def update_attachment(self, pos_gain=0.6, rot_gain=0.5):
        if not self.is_attached:
            return
 
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
 
        ee_pos = self.rb_states[self.ee_index_global, 0:3]
        ee_rot = self.rb_states[self.ee_index_global, 3:7]
        ee_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
 
        attach_world_offset = ee_q.rotate(self.snap_ee_local_offset)
        attach_world_pos = gymapi.Vec3(
            ee_pos[0].item() + attach_world_offset.x,
            ee_pos[1].item() + attach_world_offset.y,
            ee_pos[2].item() + attach_world_offset.z,
        )
 
        target_offset_world = ee_q.rotate(self.snap_local_pos_offset)
        target_pos = gymapi.Vec3(
            attach_world_pos.x + target_offset_world.x,
            attach_world_pos.y + target_offset_world.y,
            attach_world_pos.z + target_offset_world.z,
        )
        target_rot = ee_q * self.snap_local_rot_offset
 
        # --- root_states에서 현재 상태 읽기 (rb_states 대신) ---
        cur_pos = self.root_states[self.attached_actor_index, 0:3]
        cur_rot = self.root_states[self.attached_actor_index, 3:7]
        cur_q = gymapi.Quat(cur_rot[0].item(), cur_rot[1].item(), cur_rot[2].item(), cur_rot[3].item())
 
        dt = self.dt
 
        # --- 위치: 목표 지점으로 gain만큼 블렌딩(부분 텔레포트) ---
        # 순수 속도 명령만으로는 솔버 감쇠/충돌 때문에 따라가지 못하는 경우가 있어,
        # FrankaController.update_snapping_object() 와 동일하게 위치/회전도 함께 보정한다.
        next_pos_x = cur_pos[0].item() + (target_pos.x - cur_pos[0].item()) * pos_gain
        next_pos_y = cur_pos[1].item() + (target_pos.y - cur_pos[1].item()) * pos_gain
        next_pos_z = cur_pos[2].item() + (target_pos.z - cur_pos[2].item()) * pos_gain
 
        next_q = slerp_quat(cur_q, target_rot, rot_gain)
 
        vel_x = (target_pos.x - cur_pos[0].item()) / dt * pos_gain
        vel_y = (target_pos.y - cur_pos[1].item()) / dt * pos_gain
        vel_z = (target_pos.z - cur_pos[2].item()) / dt * pos_gain
 
        rot_err = target_rot * cur_q.inverse()
        if rot_err.w < 0:
            rot_err.x, rot_err.y, rot_err.z, rot_err.w = -rot_err.x, -rot_err.y, -rot_err.z, -rot_err.w
        omega_x = rot_err.x * 2.0 / dt * rot_gain
        omega_y = rot_err.y * 2.0 / dt * rot_gain
        omega_z = rot_err.z * 2.0 / dt * rot_gain
 
        # --- 흡착 물체 액터 하나만 인덱싱해서 갱신 ---
        new_state = self.root_states[self.attached_actor_index].clone()
        new_state[0]  = next_pos_x
        new_state[1]  = next_pos_y
        new_state[2]  = next_pos_z
        new_state[3]  = next_q.x
        new_state[4]  = next_q.y
        new_state[5]  = next_q.z
        new_state[6]  = next_q.w
        new_state[7]  = vel_x
        new_state[8]  = vel_y
        new_state[9]  = vel_z
        new_state[10] = omega_x
        new_state[11] = omega_y
        new_state[12] = omega_z
 
        # 🔑 [핵심 수정] 계산한 new_state를 실제로 self.root_states에 반영해야
        # 아래 set_actor_root_state_tensor_indexed 호출이 의미를 가진다.
        # (과거에는 new_state만 만들고 실제 텐서에는 쓰지 않아 흡착이 전혀 동작하지 않았음)
        self.root_states[self.attached_actor_index] = new_state
 
        actor_indices = torch.tensor(
            [self.attached_actor_index], dtype=torch.int32, device=self.root_states.device
        )
 
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            1
        )