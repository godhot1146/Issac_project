import numpy as np
from isaacgym import gymapi

gym = gymapi.acquire_gym()

# --- sim 생성 ---
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# --- asset 로드 ---
asset_root = "/home/henry/Desktop/Issac_asset/isaac_assets/urdf/doosan_a0509"
asset_file = "a0509.urdf"
opts = gymapi.AssetOptions()
opts.fix_base_link = True
opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset = gym.load_asset(sim, asset_root, asset_file, opts)

# --- env / actor ---
env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
pose = gymapi.Transform()
actor = gym.create_actor(env, asset, pose, "a0509", 0, 0)

# 모든 DOF를 0으로
num_dof = gym.get_actor_dof_count(env, actor)
dof_states = np.zeros(num_dof, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

gym.simulate(sim); gym.fetch_results(sim, True)

# --- 링크(rigid body) 월드 좌표 읽기 ---
names = gym.get_actor_rigid_body_names(env, actor)
states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
pos = states['pose']['p']

print("=== 링크별 위치 ===")
p = {}
for n, q in zip(names, pos):
    p[n] = np.array([q['x'], q['y'], q['z']])
    print(f"{n:12s}: {p[n]}")

def dist(a, b): return np.linalg.norm(p[a] - p[b])

print("\n=== 링크 길이 vs 기준값 ===")
print(f"base_link -> link_1 : {dist('base_link','link_1'):.3f} m  (기준 0.156)")
print(f"link_2 -> link_3    : {dist('link_2','link_3'):.3f} m  (기준 0.409)")
print(f"link_3 -> link_4    : {dist('link_3','link_4'):.3f} m  (기준 0.367)")
print(f"link_5 -> link_6    : {dist('link_5','link_6'):.3f} m  (기준 0.124)")