"""Film a get-up brain from the supine event start: contact sheet + joint speed / torque statistics."""
import math, sys
import mujoco, numpy as np, onnxruntime as ort
from poolympic import contract as C
onnx, png, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000
rng = np.random.default_rng(seed)
m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML)); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
ath = C.Athlete.bind(m); r = ath.root_qposadr; torso = m.body("torso").id
roll, pitch, yaw = rng.uniform(-0.3, 0.3), -math.pi / 2 + rng.uniform(-0.2, 0.2), 0.0
q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "XYZ")
d.qpos[r:r + 3] = [0, 0, 0.22]; d.qpos[r + 3:r + 7] = q
mujoco.mj_forward(m, d)
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
last, cmd = np.zeros(C.NUM_ACTIONS), np.zeros(3)
ren = mujoco.Renderer(m, 300, 240); cam = mujoco.MjvCamera(); cam.distance = 3.4; cam.azimuth = 90; cam.elevation = -8
dofs = [m.jnt_dofadr[m.actuator_trnid[a][0]] for a in ath.actuator_ids]
names = list(ath.actuator_names)
shots, vmax, over, n, tq = [], np.zeros(len(dofs)), 0, 0, np.zeros(len(dofs))
hands = 0
ground = m.geom("ground").id
arm_bodies = {m.body(b).id for b in ("forearm_l", "forearm_r", "upper_arm_l", "upper_arm_r")}
for tick in range(150):
    if tick % 8 == 0 and len(shots) < 14:
        cam.lookat[:] = [d.qpos[r], d.qpos[r + 1], 0.55]; ren.update_scene(d, cam); shots.append(ren.render().copy())
    obs = C.build_obs(ath, d.qpos, d.qvel, cmd, 0.0, last)
    ctrl, act = sess.run(None, {"obs": obs[None]})
    d.ctrl[ath.actuator_ids] = ctrl[0]; last = act[0].astype(np.float64)
    for _ in range(C.DECIMATION):
        mujoco.mj_step(m, d)
        v = np.abs(d.qvel[dofs]); vmax = np.maximum(vmax, v); over += int((v > 18).any()); n += 1
        tq = np.maximum(tq, np.abs(d.actuator_force[ath.actuator_ids]))
    for c in d.contact[: d.ncon]:
        o = c.geom2 if c.geom1 == ground else c.geom1 if c.geom2 == ground else -1
        if o >= 0 and m.geom_bodyid[o] in arm_bodies:
            hands += 1; break
    tilt = math.degrees(math.acos(max(-1, min(1, d.xmat[torso][8]))))
    if tick % 10 == 9:
        print(f"t {(tick + 1) * 0.02:.1f}s pelvis z {d.qpos[r + 2]:.2f} tilt {tilt:3.0f}")
from PIL import Image
Image.fromarray(np.concatenate([np.concatenate(shots[:7], axis=1), np.concatenate(shots[7:14], axis=1)], axis=0)).save(png)
top = np.argsort(-vmax)[:4]
print("fastest joints (rad/s):", ", ".join(f"{names[i]} {vmax[i]:.1f}" for i in top), f"| substeps with any joint > 18 rad/s: {100 * over / n:.1f} %")
print("peak |torque| (Nm):", ", ".join(f"{names[i]} {tq[i]:.0f}" for i in np.argsort(-tq)[:5]), f"| ticks with an arm on the ground: {hands}")
