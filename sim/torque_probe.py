"""
tachikoma / Phase 1 主成果物  ―  股ピッチのピークトルク実測

要件 §7,§8: 「MuJoCoモデルに実質量・寸法を入れて 3cm 段差通過と姿勢保持を走らせ、
股ピッチのピークトルクを実測する。その値に安全率 1.5〜2 を掛けて製品を確定する。」

ホイール径は 3cm 固定(半径0.015)。段差3cmと同高のため転がり越えは原理的に不可で、
要件どおり「脚で越える」= 股ピッチで脚を振り出し、車輪を段の上面に置き、荷重を移して
車体を引き上げる(pull-up)。トルクのピークはこの引き上げ局面に出る。

2つのシナリオでトルクをログる:
  (A) 姿勢保持スイープ : 股ピッチ角を振って各姿勢の保持トルクを測り、最悪姿勢=
                         ピーク保持トルクを出す(棒脚を寝かせるほど大きい)。
  (B) 3cm段差 pull-up  : 車輪を段の上面に置いた状態から股ピッチで車体を引き上げ、
                         その股ピッチのピークトルクを測る(=段差越えの律速トルク)。

  ※単脚の割り切り: 「空中で脚を振り出して足を段上へ置く」局面は、その足が唯一の
    支持点になるため単脚では自らを持ち上げられず、他の3脚の支持が要る(=多脚/Phase2)。
    本rigでは車輪を段上に置いた状態を初期姿勢とし、荷重移動+引き上げのトルクを測る。

荷重は要件 §6 の安全側 総重量 1.5kg を単脚に載せる(=全荷重が1脚に乗る最悪ケース)。
参考として静的4脚分担 総重量/4 = 0.375kg でも出す。

使い方:
    uv run python sim/torque_probe.py
    uv run python sim/torque_probe.py --load 0.375
"""
import argparse
import numpy as np
import mujoco

XML = "sim/leg_single.xml"


def make(load_kg):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    # carriage 質量を load_kg に設定(慣性も比例スケール)
    bid = m.body("carriage").id
    scale = load_kg / 1.5
    m.body_mass[bid] = load_kg
    m.body_inertia[bid] *= scale
    A = {m.actuator(i).name: i for i in range(m.nu)}
    return m, d, A


def settle(m, d, steps):
    for _ in range(steps):
        mujoco.mj_step(m, d)


def scenario_hold(load_kg):
    """(A) 姿勢保持スイープ: 股ピッチ角 vs 保持トルク。ピークを返す。"""
    m, d, A = make(load_kg)
    qp = m.joint("hip_pitch").qposadr[0]
    rows = []
    for target in np.linspace(0.05, 1.05, 21):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[A["hip_yaw_pos"]] = 0.0
        d.ctrl[A["hip_pitch_pos"]] = target
        d.ctrl[A["wheel_drive"]] = 0.0
        settle(m, d, 3000)
        pitch = d.qpos[qp]
        tau = d.actuator("hip_pitch_pos").force[0]
        cz = d.body("carriage").xpos[2]
        contact = d.body("wheel").xpos[2] < 0.05
        rows.append((target, pitch, tau, cz, contact))
    valid = [r for r in rows if r[4]]
    peak = max(valid or rows, key=lambda r: abs(r[2]))
    return rows, peak


# --- (B) pull-up 初期姿勢 (1.5kg で車輪が段の縁上に載るよう調整済み) ---
PULLUP_INIT = dict(slide_x=0.086, slide_z=-0.041, pitch=-0.9)
PULLUP_TARGET = -0.30          # 引き上げ後の股ピッチ(車体が段上に載る)
PULLUP_RAMP_S = 1.0            # 引き上げ所要時間[s]


def scenario_pullup(load_kg, ramp_s=PULLUP_RAMP_S):
    """(B) 3cm段差 pull-up: 車輪を段上面に置いた状態から股ピッチで車体を引き上げ、
    股ピッチのピークトルクを測る。車体が ~3cm 上がれば段差越え成功。"""
    m, d, A = make(load_kg)
    qsx = m.joint("slide_x").qposadr[0]
    qsz = m.joint("slide_z").qposadr[0]
    qpi = m.joint("hip_pitch").qposadr[0]

    # 車輪を段の縁上へ置いた初期姿勢
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[qsx] = PULLUP_INIT["slide_x"]
    d.qpos[qsz] = PULLUP_INIT["slide_z"]
    d.qpos[qpi] = PULLUP_INIT["pitch"]
    d.ctrl[A["hip_yaw_pos"]] = 0.0
    d.ctrl[A["hip_pitch_pos"]] = PULLUP_INIT["pitch"]
    d.ctrl[A["wheel_drive"]] = 0.0
    mujoco.mj_forward(m, d)
    settle(m, d, 300)                                   # 縁での座り(短く=ドリフト最小)
    placed_on_step = d.body("wheel").xpos[2] > 0.035
    z0 = d.body("carriage").xpos[2]

    peak = 0.0
    peak_t = 0.0
    traj = []
    dt = m.opt.timestep
    n = int((ramp_s + 1.0) / dt)                        # 引き上げ + 1秒保持
    for k in range(n):
        t = k * dt
        cmd = min(PULLUP_INIT["pitch"] + (PULLUP_TARGET - PULLUP_INIT["pitch"]) * (t / ramp_s),
                  PULLUP_TARGET)
        d.ctrl[A["hip_pitch_pos"]] = cmd
        d.ctrl[A["wheel_drive"]] = 0.0                  # 車輪はブレーキ(段上に保持)
        mujoco.mj_step(m, d)
        tau = d.actuator("hip_pitch_pos").force[0]
        if abs(tau) > abs(peak):
            peak = tau
            peak_t = t
        traj.append((t, d.body("carriage").xpos[0], d.body("carriage").xpos[2], tau))
    lift = d.body("carriage").xpos[2] - z0
    climbed = lift > 0.025                              # 3cm段を登り切ったか(目安)
    return traj, peak, peak_t, placed_on_step, climbed, lift


def report(load_kg):
    print("=" * 66)
    print("荷重 load = %.3f kg  (総重量%.2fkg %s)" %
          (load_kg, load_kg, "そのもの=安全側" if abs(load_kg - 1.5) < 1e-6 else "の分担"))
    print("=" * 66)

    rows, peak = scenario_hold(load_kg)
    print("\n[A] 姿勢保持スイープ  target -> 実pitch  保持トルク[N*m]  車体高[m] 接地")
    for target, pitch, tau, cz, contact in rows[::2]:
        print("     %.2f -> %+.3f   %+7.3f     %.3f   %s" %
              (target, pitch, tau, cz, "接地" if contact else " 浮"))
    hold_peak = abs(peak[2])
    print("  → 保持ピーク: pitch=%+.3f で |tau|=%.3f N*m" % (peak[1], hold_peak))

    traj, speak, speak_t, placed, climbed, lift = scenario_pullup(load_kg)
    print("\n[B] 3cm段差 pull-up  (車輪を段上に置き、股ピッチで車体を引き上げ)")
    print("     車輪の段上設置=%s  引き上げ量=%.3f m  段差越え=%s" %
          ("OK" if placed else "NG", lift, "成功" if climbed else "未達"))
    step_peak = abs(speak)
    print("  → 引き上げピーク: t=%.2fs で |tau|=%.3f N*m" % (speak_t, step_peak))

    frng = float(mujoco.MjModel.from_xml_path(XML).actuator("hip_pitch_pos").forcerange[1])
    saturated = max(hold_peak, step_peak) > frng - 1e-3

    overall = max(hold_peak, step_peak)
    print("\n[結論] load=%.3fkg" % load_kg)
    print("  姿勢保持ピーク       = %.3f N*m (最悪リーチ姿勢)" % hold_peak)
    print("  3cm段差pull-upピーク = %.3f N*m (段差越えの律速)" % step_peak)
    if saturated:
        print("  ※ forcerange %.0f N*m に飽和 → 広げて再測を" % frng)
    print("  --------------------------------------------------------------")
    print("  股ピッチ サイジング値 = %.3f N*m  → x1.5=%.3f  x2.0=%.3f" %
          (overall, overall * 1.5, overall * 2.0))
    print("  (Feetech STS3215 ~1.5-3 N*m / STS3250 ~3-4 / Dynamixel XM430 等と比較)")
    return overall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=float, default=None,
                    help="単脚に載せる荷重[kg]. 省略時は 1.5(安全側) と 0.375(4脚分担) 両方")
    args = ap.parse_args()
    if args.load is not None:
        report(args.load)
    else:
        report(1.5)
        print()
        report(0.375)
