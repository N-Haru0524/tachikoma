"""
tachikoma / Phase 2a 案B ― tank軽量化+偏心是正 の 3cm段差リフト ビューア

phase2a_stepclimb の核心(tank側前脚 F_1 の持ち上げ可否)を可視化する。
  ・recommended : tank 280g・重心を y=−35mm へ寄せた推奨構成で、tank側前脚(feet_1)を
                  リフト→戻す。転倒せず持ち上がる(段越えの鍵)。
  ・current     : 現状 tank 580g・y=−70mm で同じ feet_1 リフト。偏重心で転倒する(不可)。

使い方:
    uv run python sim/phase2a_stepclimb_viewer.py               # 推奨構成(成立)
    uv run python sim/phase2a_stepclimb_viewer.py current       # 現状(転倒)
    uv run python sim/phase2a_stepclimb_viewer.py headless      # 画面なし数値検証
"""
import argparse
import time
import numpy as np
import mujoco

import urdf_model as U
from phase2a_stepclimb import _lift_pitch, _tilt
from quad_stability import leg_normal_force

PRESETS = {
    "recommended": dict(tank_mass=0.280, tank_y=-0.035, lead="feet_1"),
    "current":     dict(tank_mass=None,  tank_y=None,    lead="feet_1"),
}
SEG = 2.0
P = 1.1
LIFT_TIME = 0.30


def _setup(preset):
    p = PRESETS[preset]
    m, d, A = U.make_stand(P=P, yaw_splay=0.0, tank_mass=p["tank_mass"],
                           tank_y=p["tank_y"], settle=1200)
    stance = {f: U.PITCH_SIGN[f] * P for f in U.FEET}
    lead = p["lead"]
    lift_target, _ = _lift_pitch(m, d, lead)
    return m, d, A, stance, lead, lift_target


def _phase_ctrl(t, stance, lead, lift_target):
    ph = t % (4 * SEG)
    if ph < SEG:
        return stance[lead], "STAND  4脚立位"
    elif ph < 2 * SEG:
        a = min(1.0, (ph - SEG) / LIFT_TIME)
        return stance[lead] + a * (lift_target - stance[lead]), "LIFT   tank側前脚 feet_1 を段へ"
    elif ph < 3 * SEG:
        return lift_target, "HOLD   滞空"
    else:
        a = min(1.0, (ph - 3 * SEG) / LIFT_TIME)
        return lift_target + a * (stance[lead] - lift_target), "BACK   戻す→回復"


def run_headless(preset):
    m, d, A, stance, lead, lift_target = _setup(preset)
    dt = m.opt.timestep
    peak, last = 0.0, None
    for k in range(int(4 * SEG / dt)):
        t = k * dt
        p, tag = _phase_ctrl(t, stance, lead, lift_target)
        for f in U.FEET:
            d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
            d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = 0.0
        d.ctrl[A[f"pitchpos_{lead}"]] = p
        mujoco.mj_step(m, d)
        peak = max(peak, _tilt(m, d))
        if tag != last:
            nsup = sum(1 for f in U.FEET if leg_normal_force(m, d, f) > 0.05)
            print("  [%4.1fs] %-32s tilt=%5.1f° 接地=%d" % (t, tag, _tilt(m, d), nsup))
            last = tag
    verdict = "○持ち上げ成立" if peak < 35 and _tilt(m, d) < 6 else "×転倒"
    print("  preset=%-12s ピーク傾き=%.1f° 最終=%.1f° → %s\n" % (preset, peak, _tilt(m, d), verdict))


def main(preset="recommended"):
    m, d, A, stance, lead, lift_target = _setup(preset)
    print(__doc__); print("preset =", preset, PRESETS[preset])
    import mujoco.viewer
    dt = m.opt.timestep
    last = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            p, tag = _phase_ctrl(t, stance, lead, lift_target)
            for f in U.FEET:
                d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
                d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = 0.0
            d.ctrl[A[f"pitchpos_{lead}"]] = p
            if tag != last:
                print("  [%5.1fs] %s" % (t, tag)); last = tag
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="recommended",
                    help="プリセット (%s) または headless [recommended]" % "/".join(PRESETS))
    U.add_robot_arg(ap)
    args = ap.parse_args()
    U.set_robot(args.robot)
    if args.mode == "headless":
        run_headless("recommended"); run_headless("current")
    else:
        main(args.mode if args.mode in PRESETS else "recommended")
