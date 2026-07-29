"""
tachikoma / Phase 2a  案(b) 動的リフト  ビューア(URDF直読みモデル)

URDF直読み4脚(立位)で、案(b)の単隅リフト挙動を可視化する。
成立域の推奨(中庸開脚・遅リフト)と、不可域(速リフト/COM高)を見比べられる。

使い方:
    uv run python sim/phase2a_urdf_viewer.py            # 推奨(成立)を巡回
    uv run python sim/phase2a_urdf_viewer.py fast       # 速リフト(不可域)
    uv run python sim/phase2a_urdf_viewer.py headless    # 画面なし数値検証

局面: STAND(4輪) → LIFT(前脚上げ→対角抜け) → HOLD(段へ運ぶ想定) → BACK(戻して回復)。
"""
import argparse
import time
import numpy as np
import mujoco

import urdf_model as U
from phase2a_urdf import _lead_leg, _lift_pitch, _tilt
from quad_stability import leg_normal_force

# 総重量1.606kg(URDF実値)＋柔らかい足首サス(2500)では、P≲0.7 の高め立位は
# 起立時に横倒れ→天地逆になる(要 P≳0.8 で安定。旧 P=0.6/0.4 は1.3kg時代の値)。
PRESETS = {
    "recommended": dict(splay=0.45, P=0.8, lift_time=0.30),
    "fast": dict(splay=0.20, P=0.8, lift_time=0.15),
}
SEG = 2.0            # LIFT / HOLD / BACK 各局面の秒数
STAND_SEG = 5.0      # 初期姿勢(STAND)の表示秒数(--stand で変更可)


def _setup(preset):
    p = PRESETS[preset]
    m, d, A = U.make_stand(P=p["P"], yaw_splay=p["splay"], settle=1200)
    stance = {f: U.PITCH_SIGN[f] * p["P"] for f in U.FEET}
    ysign = {"feet": +1, "feet_2": +1, "feet_1": -1, "feet_3": -1}
    yaw_ctrl = {f: ysign[f] * p["splay"] for f in U.FEET}
    lead = _lead_leg(m, d)
    lift_target, _ = _lift_pitch(m, d, lead, stance)
    return m, d, A, stance, yaw_ctrl, lead, lift_target, p["lift_time"]


def cycle_len():
    return STAND_SEG + 3 * SEG


def _phase_ctrl(t, stance, lead, lift_target, lift_time):
    ph = t % cycle_len()
    if ph < STAND_SEG:                                  # 初期姿勢(長め)
        return stance[lead], "STAND  4脚で立つ"
    ph -= STAND_SEG
    if ph < SEG:
        a = min(1.0, ph / lift_time)
        return stance[lead] + a * (lift_target - stance[lead]), "LIFT   前脚上げ→対角抜け(接地2輪)"
    elif ph < 2 * SEG:
        return lift_target, "HOLD   滞空(段へ足を運ぶ想定)"
    else:
        a = min(1.0, (ph - 2 * SEG) / lift_time)
        return lift_target + a * (stance[lead] - lift_target), "BACK   足を戻す→回復"


def run_headless(preset="recommended"):
    m, d, A, stance, yaw_ctrl, lead, lift_target, lt = _setup(preset)
    dt = m.opt.timestep
    last = None
    peak = 0.0
    for k in range(int(cycle_len() / dt)):
        t = k * dt
        p, tag = _phase_ctrl(t, stance, lead, lift_target, lt)
        for f in U.FEET:
            d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
            d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = yaw_ctrl[f]
        d.ctrl[A[f"pitchpos_{lead}"]] = p
        mujoco.mj_step(m, d)
        peak = max(peak, _tilt(m, d))
        if tag != last:
            nsup = sum(1 for f in U.FEET if leg_normal_force(m, d, f) > 0.05)
            print("  [%4.1fs] %-34s tilt=%4.1f° 接地=%d bz=%.3f" %
                  (t, tag, _tilt(m, d), nsup, d.body("body").xpos[2]))
            last = tag
    print("  preset=%s → ピーク傾き=%.1f° 最終=%.1f°" % (preset, peak, _tilt(m, d)))


def main(preset="recommended"):
    m, d, A, stance, yaw_ctrl, lead, lift_target, lt = _setup(preset)
    print(__doc__); print("preset =", preset, PRESETS[preset], " lead =", lead)
    import mujoco.viewer
    last = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            p, tag = _phase_ctrl(t, stance, lead, lift_target, lt)
            for f in U.FEET:
                d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
                d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = yaw_ctrl[f]
            d.ctrl[A[f"pitchpos_{lead}"]] = p
            if tag != last:
                print("  [%5.1fs] %s" % (t, tag)); last = tag
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(dt if (dt := m.opt.timestep) else 0.002)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="recommended",
                    help="プリセット (%s) または headless [recommended]" % "/".join(PRESETS))
    ap.add_argument("--stand", type=float, default=STAND_SEG,
                    help="初期姿勢(STAND)の表示秒数 [%(default)s]")
    U.add_robot_arg(ap)
    args = ap.parse_args()
    U.set_robot(args.robot)
    STAND_SEG = args.stand
    if args.mode == "headless":
        run_headless("recommended"); print(); run_headless("fast")
    else:
        main(args.mode if args.mode in PRESETS else "recommended")
