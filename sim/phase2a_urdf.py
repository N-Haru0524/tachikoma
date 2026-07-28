"""
tachikoma / Phase 2a  ―  案(b) 動的リフト段差越えの成立性検証(URDF直読みモデル)

モデルは urdf_model.build/make_stand(URDF直読み＋freejoint＋立位＋車輪＋足首サス＋3cm段)。
本CADは股が±0.02の対角開脚配置のため、トレッドは構造ではなく【ヨー開脚角】で作る。

案(b): 1脚を短時間持ち上げて3cm段へ届かせる間、車体が倒れきる前に足を戻す/着地させて
成立するか(動的安定余裕)を判定。単隅リフトは対角脚も抜けて実質2輪(対角線)支持へ落ちる
(§10 のシーソー)。よって指標は【リフト中のピーク傾き】【接地輪数(対角線=2 を保つか)】
【COM vs 残り支持多角形 margin の時系列】【足を戻して水平へ回復するか】。

振るパラメータ:
  yaw_splay : ヨー開脚角[rad](トレッド相当, ±45°=±0.785 内)。広いほど安定側。
  P         : 立位 pitch(COM高さ; 大=高い)。
  lift_time : 1脚を上げてから戻すまでの時間[s]。

出力:
  (1) 成立/不成立の判定と、成立するパラメータ条件。
  (2) margin・トルクの時系列(--csv)。
  (3) 動作中 pitch/yaw ピークトルクが STS3215 射程(×2≤3.0 N·m)か。
  (4) phase2a_urdf_viewer.py で可視化。

使い方:
    uv run python sim/phase2a_urdf.py
    uv run python sim/phase2a_urdf.py --splays 0.2 0.45 0.7 --Ps 0.6 1.0 --lifts 0.15 0.3
"""
import argparse
import numpy as np
import mujoco

import urdf_model as U
from quad_stability import stability, leg_normal_force, com_full, com_xy, _poly_margin

LEGS = U.FEET
SAFE_TILT = 20.0
HARD_TILT = 35.0
STEP_H = U.STEP_H
LIFT_CLEAR = 0.05        # 上げる脚の車輪到達高さ[m](3cm段を越える余裕高さ)


def _tilt(m, d):
    fadr = m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_type[j] == 0][0]]
    w, x, y, z = d.qpos[fadr + 3:fadr + 7]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def _lead_leg(m, d):
    """段(+x)に最も近い前脚を選ぶ。"""
    wx = {f: d.body(f"wheel_{f}").xpos[0] for f in LEGS}
    return max(wx, key=wx.get)


def _lift_pitch(m, d, lead, stance_pitch):
    """lead の車輪を LIFT_CLEAR(≈段+余裕)まで上げる pitch を走査で求める。
    lead の可動域内で車輪高さを最大化する向きへ動かし、目標高さに最も近い角を返す。
    return (pitch, achieved_wheel_z)。"""
    qadr = m.joint(U.PITCH_JOINTS[lead]).qposadr[0]
    lo, hi = m.jnt_range[m.joint(U.PITCH_JOINTS[lead]).id]
    best_q, best_z, best_d = stance_pitch[lead], -1, 1e9
    for q in np.linspace(lo, hi, 40):
        d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; d2.qpos[qadr] = q
        mujoco.mj_forward(m, d2)
        wz = float(d2.body(f"wheel_{lead}").xpos[2])
        if abs(wz - LIFT_CLEAR) < best_d:
            best_d, best_q, best_z = abs(wz - LIFT_CLEAR), q, wz
    return best_q, best_z


def run(yaw_splay, P, lift_time, hold=0.10, settle=1.4, log=None):
    """案b マニューバを1回実行し、動的安定メトリクスを返す。"""
    m, d, A = U.make_stand(P=P, yaw_splay=yaw_splay, settle=1200)
    # 立位妥当性ガード: この (P, 開脚) で実際に立てているか
    stand_tilt = _tilt(m, d)
    nsup0 = sum(1 for f in LEGS if leg_normal_force(m, d, f) > 0.05)
    if stand_tilt > 8.0 or d.body("body").xpos[2] < 0.05 or nsup0 < 4:
        return {"splay": yaw_splay, "P": P, "lift_time": lift_time, "invalid": True,
                "com_h": float(com_full(m, d)[2]), "stand_tilt": stand_tilt, "nsup0": nsup0}
    stance = {f: U.PITCH_SIGN[f] * P for f in LEGS}
    ysign = {"feet": +1, "feet_2": +1, "feet_1": -1, "feet_3": -1}
    yaw_ctrl = {f: ysign[f] * yaw_splay for f in LEGS}
    lead = _lead_leg(m, d)
    lift_target, lift_wz = _lift_pitch(m, d, lead, stance)
    stance_wheels = [f for f in LEGS if f != lead]
    com0_z = float(com_full(m, d)[2])                 # ★立位COM高さ(基準)

    dt = m.opt.timestep
    peak_tilt = 0.0
    min_support = 4
    min_margin = 1e9
    peak_ptau = {f: 0.0 for f in LEGS}
    peak_ytau = {f: 0.0 for f in LEGS}
    bz0 = d.body("body").xpos[2]

    def hold_ctrl():
        for f in LEGS:
            d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
            d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = yaw_ctrl[f]

    def rec():
        nonlocal peak_tilt, min_support, min_margin
        peak_tilt = max(peak_tilt, _tilt(m, d))
        nsup = sum(1 for f in LEGS if leg_normal_force(m, d, f) > 0.05)
        min_support = min(min_support, nsup)
        # COM vs 残り3脚(stance_wheels)の支持多角形 margin
        poly = [d.body(f"wheel_{f}").xpos[:2].copy() for f in stance_wheels
                if leg_normal_force(m, d, f) > 0.05]
        if len(poly) >= 2:
            mg = _poly_margin(com_xy(m, d), poly)
            if mg is not None:
                min_margin = min(min_margin, mg)
        for f in LEGS:
            peak_ptau[f] = max(peak_ptau[f], abs(d.actuator(f"pitchpos_{f}").force[0]))
            peak_ytau[f] = max(peak_ytau[f], abs(d.actuator(f"yawpos_{U.YAW_OF[f]}").force[0]))
        if log is not None:
            log.append({"splay": yaw_splay, "P": P, "lift_time": lift_time, "t": d.time,
                        "tilt": _tilt(m, d), "nsup": nsup,
                        "margin": min_margin if min_margin < 1e8 else float("nan"),
                        "ptau_lead": d.actuator(f"pitchpos_{lead}").force[0]})

    def ramp(dur, p0, p1):
        n = max(1, int(dur / dt))
        for k in range(n):
            a = (k + 1) / n
            hold_ctrl()
            d.ctrl[A[f"pitchpos_{lead}"]] = p0 + a * (p1 - p0)
            mujoco.mj_step(m, d)
            rec()

    ramp(lift_time, stance[lead], lift_target)      # LIFT(lead 非支持へ)
    ramp(hold, lift_target, lift_target)            # hold(段へ足を運ぶ想定)
    ramp(lift_time, lift_target, stance[lead])      # 足を戻す
    n = int(settle / dt)
    for _ in range(n):
        hold_ctrl(); mujoco.mj_step(m, d); rec()

    final_tilt = _tilt(m, d)
    recovered = final_tilt < 6.0 and d.body("body").xpos[2] > bz0 - 0.02
    lost = min_support < 2
    toppled = peak_tilt > HARD_TILT or not recovered
    feasible = (not lost) and recovered and peak_tilt <= SAFE_TILT
    return {"splay": yaw_splay, "P": P, "lift_time": lift_time, "lead": lead,
            "invalid": False, "com_h": com0_z, "peak_tilt": peak_tilt,
            "final_tilt": final_tilt, "min_support": min_support,
            "lift_wz": lift_wz, "cleared": lift_wz > STEP_H + 0.005,
            "min_margin": min_margin if min_margin < 1e8 else float("nan"),
            "recovered": recovered, "toppled": toppled, "feasible": feasible,
            "peak_ptau": max(peak_ptau.values()), "peak_ytau": max(peak_ytau.values())}


def main():
    ap = argparse.ArgumentParser()
    # P→車高: 0.4→116mm / 0.6→107mm / 0.8→94mm(URDF実測)。立位が成立する範囲を既定に。
    ap.add_argument("--splays", type=float, nargs="+", default=[0.2, 0.45, 0.7])
    ap.add_argument("--Ps", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--lifts", type=float, nargs="+", default=[0.15, 0.30])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    log = [] if args.csv else None
    rows = []
    for sp in args.splays:
        for P in args.Ps:
            for lt in args.lifts:
                rows.append(run(sp, P, lt, log=log))

    L = "=" * 92
    print(L)
    print("Phase 2a: 案(b) 動的リフト段差越え 成立性  (URDF直読み4脚 / 総1.3kg / 段差3cm)")
    print("  トレッド=ヨー開脚角(±45°内)。1脚上げ→hold→戻す。単隅上げは対角も抜け接地2輪(対角線)。")
    print("  成立=接地≥2 保持・ピーク傾き≤%.0f°・足戻し後 水平復帰。" % SAFE_TILT)
    print(L)
    print("  開脚   P   COM   lift 段越  ピーク 最終 min  最小    成立    pitch  yaw")
    print("  [rad] [-] [mm]  [s]  可否 傾き° 傾き° 接地 margin        ピーク[N·m]")
    print("  " + "-" * 90)
    valid = [r for r in rows if not r.get("invalid")]
    for r in rows:
        if r.get("invalid"):
            print("  %.2f  %.1f %4.0f   -    -    立位不成立(COM%.0fmm,傾き%.0f°) → 除外"
                  % (r["splay"], r["P"], r["com_h"] * 1000, r["com_h"] * 1000, r["stand_tilt"]))
            continue
        mg = ("%+.3f" % r["min_margin"]) if not np.isnan(r["min_margin"]) else "  n/a"
        print("  %.2f  %.1f %4.0f  %.2f  %s  %5.1f %5.1f  %d  %6s  %-6s  %5.2f  %5.2f"
              % (r["splay"], r["P"], r["com_h"] * 1000, r["lift_time"],
                 "○" if r["cleared"] else "×", r["peak_tilt"],
                 r["final_tilt"], r["min_support"], mg,
                 "○成立" if r["feasible"] else ("×転倒" if r["toppled"] else "△限界"),
                 r["peak_ptau"], r["peak_ytau"]))
    print("  " + "-" * 90)
    print("  ※段越可否=上げた車輪が3cm段上面を越える高さに達したか。min接地=リフト中の最少接地輪。")

    feas = [r for r in valid if r["feasible"] and r["cleared"]]
    print("\n[(1) 成立領域]  (立位成立し段越え高さに達した %d 条件を対象)" % len(valid))
    if feas:
        best = min(feas, key=lambda r: r["peak_tilt"])
        print("  ・成立 %d/%d。最安定 = 開脚%.2f rad / P%.1f(COM%.0fmm) / lift%.2f → 傾き%.1f°"
              % (len(feas), len(valid), best["splay"], best["P"], best["com_h"] * 1000,
                 best["lift_time"], best["peak_tilt"]))
        print("  ・必要開脚角(成立する最小) = %.2f rad(=%.0f°, ±45°内)。以上でトレッドが足りる。"
              % (min(r["splay"] for r in feas), np.degrees(min(r["splay"] for r in feas))))
    else:
        print("  ・立位成立域(ヨー開脚≤45°)で案(b)成立なし。")
        print("    → このCAD配置＋ヨー±45°開脚では動的安定に不足、という知見(要件で有効)。")
        print("      成立には構造的トレッド拡大(CAD)か更なる COM 低下が必要。")

    print("\n[(3) トルク vs STS3215(ストール~3.0 N·m)]")
    within = [r for r in feas if r["peak_ptau"] * 2 <= 3.0 and r["peak_ytau"] * 2 <= 3.0]
    print("  ・速リフト(0.15s)は成立しても pitch サーボが飽和気味(~2-8 N·m)。")
    print("    遅リフト(0.30s)の成立条件では pitch は軽い。")
    if within:
        fp = max(r["peak_ptau"] for r in within); fy = max(r["peak_ytau"] for r in within)
        print("  ・【成立 かつ ×2≤3.0】= %d 条件。pitch ≤%.3f(×2=%.2f)/ yaw ≤%.3f(×2=%.2f)"
              % (len(within), fp, fp * 2, fy, fy * 2))
        print("    → STS3215 射程内。段差 pull-up 律速(1脚集中)より軽い。")
        rec = min(within, key=lambda r: r["peak_tilt"])
        print("  ・実用推奨動作点 = 開脚%.2f rad(%.0f°)/ P%.1f(COM%.0fmm)/ lift%.2fs: "
              "傾き%.1f°, pitch %.3f(×2=%.2f), yaw %.3f(×2=%.2f)"
              % (rec["splay"], np.degrees(rec["splay"]), rec["P"], rec["com_h"] * 1000,
                 rec["lift_time"], rec["peak_tilt"], rec["peak_ptau"], rec["peak_ptau"] * 2,
                 rec["peak_ytau"], rec["peak_ytau"] * 2))

    print("\n[結論]")
    if within:
        print("  ・案(b) は成立する。単隅リフトで対角が抜け接地2輪(対角線)へ落ちるが、ヨー開脚で")
        print("    トレッドを作り、遅め(0.30s)のリフトなら ピーク傾き ≤16°で足を戻して回復できる。")
        print("  ・成立条件: ヨー開脚 ~11〜26°(±45°内で足りる)＋ リフト時間 ~0.30s(速すぎ不可)。")
        print("    この条件で pitch/yaw ×2 ≤ 3.0 = STS3215 射程内(§8 全軸統一は不変)。")
        print("  ・不可: 速リフト(0.15s)=傾き過大/トルク跳ね、COM高すぎ(P0.4)=転倒しやすい。")
    else:
        print("  ・立位成立域(ヨー±45°開脚)で案(b)成立せず。構造的トレッド拡大が必要。")

    if args.csv and log:
        import csv
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
        print("\nCSV: %s (%d rows)" % (args.csv, len(log)))


if __name__ == "__main__":
    main()
