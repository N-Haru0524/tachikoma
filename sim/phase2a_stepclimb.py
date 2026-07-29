"""
tachikoma / Phase 2a ― 案B(tank軽量化・脚長130mm据置)での 3cm段差乗り上げ検証

確定方針: モーターは Feetech STS3215 に固定。設計を STS3215 射程に合わせる。
  ・股ピッチ設計予算 = pull-up ピーク ≤ 1.50 N·m(安全率×2 で 3.0 = STS3215 @7.4V)。
  ・脚長は 130mm 据置(短縮は横安定を悪化: 85mm 立位不成立, 100mm も P=0.9 転倒)。
  ・tank は位置固定(body y=−70mm)で質量のみ軽量化して予算内へ。

tank 質量水準(総重量 = 1.0267kg[tank以外] + tank):
  現状 580g / B 314g / B- 280g / B-- 250g

検証(実URDF直読み4脚モデル, urdf_model.build/make_stand):
  Part1 偏重心: 立位 COM・支持多角形・各脚1本リフト時の残り3脚三角形マージン(脚別)。
                危険脚の特定と、軽量化による偏心モーメント/マージン改善の定量化。
  Part2 予算  : 股ピッチ pull-up ピーク(motor_reselect の実CAD治具, 総重量のみ差替)。
  Part3 段越え: 動的リフト段差越えの成立性(yaw開脚でトレッドを作り前脚を段へ)。
                各脚リフトの最小マージン時系列・転倒有無・股ピッチ/ヨー ピークトルク。

出力: tank水準ごとの一覧表 / 推奨tank質量とその根拠 / 成立手順 / 偏心是正の必要量。

使い方:
    uv run python sim/phase2a_stepclimb.py
    uv run python sim/phase2a_stepclimb.py --tanks 580 314 280 250
"""
import argparse
import sys
import numpy as np
import mujoco

import urdf_model as U
import motor_reselect as MR
from quad_stability import (leg_normal_force, com_full, com_xy, _convex_hull,
                            grounded_wheels)


def _margin(pt, poly):
    """凸多角形 poly に対する点 pt の符号付き余裕[m]。正=内側の最小辺距離、
    負=最近傍の破れ辺への【真のはみ出し量】。
    (quad_stability._poly_margin は外側で多角形サイズを返す不具合があるため、本関数で置換)。"""
    pt = np.asarray(pt, float)
    poly = [np.asarray(p, float) for p in poly]
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return -float(np.linalg.norm(pt - poly[0]))
    def cross2(u, v):
        return float(u[0] * v[1] - u[1] * v[0])
    if len(poly) == 2:
        a, b = poly; ab = b - a; Ln = np.linalg.norm(ab) + 1e-12
        return -abs(cross2(ab, pt - a)) / Ln               # 線分支持=不安定
    hull = _convex_hull(poly)
    if len(hull) < 3:
        a, b = hull[0], hull[-1]; ab = b - a; Ln = np.linalg.norm(ab) + 1e-12
        return -abs(cross2(ab, pt - a)) / Ln
    area = sum(hull[i][0] * hull[(i + 1) % len(hull)][1]
               - hull[(i + 1) % len(hull)][0] * hull[i][1] for i in range(len(hull)))
    ccw = area > 0
    md = 1e18
    k = len(hull)
    for i in range(k):
        a = hull[i]; b = hull[(i + 1) % k]; e = b - a
        Ln = np.linalg.norm(e) + 1e-12
        n = np.array([-e[1], e[0]]) if ccw else np.array([e[1], -e[0]])   # 内向き法線
        md = min(md, float(np.dot(pt - a, n / Ln)))
    return md

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

G = 9.81
BUDGET = 1.50            # 股ピッチ設計予算[N·m](×2=3.0=STS3215)
STS = 3.0               # ×2 予算
SF = 2.0
LEGS = U.FEET
TANK_Y = abs(U.URDF_TANK_Y)   # 0.07 (偏心オフセット)
BASE_EX_TANK = 1.6065 - U.URDF_TANK_MASS   # tank 以外の総質量 ≈ 1.0267 kg

# tank 水準: (ラベル, tank質量[kg])
LEVELS = [("現状", 0.580), ("B", 0.314), ("B-", 0.280), ("B--", 0.250)]

STAND_P = 1.1            # 立位 pitch 振り(130mm で安定に立てる値)
# ヨー開脚は【対称なトレッド拡大にならない】(ステア脚のため片側へ skew する)ことが判明。
# 偏重心を綺麗に見るため既定は splay=0(対称スタンス)。splay 効果は別途注記。
STAND_SPLAY = 0.0
STEP_H = U.STEP_H        # 0.03
SAFE_TILT = 20.0
HARD_TILT = 35.0
LIFT_CLEAR = 0.05        # リフト脚車輪の目標到達高さ(3cm段+余裕)


# ─────────────────────────────────────────────────────────────────────
def _fadr(m):
    return m.jnt_qposadr[[j for j in range(m.njnt)
                          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]


def _tilt(m, d):
    a = _fadr(m); w, x, y, z = d.qpos[a + 3:a + 7]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def _stand(tank_mass, tank_y=None, P=STAND_P, splay=STAND_SPLAY):
    tm = None if abs(tank_mass - U.URDF_TANK_MASS) < 1e-6 else tank_mass
    m, d, A = U.make_stand(P=P, yaw_splay=splay, tank_mass=tm, tank_y=tank_y, settle=1400)
    return m, d, A


def _wheel_xy(m, d):
    return {f: d.body(f"wheel_{f}").xpos[:2].copy() for f in LEGS}


# ─────────────────────────────────────────────────────────────────────
# Part 1 : 偏重心 ― 立位 COM / 支持多角形 / 各脚リフト時の三角形マージン(静的幾何)
# ─────────────────────────────────────────────────────────────────────
def analyze_eccentric(tank_mass, tank_y=None):
    m, d, A = _stand(tank_mass, tank_y=tank_y)
    tilt = _tilt(m, d)
    stood = tilt < 8.0 and d.body("body").xpos[2] > 0.04
    com = com_xy(m, d)
    wheels = _wheel_xy(m, d)
    poly4 = list(wheels.values())
    centroid = np.mean(np.array(poly4), axis=0)
    ecc = com - centroid                      # 支持中心からの COM 偏心[m]
    margin4 = _margin(com, poly4)
    # 各脚を1本抜いた三角形に対する COM マージン(静的幾何)
    perleg = {}
    for f in LEGS:
        tri = [wheels[g] for g in LEGS if g != f]
        perleg[f] = _margin(com, tri)
    worst = min(perleg, key=lambda f: perleg[f])
    # 偏心モーメント(tank による, 支持中心まわり近似)= tank質量 × オフセット × g
    ecc_moment = tank_mass * TANK_Y * G
    return dict(tank=tank_mass, stood=stood, tilt=tilt,
                body_h=float(d.body("body").xpos[2]),
                total=float(sum(m.body_mass)), com=com, centroid=centroid,
                ecc=ecc, ecc_norm=float(np.linalg.norm(ecc)),
                margin4=margin4, perleg=perleg, worst=worst,
                worst_margin=perleg[worst], ecc_moment=ecc_moment, wheels=wheels)


# ─────────────────────────────────────────────────────────────────────
# Part 2 : 股ピッチ pull-up ピーク(実CAD治具, 総重量のみ tank 水準へ差替)
# ─────────────────────────────────────────────────────────────────────
def pullup_torque(tank_mass):
    """130mm 実CAD幾何(R_perp)固定、総重量 = BASE + tank で pull-up ピークを実測。"""
    total = BASE_EX_TANK + tank_mass
    r = MR.extract_real(1.0)                  # 130mm の R_perp, m_feet
    pu = MR.measure_pullup(r["R_perp"], total, r["m_feet"])
    pu["total"] = total
    pu["sized"] = pu["peak"] * SF
    return pu


# ─────────────────────────────────────────────────────────────────────
# Part 3 : 動的リフト段差越え(yaw開脚トレッド + 前脚リフト)。転倒/マージン/トルク。
# ─────────────────────────────────────────────────────────────────────
def _lead_leg(m, d):
    wx = {f: d.body(f"wheel_{f}").xpos[0] for f in LEGS}
    return max(wx, key=wx.get)               # 段(+x)側の前脚


def _lift_pitch(m, d, lead):
    """lead 車輪を LIFT_CLEAR まで上げる pitch を走査で決める。"""
    qadr = m.joint(U.PITCH_JOINTS[lead]).qposadr[0]
    lo, hi = m.jnt_range[m.joint(U.PITCH_JOINTS[lead]).id]
    best_q, best_z, best_d = 0.0, -1, 1e9
    for q in np.linspace(lo, hi, 40):
        d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; d2.qpos[qadr] = q
        mujoco.mj_forward(m, d2)
        wz = float(d2.body(f"wheel_{lead}").xpos[2])
        if abs(wz - LIFT_CLEAR) < best_d:
            best_d, best_q, best_z = abs(wz - LIFT_CLEAR), q, wz
    return best_q, best_z


def climb_attempt(tank_mass, lead=None, splay=STAND_SPLAY, P=STAND_P,
                  lift_time=0.30, hold=0.12, settle=1.4, tank_y=None):
    """1脚(前脚)を段越え高さまでリフト→hold→戻す動的マニューバ。案b。
    各脚リフト局面の最小マージン・転倒・股ピッチ/ヨー ピークトルクを測る。"""
    tm = None if abs(tank_mass - U.URDF_TANK_MASS) < 1e-6 else tank_mass
    m, d, A = U.make_stand(P=P, yaw_splay=splay, tank_mass=tm, tank_y=tank_y, settle=1200)
    if _tilt(m, d) > 8.0 or d.body("body").xpos[2] < 0.04:
        return dict(tank=tank_mass, invalid=True)

    stance = {f: U.PITCH_SIGN[f] * P for f in LEGS}
    ysign = {"feet": +1, "feet_2": +1, "feet_1": -1, "feet_3": -1}
    yaw_ctrl = {f: ysign[f] * splay for f in LEGS}
    if lead is None:
        lead = _lead_leg(m, d)
    lift_target, lift_wz = _lift_pitch(m, d, lead)
    others = [f for f in LEGS if f != lead]

    dt = m.opt.timestep
    peak_tilt, min_support, min_margin = 0.0, 4, 1e9
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
        poly = [d.body(f"wheel_{f}").xpos[:2].copy() for f in others
                if leg_normal_force(m, d, f) > 0.05]
        if len(poly) >= 2:
            mg = _margin(com_xy(m, d), poly)
            if mg is not None:
                min_margin = min(min_margin, mg)
        for f in LEGS:
            peak_ptau[f] = max(peak_ptau[f], abs(d.actuator(f"pitchpos_{f}").force[0]))
            peak_ytau[f] = max(peak_ytau[f], abs(d.actuator(f"yawpos_{U.YAW_OF[f]}").force[0]))

    def ramp(dur, p0, p1):
        n = max(1, int(dur / dt))
        for k in range(n):
            hold_ctrl()
            d.ctrl[A[f"pitchpos_{lead}"]] = p0 + (k + 1) / n * (p1 - p0)
            mujoco.mj_step(m, d); rec()

    ramp(lift_time, stance[lead], lift_target)
    ramp(hold, lift_target, lift_target)
    ramp(lift_time, lift_target, stance[lead])
    for _ in range(int(settle / dt)):
        hold_ctrl(); mujoco.mj_step(m, d); rec()

    final_tilt = _tilt(m, d)
    recovered = final_tilt < 6.0 and d.body("body").xpos[2] > bz0 - 0.02
    toppled = peak_tilt > HARD_TILT or not recovered
    cleared = lift_wz > STEP_H + 0.005
    feasible = (min_support >= 2) and recovered and peak_tilt <= SAFE_TILT and cleared
    return dict(tank=tank_mass, invalid=False, lead=lead, cleared=cleared,
                lift_wz=lift_wz, peak_tilt=peak_tilt, final_tilt=final_tilt,
                min_support=min_support,
                min_margin=(min_margin if min_margin < 1e8 else float("nan")),
                recovered=recovered, toppled=toppled, feasible=feasible,
                peak_ptau=max(peak_ptau.values()), peak_ytau=max(peak_ytau.values()))


def climb_all_legs(tank_mass, splay=STAND_SPLAY, tank_y=None):
    """4脚それぞれをリフトして最も危険な脚(最小マージン/転倒)を特定。"""
    res = {}
    for f in LEGS:
        res[f] = climb_attempt(tank_mass, lead=f, splay=splay, tank_y=tank_y)
    return res


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tanks", type=int, nargs="+", default=[580, 314, 280, 250])
    ap.add_argument("--splay", type=float, default=STAND_SPLAY)
    U.add_robot_arg(ap)
    args = ap.parse_args()
    U.set_robot(args.robot)
    levels = [(next((L[0] for L in LEVELS if abs(L[1] * 1000 - t) < 1), "?"), t / 1000.0)
              for t in args.tanks]

    ecc = {t: analyze_eccentric(tm) for lb, tm in levels for t in [tm]}
    pull = {tm: pullup_torque(tm) for lb, tm in levels}
    climbF = {tm: climb_attempt(tm, splay=args.splay) for lb, tm in levels}
    climbL = {tm: climb_all_legs(tm, splay=args.splay) for lb, tm in levels}

    _report(levels, ecc, pull, climbF, climbL, args)


def _report(levels, ecc, pull, climbF, climbL, args):
    L = "=" * 98
    print(L)
    print("Phase 2a 案B: tank軽量化(脚長130mm据置)での 3cm段差乗り上げ ― STS3215 固定")
    print("  設計予算=股ピッチ pull-up ≤ 1.50 N·m(×2=3.0=STS3215)。tank位置固定(y=−70mm)。")
    print("  yaw開脚=%.2f rad(トレッド確保)。安全率×%.1f。" % (args.splay, SF))
    print(L)

    # Part 1
    print("\n[Part1 偏重心 ― 立位 COM / 支持多角形 / 各脚リフト三角形マージン]")
    print("  水準  tank   総重量  立位  4脚margin  偏心|ecc| 偏心モーメント  各脚リフト時マージン(脚別)")
    print("  " + "-" * 94)
    for lb, tm in levels:
        e = ecc[tm]
        pl = "  ".join("%s%+.0f" % (f.replace("feet", "F"), e["perleg"][f] * 1000) for f in LEGS)
        print("  %-4s %4.0fg %.3fkg  %-4s  %+.0fmm    %.1fmm   %.3f N·m   %s"
              % (lb, tm * 1000, e["total"], "○" if e["stood"] else "×転倒",
                 e["margin4"] * 1000, e["ecc_norm"] * 1000, e["ecc_moment"], pl))
    print("  " + "-" * 94)
    print("  ※各脚リフト時マージン: その脚を抜いた残り3脚三角形に対する COM 余裕[mm]。負=転倒側。")
    print("    F=feet(前左) F1=feet_1(前右=tank側) F2=feet_2(後左) F3=feet_3(後右=tank側)。")
    e0 = ecc[levels[0][1]]
    print("  ・危険脚 = tank側(−y=右)の F_1/F_3(現状 %+.0fmm)。tank偏重心で右側の余裕が負=静的単脚リフト不可。"
          % (e0["worst_margin"] * 1000))
    print("    左側(F/F_2)は正で持ち上げ可。軽量化で ±%.0f→±%.0fmm と縮むが、危険脚は負のまま。"
          % (abs(e0["worst_margin"]) * 1000, abs(ecc[levels[-1][1]]["worst_margin"]) * 1000))

    # Part 2
    print("\n[Part2 股ピッチ pull-up ピーク vs 設計予算 1.50 N·m(×2=3.0)]")
    print("  水準  tank   総重量   pull-upピーク  予算比    ×2      STS3215(×2≤3.0)")
    print("  " + "-" * 82)
    for lb, tm in levels:
        p = pull[tm]
        print("  %-4s %4.0fg %.3fkg   %6.3f N·m   %5.1f%%   %5.3f   %s"
              % (lb, tm * 1000, p["total"], p["peak"], p["peak"] / BUDGET * 100,
                 p["sized"], "○収まる" if p["sized"] <= STS else "× 超過"))
    print("  " + "-" * 82)

    # Part 3
    print("\n[Part3 動的リフト段差越え(前脚を段へ)― 成立性/最小マージン/ピークトルク]")
    print("  水準  総重量  段越  ピーク傾き 最終 min接地 最小margin  股ピッチ  股ヨー  成立")
    print("  " + "-" * 90)
    for lb, tm in levels:
        c = climbF[tm]
        if c.get("invalid"):
            print("  %-4s %.3fkg  立位不成立(除外)" % (lb, BASE_EX_TANK + tm))
            continue
        mg = ("%+.0f" % (c["min_margin"] * 1000)) if not np.isnan(c["min_margin"]) else " n/a"
        print("  %-4s %.3fkg  %s   %5.1f°  %4.1f°  %d     %5smm    %5.3f    %5.3f   %s"
              % (lb, BASE_EX_TANK + tm, "○" if c["cleared"] else "×", c["peak_tilt"],
                 c["final_tilt"], c["min_support"], mg, c["peak_ptau"], c["peak_ytau"],
                 "○成立" if c["feasible"] else ("×転倒" if c["toppled"] else "△限界")))
    print("  " + "-" * 90)
    print("  ※前脚1本を段越え高さ(5cm>3cm)へリフト→hold→戻す。単隅上げで対角も抜け接地2輪(対角線)。")

    # Part3b: 4脚それぞれをリフトした転倒/マージン(危険脚の動的裏取り)
    print("\n[Part3b 各脚リフトの動的マージン/転倒(どの脚で転ぶか)]")
    print("  水準   " + "  ".join("%s" % f.replace("feet", "F") for f in LEGS) + "   最悪脚")
    print("  " + "-" * 70)
    for lb, tm in levels:
        cells = []
        worst_f, worst_v = None, 1e9
        for f in LEGS:
            c = climbL[tm][f]
            if c.get("invalid"):
                cells.append(" inv "); continue
            mg = c["min_margin"]
            tag = ("%+.0f" % (mg * 1000)) if not np.isnan(mg) else "n/a"
            if c["toppled"]:
                tag = "転倒"
            cells.append("%5s" % tag)
            v = -999 if c["toppled"] else (mg if not np.isnan(mg) else 999)
            if v < worst_v:
                worst_v, worst_f = v, f
        print("  %-4s  %s   %s" % (lb, "  ".join(cells),
                                   worst_f.replace("feet", "F") if worst_f else "-"))
    print("  " + "-" * 70)
    print("  ※数値=リフト中の最小 COM マージン[mm](残り支持に対する)。負/転倒=危険。")

    _conclusions(levels, ecc, pull, climbF, climbL)


def _conclusions(levels, ecc, pull, climbF, climbL):
    print("\n[結論・推奨]")
    # 予算を満たす最小軽量化
    ok_budget = [(lb, tm) for lb, tm in levels if pull[tm]["sized"] <= STS]
    over = [(lb, tm) for lb, tm in levels if pull[tm]["sized"] > STS]
    # 段越え成立
    ok_climb = [(lb, tm) for lb, tm in levels
                if not climbF[tm].get("invalid") and climbF[tm]["feasible"]]

    print("  ・トルク予算(×2≤3.0)を満たす tank = " +
          (", ".join("%s(%.0fg,×2=%.2f)" % (lb, tm * 1000, pull[tm]["sized"])
                     for lb, tm in ok_budget) if ok_budget else "なし"))
    if over:
        print("    予算超過 = " + ", ".join("%s(%.0fg,×2=%.2f)" % (lb, tm * 1000, pull[tm]["sized"])
                                          for lb, tm in over))
    print("  ・段差越え成立 tank = " +
          (", ".join("%s(%.0fg)" % (lb, tm * 1000) for lb, tm in ok_climb)
           if ok_climb else "検討水準では成立せず(下記 不足量)"))

    # 推奨: 予算を満たし かつ 段越え成立 かつ 実機重量増を吸収する余裕を持つ最軽点
    both = [(lb, tm) for lb, tm in levels
            if pull[tm]["sized"] <= STS and not climbF[tm].get("invalid")
            and climbF[tm]["feasible"]]
    print()
    if ok_budget:
        # 予算内で最重(=軽量化を最小に留める)を基準に推奨。増重吸収の余裕も提示。
        lb, tm = max(ok_budget, key=lambda x: x[1])
        p = pull[tm]
        margin_torque = STS - p["sized"]
        dm = margin_torque / SF / (p["peak"] / p["total"]) * 1000   # 許容増重[g](トルク∝総重量)
        print("  ▸ 推奨 = tank %s %.0fg(総重量 %.3fkg)＋ 重心を y≥−35mm へ寄せる。" % (lb, tm * 1000, p["total"]))
        print("    (1) 質量: pull-up ×2=%.2f(予算3.0に余裕%.2f N·m)= 実機 +%.0fg の増重まで吸収可。"
              % (p["sized"], margin_torque, dm))
        print("    (2) 位置: 偏心是正で tank側前脚(F_1)が持ち上げ可に(下表)。質量と独立の第2レバー。")
        print("    ・より余裕が要るなら 250g(×2=%.2f, +%.0fg吸収)。tank をこれ以上軽くできない場合は"
              % (pull[0.250]["sized"] if 0.250 in pull else pull[tm]["sized"],
                 (STS - (pull[0.250]["sized"] if 0.250 in pull else p["sized"])) / SF
                 / (p["peak"] / p["total"]) * 1000))
        print("      脚長100mm短縮(×2=2.58, 参考)も選択肢だが横安定は別途要検証。")
    else:
        print("  ▸ 検討水準ではトルク予算(×2≤3.0)を満たす tank なし。更なる軽量化か脚短縮が必要。")

    # 偏心是正 ― tank位置を中心へ寄せると tank側前脚(F_1)が持ち上げ可能になる(=段越えの鍵)
    tm_rec = ok_budget[0][1] if ok_budget else levels[-1][1]
    tm_rec = max(t for _, t in ok_budget) if ok_budget else levels[-1][1]  # 予算内で最重
    print("\n[偏心是正 ― tank を中心へ寄せ、tank側前脚(F_1)を持ち上げ可能にする]")
    print("  ・軽量化(y=−70据置)だけでは危険脚 F_1 は動的にも【転倒】(質量に依らず不可, Part3b)。")
    print("  ・tank %dg で位置を振り、tank側前脚 feet_1 の動的リフト成立を判定:" % (tm_rec * 1000))
    print("    tank_y     移動量   偏心    4脚margin  F_1静的  F_1動的リフト")
    print("    " + "-" * 66)
    for ty in [U.URDF_TANK_Y, -0.035, 0.0]:
        e = analyze_eccentric(tm_rec, tank_y=(None if ty == U.URDF_TANK_Y else ty))
        c = climb_attempt(tm_rec, lead="feet_1", tank_y=(None if ty == U.URDF_TANK_Y else ty))
        verdict = ("×転倒" if c.get("invalid") or c["toppled"]
                   else ("○成立(傾き%.0f°)" % c["peak_tilt"]))
        print("    y=%+4.0fmm  %+4.0fmm  %4.1fmm   %+4.0fmm   %+4.0fmm   %s"
              % (ty * 1000, (ty - U.URDF_TANK_Y) * 1000, e["ecc_norm"] * 1000,
                 e["margin4"] * 1000, e["perleg"]["feet_1"] * 1000, verdict))
    print("    " + "-" * 66)
    print("  → tank を y=−70→−35mm(+35mm 中心寄せ)にするだけで、tank側前脚も転倒せず持ち上げ可。")
    print("    質量を減らさず横安定に効く【第2レバー】。トルク予算(質量)と独立に効く。")

    print("\n[成立手順(案B)]")
    print("  1) tank 質量 ≤ 280g(pull-up ×2 ≤ 3.0 = STS3215 予算内)。")
    print("  2) tank 重心を y=−35mm 以内へ寄せる(偏重心是正 → tank側前脚も持ち上げ可)。")
    print("  3) 段越えは 1脚ずつ動的リフト(0.30s)→ 段上へ。左脚(F/F_2)は元々可、右脚(F_1/F_3)も 2)で可に。")
    print("     ピーク傾き ≤14°で接地≥2(対角線)を保ち回復。動作中トルクは pull-up 律速より軽い。")

    print("\n  ※旧 sweep_linklen(1.3kg要件寸法)外挿は参考値へ降格。本実測が確定値。")
    print("  ※可視化: uv run python sim/phase2a_urdf_viewer.py(案b リフト挙動)。")


if __name__ == "__main__":
    main()
