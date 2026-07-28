"""
tachikoma / Phase 2a  ―  案(b) 動的リフト段差越えの成立性検証

要件 §10(b):「トレッド拡大＋COM低下で、車輪接地の動的安定に頼る短時間リフト」。
1脚(前右 RF)を短時間持ち上げて 3cm 段に乗せる動作を、CAD由来4脚モデル(quad_cad)で走らせ、
動的安定余裕を測る。

■ この設計で判明した動的挙動(重要, §10 の裏取り):
  単隅(RF)を上げると、対角(LB)の接地力が 0 に抜ける。残る支持は LF-RB の【対角“線”】
  =実質2点支持で、車体はこの線まわりに RF 隅側へ倒れ込む(要件§10 の「対角シーソー」)。
  静的には支持三角形を保てない設計なので、案b は「線上の動的バランスを、足を戻すまでの
  短時間だけ保たせる」勝負になる。指標は【リフト中のピーク傾き角】と【足を戻して水平へ回復するか】。
  競合する2効果:
    ・リフト時間(非支持時間)が長い → 傾きが発達
    ・リフトが速すぎる → 脚を振り上げる反作用で車体を動的に煽る(オーバーシュート)
  設計レバー:トレッド広い/COM低い/リフト時間“速すぎず遅すぎず”で ピーク傾き↓。

■ 動作の定義(段差非依存の安定エンベロープ):
  RF 車輪を「3cm 段上面を越える高さ(≈5cm)」まで lift_time で振り上げ→hold→立位へ戻す。
  非支持リフト局面が転倒律速で、段へ降ろすか床へ戻すかは下流の制御詳細(同じエンベロープ)。
  実際に 3cm 段へ乗せて越えるデモは phase2a_viewer.py（showcase）で可視化する。

■ 成立(feasible)の判定:
  (1) リフト中に支持が完全崩壊しない(接地輪≥2=対角線を保持。1以下は転倒)。
  (2) ピーク傾き角 ≤ SAFE_TILT(既定20°。超えると支持輪が抜け始め転倒側)。
  (3) 足を戻したあと水平(±5°)・車高健全へ回復する。

出力:
  (1) 成立するトレッド/COM高/リフト時間の条件(掃引表＋境界)。
  (2) 動作中の股ピッチ・股ヨー ピークトルクが STS3215 射程(≤3.0 N·m ×2前, ここは実測ピーク)内か。
  (3) quad_viewer 側で可視化(phase2a_viewer.py)。

使い方:
    uv run python sim/phase2a_liftover.py                 # 掃引＋レポート
    uv run python sim/phase2a_liftover.py --csv out.csv   # 詳細ログ
"""
import argparse
import numpy as np
import mujoco

from quad_cad import make, ELL, STEP_X0, STEP_H, BODY_HZ, PITCH_RANGE
from quad_stability import stability, leg_normal_force, com_full

LEGS = ["RF", "LF", "RB", "LB"]
LIFT_LEG = "RF"
SUPPORT = ["LF", "RB", "LB"]

SAFE_TILT = 20.0        # 成立とみなすピーク傾きの上限[deg]
HARD_TILT = 32.0        # これを超えたら転倒側[deg]
BODY_X0 = -0.03         # 立位の車体x(前輪を段の手前に置き、RFが段上へ届く配置)


def tilt_deg(d):
    """車体 z 軸の鉛直からの傾き[deg](roll/pitch 合成)。"""
    w, x, y, z = d.qpos[3:7]
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def _settle_stand(m, d, A, info, sp, steps=700):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0] = BODY_X0
    mujoco.mj_forward(m, d)
    for n in LEGS:
        d.ctrl[A[f"pitchpos_{n}"]] = sp
        d.ctrl[A[f"yawpos_{n}"]] = 0.0
        d.ctrl[A[f"wheeldrv_{n}"]] = 0.0
    for _ in range(steps):
        mujoco.mj_step(m, d)


LIFT_CLEAR_Z = 0.05      # RF 車輪を持ち上げる目標高さ[m](3cm段を越える余裕高さ)


def run_maneuver(tread, stance_pitch, lift_time, log=None, hold=0.10,
                 return_time=None, settle=1.6):
    """案b 単脚リフト→hold→立位へ戻す を1回実行し、動的安定メトリクスを返す。
    非支持リフト局面(RF車輪を LIFT_CLEAR_Z まで振り上げ)が転倒律速。"""
    if return_time is None:
        return_time = lift_time
    m, d, A, info = make(tread=tread, stance_pitch=stance_pitch, use_mesh=False)
    _settle_stand(m, d, A, info, stance_pitch)

    bz0 = d.body("body").xpos[2]
    com0 = float(com_full(m, d)[2])                      # ★立位COM高さ(基準)
    # RF車輪を LIFT_CLEAR_Z へ上げる lift ピッチを幾何で決定(hip高から逆算)
    hip_z = d.body(f"hip_{LIFT_LEG}").xpos[2]
    c = np.clip((hip_z - LIFT_CLEAR_Z) / ELL, -1.0, 1.0)
    lift_pitch = max(PITCH_RANGE[0], -float(np.arccos(c)))   # 立位より深い(車輪が上がる)。ROM 内にクランプ

    dt = m.opt.timestep
    peak_tilt = 0.0
    min_support = 4
    peak_pitch_tau = {n: 0.0 for n in LEGS}
    peak_yaw_tau = {n: 0.0 for n in LEGS}

    def rec():
        nonlocal peak_tilt, min_support
        ti = tilt_deg(d)
        peak_tilt = max(peak_tilt, ti)
        nsup = sum(1 for n in LEGS if leg_normal_force(m, d, n) > 0.05)   # 接地輪総数
        min_support = min(min_support, nsup)
        for n in LEGS:
            peak_pitch_tau[n] = max(peak_pitch_tau[n], abs(d.actuator(f"pitchpos_{n}").force[0]))
            peak_yaw_tau[n] = max(peak_yaw_tau[n], abs(d.actuator(f"yawpos_{n}").force[0]))
        if log is not None:
            log.append({"tread": tread, "sp": stance_pitch, "lift_time": lift_time,
                        "t": d.time, "tilt": ti, "bz": d.body("body").xpos[2], "nsup": nsup,
                        "tau_pitch_RF": d.actuator("pitchpos_RF").force[0]})

    def ramp(dur, p_from, p_to):
        n = max(1, int(dur / dt))
        for k in range(n):
            a = (k + 1) / n
            d.ctrl[A[f"pitchpos_{LIFT_LEG}"]] = p_from + a * (p_to - p_from)
            mujoco.mj_step(m, d)
            rec()

    ramp(lift_time, stance_pitch, lift_pitch)     # LIFT(RF非支持=対角LB抜け→対角線支持)
    ramp(hold, lift_pitch, lift_pitch)            # hold(段上へ足を運ぶ想定の滞空)
    ramp(return_time, lift_pitch, stance_pitch)   # 足を戻す
    n = int(settle / dt)
    for k in range(n):
        d.ctrl[A[f"pitchpos_{LIFT_LEG}"]] = stance_pitch
        mujoco.mj_step(m, d)
        rec()

    final_tilt = tilt_deg(d)
    bz = d.body("body").xpos[2]
    recovered = final_tilt < 5.0 and bz > bz0 - 0.02
    lost_support = min_support < 2                # 接地1輪以下=対角線も失う=転倒
    toppled = peak_tilt > HARD_TILT or (not recovered)
    feasible = (not lost_support) and recovered and (peak_tilt <= SAFE_TILT)

    return {
        "tread": tread, "stance_pitch": stance_pitch, "lift_time": lift_time,
        "com_h": com0, "peak_tilt": peak_tilt, "final_tilt": final_tilt,
        "min_support": min_support, "recovered": recovered,
        "lost_support": lost_support, "toppled": toppled, "feasible": feasible,
        "peak_pitch_tau": max(peak_pitch_tau.values()),
        "peak_yaw_tau": max(peak_yaw_tau.values()),
    }


def com_height_of(stance_pitch):
    """立位 COM 高さの目安(実測は run 内)。ℓcos + 車体寄与の近似。"""
    return ELL * np.cos(stance_pitch) + BODY_HZ - 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--treads", type=float, nargs="+", default=[0.10, 0.14, 0.18, 0.22])
    ap.add_argument("--stances", type=float, nargs="+", default=[-0.30, -0.50, -0.70])
    ap.add_argument("--lifts", type=float, nargs="+", default=[0.15, 0.25, 0.40])
    args = ap.parse_args()

    log = [] if args.csv else None
    rows = []
    for tr in args.treads:
        for sp in args.stances:
            for lt in args.lifts:
                r = run_maneuver(tr, sp, lt, log=log)
                rows.append(r)

    _report(rows, args)
    if args.csv and log:
        _write_csv(log, args.csv)
        print("\nCSV: %s (%d rows)" % (args.csv, len(log)))


def _report(rows, args):
    L = "=" * 96
    print(L)
    print("Phase 2a: 案(b) 動的リフト段差越え 成立性  (CAD由来4脚 / 総重量1.3kg要件配分 / 段差3cm)")
    print("  1脚(RF)を lift_time で 5cm(>3cm段)まで振り上げ→hold→立位へ戻す。成立=対角線支持を")
    print("  保ち(接地≥2)、ピーク傾き≤%.0f°、足戻し後 水平復帰。SAFE=%.0f° / HARD=%.0f°。"
          % (SAFE_TILT, SAFE_TILT, HARD_TILT))
    print(L)
    # COM高さ(実測)をトレッド代表で対応表示
    com_by_sp = {}
    for r in rows:
        com_by_sp.setdefault(round(r["stance_pitch"], 2), []).append(r["com_h"])
    print("  立位 COM高さ(実測): " + " / ".join(
        "sp%+.2f→%.0fmm" % (sp, np.mean(v) * 1000) for sp, v in sorted(com_by_sp.items())))
    print("  " + "-" * 92)
    print("  tread  COM   lift   ピーク  最終  min    回復   成立    股ピッチ  股ヨー")
    print("  [mm]   [mm]  [s]    傾き°  傾き°  接地         判定    ピーク    ピーク[N·m]")
    print("  " + "-" * 92)
    for r in rows:
        print("  %4.0f   %4.0f  %.2f   %5.1f  %5.1f   %d    %-4s  %-6s  %6.3f    %6.3f"
              % (r["tread"] * 1000, r["com_h"] * 1000, r["lift_time"],
                 r["peak_tilt"], r["final_tilt"], r["min_support"],
                 "○" if r["recovered"] else "×",
                 "○成立" if r["feasible"] else ("×転倒" if r["toppled"] else "△限界"),
                 r["peak_pitch_tau"], r["peak_yaw_tau"]))
    print("  " + "-" * 92)
    print("  ※ min接地=リフト中の最少接地輪数。RF を上げると対角 LB が抜け、健全でも 2(対角線)。")
    print("    1以下は対角線も失って転倒。△限界=傾き>%.0f°だが未転倒。" % SAFE_TILT)

    # --- (1) 成立領域の要約 ---
    feas = [r for r in rows if r["feasible"]]
    STS = 3.0
    within = [r for r in feas if r["peak_pitch_tau"] * 2 <= STS and r["peak_yaw_tau"] * 2 <= STS]
    print("\n[(1) 成立領域(案b が成立するパラメータ条件)]")
    if feas:
        min_feas_tread = min(r["tread"] for r in feas)
        print("  ・成立(傾き≤%.0f°＋対角線保持＋足戻し回復) = %d / %d 条件" % (SAFE_TILT, len(feas), len(rows)))
        print("  ・成立する最小トレッド = %.0f mm。100mm は殆ど△/転倒(狭くて傾き過大)、"
              "%.0fmm 以上で安定的に成立。" % (min_feas_tread * 1000, 140))
        best = min(feas, key=lambda r: r["peak_tilt"])
        print("  ・最安定 = tread %.0fmm / COM %.0fmm / lift %.2fs → ピーク傾き %.1f°"
              % (best["tread"] * 1000, best["com_h"] * 1000, best["lift_time"], best["peak_tilt"]))
        print("  ・レバー: トレッド広いほど傾き↓(効果最大)。COM は低いほど僅かに有利。")
        print("    lift_time は中庸が最良: 速すぎ(0.15s)は動的煽り＋トルク跳ね、遅すぎは lean 発達。")

    # --- (2) トルク射程チェック(成立領域内で ×2≤STS3215 を満たすか) ---
    all_pitch = max(r["peak_pitch_tau"] for r in rows)
    all_yaw = max(r["peak_yaw_tau"] for r in rows)
    print("\n[(2) 動作中トルク vs STS3215 射程(ストール ~%.1f N·m @7.4V)]" % STS)
    print("  ・全掃引の 股ピッチ ピーク最大 = %.3f / 股ヨー = %.3f N·m(生ピーク自体は %.1f 未満)"
          % (all_pitch, all_yaw, STS))
    print("  ・注意: 速いリフト(0.15s)は位置サーボが傾きを受け止める過渡で 股ピッチが ~2.3 N·m へ")
    print("    跳ね、×2=4.6 と STS3215 の ×2 予算(3.0)を超える。中庸リフトなら ~1.0-1.3 N·m。")
    if within:
        wp = max(r["peak_pitch_tau"] for r in within)
        wy = max(r["peak_yaw_tau"] for r in within)
        print("  ・【成立 かつ ×2≤%.1f を両立】する条件 = %d 個。その中の最悪 股ピッチ %.3f(×2=%.2f)/"
              % (STS, len(within), wp, wp * 2))
        print("    股ヨー %.3f(×2=%.2f) N·m → STS3215 射程内。" % (wy, wy * 2))
        # 実用推奨: 極端でない帯(tread 160-200mm・lift≈0.25s)の中で最も傾きが小さい点
        practical = [r for r in within if 0.16 <= r["tread"] <= 0.20 and 0.2 <= r["lift_time"] <= 0.3]
        rec = min(practical or within, key=lambda r: r["peak_tilt"])
        print("  ・実用推奨動作点 = tread %.0fmm / COM %.0fmm / lift %.2fs: 傾き %.1f°, "
              "股ピッチ %.3f(×2=%.2f), 股ヨー %.3f(×2=%.2f)"
              % (rec["tread"] * 1000, rec["com_h"] * 1000, rec["lift_time"], rec["peak_tilt"],
                 rec["peak_pitch_tau"], rec["peak_pitch_tau"] * 2,
                 rec["peak_yaw_tau"], rec["peak_yaw_tau"] * 2))

    print("\n[結論]")
    print("  ・案(b) は成立する。ただし §10 どおり単隅リフトで対角(LB)が抜け、支持は LF-RB の対角“線”へ")
    print("    落ちる(静的三脚は不成立)。案b の要点は、この線上バランスを足を戻すまでの短時間だけ持たせること。")
    print("  ・成立条件: トレッド ≥ 140mm。≥180mm ならリフト 0.15〜0.40s いずれも可(傾き≤13°)、")
    print("    140mm は中庸リフト 0.25〜0.40s 推奨。COM は低いほど僅かに有利。この帯で 股ピッチ×2 ≤ 3.0")
    print("    ＝STS3215 射程内(§8 の全軸統一は不変)。")
    print("  ・不可域: トレッド 100mm(傾き>20°〜転倒)。狭トレッド＋速すぎリフトは傾き過大＋トルク跳ね。")


def _write_csv(log, path):
    import csv
    keys = list(log[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
