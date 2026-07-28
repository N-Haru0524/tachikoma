"""
tachikoma / 感度スタディ  ―  脚リンク長 vs 股ピッチ pull-up トルク＋段差成立性

狙い: 軽量化と切り離し、「脚を短くする(=股ピッチの荷重腕を縮める)」という純粋な
幾何だけで、股ピッチ pull-up トルクをどこまで下げられるか。総重量は現実値 1.3kg 固定。
短くするとトルクは下がるが、いつか (i)車体地上高 と (ii)3cm段への足先リーチ が破綻する。
その両立レンジと、STS3215(ストール ~3 N·m @7.4V)射程へ届く最短リンク長を出す。

前提の rig は Phase 1 単脚 (leg_single.xml) と同一構成を、リンク長 L を可変にして再現:
  股ヨー(剛) + 股ピッチ(剛, ★トルク実測★) + 足首受動サス + 脚先ホイール(径3cm固定)。
  全荷重を1脚に集中(=段差越えサイジングの律速, Phase1 と同じ最悪ケース)。

幾何(pitch軸→ホイール接地までの距離): ℓ = L + 0.015(足首) + 0.015(ホイール半径) = L + 0.03
  現行 L=0.10 → ℓ=0.13。 リンク長% は L=0.10 を 100% とする。

メソド:
  ・pull-up トルクは準静的(ramp を長く=3.0s)で測り、起動時の動的過渡ノイズを抑える。
  ・pull-up 初期姿勢は L ごとに幾何で再導出(ホイールを段上面 z=0.03 に置く)。
    L=0.10 で Phase1 の手調整 PULLUP_INIT を厳密に再現することを確認済み。
  ・成立性:
    (i) 地上高  = 直立時の車体底高 = ℓ [m]。3cm を超えるか(段に車体底が干渉しないか)。
    (ii) リーチ = pull-up 初期にホイールが段上面へ実際に載ったか(sim フラグ placed)
                 ＋ 幾何リーチ角が股ピッチ可動範囲 ±1.4rad 内か。
    (iii) 段越え = pull-up で車体が段上へ載り切ったか(sim フラグ climbed)＋登坂後の
                  段上面に対する車体底クリアランス。

使い方:
    uv run python sim/sweep_linklen.py
    uv run python sim/sweep_linklen.py --pcts 100 85 70 55 40
"""
import argparse
import numpy as np
import mujoco

MASS = 1.3               # 総重量[kg] 固定(現実値)。全荷重を1脚に集中。
L0 = 0.10                # 現行リンク長(=100%)[m]
STEP_TOP = 0.03          # 3cm 段の上面 z
STEP_EDGE_X = 0.15       # 段の手前エッジ x
PITCH_ROM = 1.4          # 股ピッチ可動範囲 ±[rad]

# pull-up パラメータ(Phase1 torque_probe と同一の思想)
# 段差越えは「車体を正味 3cm 持ち上げて段上へ載る」動作。短い脚ほど 1 ストロークの
# 上昇量が小さいので、より深いリーチ(股ピッチをより寝かせた開始姿勢)から立て起こす
# 必要がある。各脚長で「正味 3cm 登坂を成立させる最も浅い開始リーチ角」を自動探索し、
# その動作点でのピークトルクを測る(=その脚長で 3cm 段を実際に越えるのに要るトルク)。
# 深いリーチほどトルクは増えるので、これは短脚に不利側=保守側の評価。
PULLUP_TARGET = -0.05    # 引き上げ後(脚をほぼ立てて車体を段上へ)
QS_RAMP_S = 3.0          # 準静的ランプ[s](動的過渡ノイズ抑制)
MOUNT_LIFT = 0.028       # 正味 3cm 登坂の成立判定(足首沈み込み分を見込み 2.8cm)
THI_SHALLOW = -0.80      # 開始リーチ角の探索範囲(浅い側)
THI_DEEP = -1.40         # 同(深い側=股ピッチ ROM 限界 ±1.4)
THI_STEP = 0.05


def ell_of(L):
    """pitch 軸→ホイール接地までの距離[m]。"""
    return L + 0.03


def build_xml(L, mass):
    """リンク長 L・荷重 mass の単脚 rig(leg_single.xml をパラメタ化)。"""
    az = -L                      # 足首 body の z オフセット(=ロッド長)
    return f"""<mujoco model="tachikoma_leg_len">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom friction="1.0 0.02 0.001" density="500" contype="0" conaffinity="0"/>
    <joint damping="0.02" armature="0.003"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"
          friction="1.2 0.02 0.001" contype="1" conaffinity="1"/>
    <geom name="step" type="box" pos="5.15 0 0.015" size="5.0 0.30 0.015"
          friction="1.2 0.02 0.001" contype="1" conaffinity="1"/>
    <body name="carriage" pos="0 0 0.18">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.8"/>
      <joint name="slide_z" type="slide" axis="0 0 1" damping="0.5"/>
      <inertial pos="0 0 0" mass="{mass:.4f}" diaginertia="0.006 0.006 0.004"/>
      <geom name="body_geom" type="box" size="0.10 0.06 0.03" density="0"/>
      <body name="yaw" pos="0 0 -0.03">
        <joint name="hip_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.1"/>
        <inertial pos="0 0 -0.01" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <body name="leg" pos="0 0 0">
          <joint name="hip_pitch" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="0.1"/>
          <geom name="leg_rod" type="capsule" fromto="0 0 0  0 0 {az:.4f}" size="0.008"
                density="400"/>
          <body name="ankle" pos="0 0 {az:.4f}">
            <joint name="ankle_sus" type="slide" axis="0 0 1" range="-0.002 0.020"
                   stiffness="2500" damping="40" springref="0"/>
            <geom name="ankle_geom" type="sphere" size="0.010"/>
            <body name="wheel" pos="0 0 -0.015">
              <joint name="wheel_spin" type="hinge" axis="0 1 0" damping="0.01"/>
              <geom name="wheel_geom" type="cylinder" fromto="0 -0.010 0  0 0.010 0"
                    size="0.015" friction="1.6 0.02 0.001" density="800"
                    contype="1" conaffinity="1"
                    solref="0.005 1" solimp="0.95 0.99 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_yaw_pos" joint="hip_yaw" kp="20" kv="1"
              ctrlrange="-1.57 1.57" forcerange="-4 4"/>
    <position name="hip_pitch_pos" joint="hip_pitch" kp="60" kv="3"
              ctrlrange="-1.4 1.4" forcerange="-20 20"/>
    <velocity name="wheel_drive" joint="wheel_spin" kv="1.5" ctrlrange="-60 60"
              forcerange="-1.5 1.5"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0 0.30 0 0" ctrl="0 0.30 0"/>
  </keyframe>
</mujoco>"""


def make(L, mass=MASS):
    m = mujoco.MjModel.from_xml_string(build_xml(L, mass))
    d = mujoco.MjData(m)
    A = {m.actuator(i).name: i for i in range(m.nu)}
    return m, d, A


def settle(m, d, steps):
    for _ in range(steps):
        mujoco.mj_step(m, d)


def pullup_init_pose(L, theta_init):
    """L・開始リーチ角 theta_init での pull-up 初期姿勢(ホイールを段上面に置く)を幾何で
    導出。theta_init=-0.9 の L=0.10 で Phase1 手調整値を厳密に再現する式系。"""
    ell = ell_of(L)
    th = -theta_init                       # 正の大きさ
    C = 0.058 + ell * np.cos(th)           # carriage 中心 z(足先が段上面 z≈0.03)
    slide_z = C - 0.18
    slide_x = 0.188 - ell * np.sin(th)     # 足先 x≈0.188(段プラトー上)
    return dict(slide_x=slide_x, slide_z=slide_z, pitch=theta_init)


def _run_pullup(L, theta_init, mass, ramp_s):
    """単発 pull-up。開始リーチ角 theta_init から PULLUP_TARGET まで準静的に立て起こす。"""
    m, d, A = make(L, mass)
    qsx = m.joint("slide_x").qposadr[0]
    qsz = m.joint("slide_z").qposadr[0]
    qpi = m.joint("hip_pitch").qposadr[0]
    init = pullup_init_pose(L, theta_init)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[qsx] = init["slide_x"]
    d.qpos[qsz] = init["slide_z"]
    d.qpos[qpi] = init["pitch"]
    d.ctrl[A["hip_yaw_pos"]] = 0.0
    d.ctrl[A["hip_pitch_pos"]] = init["pitch"]
    d.ctrl[A["wheel_drive"]] = 0.0
    mujoco.mj_forward(m, d)
    settle(m, d, 300)
    placed = d.body("wheel").xpos[2] > 0.035          # ホイールが段上面に載ったか
    z0 = d.body("carriage").xpos[2]

    peak = 0.0
    dt = m.opt.timestep
    n = int((ramp_s + 1.0) / dt)                      # 引き上げ + 1s 保持
    for k in range(n):
        t = k * dt
        cmd = min(theta_init + (PULLUP_TARGET - theta_init) * (t / ramp_s), PULLUP_TARGET)
        d.ctrl[A["hip_pitch_pos"]] = cmd
        d.ctrl[A["wheel_drive"]] = 0.0
        mujoco.mj_step(m, d)
        tau = d.actuator("hip_pitch_pos").force[0]
        if abs(tau) > abs(peak):
            peak = tau
    cz = d.body("carriage").xpos[2]
    return {"peak": abs(peak), "placed": placed, "lift": cz - z0, "ride_cz": cz}


def scenario_mount(L, mass=MASS, ramp_s=QS_RAMP_S):
    """その脚長で 3cm 段を実際に越える動作点を探す。
    正味 MOUNT_LIFT 以上の登坂を成立させる「最も浅い開始リーチ角」を探索し(浅い=低トルク)、
    その動作点のピークトルクを返す。ROM(±1.4)内に成立角が無ければ mountable=False。"""
    best = None
    th = THI_SHALLOW
    while th >= THI_DEEP - 1e-9:
        r = _run_pullup(L, round(th, 3), mass, ramp_s)
        if r["placed"] and r["lift"] >= MOUNT_LIFT:
            best = {**r, "theta_init": round(th, 3)}
            break
        th -= THI_STEP
    if best is None:
        # ROM 限界でも 3cm 登坂に届かない(=リーチ破綻)。最深角の結果を参考に返す。
        r = _run_pullup(L, THI_DEEP, mass, ramp_s)
        return {**r, "theta_init": THI_DEEP, "mountable": False}
    best["mountable"] = True
    return best


def classify_motor(sized):
    """×2 サイジング値が入る Feetech 射程。STS3215 ~1.5-3 / STS3250 ~3-4 N·m。"""
    if sized <= 3.0:
        return "STS3215 内"
    if sized <= 4.0:
        return "STS3250 内"
    return "3250 超過"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcts", type=float, nargs="+",
                    default=[100, 85, 70, 55, 40, 30],
                    help="リンク長%(L=0.10 を 100%)")
    ap.add_argument("--sf", type=float, default=2.0)
    ap.add_argument("--sts3215", type=float, default=3.0)
    args = ap.parse_args()

    rows = []
    for pct in args.pcts:
        L = L0 * pct / 100.0
        r = scenario_mount(L)
        sized = r["peak"] * args.sf
        ground_clear = ell_of(L)                       # 直立時の車体底地上高 = ℓ
        rom_head = PITCH_ROM - abs(r["theta_init"])    # 開始リーチ角の ROM 余裕[rad]
        feasible = r["mountable"] and (ground_clear > STEP_TOP)
        rows.append({
            "pct": pct, "L": L, "ell": ell_of(L),
            "peak": r["peak"], "sized": sized, "motor": classify_motor(sized),
            "sts3215": sized <= args.sts3215,
            "theta_init": r["theta_init"], "rom_head": rom_head,
            "ground_clear": ground_clear,
            "ride_clear": (r["ride_cz"] - 0.03) - STEP_TOP,   # 登坂後 車体底 vs 段上面
            "mountable": r["mountable"], "feasible": feasible,
        })

    ln = "=" * 92
    print(ln)
    print("感度表: 脚リンク長 vs 股ピッチ pull-up トルク ＋ 3cm段差成立性  (総重量1.3kg固定)")
    print("  rig=単脚(全荷重1脚集中) / 段差3cm / ホイール径3cm / 準静的ランプ%.1fs / 安全率×%.1f"
          % (QS_RAMP_S, args.sf))
    print("  各脚長で『正味3cm登坂を成立させる最も浅い開始リーチ角』の動作点で実測(短脚に保守側)")
    print(ln)
    print("  リンク長     ℓ     開始リーチ  pull-up   ×%.1f    射程判定    地上高  登坂後底  段越え"
          % args.sf)
    print("  (%%/mm)     [mm]   角/ROM余裕  ピーク[N·m][N·m]  (3215≤%.1f)  [mm]   余裕[mm]  成立"
          % args.sts3215)
    print("  " + "-" * 88)
    for r in rows:
        print("  %3.0f%%/%4.1f  %5.1f   %+.2f/%+.2f   %6.3f   %6.3f  %-9s  %4.0f   %+6.1f    %s"
              % (r["pct"], r["L"]*1000, r["ell"]*1000, r["theta_init"], r["rom_head"],
                 r["peak"], r["sized"], r["motor"], r["ground_clear"]*1000,
                 r["ride_clear"]*1000, "○" if r["mountable"] else "× 不可"))
    print("  " + "-" * 88)
    print("  ※射程判定: STS3215=ストール~3.0 / STS3250=~3-4 N·m @7.4V。")
    print("    段越え成立=地上高>30mm(車体が段に干渉せず)＋開始リーチ角が股ピッチROM±1.4内で")
    print("    正味3cm登坂を達成。ROM余裕がマイナス化する脚長で成立破綻(リーチ不足)。")

    # --- 境界の抽出 ---
    into_3215 = [r for r in rows if r["sts3215"]]
    feasible = [r for r in rows if r["feasible"]]
    print("\n[境界]")
    if into_3215:
        longest_into = max(r["pct"] for r in into_3215)
        print("  ・トルクが STS3215 射程(×%.1f ≤ %.1f N·m)に入る最長リンク長 = %.0f%% (L=%.1fmm)"
              % (args.sf, args.sts3215, longest_into, L0*longest_into/100*1000))
        print("    (これより長いと STS3250 級が要る。=短くするほどトルクが下がり 3215 に届く側)")
    else:
        print("  ・掃引内に STS3215 射程へ入るリンク長なし(全て 3.0 N·m 超過)。")
    if feasible:
        shortest_feasible = min(r["pct"] for r in feasible)
        print("  ・3cm段差成立(地上高>3cm＋ROM内で正味3cm登坂)を保てる最短リンク長 = %.0f%% (L=%.1fmm)"
              % (shortest_feasible, L0*shortest_feasible/100*1000))
        print("    (これより短いと開始リーチ角が股ピッチ ROM±1.4 を超え、段上へ登り切れない)")
    else:
        print("  ・掃引内に段差成立するリンク長なし。")

    # 両立レンジ = STS3215射程内 かつ 成立
    both = [r for r in rows if r["sts3215"] and r["feasible"]]
    print("\n[両立レンジ: STS3215射程 ∩ 3cm段差成立]")
    if both:
        lo = min(r["pct"] for r in both)
        hi = max(r["pct"] for r in both)
        print("  → リンク長 %.0f%%〜%.0f%% (L=%.1f〜%.1fmm) で、軽量化なし・幾何だけで"
              % (lo, hi, L0*lo/100*1000, L0*hi/100*1000))
        print("     股ピッチ STS3215 統一(×%.1f 確保)と 3cm段差成立が両立する。"
              % args.sf)
        print("     ∴ 0.8kg 級の軽量化地獄に頼らず、脚を %.0f〜%.0f%% に詰めるだけで STS3215 統一に届く。"
              % (lo, hi))
        # ROM 余裕を残した実用推奨(帯の長い側=現行から最小変更)
        rec = max(both, key=lambda r: r["pct"])
        print("     実用推奨=帯の最長側 %.0f%% (L=%.1fmm): ×%.1f=%.3f N·m, 地上高%.0fmm, ROM余裕%+.2f。"
              % (rec["pct"], rec["L"]*1000, args.sf, rec["sized"],
                 rec["ground_clear"]*1000, rec["rom_head"]))
        print("     現行からの短縮量が最小で ROM 余裕も最大。ここを起点に詰める。")
    else:
        print("  → 両立するリンク長は掃引内に無し。トルク射程と成立性が排他的な区間のみ。")


if __name__ == "__main__":
    main()
