"""
tachikoma / モーター再選定 ― 実CAD(URDF実質量1.606kg)での股ピッチトルク実測

背景(2026-07, 前提更新):
  ・Onshape 焼き付き解消 → pitch 可動域は全脚 ±80° 対称(URDF limit と一致)。
  ・PLA 充填で URDF 実質量が入った。総重量 1.6065kg(body0.484/tank0.580/feet0.131×4/
    joint0.005×4)。これが「総重量」(電装込み代用)。1.3kg 上書きは【廃止】= 実値を使う。

この実測が【確定値】。旧 sweep_linklen(1.3kg・要件寸法)の外挿は参考値へ降格。

── 2つの測定 ──────────────────────────────────────────────────────────
(A) 立位 姿勢保持スイープ  : 実URDF直読み4脚(urdf_model.make_stand)で pitch を振り、
    各姿勢の股ピッチ保持トルク(4脚最大)と車体高を実測。荷重は4脚に分散(=軽い)。

(B) 3cm段差 pull-up (律速) : Phase 1 と同じ「段上にアンカー→車体を引き上げる」単脚治具。
    ただし数値は全て実CAD由来 ― 総重量 1.6065kg・面内リーチ R_perp(股ピッチ軸への
    垂直距離=実測)・feet 実質量 ― を注入。全荷重を1脚に集中(保守側=段差サイジング律速)。
    ・治具構造は Phase 1(sweep_linklen)の検証済み rig と同一。質量配分・寸法だけ実CAD値。
    ・各脚長で「正味3cm登坂を成立させる最も浅い開始リーチ角」の動作点でピークトルクを測る
      (短脚に不利=保守側)。

── 脚長感度 ───────────────────────────────────────────────────────────
  脚を短くすると feet 質量も減る(体積∝長さ=線形)。総重量もその分下がる。
    130mm(leg_scale1.00, 総1.606kg) / 100mm(0.72, 1.460kg) / 85mm(0.59, 1.392kg)

出力:
  (A)(B) 実測値と脚長感度の一覧表 / STS3215(~3.0)・STS3250(3〜4) に収まる条件 /
  収まらない場合の必要トルクと必要短縮量の逆算。

使い方:
    uv run python sim/motor_reselect.py
    uv run python sim/motor_reselect.py --mm 130 100 85 --Ps 0.6 0.9 1.2
"""
import argparse
import sys
import numpy as np
import mujoco

import urdf_model as U

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp932 コンソールでも「·」等で落ちない
except Exception:
    pass

G = 9.81
WHEEL_R = U.WHEEL_R                       # 0.015
ANKLE_OFF = 0.015                         # 足首オフセット(rig)
STEP_H = U.STEP_H                         # 0.03
SF = 2.0                                  # サイジング安全率 ×2(既定, §8 踏襲)
STS3215 = 3.0                             # ストール ~3.0 N·m @7.4V
STS3250 = 4.0                             # ~3〜4 N·m @7.4V

# 検討する脚長[mm] → leg_scale。130mm=実CAD現状(scale1.0)。
#   leg_scale は「pitch軸→接地」の reach を線形短縮し、feet 質量も同率で縮める。
LEG_MM = {130: 1.00, 100: 0.72, 85: 0.59}


# ─────────────────────────────────────────────────────────────────────
# 実CAD値の抽出(URDF直読みモデルから、脚長ごとに実測)
# ─────────────────────────────────────────────────────────────────────
def extract_real(leg_scale):
    """実URDF4脚モデルから、その脚長での確定パラメータを抽出。
      total    : 総質量[kg](URDF実値, 車輪/足首は feet から切出済で総重量不変)
      m_feet   : feet 1本の質量[kg]
      R_perp   : 股ピッチ軸線への接地点の垂直距離[m] = pitch トルクの真の面内リーチ
      arm_max  : ±80° ROM 内で到達できる最大水平モーメントアーム[m]
    """
    m, d = U.build(leg_scale=leg_scale)
    jid = m.joint(U.PITCH_JOINTS["feet"]).id
    qadr = m.joint(U.PITCH_JOINTS["feet"]).qposadr[0]
    total = float(sum(m.body_mass))
    m_feet = float(m.body("feet").mass[0] + m.body("wheel_feet").mass[0]
                   + m.body("ankle_feet").mass[0])   # 切出分を戻した実 feet 質量
    d0 = mujoco.MjData(m); d0.qpos[qadr] = 0.0; mujoco.mj_forward(m, d0)
    anc = d0.xanchor[jid].copy(); ax = d0.xaxis[jid].copy(); ax /= np.linalg.norm(ax)
    wc = d0.body("wheel_feet").xpos.copy(); wc[2] -= WHEEL_R
    v = wc - anc
    R_perp = float(np.linalg.norm(v - np.dot(v, ax) * ax))
    arm_max = 0.0
    for th in np.linspace(-U.PITCH_ROM, U.PITCH_ROM, 401):
        dd = mujoco.MjData(m); dd.qpos[qadr] = th; mujoco.mj_forward(m, dd)
        a = dd.xanchor[jid]; w = dd.body("wheel_feet").xpos.copy(); w[2] -= WHEEL_R
        vv = w - a; vv = vv - np.dot(vv, ax) * ax
        arm_max = max(arm_max, float(np.hypot(vv[0], vv[1])))
    return dict(total=total, m_feet=m_feet, R_perp=R_perp, arm_max=arm_max)


# ─────────────────────────────────────────────────────────────────────
# (A) 立位 姿勢保持スイープ ― 実URDF直読み4脚モデル(荷重は4脚に分散)
# ─────────────────────────────────────────────────────────────────────
def measure_stand(leg_scale, P):
    """立位(pitch振り P)で静定させ、股ピッチ保持トルク(4脚最大)と車体高を実測。
    return None は立位不成立(転倒/沈み込み)。"""
    m, d, A = U.make_stand(P=P, yaw_splay=0.0, leg_scale=leg_scale, settle=1200)
    fadr = m.jnt_qposadr[[j for j in range(m.njnt)
                          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
    w, x, y, z = d.qpos[fadr + 3:fadr + 7]
    tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
    body_h = float(d.body("body").xpos[2])
    if tilt > 8.0 or body_h < 0.04:
        return dict(P=P, invalid=True, tilt=tilt, body_h=body_h)
    hold = max(abs(float(d.actuator(f"pitchpos_{f}").force[0])) for f in U.FEET)
    return dict(P=P, invalid=False, tilt=tilt, body_h=body_h, hold=hold)


# ─────────────────────────────────────────────────────────────────────
# (B) 3cm段差 pull-up ― Phase 1 と同一の単脚アンカー治具(数値は実CAD)
# ─────────────────────────────────────────────────────────────────────
PULLUP_TARGET = -0.05    # 引き上げ後(脚をほぼ立てて車体を段上へ)
QS_RAMP_S = 3.0          # 準静的ランプ[s](動的過渡ノイズ抑制)
MOUNT_LIFT = 0.028       # 正味3cm登坂の成立判定(足首沈み込み見込み2.8cm)
THI_SHALLOW = -0.80      # 開始リーチ角 探索範囲(浅い側)
THI_DEEP = -U.PITCH_ROM  # 同(深い側=股ピッチ ROM 限界 ±80°)
THI_STEP = 0.05


def _rig_xml(ell, mass_carriage, mass_leg):
    """Phase 1 単脚 pull-up 治具(carriage=全荷重, 剛な股ヨー/ピッチ, 足首受動サス, 径3cm輪)。
      ell        : 股ピッチ軸→ホイール接地 の面内リーチ[m](=実CAD R_perp)
      mass_carriage : 車体側集中質量[kg](=総質量 − feet1本)
      mass_leg   : 脚(feet)実質量[kg]。rod 中程に付与。
    治具構造は sweep_linklen(検証済)と同一。ℓ・質量のみ実CAD値へ差し替え。"""
    L = ell - (ANKLE_OFF + WHEEL_R)      # rod 長(ℓ = L + 0.03)
    az = -L
    return f"""<mujoco model="tachikoma_pullup_realcad">
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
      <inertial pos="0 0 0" mass="{mass_carriage:.4f}" diaginertia="0.006 0.006 0.004"/>
      <geom name="body_geom" type="box" size="0.09 0.08 0.025" density="0"/>
      <body name="yaw" pos="0 0 -0.02">
        <joint name="hip_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.1"/>
        <inertial pos="0 0 -0.008" mass="0.005" diaginertia="1e-6 1e-6 1e-6"/>
        <body name="leg" pos="0 0 0">
          <joint name="hip_pitch" type="hinge" axis="0 1 0" range="{-U.PITCH_ROM} {U.PITCH_ROM}" damping="0.1"/>
          <geom name="leg_rod" type="capsule" fromto="0 0 0  0 0 {az:.4f}" size="0.008" density="0"/>
          <inertial pos="0 0 {az/2:.4f}" mass="{mass_leg:.4f}" diaginertia="1.5e-4 1.5e-4 3e-5"/>
          <body name="ankle" pos="0 0 {az:.4f}">
            <joint name="ankle_sus" type="slide" axis="0 0 1" range="-0.002 0.020"
                   stiffness="2500" damping="40" springref="0"/>
            <geom name="ankle_geom" type="sphere" size="0.010"/>
            <body name="wheel" pos="0 0 {-ANKLE_OFF:.4f}">
              <joint name="wheel_spin" type="hinge" axis="0 1 0" damping="0.01"/>
              <geom name="wheel_geom" type="cylinder" fromto="0 -0.010 0  0 0.010 0"
                    size="{WHEEL_R:.4f}" friction="1.6 0.02 0.001" density="800"
                    contype="1" conaffinity="1" solref="0.005 1" solimp="0.95 0.99 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_yaw_pos" joint="hip_yaw" kp="20" kv="1" ctrlrange="-1.57 1.57" forcerange="-4 4"/>
    <position name="hip_pitch_pos" joint="hip_pitch" kp="80" kv="4"
              ctrlrange="{-U.PITCH_ROM} {U.PITCH_ROM}" forcerange="-30 30"/>
    <velocity name="wheel_drive" joint="wheel_spin" kv="1.5" ctrlrange="-60 60" forcerange="-1.5 1.5"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0 0.30 0 0" ctrl="0 0.30 0"/>
  </keyframe>
</mujoco>"""


def _make_rig(ell, mass_carriage, mass_leg):
    m = mujoco.MjModel.from_xml_string(_rig_xml(ell, mass_carriage, mass_leg))
    d = mujoco.MjData(m)
    A = {m.actuator(i).name: i for i in range(m.nu)}
    return m, d, A


def _pullup_init_pose(ell, theta_init):
    """開始リーチ角 theta_init で車輪を段上面(z≈0.03)に置く carriage 初期姿勢を幾何導出
    (sweep_linklen と同一式系。ℓ=実CAD 面内リーチ)。"""
    th = -theta_init
    C = 0.058 + ell * np.cos(th)           # carriage 中心 z
    return dict(slide_x=0.188 - ell * np.sin(th), slide_z=C - 0.18, pitch=theta_init)


def _run_pullup(ell, mass_carriage, mass_leg, theta_init):
    """単発 pull-up を【準静的】に実施しピークトルクを測る。
    theta_init(最深リーチ=最大アーム)から PULLUP_TARGET まで pitch を小刻みに指令し、
    各角で静定させてから【定常】トルクを採る。→ サーボ係合の1〜2ステップ過渡スパイクを
    排し、実際の保持トルク(=W×水平アーム+脚自重項+登坂摩擦項)だけを測る。"""
    m, d, A = _make_rig(ell, mass_carriage, mass_leg)
    qsx = m.joint("slide_x").qposadr[0]; qsz = m.joint("slide_z").qposadr[0]
    qpi = m.joint("hip_pitch").qposadr[0]
    init = _pullup_init_pose(ell, theta_init)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[qsx] = init["slide_x"]; d.qpos[qsz] = init["slide_z"]; d.qpos[qpi] = init["pitch"]
    d.ctrl[A["hip_yaw_pos"]] = 0.0; d.ctrl[A["hip_pitch_pos"]] = init["pitch"]
    d.ctrl[A["wheel_drive"]] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(400):                     # 初期姿勢で静定(過渡を減衰)
        mujoco.mj_step(m, d)
    placed = d.body("wheel").xpos[2] > 0.035
    z0 = d.body("carriage").xpos[2]

    peak = 0.0
    for cmd in np.arange(theta_init, PULLUP_TARGET + 1e-6, 0.03):  # 最深→立位へ準静的に
        d.ctrl[A["hip_pitch_pos"]] = float(cmd)
        d.ctrl[A["wheel_drive"]] = 0.0
        for _ in range(120):                 # この角で静定
            mujoco.mj_step(m, d)
        taus = []
        for _ in range(30):                  # 定常トルクを平均(過渡除去)
            mujoco.mj_step(m, d)
            taus.append(abs(float(d.actuator("hip_pitch_pos").force[0])))
        peak = max(peak, float(np.mean(taus)))
    cz = d.body("carriage").xpos[2]
    return dict(peak=peak, placed=placed, lift=cz - z0)


def measure_pullup(ell, total, m_feet):
    """その脚長で 3cm 段を越える動作点(正味3cm登坂を成立させる最も浅い開始リーチ角)を
    探し、そのピークトルクを返す。ROM 内で成立しなければ mountable=False(最深角の値)。"""
    mass_carriage = total - m_feet
    th = THI_SHALLOW
    while th >= THI_DEEP - 1e-9:
        r = _run_pullup(ell, mass_carriage, m_feet, round(th, 3))
        if r["placed"] and r["lift"] >= MOUNT_LIFT:
            return dict(**r, theta_init=round(th, 3), mountable=True)
        th -= THI_STEP
    r = _run_pullup(ell, mass_carriage, m_feet, THI_DEEP)
    return dict(**r, theta_init=THI_DEEP, mountable=False)


def classify(sized):
    if sized <= STS3215:
        return "STS3215 内"
    if sized <= STS3250:
        return "STS3250 内"
    return "3250 超過"


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", type=int, nargs="+", default=[130, 100, 85])
    ap.add_argument("--Ps", type=float, nargs="+", default=[0.9, 1.1, 1.3])
    args = ap.parse_args()

    scales = {mm: LEG_MM.get(mm) for mm in args.mm}
    for mm, s in scales.items():
        if s is None:   # 未登録脚長は reach 比から scale 概算(130mm 基準)
            scales[mm] = (mm / 1000 - (ANKLE_OFF + WHEEL_R)) / (0.1243 - (ANKLE_OFF + WHEEL_R)) * 1.0

    real = {mm: extract_real(scales[mm]) for mm in args.mm}

    # ---- (A) 立位保持スイープ ----
    stand = {mm: [measure_stand(scales[mm], P) for P in args.Ps] for mm in args.mm}

    # ---- (B) pull-up ----
    pull = {}
    for mm in args.mm:
        r = real[mm]
        pu = measure_pullup(r["R_perp"], r["total"], r["m_feet"])
        pu["sized"] = pu["peak"] * SF
        pu["motor"] = classify(pu["sized"])
        pull[mm] = pu

    _report(args, scales, real, stand, pull)


def _report(args, scales, real, stand, pull):
    L = "=" * 96
    print(L)
    print("モーター再選定: 実CAD(URDF実質量)での股ピッチトルク実測  ―  確定値")
    print("  総重量=URDF実値(1.6065kg@130mm, 電装込み代用)。pitch±80°対称。安全率×%.1f。" % SF)
    print("  (A) 立位=実URDF直読み4脚(荷重4脚分散)。(B) pull-up=Phase1単脚アンカー治具に実CAD値注入。")
    print(L)

    # 実CAD抽出値
    print("\n[実CAD抽出値(脚長ごと)]")
    print("  脚長  leg_scale  総質量   feet/本  面内リーチR_perp  ±80°内 最大アーム")
    print("  " + "-" * 74)
    for mm in args.mm:
        r = real[mm]
        print("  %3dmm  %.3f    %.3fkg  %.3fkg    %.4f m        %.4f m"
              % (mm, scales[mm], r["total"], r["m_feet"], r["R_perp"], r["arm_max"]))

    # (A) 立位保持
    print("\n[(A) 立位 姿勢保持スイープ ― 股ピッチ保持トルク(4脚最大)]")
    print("  脚長   pitch振りP   車体高    傾き    股ピッチ保持   保持×2")
    print("  " + "-" * 66)
    for mm in args.mm:
        for s in stand[mm]:
            if s["invalid"]:
                print("  %3dmm   P=%.1f       %4.0fmm   %4.1f°   立位不成立(除外)"
                      % (mm, s["P"], s["body_h"] * 1000, s["tilt"]))
            else:
                print("  %3dmm   P=%.1f       %4.0fmm   %4.1f°   %6.3f N·m   %5.2f"
                      % (mm, s["P"], s["body_h"] * 1000, s["tilt"], s["hold"], s["hold"] * SF))
    print("  ※立位は4脚で荷重分散するため保持トルク=0.4〜0.55 N·m(×2でも~1.1)。pull-up 律速の 1/3〜1/4。")
    print("  ※85mm は yaw=0 静的立位が不成立(短脚で横方向の支持幅が狭く、tank偏重心で横転)。")
    print("    保持トルク自体は軽い(質量減で更に小)ので選定には無影響。段差 pull-up が律速。")

    # (B) pull-up
    print("\n[(B) 3cm段差 pull-up ― 全荷重1脚集中(段差サイジング律速)]")
    print("  脚長   総質量   開始リーチ角  段越  pull-upピーク  ×%.1f サイジング  射程判定" % SF)
    print("  " + "-" * 78)
    for mm in args.mm:
        p = pull[mm]
        print("  %3dmm  %.3fkg   %+.2f rad    %s   %6.3f N·m     %6.3f N·m     %s"
              % (mm, real[mm]["total"], p["theta_init"],
                 "○" if p["mountable"] else "×", p["peak"], p["sized"], p["motor"]))
    print("  " + "-" * 78)
    print("  ※段越○=ROM±80°内で正味3cm登坂を成立。×=リーチ不足(ROM 限界でも届かず)。")

    # ---- 総合一覧 ----
    print("\n[脚長感度 一覧(pull-up 律速)]")
    print("  脚長   総重量    pull-upピーク   ×2 サイジング   STS3215(≤3.0)   STS3250(≤4.0)")
    print("  " + "-" * 80)
    for mm in args.mm:
        p = pull[mm]
        in15 = p["sized"] <= STS3215
        in50 = p["sized"] <= STS3250
        print("  %3dmm  %.3fkg    %6.3f N·m    %6.3f N·m     %-8s      %-8s"
              % (mm, real[mm]["total"], p["peak"], p["sized"],
                 "○射程内" if in15 else "× 超過", "○射程内" if in50 else "× 超過"))

    # ---- 結論・条件・逆算 ----
    print("\n[結論 ― STS3215 / STS3250 に収まる条件]")
    in15 = [mm for mm in args.mm if pull[mm]["sized"] <= STS3215 and pull[mm]["mountable"]]
    in50 = [mm for mm in args.mm if pull[mm]["sized"] <= STS3250 and pull[mm]["mountable"]]
    if in15:
        mm = max(in15)
        print("  ・STS3215(×2≤3.0) 射程 = 脚長 %dmm 以下(総重量 %.3fkg, ×2=%.2f N·m)。"
              % (mm, real[mm]["total"], pull[mm]["sized"]))
    else:
        print("  ・STS3215(×2≤3.0): 検討脚長すべてで超過(下記 逆算参照)。")
    if in50:
        mm = max(in50)
        print("  ・STS3250(×2≤4.0) 射程 = 脚長 %dmm 以下(総重量 %.3fkg, ×2=%.2f N·m)。"
              % (mm, real[mm]["total"], pull[mm]["sized"]))
    else:
        print("  ・STS3250(×2≤4.0): 検討脚長すべてで超過(下記 逆算参照)。")
    # 推奨(現状130mm を基準に)
    p130 = pull.get(130)
    if p130:
        if p130["sized"] <= STS3215:
            print("  ▸ 推奨: 現状130mm のまま STS3215 で全12軸統一(§8)可。")
        elif p130["sized"] <= STS3250:
            best15 = max(in15) if in15 else None
            print("  ▸ 推奨: 現状130mm なら【STS3250】採用が確実(×2=%.2f, 射程内)。" % p130["sized"])
            if best15:
                print("         STS3215 で統一したいなら 脚を %dmm へ短縮(×2=%.2f)。"
                      % (best15, pull[best15]["sized"]))
        else:
            print("  ▸ 推奨: 130mm では STS3250 も超過。脚短縮 or 軽量化が必須(下記 逆算)。")

    # 現状(130mm)の逆算 ― STS3215 に収める必要条件、および STS3250 余裕
    p130 = pull.get(130)
    if p130:
        r130 = real[130]
        need = STS3215 / SF        # STS3215 に収める必要ピーク[N·m](×2=3.0)
        print("\n[逆算 ― 現状130mm(総重量%.3fkg) を各サーボに収める条件]" % r130["total"])
        print("  ・現状130mm 実測: pull-up ピーク %.3f / ×2 = %.3f N·m。"
              % (p130["peak"], p130["sized"]))
        # トルク ∝ 総重量 W かつ ∝ 水平アーム(≈脚長)。2レバー(短縮/軽量化)で逆算。
        if p130["sized"] <= STS3250:
            print("  ・STS3250(×2≤4.0): 130mm 現状で【射程内】(余裕 %.2f N·m 相当 = 総重量 +%dg まで可)。"
                  % (STS3250 - p130["sized"],
                     round((r130["total"] * (STS3250 / SF) / p130["peak"] - r130["total"]) * 1000)))
        else:
            m50 = r130["total"] * (STS3250 / SF) / p130["peak"]
            print("  ・STS3250(×2≤4.0): 130mm 据置なら 総重量 %.3fkg 以下(現状比 %dg 減)が必要。"
                  % (m50, round((r130["total"] - m50) * 1000)))
        if p130["sized"] <= STS3215:
            print("  ・STS3215(×2≤3.0): 130mm 現状で射程内。")
        else:
            print("  ・STS3215(×2≤3.0) に収めるには pull-up ピーク %.3f N·m 以下が必要"
                  "(超過分 %.2f N·m)。二択:" % (need, p130["sized"] - STS3215))
            # (i) 脚短縮のみ(質量も脚長で連動して減る=感度表から)
            ok_mm = [mm for mm in args.mm
                     if mm < 130 and pull[mm]["sized"] <= STS3215 and pull[mm]["mountable"]]
            if ok_mm:
                mm = max(ok_mm)
                print("     (i) 脚短縮のみ  : 脚長 %dmm(総重量%.3fkg, ×2=%.2f)。現状から %dmm 短縮。"
                      % (mm, real[mm]["total"], pull[mm]["sized"], 130 - mm))
            # (ii) 軽量化のみ(脚長130mm据置, トルク∝総重量)
            m_target = r130["total"] * need / p130["peak"]
            print("     (ii)軽量化のみ  : 脚長130mm据置なら 総重量 %.3fkg 以下"
                  "(現状 %.3fkg から %dg 減)。" % (m_target, r130["total"],
                                              round((r130["total"] - m_target) * 1000)))

    print("\n  ※本実測が確定値。旧 sweep_linklen(1.3kg・要件寸法)の外挿は参考値へ降格。")
    print("  ※Phase 2a(動的リフト段差越え)は本選定が確定してから着手。")


if __name__ == "__main__":
    main()
