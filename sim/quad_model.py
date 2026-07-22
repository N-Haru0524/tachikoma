"""
tachikoma / Phase 2  ―  4脚フルモデル (自由浮遊ベース + 4本脚) のジェネレータ

Phase 1 の脚1本 rig (leg_single.xml) を 4脚全体へ拡張する。要件 §3,§4,§5,§6 準拠。

Phase 1 との本質的な違い:
  - Phase 1 は carriage を slide_x + slide_z の 2 自由度に拘束していた
    (「車体ピッチ回転は他3脚が支える前提」で拘束)。
  - Phase 2 はその前提こそを検証対象にするため、車体を <freejoint> の
    自由浮遊ベース(6自由度)にする。3脚で支えながら1脚を振り出したときに
    実際に転ばないか(重心が支持三角形に留まるか)を物理で確かめる。

構成 (要件 §5):
  脚1本 = 股ヨー(能動 position) + 股ピッチ(能動 position) + 足首サス(受動 slide ばね)
           + 脚先ホイール(能動 velocity)。 これを ×4 (矩形配置 前後2×左右2)。
  能動 12 軸 (yaw4 + pitch4 + wheel4) / 受動 4 (ankle4)。

座標系: +x=前, +y=左, +z=上。 前方リーチ = 負 pitch (Phase 1 と同符号)。
脚の並び: RF(右前) LF(左前) RB(右後) LB(左後)。 右 = -y。

物理 (要件 §6): 総重量 1.3kg(目標) / 段差 3cm / ホイール径 3cm(半径0.015)。
"""
import mujoco
import numpy as np

# ---- 幾何パラメータ (すべてここに集約) --------------------------------------
BODY_HX, BODY_HY, BODY_HZ = 0.12, 0.08, 0.025   # 車体 半寸法 (体長0.24×幅0.16×厚0.05)
HIP_X, HIP_Y = 0.10, 0.07                        # 股関節の車体からのオフセット
ROD_LEN = 0.10                                   # 棒脚の長さ
WHEEL_R = 0.015                                  # ホイール半径 (径3cm固定)
ANKLE_R = 0.010                                  # 足首球の半径

# 脚: (名前, x符号, y符号)。 y: 左=+1, 右=-1
LEGS = [
    ("RF", +1, -1),   # 右前
    ("LF", +1, +1),   # 左前
    ("RB", -1, -1),   # 右後
    ("LB", -1, +1),   # 左後
]

# 標準立位の脚先高さ: hip から wheel 底面まで = ROD_LEN + WHEEL_R*2 (pitch=0 時)
STAND_HIP_Z = ROD_LEN + WHEEL_R * 2              # = 0.13
# 車体中心の標準高さ (股は車体底 -BODY_HZ に付く)
STAND_BODY_Z = STAND_HIP_Z + BODY_HZ            # = 0.155

STEP_H = 0.03                                    # 段差高さ 3cm
STEP_X0 = 0.15                                   # 段差の手前側エッジ x 座標


def _leg_xml(name, sx, sy, stand_pitch, ankle_stiff, ankle_damp, wheel_solref):
    """脚1本ぶんの body ツリーを生成。 sx,sy は前後/左右の符号。"""
    hx = HIP_X * sx
    hy = HIP_Y * sy
    return f"""
      <body name="hip_{name}" pos="{hx:.4f} {hy:.4f} {-BODY_HZ:.4f}">
        <!-- (1) 股ヨー: 能動・剛 (操舵/脚を向ける) -->
        <joint name="yaw_{name}" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.1"/>
        <inertial pos="0 0 -0.01" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="hipg_{name}" type="box" size="0.012 0.012 0.010" rgba="0.4 0.42 0.48 1" density="0"/>

        <!-- (2) 股ピッチ: 能動・剛 (主軸/重量支持/段差越え). ★トルク実測対象★ -->
        <body name="leg_{name}" pos="0 0 0">
          <joint name="pitch_{name}" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="0.1"/>
          <geom name="rod_{name}" type="capsule" fromto="0 0 0  0 0 {-ROD_LEN:.4f}" size="0.008"
                rgba="0.7 0.72 0.78 1" density="400"/>

          <!-- (3) 足首ショックアブソーバー: 受動 slide ばね (脚軸方向) -->
          <body name="ankle_{name}" pos="0 0 {-ROD_LEN:.4f}">
            <joint name="ankle_{name}" type="slide" axis="0 0 1" range="-0.002 0.020"
                   stiffness="{ankle_stiff:.0f}" damping="{ankle_damp:.0f}" springref="0"/>
            <geom name="ankleg_{name}" type="sphere" size="{ANKLE_R:.4f}" rgba="0.5 0.5 0.55 1"/>

            <!-- (4) 脚先ホイール: 能動駆動 (径3cm=半径0.015固定) -->
            <body name="wheel_{name}" pos="0 0 {-WHEEL_R:.4f}">
              <joint name="wheel_{name}" type="hinge" axis="0 1 0" damping="0.01"/>
              <geom name="wheelg_{name}" type="cylinder" fromto="0 -0.010 0  0 0.010 0"
                    size="{WHEEL_R:.4f}" rgba="0.15 0.15 0.18 1"
                    friction="1.6 0.02 0.001" density="800"
                    contype="1" conaffinity="1"
                    solref="{wheel_solref}" solimp="0.9 0.98 0.001"/>
            </body>
          </body>
        </body>
      </body>"""


def _actuator_xml(name):
    return f"""
    <position name="yawpos_{name}" joint="yaw_{name}" kp="20" kv="1"
              ctrlrange="-1.57 1.57" forcerange="-4 4"/>
    <position name="pitchpos_{name}" joint="pitch_{name}" kp="60" kv="3"
              ctrlrange="-1.4 1.4" forcerange="-20 20"/>
    <velocity name="wheeldrv_{name}" joint="wheel_{name}" kv="1.5"
              ctrlrange="-60 60" forcerange="-1.5 1.5"/>"""


def build_xml(body_mass=1.05, stand_pitch=0.0, step_x0=STEP_X0, step_h=STEP_H,
              ankle_stiff=2500, ankle_damp=40, wheel_solref="0.02 1"):
    """4脚フルモデルの MJCF 文字列を生成する。

    body_mass: 車体本体の質量[kg]。脚部品込みで総重量 ~1.3kg になるよう既定調整。
    stand_pitch: 標準立位の股ピッチ角(0=脚を真下、正で後傾)。
    ankle_damp: 足首サスの減衰(受動サスの主減衰)。
    wheel_solref: ホイール接地の solref。段差と同高の小径ホイールなので
                  やや軟らかめ(timeconst大)にして接地バウンドを抑える。
    """
    legs_xml = "".join(
        _leg_xml(n, sx, sy, stand_pitch, ankle_stiff, ankle_damp, wheel_solref)
        for n, sx, sy in LEGS)
    act_xml = "".join(_actuator_xml(n) for n, _, _ in LEGS)

    # 段差 plateau: 上面が z=step_h。手前エッジ x=step_x0 から +x 側へ広く伸ばす。
    step_half = 2.5
    step_cx = step_x0 + step_half
    step_cz = step_h / 2

    # keyframe qpos: freejoint(7) + 各脚[yaw,pitch,ankle,wheel](4)×4
    # freejoint: x y z (body中心) + quat(1 0 0 0)
    leg_q = f"0 {stand_pitch:.3f} 0 0"
    kf_qpos = f"0 0 {STAND_BODY_Z:.4f} 1 0 0 0 " + " ".join([leg_q] * 4)
    # ctrl: 各脚 [yawpos, pitchpos, wheeldrv]
    kf_ctrl = " ".join([f"0 {stand_pitch:.3f} 0"] * 4)

    return f"""<mujoco model="tachikoma_quad_v2">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>

  <default>
    <geom friction="1.0 0.02 0.001" density="500" contype="0" conaffinity="0"/>
    <joint damping="0.02" armature="0.003"/>
  </default>

  <worldbody>
    <light pos="0 0 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"
          rgba="0.35 0.4 0.45 1" friction="1.2 0.02 0.001"
          contype="1" conaffinity="1"/>

    <!-- {step_h*100:.0f}cm 段差 (要件 §6). x={step_x0} から先が高い plateau。 -->
    <geom name="step" type="box" pos="{step_cx:.4f} 0 {step_cz:.4f}"
          size="{step_half:.4f} 0.60 {step_cz:.4f}"
          rgba="0.30 0.34 0.40 1" friction="1.2 0.02 0.001"
          contype="1" conaffinity="1"/>

    <!-- ================= 車体(自由浮遊ベース) + 脚4本 ================= -->
    <body name="body" pos="0 0 {STAND_BODY_Z:.4f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{body_mass:.4f}"
                diaginertia="0.004 0.008 0.010"/>
      <geom name="bodyg" type="box" size="{BODY_HX:.4f} {BODY_HY:.4f} {BODY_HZ:.4f}"
            rgba="0.55 0.58 0.62 1" density="0"/>
      <!-- 前方マーカ (向きが分かるように) -->
      <geom name="nose" type="box" pos="{BODY_HX+0.015:.4f} 0 0" size="0.015 0.03 0.012"
            rgba="0.80 0.55 0.20 1" density="0"/>
{legs_xml}
    </body>
  </worldbody>

  <actuator>{act_xml}
  </actuator>

  <keyframe>
    <key name="stand" qpos="{kf_qpos}" ctrl="{kf_ctrl}"/>
  </keyframe>
</mujoco>
"""


# ---- ロード用ヘルパ ---------------------------------------------------------
def make(total_mass=1.3, stand_pitch=0.0, **kw):
    """総重量 total_mass[kg] に合わせて車体質量を逆算し、モデルを構築して返す。

    return: (m, d, A, info)
      A    : {actuator_name: id}
      info : 便利な id/adr 群 (下記 build_info 参照)
    """
    # まず暫定質量で作り、脚部品の合計質量を測って車体質量を逆算する
    xml0 = build_xml(body_mass=1.0, stand_pitch=stand_pitch, **kw)
    m0 = mujoco.MjModel.from_xml_string(xml0)
    parts = m0.body_subtreemass[m0.body("body").id] - m0.body_mass[m0.body("body").id]
    body_mass = max(0.05, total_mass - parts)

    xml = build_xml(body_mass=body_mass, stand_pitch=stand_pitch, **kw)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    A = {m.actuator(i).name: i for i in range(m.nu)}
    return m, d, A, build_info(m)


def build_info(m):
    """脚ごとの関節 qpos アドレス等をまとめて返す。"""
    info = {"legs": [n for n, _, _ in LEGS], "wheel_bid": {}, "pitch_qadr": {},
            "yaw_qadr": {}, "ankle_qadr": {}, "wheel_qadr": {}}
    for n, _, _ in LEGS:
        info["wheel_bid"][n] = m.body(f"wheel_{n}").id
        info["pitch_qadr"][n] = m.joint(f"pitch_{n}").qposadr[0]
        info["yaw_qadr"][n] = m.joint(f"yaw_{n}").qposadr[0]
        info["ankle_qadr"][n] = m.joint(f"ankle_{n}").qposadr[0]
        info["wheel_qadr"][n] = m.joint(f"wheel_{n}").qposadr[0]
    info["body_bid"] = m.body("body").id
    return info


if __name__ == "__main__":
    m, d, A, info = make(1.3)
    print("nq=%d nv=%d nu=%d" % (m.nq, m.nv, m.nu))
    print("総重量 = %.4f kg (目標 1.3)" % m.body_subtreemass[info["body_bid"]])
    print("actuators:", list(A))
    print("legs:", info["legs"])
    print("標準車体高 STAND_BODY_Z = %.4f m" % STAND_BODY_Z)
