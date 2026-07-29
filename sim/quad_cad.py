"""
tachikoma / 4脚タチコマ 生成モデル(唯一の真実)  ―  MJCF をコードで生成

【モデル定義の唯一の真実はこの生成スクリプト】。静的 mjmodel.xml は廃止(_deprecated)。
可動域・質量配分・トレッド・脚長などを変数で持ち、Phase 2a のパラメータ探索を容易にする。

CAD(test_robot.urdf, Onshape 出力)からは視覚メッシュのみ流用。物理は要件で規定:
  ・質量/慣性 : URDF値(合計<1g)は捨て、総重量 1.300kg を要件配分で割り当てる。
        body(Raspi＋バッテリー＋基板) = 0.66kg
        各脚 = 股ヨーサーボ 0.06 + 股ピッチサーボ 0.06 + ロッド 0.02 + 足首 0.004
               + ホイール 0.016 = 0.16kg  (サーボは付け根集約 → 脚先は軽い, 要件§4)
        → 0.66 + 4×0.16 = 1.30kg。質量は各 body の <inertial> で与える(geom非依存)。
  ・軸の意味  : CADの親子表記に依らず、運動学は「股ヨー(操舵)＋股ピッチ(脚の振り)」の
        2軸・棒脚(膝なし)。前脚は前方へ、後脚は後方へ振れるよう mount を鏡像化。
  ・可動範囲  : ヨー ±45°(±0.785, 一律・案i)。ピッチ −90〜+20°(−1.57〜+0.35, 確定値)。
        pitchpos アクチュエータの ctrlrange も同値に同期。0=脚を真下、負=外側へ振り出し。
  ・ジオメトリ: 表示用プリム(bodyg/nose/hipg/rod/ankleg)は生成しない。視覚は CAD メッシュ
        (body/joint/feet.obj)のみ。feet.obj が脚領域を覆うので rod 無しでも脚視覚は欠けない。
        衝突はホイール(wheelg_*, contype/conaffinity=1)のみ。
  ・寸法/新規: 脚リンク長 100mm(130→70%短縮, STS3215 射程確定値)。脚先ホイール
        (径3cm=半径0.015・駆動・既存 solref/solimp)＋足首受動サス(stiffness/damping 可変)。

パラメータ(案b 動的リフト検証の設計変数):
  tread     : 左右トレッド[m] = 2×HIP_Y。広いほど転倒までの時間が伸びる。
  stance_deg: 立位の股ピッチ振り出し角[deg]。大きいほど脚を寝かせて車高(COM)が下がる。
  (lift_time はシーケンス側パラメータ。本ファイルはモデルのみ)

座標: +x=前, +y=左, +z=上。 脚配置 RF(右前) LF(左前) RB(右後) LB(左後)。右=-y。
"""
import os
import numpy as np
import mujoco

# ---- 幾何(要件値) ---------------------------------------------------------
BODY_HX, BODY_HY, BODY_HZ = 0.09, 0.08, 0.025    # 車体半寸法(体長0.18×幅0.16×厚0.05)
HIP_X = 0.08                                      # 股の前後オフセット(体端寄り)
ROD_LEN = 0.10                                    # 棒脚 100mm(要件採用値=130→70%)
WHEEL_R = 0.015                                   # ホイール半径(径3cm固定)
ANKLE_R = 0.010
ELL = ROD_LEN + 0.015 + WHEEL_R                   # pitch軸→ホイール接地 = 0.13

STEP_H = 0.03
STEP_X0 = 0.15

# ---- 質量配分(要件, 合計1.30kg) -------------------------------------------
M_YAW_SERVO = 0.06        # 股ヨー STS3215(付け根)
M_PITCH_SERVO = 0.06      # 股ピッチ STS3215(付け根)
M_ROD = 0.02              # 棒リンク(3Dプリント)
M_ANKLE = 0.004           # 足首サス部
M_WHEEL = 0.016           # ホイール＋駆動
M_LEG = M_YAW_SERVO + M_PITCH_SERVO + M_ROD + M_ANKLE + M_WHEEL   # 0.16
M_BODY = round(1.30 - 4 * M_LEG, 4)   # = 0.66  (Raspi＋バッテリー＋基板)

# 0=脚を真下、負=外側へ振り出し(pull-up は −1.4 rad 付近を使用)
YAW_RANGE = (-0.785, 0.785)           # 股ヨー ±45°(一律, 案i)
PITCH_RANGE = (-1.57, 0.35)           # 股ピッチ −90〜+20°(確定値). 旧 −2.79(CAD自己干渉上限)は不使用

# 脚: (名前, x符号, y符号, 後脚か). 後脚は mount を 180°回して「振り出し=後方」にする。
LEGS = [
    ("RF", +1, -1, False),
    ("LF", +1, +1, False),
    ("RB", -1, -1, True),
    ("LB", -1, +1, True),
]

MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "robot", "test_robot", "meshes", "obj")

# feet 視覚メッシュの配置(脚 body ローカル). 脚は -z 方向に伸び、脚先ホイールは z=-0.115。
# feet.obj はネイティブで長軸 +z(0〜0.11)。これを脚の -z(下向き)へ向けると足先がホイールへ
# 自然に繋がる = Y軸180°回転(euler "0 π 0")。URDF の feet visual は Y軸回転(rpy 0 ?  0)だが、
# その rpy はメッシュファイル外(URDF ノード)で OBJ には焼き込まれていないため、値は URLF の
# 45°/80° をそのまま使わず、MuJoCo 表示で車輪・脚が自然に繋がる向きに合わせ込んだ(前脚/後脚とも
# 確認済み。後脚は hip の Z180°反転を継承するが本回転で狂いなし)。
# 旧値 euler "-1.5708 0 0"(X−90°)は長軸を側方(-y)へ倒す座標変換ミスだった。
FEET_XFORM = 'pos="0 0 0" euler="0 3.14159 0"'


def _box_inertia(mass, hx, hy, hz):
    ix = mass * (hy * hy + hz * hz) / 3.0
    iy = mass * (hx * hx + hz * hz) / 3.0
    iz = mass * (hx * hx + hy * hy) / 3.0
    return ix, iy, iz


def _leg_xml(name, sx, sy, rear, ankle_stiff, ankle_damp, wheel_solref, use_mesh):
    hx = HIP_X * sx
    hy = HIP_Y * sy
    # 後脚は 180°回転 mount(局所+x が世界-x): ピッチ振り出しが後方になる
    euler = "0 0 3.14159" if rear else "0 0 0"
    # 視覚メッシュ(質量0・非衝突)。joint=股ブラケット, feet=脚+足。
    joint_mesh = f'<geom type="mesh" mesh="joint" density="0" contype="0" conaffinity="0" rgba="0.62 0.65 0.66 1"/>' if use_mesh else ""
    # feet メッシュの配置(向き)は FEET_XFORM で一元管理(下記定数参照)。
    feet_mesh = (f'<geom type="mesh" mesh="feet" {FEET_XFORM} '
                 f'density="0" contype="0" conaffinity="0" rgba="0.62 0.81 0.93 1"/>') if use_mesh else ""
    ix_y, iy_y, iz_y = 1e-5, 1e-5, 1e-5
    pix, piy, piz = _box_inertia(M_PITCH_SERVO + M_ROD, 0.012, 0.012, 0.05)
    # 表示用プリム(hipg/rod/ankleg)は生成しない。視覚は CAD メッシュ、衝突はホイールのみ。
    # 質量は各 body の <inertial> で与える(geom には依存しない)。
    return f"""
      <body name="hip_{name}" pos="{hx:.4f} {hy:.4f} {-BODY_HZ:.4f}" euler="{euler}">
        <!-- 股ヨー(操舵, ±45°). サーボ質量は付け根に集約 -->
        <joint name="yaw_{name}" type="hinge" axis="0 0 1"
               range="{YAW_RANGE[0]} {YAW_RANGE[1]}" damping="0.1"/>
        <inertial pos="0 0 -0.008" mass="{M_YAW_SERVO}" diaginertia="{ix_y} {iy_y} {iz_y}"/>
        {joint_mesh}
        <!-- 股ピッチ(脚の振り, {PITCH_RANGE[0]}〜{PITCH_RANGE[1]} rad). ★トルク実測対象★ -->
        <body name="leg_{name}" pos="0 0 0">
          <joint name="pitch_{name}" type="hinge" axis="0 1 0"
                 range="{PITCH_RANGE[0]} {PITCH_RANGE[1]}" damping="0.1"/>
          <inertial pos="0 0 -0.02" mass="{M_PITCH_SERVO + M_ROD}"
                    diaginertia="{pix:.2e} {piy:.2e} {piz:.2e}"/>
          {feet_mesh}
          <!-- 足首 受動サス(脚軸ばね) -->
          <body name="ankle_{name}" pos="0 0 {-ROD_LEN:.4f}">
            <joint name="ankle_{name}" type="slide" axis="0 0 1" range="-0.002 0.020"
                   stiffness="{ankle_stiff:.0f}" damping="{ankle_damp:.0f}" springref="0"/>
            <inertial pos="0 0 0" mass="{M_ANKLE}" diaginertia="1e-6 1e-6 1e-6"/>
            <!-- 脚先ホイール(径3cm・駆動). ★衝突はこの geom のみ★ -->
            <body name="wheel_{name}" pos="0 0 {-WHEEL_R:.4f}">
              <joint name="wheel_{name}" type="hinge" axis="0 1 0" damping="0.01"/>
              <inertial pos="0 0 0" mass="{M_WHEEL}" diaginertia="2e-6 3e-6 2e-6"/>
              <geom name="wheelg_{name}" type="cylinder" fromto="0 -0.010 0  0 0.010 0"
                    size="{WHEEL_R:.4f}" rgba="0.15 0.15 0.18 1"
                    friction="1.6 0.02 0.001" density="0"
                    contype="1" conaffinity="1"
                    solref="{wheel_solref}" solimp="0.9 0.98 0.001"/>
            </body>
          </body>
        </body>
      </body>"""


def _actuator_xml(name):
    return f"""
    <position name="yawpos_{name}" joint="yaw_{name}" kp="20" kv="1"
              ctrlrange="{YAW_RANGE[0]} {YAW_RANGE[1]}" forcerange="-4 4"/>
    <position name="pitchpos_{name}" joint="pitch_{name}" kp="60" kv="3"
              ctrlrange="{PITCH_RANGE[0]} {PITCH_RANGE[1]}" forcerange="-20 20"/>
    <velocity name="wheeldrv_{name}" joint="wheel_{name}" kv="1.5"
              ctrlrange="-60 60" forcerange="-1.5 1.5"/>"""


def build_xml(tread=0.18, stance_pitch=-0.40, step_x0=STEP_X0, step_h=STEP_H,
              ankle_stiff=2500, ankle_damp=40, wheel_solref="0.01 1", use_mesh=True):
    """CAD由来4脚モデルの MJCF。tread=左右トレッド[m], stance_pitch=立位の股ピッチ[rad]。"""
    global HIP_Y
    HIP_Y = tread / 2.0

    mesh_ok = use_mesh and os.path.isdir(MESH_DIR) and \
        all(os.path.exists(os.path.join(MESH_DIR, f)) for f in ("body.obj", "joint.obj", "feet.obj"))
    if use_mesh and not mesh_ok:
        use_mesh = False

    asset = ""
    compiler_meshdir = ""
    body_mesh = ""
    if use_mesh:
        # meshdir に(スクリプト位置から導いた)メッシュ dir を与え、file はベース名で参照。
        # from_xml_string は相対パスの起点を持たないため、cwd 非依存にするには絶対 meshdir が要る。
        # ハードコード絶対パスではなく __file__ から算出するので機種・cwd に依存しない。
        compiler_meshdir = f' meshdir="{MESH_DIR}"'
        asset = """
  <asset>
    <mesh name="body" file="body.obj"/>
    <mesh name="joint" file="joint.obj"/>
    <mesh name="feet" file="feet.obj"/>
  </asset>"""
        body_mesh = '<geom type="mesh" mesh="body" density="0" contype="0" conaffinity="0" rgba="0.62 0.81 0.93 1"/>'

    legs_xml = "".join(_leg_xml(n, sx, sy, rear, ankle_stiff, ankle_damp, wheel_solref, use_mesh)
                       for n, sx, sy, rear in LEGS)
    act_xml = "".join(_actuator_xml(n) for n, _, _, _ in LEGS)

    # 立位の車体高: pitch=stance_pitch のとき pitch軸 = ℓ·cos(pitch) 上, +BODY_HZ
    stand_body_z = ELL * np.cos(stance_pitch) + BODY_HZ + 0.002

    ix, iy, iz = _box_inertia(M_BODY, BODY_HX, BODY_HY, BODY_HZ)

    step_half = 2.5
    step_cx = step_x0 + step_half
    step_cz = step_h / 2

    # keyframe: freejoint(7) + 各脚[yaw,pitch,ankle,wheel]. 前後脚とも局所 pitch=stance_pitch。
    leg_q = f"0 {stance_pitch:.4f} 0 0"
    kf_qpos = f"0 0 {stand_body_z:.4f} 1 0 0 0 " + " ".join([leg_q] * 4)
    kf_ctrl = " ".join([f"0 {stance_pitch:.4f} 0"] * 4)

    return f"""<mujoco model="tachikoma_quad_cad">
  <compiler angle="radian" autolimits="true"{compiler_meshdir}/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>
  <default>
    <geom friction="1.0 0.02 0.001" density="500" contype="0" conaffinity="0"/>
    <joint damping="0.02" armature="0.003"/>
  </default>{asset}
  <worldbody>
    <light pos="0 0 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"
          rgba="0.35 0.4 0.45 1" friction="1.2 0.02 0.001" contype="1" conaffinity="1"/>
    <geom name="step" type="box" pos="{step_cx:.4f} 0 {step_cz:.4f}"
          size="{step_half:.4f} 0.60 {step_cz:.4f}"
          rgba="0.30 0.34 0.40 1" friction="1.2 0.02 0.001" contype="1" conaffinity="1"/>
    <body name="body" pos="0 0 {stand_body_z:.4f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{M_BODY}" diaginertia="{ix:.5f} {iy:.5f} {iz:.5f}"/>
      {body_mesh}
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


def make(tread=0.18, stance_pitch=-0.40, **kw):
    """CAD由来4脚モデルを構築。return (m, d, A, info)。総重量は要件配分で 1.30kg 固定。"""
    xml = build_xml(tread=tread, stance_pitch=stance_pitch, **kw)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    A = {m.actuator(i).name: i for i in range(m.nu)}
    return m, d, A, build_info(m)


def build_info(m):
    info = {"legs": [n for n, _, _, _ in LEGS], "rear": {n: r for n, _, _, r in LEGS},
            "wheel_bid": {}, "pitch_qadr": {}, "yaw_qadr": {}, "ankle_qadr": {}, "wheel_qadr": {}}
    for n, _, _, _ in LEGS:
        info["wheel_bid"][n] = m.body(f"wheel_{n}").id
        info["pitch_qadr"][n] = m.joint(f"pitch_{n}").qposadr[0]
        info["yaw_qadr"][n] = m.joint(f"yaw_{n}").qposadr[0]
        info["ankle_qadr"][n] = m.joint(f"ankle_{n}").qposadr[0]
        info["wheel_qadr"][n] = m.joint(f"wheel_{n}").qposadr[0]
    info["body_bid"] = m.body("body").id
    return info


if __name__ == "__main__":
    m, d, A, info = make(tread=0.18, stance_pitch=-0.40)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print("nq=%d nv=%d nu=%d" % (m.nq, m.nv, m.nu))
    print("総重量 = %.4f kg (要件配分 body%.2f + 4脚×%.2f)" %
          (m.body_subtreemass[info["body_bid"]], M_BODY, M_LEG))
    for _ in range(600):
        mujoco.mj_step(m, d)
    com = d.subtree_com[info["body_bid"]]
    print("静定後: body_z=%.4f  COM_z=%.4f  roll/pitch すべり確認" %
          (d.body("body").xpos[2], com[2]))
    print("接地脚 wheel z:", {n: round(float(d.body(f"wheel_{n}").xpos[2]), 4) for n in info["legs"]})
    print("wheel x (前後対称?):", {n: round(float(d.body(f"wheel_{n}").xpos[0]), 4) for n in info["legs"]})
    print("wheel y (左右対称?):", {n: round(float(d.body(f"wheel_{n}").xpos[1]), 4) for n in info["legs"]})
