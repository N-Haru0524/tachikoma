"""
tachikoma / 4脚モデル ―  URDF 直読み方式(メッシュ向きズレの根本解決)

方針(ユーザー決定・確定):
  test_robot.urdf(Onshape 出力)を MuJoCo に直接ロードする。メッシュ向き・関節配置・
  リンク寸法は URDF を唯一の真実とし、MuJoCo に解釈させる。手で quat を打ち込まない。
  → feet の向きは URDF の <visual rpy> から MuJoCo が自動生成 = ズレが原理的に発生しない。
  (検証: MuJoCo は URDF の visual rpy を正しく適用することを最小例で確認済み。)

URDF に無い要素だけをプログラムで足す(向きの曖昧さが無い要素):
  ・車輪    : 各脚先(feet リンクの足先)に径3cm(半径0.015)駆動ホイール(hinge)
  ・足首サス: 足先と車輪の間に受動 slide(stiffness≈2500, damping≈40, range −0.002〜0.02)
  ・衝突    : ホイールのみ(contype/conaffinity=1)。URDF由来メッシュは視覚専用(contype=0)
  ・質量    : 総 1.300kg(body 0.66 + 脚1本 0.16×4)で上書き(URDF inertial は非現実値)
  ・可動域  : pitch −1.57〜0.35、yaw ±0.785。アクチュエータ ctrlrange も同期
  ・環境    : floor(plane)＋ step(3cm段, box)

メッシュ:
  MuJoCo 3.10 は glTF デコーダを持たないため、URDF の *.gltf を *.obj に変換して参照する
  (変換は座標を保存: trimesh force='mesh'。yourdfpy=標準URDFツールと同一頂点)。
  meshdir は URDF ファイルからの相対パス(../meshes/obj)。絶対パスは使わない。

使い方:
    uv run python sim/urdf_model.py            # 生成・検証(質量/可動域/対称)
    uv run python sim/urdf_model.py render      # オフスクリーン描画を PNG 保存
"""
import argparse
import glob
import os
import re
import numpy as np
import mujoco

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)                          # リポジトリ直下
ROBOT_ROOT = os.path.join(ROOT, "robot")              # 各ロボットは robot/<name>/ に配置
DEFAULT_ROBOT = "test_robot"

# 対象ロボットのパス群(set_robot() で切替。build/make_stand/ensure_meshes が参照)。
# 既定 test_robot で下の set_robot() が初期化する。
ROBOT_NAME = DEFAULT_ROBOT
ROBOT_DIR = URDF_SRC = URDF_MJC = MESH_GLTF_DIR = MESH_OBJ_DIR = None


def robot_paths(name_or_path=DEFAULT_ROBOT):
    """ロボット名 or ディレクトリパスから各アセットのパスを解決して dict で返す。
    ・名前(例 'test_robot')      → robot/<名前>/ を使う
    ・パス(区切りを含む/実在dir) → それを robot ルートとして使う
    URDF は urdf/<名前>.urdf を優先。無ければ urdf/ 内の(生成物 *_mjc を除く)最初の *.urdf。
    meshes は <root>/meshes/、変換 OBJ は <root>/meshes/obj/。"""
    if os.sep in name_or_path or (os.altsep and os.altsep in name_or_path) \
            or os.path.isdir(name_or_path):
        rdir = os.path.abspath(name_or_path)
        name = os.path.basename(os.path.normpath(rdir))
    else:
        name = name_or_path
        rdir = os.path.join(ROBOT_ROOT, name)
    udir = os.path.join(rdir, "urdf")
    src = os.path.join(udir, name + ".urdf")
    if not os.path.exists(src):
        cands = [c for c in sorted(glob.glob(os.path.join(udir, "*.urdf")))
                 if not c.endswith("_mjc.urdf")]
        if cands:
            src = cands[0]
    stem = os.path.splitext(os.path.basename(src))[0]
    return dict(name=name, dir=rdir, urdf_src=src,
                urdf_mjc=os.path.join(udir, stem + "_mjc.urdf"),
                mesh_gltf=os.path.join(rdir, "meshes"),
                mesh_obj=os.path.join(rdir, "meshes", "obj"))


def set_robot(name_or_path=DEFAULT_ROBOT):
    """対象ロボットを切り替える(モジュールのパス群を再設定して dict を返す)。"""
    global ROBOT_NAME, ROBOT_DIR, URDF_SRC, URDF_MJC, MESH_GLTF_DIR, MESH_OBJ_DIR
    p = robot_paths(name_or_path)
    if not os.path.exists(p["urdf_src"]):
        avail = sorted(os.path.basename(x) for x in glob.glob(os.path.join(ROBOT_ROOT, "*"))
                       if os.path.isdir(x) and glob.glob(os.path.join(x, "urdf", "*.urdf")))
        raise FileNotFoundError(
            "ロボット '%s' の URDF が見つかりません: %s\n"
            "  --robot に指定できる robot/ 直下: %s"
            % (name_or_path, p["urdf_src"], ", ".join(avail) or "(なし)"))
    ROBOT_NAME, ROBOT_DIR = p["name"], p["dir"]
    URDF_SRC, URDF_MJC = p["urdf_src"], p["urdf_mjc"]
    MESH_GLTF_DIR, MESH_OBJ_DIR = p["mesh_gltf"], p["mesh_obj"]
    return p


def add_robot_arg(parser):
    """argparse パーサに共通の --robot を追加する(全エントリで統一)。"""
    parser.add_argument("--robot", default=DEFAULT_ROBOT,
                        help="対象ロボット名(robot/<name>/) or robot ディレクトリのパス [%(default)s]")
    return parser


set_robot(DEFAULT_ROBOT)                              # 既定ロボットでパスを初期化

WHEEL_R = 0.015
STEP_H = 0.03
STEP_X0 = 0.15

# 脚: pitch関節 -> feet リンク / yaw関節 -> joint リンク の対応(URDF より)
FEET = ["feet", "feet_1", "feet_2", "feet_3"]
PITCH_JOINTS = {"feet": "revolute_1_3", "feet_1": "revolute_1_1",
                "feet_2": "revolute_1_2", "feet_3": "revolute_1"}
# feet リンクごとの yaw(操舵)関節。joint リンク側の revolute。
YAW_OF = {"feet": "revolute_1_4", "feet_1": "revolute_4",
          "feet_2": "revolute_2", "feet_3": "revolute_3"}
YAW_JOINTS = list(YAW_OF.values())

# 鏡像実装の脚グループ: A は負 pitch で脚が下がる、B(π反転 mount)は正 pitch で下がる。
# Onshape 焼き付き解消済 → pitch 可動域は全脚 ±80°(±1.39626)で対称(URDF limit と一致)。
# PITCH_SIGN は「脚を下へ向ける」符号だけを鏡像化する(ROM 量・対称性は不変)。
PITCH_ROM = 1.39626                    # 股ピッチ ±80°(URDF revolute limit)
GROUP_A = ["feet", "feet_3"]           # down = 負 pitch
GROUP_B = ["feet_1", "feet_2"]         # down = 正 pitch(mount π反転)
PITCH_SIGN = {"feet": -1, "feet_3": -1, "feet_1": +1, "feet_2": +1}
def pitch_range_of(f):
    return (-PITCH_ROM, PITCH_ROM)     # 全脚 ±80° 対称

# 質量: URDF の実 inertial(PLA充填の実測代用)を「唯一の真実」とする。総 1.606kg。
#   旧「1.300kg 要件配分に上書き」は【廃止】(2026-07 モーター再選定)。URDF 実値を使う。
#   内訳(URDF): body 0.4836 / tank 0.5798 / feet 0.1308×4 / joint 0.00497×4 = 1.6065kg。
# 追加する車輪/足首は接地動力学に必須の体だが、その質量は総重量を膨らませないよう
# feet 実質量から【切り出す】(feet 体積の中に車輪・足首取付部が含まれる、で整合)。
URDF_FEET_MASS = 0.130791          # feet 1本の URDF 実質量(脚長130mm時)
URDF_TANK_MASS = 0.579824          # tank の URDF 実質量(現状580g)
URDF_TANK_Y = -0.07                # tank 取付 y[m](body から −70mm = 偏重心の源)
URDF_TANK_I = [0.000547473, 0.000739403, 0.00053394,
               6.53e-13, 1.53e-13, -2.06437e-06]   # ixx,iyy,izz,ixy,ixz,iyz
M_ANKLE, M_WHEEL = 0.004, 0.016    # 接地に要る追加体(feet から切り出す=総重量不変)
YAW_RANGE = (-0.785, 0.785)
P_STAND = 1.0            # 公称立位の pitch 振り(鏡像で ±P_STAND)。足首サス軸の算定に使用

# vr2 実車輪(URDF に wheel* リンクを持つ改良版)の車輪ジオメトリ
WHEEL_R_VR2 = 0.010     # 半径(径 0.02m)
WHEEL_HW_VR2 = 0.0025   # 半幅(幅 0.005m)

# build() が設定する対象モデル依存マップ(downstream はこれを参照)
WHEEL_BODIES = {}       # {foot: 車輪 body 名}  vr1='wheel_<foot>' / vr2=URDFの wheel*
LEG_MAP = None          # vr2 の脚関節マップ {foot:{pitch,yaw,wheel_j,wheel_body,wheel_axis}} / vr1=None


def ensure_meshes():
    """glTF→OBJ 変換(無ければ生成)。座標保存のため trimesh force='mesh'。

    生成済み OBJ があれば trimesh は不要(=素の `uv run python` で動く)。
    OBJ が無く trimesh も無い場合は、変換方法を明示した分かりやすいエラーを出す。
    OBJ は派生物なので、リポジトリにコミットしておけば実行時 trimesh 依存が消える。

    変換対象はメッシュ dir 内の *.gltf すべて(名前は決め打ちしない)。ロボットごとに
    メッシュ構成が違っても(例: vr2 の wheel/pre_joint)取りこぼさない。"""
    def obj_of(g):
        return os.path.join(MESH_OBJ_DIR, os.path.splitext(os.path.basename(g))[0] + ".obj")
    gltfs = sorted(glob.glob(os.path.join(MESH_GLTF_DIR, "*.gltf")))
    todo = [g for g in gltfs if not os.path.exists(obj_of(g))]
    if not todo:
        return                       # 全 gltf 分の obj 済(or gltf 無=committed obj)→ trimesh 不要
    try:
        import trimesh
    except ImportError:
        raise RuntimeError(
            "視覚メッシュ OBJ が未生成で、変換に必要な trimesh がこの環境にありません。\n"
            "  MuJoCo 3.10 は glTF を読めないため OBJ 変換が一度だけ必要です。\n"
            "  次のいずれかで解決してください(プロジェクト環境は汚しません):\n"
            "    (A) 一度だけ変換:  uv run --with trimesh --with pygltflib "
            "python -c \"import sys;sys.path.insert(0,'sim');import urdf_model;urdf_model.ensure_meshes()\"\n"
            "    (B) 恒久化      :  上記で生成される robot/test_robot/meshes/obj/*.obj を"
            " git にコミット(以後 trimesh 不要)\n"
            f"  生成先: {MESH_OBJ_DIR}")
    os.makedirs(MESH_OBJ_DIR, exist_ok=True)
    for g in todo:
        trimesh.load(g, force="mesh").export(obj_of(g))
    print("[urdf_model] glTF→OBJ 変換完了(%d件): %s" % (len(todo), MESH_OBJ_DIR))


def write_sibling_urdf():
    """URDF を非破壊で複製し、(1) mesh を .obj 参照、(2) <mujoco> compiler 拡張を注入。
    relative meshdir + discardvisual=false(視覚メッシュ保持)+ strippath。"""
    txt = open(URDF_SRC, encoding="utf-8").read().replace(".gltf", ".obj")
    block = ('<mujoco><compiler discardvisual="false" meshdir="../meshes/obj" '
             'strippath="true" balanceinertia="true" fusestatic="false"/></mujoco>')
    # ロボット名に依存せず <robot ...> 開始タグ直後へ注入(最初の1箇所)
    txt = re.sub(r'(<robot\b[^>]*>)', lambda mm: mm.group(1) + '\n  ' + block, txt, count=1)
    open(URDF_MJC, "w", encoding="utf-8").write(txt)


def _foot_tip_local(m, d, feet_name):
    """feet リンク局所フレームでの足先(メッシュ最遠点)座標。車輪取付点。"""
    bid = m.body(feet_name).id
    for gi in range(m.ngeom):
        if m.geom(gi).bodyid[0] == bid and m.geom(gi).type[0] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom(gi).dataid[0]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[va:va + vn]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom(gi).quat); R = R.reshape(3, 3)
            Vl = (V @ R.T) + m.geom(gi).pos
            return Vl[np.argmax(np.linalg.norm(Vl, axis=1))]
    return np.array([0.108, -0.044, 0.019])


def _hinge_on(m, bid):
    """body bid を子とする hinge 関節 id(無ければ None)。"""
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            return j
    return None


def _derive_legs(m, d):
    """URDF に実車輪(feet* の子 = wheel* リンク)がある改良版(vr2)なら、脚ごとに
    {pitch, yaw, wheel_j, wheel_body, wheel_axis} を **構造から** 導出して返す。
    無ければ None(=vr1 構造: 車輪は後付け)。関節名の付番はモデル間で変わるため、
    固定リンク名(feet*/wheel*)と world 軸で判定する: hinge が world で鉛直に近い方=yaw
    (操舵)、そうでない方=pitch(脚振り)。"""
    legs = {}
    for f in FEET:
        fbid = m.body(f).id
        wbid = next((i for i in range(m.nbody)
                     if m.body_parentid[i] == fbid and m.body(i).name.startswith("wheel")), None)
        if wbid is None:
            return None                                   # vr1(URDFに車輪なし)
        jf, jp, jw = _hinge_on(m, fbid), _hinge_on(m, m.body_parentid[fbid]), _hinge_on(m, wbid)
        if None in (jf, jp, jw):
            return None

        def waxis(j):
            R = d.body(m.jnt_bodyid[j]).xmat.reshape(3, 3)
            return R @ m.jnt_axis[j]
        af, ap = waxis(jf), waxis(jp)
        yaw_j, pitch_j = (jf, jp) if abs(af[2]) > abs(ap[2]) else (jp, jf)
        legs[f] = dict(pitch=m.joint(pitch_j).name, yaw=m.joint(yaw_j).name,
                       wheel_j=m.joint(jw).name, wheel_body=m.body(wbid).name,
                       wheel_axis=[float(x) for x in m.jnt_axis[jw]])
    return legs


def _build_vr2(legs, tank_mass=None, tank_y=None):
    """vr2: URDF の実車輪(wheel*)に衝突円柱(径0.02m/幅0.005m)を付与し、URDF の車輪関節で
    駆動する。球車輪・足首サスの後付けはしない(車輪は CAD 実物を置換使用)。"""
    spec = mujoco.MjSpec.from_file(URDF_MJC)
    if tank_mass is not None or tank_y is not None:                # tank 上書き(共通)
        tb = spec.body("tank")
        if tank_mass is not None:
            sc = tank_mass / URDF_TANK_MASS
            tb.mass = tank_mass; tb.fullinertia = [v * sc for v in URDF_TANK_I]
        if tank_y is not None:
            p = list(tb.pos); p[1] += (tank_y - URDF_TANK_Y); tb.pos = p
    spec.body("root").add_freejoint()
    for g in spec.geoms:                                          # URDF メッシュは視覚専用
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = 0; g.conaffinity = 0
    pj = {legs[f]["pitch"] for f in FEET}
    yj = {legs[f]["yaw"] for f in FEET}
    wj = {legs[f]["wheel_j"] for f in FEET}
    for j in spec.joints:                                         # 可動域＋armature
        if j.name in pj:
            j.range = [-PITCH_ROM, PITCH_ROM]; j.armature = 0.01
        elif j.name in yj:
            j.range = list(YAW_RANGE); j.armature = 0.01
        elif j.name in wj:
            j.armature = 0.003; j.damping[0] = 0.01
    for f in FEET:                                                # 実車輪へ衝突円柱を付与
        wb = spec.body(legs[f]["wheel_body"])
        for mg in list(wb.geoms):                                 # 車輪の視覚メッシュは削除
            if mg.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(mg)
        a = np.array(legs[f]["wheel_axis"], float); a /= (np.linalg.norm(a) + 1e-9)
        g = wb.add_geom(name="wheelg_%s" % f, type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        g.size = [WHEEL_R_VR2, WHEEL_HW_VR2, 0]; g.pos = [0, 0, 0]
        g.quat = _axis_to_quat(a)                                 # 局所 z を車輪スピン軸へ
        g.contype = 1; g.conaffinity = 1
        g.friction = [1.6, 0.02, 0.001]
        g.solref = [0.01, 1]; g.solimp = [0.9, 0.98, 0.001, 0.5, 2]
        g.rgba = [0.15, 0.15, 0.18, 1]
        if wb.mass < 0.005:                                       # 軽すぎ発散対策の最低質量
            wb.mass = M_WHEEL; wb.fullinertia = [2e-6, 3e-6, 2e-6, 0, 0, 0]
    for f in FEET:                                                # アクチュエータ(foot名で統一)
        a = spec.add_actuator(name="pitchpos_%s" % f, target=legs[f]["pitch"],
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 60
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[1] = -60; a.biasprm[2] = -3
        a.ctrlrange = [-PITCH_ROM, PITCH_ROM]; a.forcerange = [-20, 20]
    for f in FEET:
        a = spec.add_actuator(name="yawpos_%s" % f, target=legs[f]["yaw"],
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 20
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[1] = -20; a.biasprm[2] = -1
        a.ctrlrange = list(YAW_RANGE); a.forcerange = [-4, 4]
    for f in FEET:
        a = spec.add_actuator(name="wheeldrv_%s" % f, target=legs[f]["wheel_j"],
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 1.5
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[2] = -1.5
        a.ctrlrange = [-60, 60]; a.forcerange = [-1.5, 1.5]
    m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def build(ankle_stiff=2500, ankle_damp=40, ankle_axis="vertical", leg_scale=1.0,
          tank_mass=None, tank_y=None):
    """URDF 直読み + 要素追加でモデル(MjModel, MjData)を返す。
    ankle_axis: 'vertical'=足首サスを鉛直軸(立位で正しく沈む)/'leg'=脚長方向。
    leg_scale : 脚長スケール(1.0=現状130mm)。<1 で脚を短縮し、feet 質量・reach を
                同率で縮める(rod 的に体積∝長さ=線形)。総重量もその分下がる。
    tank_mass : tank 質量[kg]を上書き(None=URDF実値580g)。案B 軽量化の検証用。
                慣性は質量比で縮尺。総重量は (1.0265 + tank_mass) 近傍になる。
    tank_y    : tank 取付 y[m]を上書き(None=−0.07)。偏重心を中心へ寄せる検証用。"""
    global WHEEL_BODIES, LEG_MAP
    ensure_meshes()
    write_sibling_urdf()

    # 改良版(vr2: URDF に実車輪あり)なら、その車輪を置換使用する経路へ分岐
    _m0 = mujoco.MjModel.from_xml_path(URDF_MJC)
    _d0 = mujoco.MjData(_m0); mujoco.mj_forward(_m0, _d0)
    legs = _derive_legs(_m0, _d0)
    if legs is not None:
        m, d = _build_vr2(legs, tank_mass=tank_mass, tank_y=tank_y)
        WHEEL_BODIES = {f: legs[f]["wheel_body"] for f in FEET}
        LEG_MAP = legs
        return m, d
    WHEEL_BODIES = {f: "wheel_%s" % f for f in FEET}
    LEG_MAP = None

    # ---- 以降 vr1(URDF に車輪なし)経路: 車輪・足首サスをコードで後付け ----
    # 第1パス: 足先位置＋(公称立位での)各脚の鉛直軸を測るためコンパイル
    m0 = mujoco.MjModel.from_xml_path(URDF_MJC)
    d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
    # 足先(車輪取付点)。leg_scale<1 なら pitch 軸に向けて縮める(reach を同率短縮)。
    tips = {f: _foot_tip_local(m0, d0, f) * leg_scale for f in FEET}
    # 公称立位姿勢(P_STAND, 鏡像 pitch)で feet の world 回転を得て、足首サスの軸を
    # 「その姿勢で world 鉛直」になる局所ベクトルに合わせる(斜め軸=不安定を回避)。
    d1 = mujoco.MjData(m0)
    for f in FEET:
        d1.qpos[m0.joint(PITCH_JOINTS[f]).qposadr[0]] = PITCH_SIGN[f] * P_STAND
    mujoco.mj_forward(m0, d1)
    vaxis = {}
    for f in FEET:
        Rf = d1.body(f).xmat.reshape(3, 3)          # feet world 回転
        vaxis[f] = Rf.T @ np.array([0, 0, -1.0])    # world 下向きを feet 局所で表現

    # 第2パス: MjSpec で編集
    spec = mujoco.MjSpec.from_file(URDF_MJC)

    # (a) 質量: URDF 実 inertial を保持(body/tank/joint は非上書き)。
    #     feet だけ leg_scale を反映し、追加する車輪+足首(M_WHEEL+M_ANKLE)を feet 実質量から
    #     切り出す(総重量を膨らませない)。慣性も leg_scale で縮尺。
    feet_mass = URDF_FEET_MASS * leg_scale
    feet_body_mass = max(1e-3, feet_mass - (M_WHEEL + M_ANKLE))   # 車輪/足首分を控除
    for b in spec.bodies:
        if b.name in FEET:
            I = max(1e-6, 1.5e-4 * leg_scale)     # URDF feet 慣性 ~1.5e-4 級を縮尺
            b.mass = feet_body_mass; b.fullinertia = [I, I, I, 0, 0, 0]
    # tank: 質量/位置の上書き(案B 軽量化・偏重心是正の検証)
    if tank_mass is not None or tank_y is not None:
        tb = spec.body("tank")
        if tank_mass is not None:
            sc = tank_mass / URDF_TANK_MASS
            tb.mass = tank_mass
            tb.fullinertia = [v * sc for v in URDF_TANK_I]
        if tank_y is not None:
            p = list(tb.pos); p[1] += (tank_y - URDF_TANK_Y); tb.pos = p

    # (b) 可動域(pitch は脚グループで鏡像化 / yaw ±45°)＋ armature(反映回転子慣性)
    #     URDF由来関節は armature=0 で、軽量な車輪/足首＋剛接触だと数値発散するため付与。
    j2f = {v: k for k, v in PITCH_JOINTS.items()}
    for j in spec.joints:
        if j.name in j2f:
            j.range = list(pitch_range_of(j2f[j.name])); j.armature = 0.01
        elif j.name in YAW_JOINTS:
            j.range = list(YAW_RANGE); j.armature = 0.01

    # ベースに freejoint(浮遊ベース = 動力学検証に必須)。トップレベル body(root)に付与。
    spec.body("root").add_freejoint()

    # (c) URDF由来メッシュ geom は視覚専用(衝突しない)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = 0; g.conaffinity = 0

    # (d) 各足先に 足首サス(slide, 鉛直軸) + 車輪(hinge, 球接触) を追加。衝突はホイールのみ。
    for f in FEET:
        tip = tips[f]
        vax = vaxis[f] / (np.linalg.norm(vaxis[f]) + 1e-9)   # 立位で world 鉛直になる局所軸
        fb = spec.body(f)
        ankle = fb.add_body(name=f"ankle_{f}", pos=list(tip))
        if ankle_stiff:                                       # None なら剛(サス無し)
            aj = ankle.add_joint(name=f"ankle_j_{f}", type=mujoco.mjtJoint.mjJNT_SLIDE)
            aj.axis = list(vax); aj.range = [-0.020, 0.002]   # 荷重で沈む向き(鉛直下)
            aj.stiffness[0] = ankle_stiff; aj.damping[0] = ankle_damp
            aj.armature = 0.001
        ankle.mass = M_ANKLE; ankle.fullinertia = [1e-6, 1e-6, 1e-6, 0, 0, 0]
        # 車輪: 駆動 hinge(軸は脚横方向)＋球接触(半径0.015=径3cm)。球は姿勢に依らず
        # クリーンに接地する(シリンダのリム縁当たり=不安定を回避)。駆動は hinge で担保。
        wheel_axle = np.cross(vax, [1, 0, 0]); n = np.linalg.norm(wheel_axle)
        wheel_axle = wheel_axle / n if n > 1e-6 else np.array([0, 1, 0])
        wb = ankle.add_body(name=f"wheel_{f}", pos=[0, 0, 0])
        wj = wb.add_joint(name=f"wheel_j_{f}", type=mujoco.mjtJoint.mjJNT_HINGE)
        wj.axis = list(wheel_axle); wj.damping[0] = 0.01; wj.armature = 0.003
        wb.mass = M_WHEEL; wb.fullinertia = [2e-6, 3e-6, 2e-6, 0, 0, 0]
        wg = wb.add_geom(name=f"wheelg_{f}", type=mujoco.mjtGeom.mjGEOM_SPHERE)
        wg.size = [WHEEL_R, 0, 0]
        wg.density = 0                                       # 質量は body.mass(0.016)で与える
        wg.contype = 1; wg.conaffinity = 1
        wg.friction = [1.6, 0.02, 0.001]
        wg.solref = [0.01, 1]; wg.solimp = [0.9, 0.98, 0.001, 0.5, 2]
        wg.rgba = [0.15, 0.15, 0.18, 1]

    # (e) アクチュエータ(位置: yaw/pitch, 速度: wheel)。ctrlrange を可動域に同期。
    for f in FEET:
        pj = PITCH_JOINTS[f]
        a = spec.add_actuator(name=f"pitchpos_{f}", target=pj,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 60
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[1] = -60; a.biasprm[2] = -3
        a.ctrlrange = list(pitch_range_of(f)); a.forcerange = [-20, 20]
    for yj in YAW_JOINTS:
        a = spec.add_actuator(name=f"yawpos_{yj}", target=yj, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 20
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[1] = -20; a.biasprm[2] = -1
        a.ctrlrange = list(YAW_RANGE); a.forcerange = [-4, 4]
    for f in FEET:
        a = spec.add_actuator(name=f"wheeldrv_{f}", target=f"wheel_j_{f}",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = 1.5
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.biasprm[2] = -1.5
        a.ctrlrange = [-60, 60]; a.forcerange = [-1.5, 1.5]

    # (f) 環境: floor + step(3cm)
    wb = spec.worldbody
    fl = wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE)
    fl.size = [0, 0, 0.05]; fl.contype = 1; fl.conaffinity = 1
    fl.friction = [1.2, 0.02, 0.001]; fl.rgba = [0.35, 0.4, 0.45, 1]
    st = wb.add_geom(name="step", type=mujoco.mjtGeom.mjGEOM_BOX)
    st.pos = [STEP_X0 + 2.5, 0, STEP_H / 2]; st.size = [2.5, 0.6, STEP_H / 2]
    st.contype = 1; st.conaffinity = 1; st.friction = [1.2, 0.02, 0.001]
    st.rgba = [0.30, 0.34, 0.40, 1]

    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def make_stand(P=P_STAND, yaw_splay=0.0, ankle_stiff=2500, settle=1500, leg_scale=1.0,
               tank_mass=None, tank_y=None):
    """立位で静定させたモデルを返す (m, d, A)。
    P: 立位 pitch 振り(大=高い/前後広・左右狭)。yaw_splay: ヨー開脚(トレッド調整)。
    脚は鏡像実装のため pitch は PITCH_SIGN で、yaw も左右対称に符号付けして開脚する。"""
    m, d = build(ankle_stiff=ankle_stiff, leg_scale=leg_scale,
                 tank_mass=tank_mass, tank_y=tank_y)
    A = {m.actuator(i).name: i for i in range(m.nu)}
    qadr = {f: m.joint(PITCH_JOINTS[f]).qposadr[0] for f in FEET}
    yadr = {f: m.joint(YAW_OF[f]).qposadr[0] for f in FEET}
    fadr = m.jnt_qposadr[[j for j in range(m.njnt)
                          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
    ysign = {"feet": +1, "feet_2": +1, "feet_1": -1, "feet_3": -1}   # 左右外向き
    d.qpos[fadr:fadr + 7] = [0, 0, 0.2, 1, 0, 0, 0]
    for f in FEET:
        d.qpos[qadr[f]] = PITCH_SIGN[f] * P
        d.qpos[yadr[f]] = ysign[f] * yaw_splay
    mujoco.mj_forward(m, d)
    wbz = min(d.body(f"wheel_{f}").xpos[2] - WHEEL_R for f in FEET)
    d.qpos[fadr + 2] = 0.2 - wbz - 0.001                 # 車輪を床にほぼ接地
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        for f in FEET:
            d.ctrl[A[f"pitchpos_{f}"]] = PITCH_SIGN[f] * P
            d.ctrl[A[f"yawpos_{YAW_OF[f]}"]] = ysign[f] * yaw_splay
        mujoco.mj_step(m, d)
    return m, d, A


def _axis_to_quat(axis):
    """局所 z 軸を axis に向ける quat(シリンダの軸合わせ)。"""
    z = np.array([0.0, 0, 1]); a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-9)
    v = np.cross(z, a); c = float(np.dot(z, a))
    if np.linalg.norm(v) < 1e-8:
        return [1, 0, 0, 0] if c > 0 else [0, 1, 0, 0]
    q = np.array([1 + c, *v]); return list(q / np.linalg.norm(q))


def main(robot=DEFAULT_ROBOT):
    set_robot(robot)
    m, d = build()
    print("ロボット: %s  (%s)" % (ROBOT_NAME, URDF_SRC))
    print("URDF 直読みモデル: nbody=%d njnt=%d nu=%d nmesh=%d ngeom=%d" %
          (m.nbody, m.njnt, m.nu, m.nmesh, m.ngeom))
    print("総質量 = %.4f kg (URDF実値 目標 1.6065)" % sum(m.body_mass))
    print("freejoint: %s" % any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                                for j in range(m.njnt)))
    for f in FEET:
        j = m.joint(PITCH_JOINTS[f])
        print("  pitch %-7s range=%s deg=%s" % (f, np.round(j.range, 3),
                                                np.round(np.degrees(j.range), 1)))
    print("  衝突geom:", [m.geom(i).name for i in range(m.ngeom)
                         if m.geom(i).contype[0] == 1])
    # 立位検証
    import numpy as _np
    m2, d2, A = make_stand(P=1.0)
    fadr = m2.jnt_qposadr[[j for j in range(m2.njnt)
                           if m2.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
    w, x, y, z = d2.qpos[fadr + 3:fadr + 7]
    tilt = _np.degrees(_np.arccos(_np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    print("立位(P=1.0): 車体高=%.3f m  傾き=%.2f°  %s" %
          (d2.body("body").xpos[2], tilt, "★立位OK★" if tilt < 8 else "転倒"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="URDF直読み4脚モデルの生成・検証")
    add_robot_arg(ap)
    main(ap.parse_args().robot)
