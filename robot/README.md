# robot/ — ロボットモデル置き場

Onshape からエクスポートしたロボットを、1体につき1ディレクトリで管理する。
テスト用に何体でも増やしてよい(`test_robot`, `tachikoma_v1`, ... )。

## ディレクトリ構成

```
robot/
└── <robot_name>/           # Onshape の URDF export そのままの構成
    ├── urdf/
    │   └── <robot_name>.urdf
    ├── meshes/
    │   ├── body.gltf
    │   └── ...
    └── launch/
        └── <robot_name>.launch
```

URDF 内のメッシュ参照は `package://<robot_name>/meshes/xxx.gltf` 形式。
ディレクトリ名 = パッケージ名 = URDF の `<robot name="...">` を揃えておくと、
ROS からもシミュレータからもパス解決が崩れない。

## Git 管理ルール

リポジトリ直下の [.gitattributes](../.gitattributes) で振り分けている。

| 種別 | 例 | 保存先 |
|---|---|---|
| メッシュ / CAD 成果物 | `.gltf` `.glb` `.stl` `.dae` `.obj` `.step` | **Git LFS** |
| ロボット記述 | `.urdf` `.xacro` `.sdf` `.launch` `.xml` `.yaml` | 通常の Git(テキスト差分あり) |

`.gltf` はテキスト(JSON)だがバッファが base64 埋め込みで実質バイナリのため LFS 側に入れている。

### 新しいロボットを追加するとき

```bash
# robot/<robot_name>/ に Onshape の export を展開してから
git add robot/<robot_name>
git lfs status          # meshes が (LFS: ...) になっているか確認
git commit -m "add <robot_name>"
```

### クローン直後 / 別マシンでのセットアップ

```bash
git lfs install
git lfs pull            # メッシュの実体を取得(未取得だとポインタ文字列のまま)
```

メッシュを開いて `version https://git-lfs.github.com/spec/v1` という3行のテキストだった場合は
`git lfs pull` を忘れている。
