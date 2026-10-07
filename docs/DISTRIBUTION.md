# Windows・Macの起動と配布

この文書はソース起動と配布手順です。配布設定を用意したことと、Windows実機やインストーラーの動作確認は区別します。実施した確認は作業結果の検証記録で確認してください。

## 開発環境

Python 3.12を使います。直接利用するパッケージは `requirements.txt`、配布用ツールは `requirements-build.txt` に分けています。PyTorchはセットアップでOSに応じた取得先を指定します。現在の固定バージョンは既存Apple Silicon環境で確認した構成を基準にしています。

### Windows x64

PowerShellでプロジェクトへ移動して実行します。

```powershell
.\setup_windows.ps1
Copy-Item config\models.example.json config\models.local.json
.\run_windows.ps1
```

例設定を実際のモデル保存先へ変更してからモデルを読み込みます。初期セットアップはCPU版 `torch==2.8.0` と `torchvision==0.23.0` を公式CPU indexから導入します。既存のCUDA版環境をこのセットアップで置換しないでください。

CUDAを使う場合は専用の環境を作り、PyTorch公式が示すそのバージョン用のCUDA wheelとGPUドライバーを確認します。起動には `NIKON_PYTHON` で専用Pythonを指定できます。CUDA版の例はPyTorch 2.8.0の公式一覧を基準にして選びます。[PyTorchのバージョン別導入](https://pytorch.org/get-started/previous-versions/)

### Apple Silicon Mac

```bash
bash setup_mac.sh
cp config/models.example.json config/models.local.json
bash run_mac.sh
```

このセットアップはPython 3.12・arm64を確認します。Intel Macは今回の固定パッケージセットの対象に含めず、利用可能なPyTorchとQtの組合せを別途検証します。既存の検証用環境を使ってソース起動を確認する場合は、環境を書き換えず、実行ファイルを指定できます。

```bash
NIKON_PYTHON="/path/to/verified/environment/bin/python" bash run_mac.sh
```

## 起動オプション

```bash
python -B -m app --config config/models.local.json --input /path/to/image.jpg --device cpu
```

`--input` は画像または動画、`--device` は `auto`、`cpu`、`cuda`、`mps` です。`--screenshot /path/to/preview.png` は初期画面の保存用です。スクリーンショット生成は提供モデルの推論成功を意味しません。

実モデルの読み込み、同じエンジンの直接呼出しとGUIワーカーの一致、GUIからの保存を確認する場合は `--verify-output` を使います。動画では全体フレーム数から解析間隔を決め、最大3フレームを処理します。確認後に終了し、JSON・結果画面・CSV・注釈画像を指定先へ保存します。この確認は正解ラベルとの精度測定とは別です。

```bash
python -B -m app --input /path/to/image.jpg --verify-output outputs/gui-verification
```

## 配布物の作成

最初はPyInstallerのonedir形式で、OSごとに作成します。Windows版はWindows、Mac版はMacでビルドします。Python自体が同梱されるため、配布先でPythonを別途導入する必要がありません。モデル重みと個人用設定はこのspecに同梱しません。[PyInstaller公式](https://pyinstaller.org/en/stable/operating-mode.html)

```bash
python -m pip install -r requirements-build.txt
python packaging/build.py
```

- Windows: `dist/NikonMockApp/NikonMockApp.exe`。
- Mac: `dist/NikonMockApp.app`。
- 共有可能な例設定を配置し、配布先で `config/models.local.json` とモデルを用意します。Windowsでは実行ファイルの親、Macでは `.app` の隣の `config/` を使います。Macの個人用設定・外部モデルを `.app` の中へ追加しないため、設定変更で署名対象のbundleを変更しません。絶対パスの外部モデル参照も利用できます。
- 起動、モデル読み込み、画像・動画、結果保存、日本語・スペースを含むパス、エラー表示を対象OSで確認します。
- Windowsコード署名、Mac署名・公証、インストーラー、自動更新は別の配布工程です。今回のspec作成だけで完了したとは扱いません。

Qt公式の `pyside6-deploy` はNuitkaを使う配布手段です。将来、起動時間やパッケージ内容を比較する際の候補にします。[Qt公式](https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html)

## SBOM

`scripts/generate_sbom.py` は追加パッケージなしで、実行中のPython環境のインストール済みパッケージ名・バージョンと指定ファイルのSHA-256をCycloneDX 1.5 JSONへ記録します。同じ環境・ファイルで並び順と内容を固定します。絶対パス、モデル内容、環境変数、取得URLは出力しません。

```bash
python scripts/generate_sbom.py --output outputs/development-sbom.cdx.json \
  --model detection=/path/to/nikon-detection-for-classificationv2.pt \
  --model classification=/path/to/best_resnet101_model_by_valacc.pth \
  --source app/inference.py
```

現在の開発環境の一覧は `development-python-environment` と記録します。専用のリリース環境から作る場合だけ `--scope release-python-environment` を指定します。環境に入っている未使用パッケージも含まれます。パッケージが宣言した単純なライセンスを記録し、曖昧なものやモデルのライセンスは `unconfirmed` とします。依存関係グラフはこの生成器の対象外です。

最終配布用SBOMでは、Windows CPU／Windows CUDA／Macの専用環境を分けて記録します。CycloneDX公式ツールで環境を取得・検証し、Syftで最終 `dist/` 内のバイナリ・ライブラリを補足します。生成器のPython一覧だけを最終配布物全体のSBOMと扱いません。

```bash
cyclonedx-py environment .venv --output-format JSON --output-file outputs/release-python.cdx.json
syft dir:dist -o cyclonedx-json=outputs/distribution.cdx.json
```

[CycloneDX Python公式](https://cyclonedx-bom-tool.readthedocs.io/en/latest/usage.html)、[Syft公式](https://oss.anchore.com/docs/guides/sbom/getting-started/)

配布時には、PySide6／QtのLGPLまたは商用ライセンスと、同梱した各依存ライブラリの条件・ライセンス文書を確認します。提供モデルの配布条件は確認情報が確定するまで記録を保留します。[Qt for Pythonのライセンス](https://doc.qt.io/qtforpython-6/licenses.html)
