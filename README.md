# ニコン DS10 / DS1000 検出・細胞分類モックアプリ

Python + PySide6（Qt Widgets）で作成した、Windows主体・Mac対応のローカルアプリです。
参考動画をもとに左の設定パネル・右の大きなプレビューを構成し、提供されたRF-DETRとResNet101で実際に推論します。追加依頼に合わせて、下部の処理情報・進捗バー・結果表・ログは非表示にしています。

## 実装した機能

- 画像／動画ファイルの読み込み、動画プレビュー再生、フレーム位置の移動。
- 物体検知＋細胞分類、物体検知のみ、分類のみ（頭部クロップ画像全体）の切り替え。
- 検出・分類しきい値の設定、赤枠と件数の表示。プレビュー再生の下に「物体検知した数」「分類しきい値を超えた数」をカードで表示します。画像上のID・スコア文字は表示しません。
- ワーカーでのモデル読み込みと解析、動画の一時停止・再開・停止。
- ズーム、ドラッグ移動、グレースケール・明るさの表示調整。
- 結果画像、CSV、入力・モデルSHA256、解析条件、環境、動画の解析履歴の保存。
- 指定モデルの内容照合と、読み込み・解析失敗のエラー表示。

初期値は検出0.50・分類0.50です。2026年10月8日の本人回答により、分類スコア・しきい値も0〜1に統一しました（0.50 = 50%相当、0.44 = 44%相当）。最適なしきい値や精度を確定した値ではありません。
設定変更は次回の解析に適用します。標準の分類スコアはclass1のsoftmax確率で、学習ラベルの意味は未確認です。医学的なDFI実測割合ではありません。新CSVは`classification_score_0_to_1`列、新JSONはschema_version 2で単位を明記します。旧CSV・JSON・予測キャッシュの0〜100値は変更せず、比較時だけ明示的に換算します。DS1000補正は既存の60%を0.60に換算し、係数・判定を維持します。
動画は指定間隔で解析します。1なら全フレーム、10なら10フレームごとで、未解析フレームの枠を補間しません。実時間処理性能は保証していません。

画像を開くと表示領域いっぱいに全体表示します。プレビュー上部の「＋」「−」またはCtrl／⌘を押しながら縦ホイールで拡大・縮小できます。「全体表示」で戻せます。通常のスクロールでは倍率を変えず、全体表示より小さく縮小しません。
分類しきい値を超えた候補を、文字のない赤枠だけで表示・画像保存します。物体検知のみのモードでは検出した枠を赤色で表示します。細胞ごとの数値はCSV・JSONで確認できます。
件数カードには実行時のしきい値を添え、未解析は「—」、解析済みで対象がなければ「0件」と表示します。動画の件数は表示中の解析フレームの結果です。物体検知のみ・分類のみの場合、行っていない処理の件数は「—」とし、未実施と説明します。

今回は本人回答により**ファイル入力を完成させる範囲**です。カメラ／専用DLL／SAM／同一細胞の追跡・運動指標は未実装です。

## このMacで起動する

ビルドした `dist/NikonMockApp.app` を開けば、Pythonを別に起動せず使えます。このMacのモデル設定は `.app` 隣接の `dist/config/models.local.json` に配置済みです。

```bash
open /Users/futoshi/mukuil/nikon-ds10-ds1000-mock-app/dist/NikonMockApp.app
```

ソースから起動する場合:

既存の検証用Python環境を変更せず流用できます。このMacの `config/models.local.json` は指定モデルの絶対パスを設定済みです。

```bash
cd /Users/futoshi/mukuil/nikon-ds10-ds1000-mock-app
NIKON_PYTHON=/Users/futoshi/mukuil/nikon-sperm-classification/.venv-mac/bin/python bash run_mac.sh
```

1. モデルの読み込み完了を待ちます。
2. 「画像 / 動画ファイルを選択」で実際の顕微鏡画像・動画を選び、機種・モード・しきい値を設定します。
3. 「処理開始」を押します。赤枠は分類しきい値を超えた候補です。
4. 左側の「検索結果CSVをダウンロード」を押します。ブラウザーでCSVのダウンロードを開始し、アプリに保存済みCSV・保存フォルダーへのリンクを表示します。

`通常モード.mp4` はUI参考の画面録画です。顕微鏡の元動画や精度評価の正解データとして扱いません。
今回のDownloads内の分類モデルは以前の同名モデルと重みが異なり、以前の精度集計をそのまま引き継ぎません。

## Windows・新しいMacへの導入

Python 3.12を用意し、プロジェクトルートで実行します。

Windows x64 / PowerShell:

```powershell
.\setup_windows.ps1
Copy-Item config\models.example.json config\models.local.json
.\run_windows.ps1
```

Apple Silicon Mac:

```bash
bash setup_mac.sh
cp config/models.example.json config/models.local.json
bash run_mac.sh
```

例設定は実際のモデル保存先へ変更してください。相対パスはプロジェクトルートから解決します。
WindowsはCPU版PyTorchを基本とし、CUDA環境は別に構成します。MacはApple Siliconを確認対象とし、Intel Macは別の依存セットの確認が必要です。
配布版はOSごとにPyInstallerでビルドします。詳細は [起動・配布手順](docs/DISTRIBUTION.md) を参照してください。

## 保存される結果

保存先にファイル名・日時付きの新しいフォルダーを作ります。原本・モデル・既存結果は上書きしません。
保存先はOSのダウンロードフォルダー配下の `NikonMockApp/` です。CSV・画像・実行条件を先にローカル保存し、ブラウザーにはCSVだけを渡します。127.0.0.1に限定した予測不能URLの一時サーバーは5分で終了し、フォルダー一覧や他のファイルは配信しません。ブラウザー側の保存先はブラウザー設定に従います。ブラウザーを開けない場合も、アプリのリンクから保存済みCSVを開けます。

| ファイル | 内容 |
| --- | --- |
| `annotated.png` | 最後に表示した解析フレーム。保存時の表示設定を反映 |
| `results.csv` | 表示フレームの細胞座標・未丸めスコア・しきい値超過判定 |
| `video_results.csv` | 動画の解析済みフレームの細胞結果 |
| `manifest.json` | 入力SHA256、機種、モード、設定、モデル、環境、全解析フレーム、終了理由 |

0件のフレームもJSONへ記録します。停止・失敗後の保存は処理済みフレームだけを対象とします。保存処理では正解比較・精度指標を算出しません。

## ソースと資料

- `app/`: 画面・ワーカー・共有推論エンジン・入力・保存。
- `config/`: モデル設定例。個人設定 `*.local.*` はGit対象外。
- `packaging/`: Windows/Mac別PyInstaller配布設定。
- `scripts/`: 読み取り専用レビュー・SBOM生成・実モデルGUI確認。
- `tests/`: 推論契約・画像/動画・GUI・停止/再開・保存・SBOMのテスト。
- `outputs/`, `build/`, `dist/`: ローカル結果・配布物。Git対象外。

資料:

- [技術構成の比較と採用理由](docs/ARCHITECTURE.md)
- [モデル構造・SHA256・未確定事項](docs/MODELS.md)
- [データセットガイドライン](docs/DATASET_GUIDELINES.md)
- [開発環境SBOM](sbom/development-macos-arm64.cdx.json)
- [今回の検証記録](docs/VALIDATION.md)

物体検知と細胞分類は同じアプリ内でモードを切り替え、共通エンジンを使います。PoC第1フェーズ／明視野の別モデルとの対応は未確定です。モデル本体はローカル参照で、Gitや配布アプリへ同梱していません。
開発環境SBOMは最終配布物全体のSBOMと区別します。

## 確認コマンド

```bash
QT_QPA_PLATFORM=offscreen NIKON_PYTHON=/Users/futoshi/mukuil/nikon-sperm-classification/.venv-mac/bin/python bash run_mac.sh --input /path/to/image.jpg --verify-output outputs/gui-verification
QT_QPA_PLATFORM=offscreen /path/to/python -B -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/review.sh
```

`--verify-output` は実モデルを読み、処理本体とGUIワーカーの最初のフレームを比較して保存・終了します。プログラムからのGUI操作確認で、精度評価やWindows実機確認の代わりではありません。
`--screenshot outputs/preview.png` は初期画面を保存して終了し、モデル推論は行いません。

## 開発運用

作業ルールの正本は [AGENTS.md](AGENTS.md) です。[背景](docs/PROJECT_CONTEXT.md)、[作業フロー](docs/WORKFLOW.md)、[依頼テンプレート](docs/TASK_TEMPLATE.md) を参照してください。
Gitは初回コミット前で、ソースは未追跡のままです。未追跡ファイルは通常の `git diff` に内容が出ないため、一覧と個別ファイルを確認します。コミット・Push・公開は個別の明示指示がある場合だけ実施します。
GitHubのmasterへの初回Pushは本人指示で実施済みです。Dotsには今夜の依頼を送信して受付確認済みです。Slackへの送信は未実施です。[DOT_BRIEF.md](DOT_BRIEF.md) は独自の引き継ぎ資料で、Dotsの公式設定ファイルではありません。
夜間の利用枠監視・停止制御と今夜限定の予約は [夜間運用](docs/dots-nightly.md) を参照してください。
