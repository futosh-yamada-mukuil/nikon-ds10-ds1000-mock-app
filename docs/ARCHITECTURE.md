# モックアプリの技術構成

対象はWindows x64を主とするローカルアプリです。MacはまずApple Siliconで検証します。Pythonで提供された物体検知・細胞分類モデルを実行し、参考動画の画面配置と操作をPySide6で再構成します。

## 採用構成と理由

**Python 3.12 + PySide6 / Qt Widgets + PyTorch + RF-DETR + ResNet101** を採用します。既存検証アプリの推論処理を再利用し、画面と処理を分けます。サーバー、Web認証、データベースは今回のファイル入力中心のモックには追加しません。

| 候補 | 利点 | 今回の判断 |
| --- | --- | --- |
| PySide6 / Qt Widgets | Pythonモデルを直接利用。画像、表、設定、ダイアログを同じコードでWindowsとMacに配置できる | 採用。既存検証アプリからの変更範囲が小さい |
| PySide6 / Qt Quick | QMLで動的な画面やアニメーションを構築できる | タッチ・アニメーション中心の要件が増えた際の候補 |
| Electron + Python | HTML/CSSで柔軟に画面を再現できる | Chromium・NodeとPythonの配布、プロセス間通信が増える |
| Tauri + Python | OSのWebViewを使い、Pythonを外部実行ファイルとして利用できる | Rust・Web・Pythonの管理とOS/CPU別sidecarが必要 |
| Web / FastAPI / Streamlit | ブラウザー経由の共有やPython検証画面に向く | ローカルファイル・動画を扱うデスクトップ配布では追加設計が必要 |

これは今回の既存コードと要件に基づく選択です。Qt公式はWidgetsを複雑なデスクトップ画面、Quickを動的・タッチ中心の画面向けと整理しています。[QtのUI比較](https://doc.qt.io/qtforpython-6/overviews/qtdoc-topics-ui.html)、[Electronのプロセス構成](https://www.electronjs.org/docs/latest/tutorial/process-model)、[TauriのPython sidecar](https://v2.tauri.app/develop/sidecar/)、[Streamlitの構成](https://docs.streamlit.io/develop/concepts/architecture/architecture)

## 責務

- 画面は入力選択、動画再生、プレビュー、設定、結果表示、保存を担当します。
- 推論エンジンは提供モデルの確認、前処理、頭部検出、クロップ、分類を担当します。
- 重いモデル読み込み・推論はワーカーで実行し、進捗・結果・エラーを画面へ返します。GUI部品はGUIスレッドで更新します。
- モデル設定は `config/models.local.json` でローカルファイルを参照します。モデル本体はGitへ追加しません。
- CLIや精度測定でも同じ推論エンジンを呼び、画面経由との条件差を調べられる構成にします。

処理モードは「物体検知＋分類」「物体検知のみ」「分類のみ」を分けます。分類のみは入力が細胞頭部のクロップであることを前提にします。全視野画像の1枚分類を個々の細胞分類として集計しません。

## モデルと実行デバイス

今回の提供ファイルはRF-DETR物体検知モデル1点とResNet101分類モデル1点です。ファイルのSHA-256を実装内の提供済みモデル情報と照合し、外部設定の記述だけで任意のチェックポイントを信用しません。モデルの来歴と分類クラスの意味は [MODELS.md](MODELS.md) を参照します。

自動選択はCUDA、MPS、CPUの順です。Windowsの既定セットアップはCPU版です。NVIDIA GPUを使うWindowsでは、対応ドライバーとCUDA版PyTorchを別環境で確認します。MacはMPSが利用可能なら選択できます。GPU利用可能であっても、速度と数値の一致は実際の提供モデルで確認します。[PyTorch CUDA](https://docs.pytorch.org/docs/2.8/cuda.html)、[PyTorch MPS](https://docs.pytorch.org/docs/2.8/notes/mps.html)

## 実装範囲

このモックは、提供モデルによるファイル入力と検出・分類を起点にします。動画のフレーム検出と、同一細胞の追跡・運動指標の算出は別の機能です。今回のモデル1点だけで専用の動体追跡モデルが提供されたとは扱いません。

PoC第1フェーズ／明視野というモデル区分、顕微鏡SDK、専用DLL、SAM、外部制御の詳細は今回の提供ファイルだけでは確定していません。追加モデルやインターフェースが確定した段階で実装を広げます。未接続・未実装の画面要素で検証済み結果を示しません。

配布設計とSBOMの確認範囲は [DISTRIBUTION.md](DISTRIBUTION.md)、精度比較の条件は [DATASET_GUIDELINES.md](DATASET_GUIDELINES.md) に記録します。
