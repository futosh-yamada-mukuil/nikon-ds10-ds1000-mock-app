# モデルを変更しないローカル比較

改善候補の測定用CLIです。GUIの既定経路・モデル構成は変更しません。
今回の入力例はすべて**架空の合成データ**です。実画像の精度・改善原因は未確認です。
再学習や新たな補正モデルの作成は行いません。

現在のEngine・比較CLIの分類スコアは **0〜1**、既定しきい値は検出・分類とも **0.5** です。
分類は未丸めの `score > threshold`、検出は `confidence >= threshold`。
0〜100の古い測定CSVは、評価用JSONに `score_unit: "0_to_100"` と明示した場合だけ100で割ります。
評価CLIのしきい値引数も0〜1です。0.5は50%を意味し、数値から単位を推測しません。

以下はプロジェクトルートから実行します。`$NIKON_PYTHON`には使用するPythonの絶対パスを指定してください。
従来の指定パス `/Users/futoshi/mukuil/nikon-sperm-classification/.venv-mac/bin/python` は
今回の作業環境では見つからず、実行確認待ちです。既存の結果ファイルは上書きしません。

## 同一ROIの分類比較

CSV列は `image,cell_id,x1,y1,x2,y2`。画像パスはimage-root配下、座標は整数・右下排他的です。
画像境界でクリップ後、幅・高さ4px以上を必要とします。同じ画像とcell_idの重複は拒否します。
全入力を検証してから分類モデルだけをSHA256検証付きで読み込みます。

```bash
"$NIKON_PYTHON" -B scripts/compare_preprocessing.py \
  --input-csv /path/to/rois.csv --image-root /path/to/images \
  --classifier /path/to/classifier.pth --device cpu --jpeg-quality 83 \
  --output-csv outputs/roi_comparison.csv --output-json outputs/roi_comparison.json
```

同一の元ROIを、直接224×224／640×640→メモリ内JPEG→224×224で比較します。
どちらもBILINEAR・同一Engineの正規化と分類器を使用します。JPEG品質83は例示値で、学習時品質は未確認です。
CSVは元座標・クリップ後座標・2スコア・差分・しきい値超過を、JSONは画像／モデルSHA256・デバイス・ライブラリ版も保存します。
新規の正解ラベルは作成しません。

## 分割検出の比較

```bash
"$NIKON_PYTHON" -B scripts/compare_detection.py \
  --image /path/to/image.png --detector /path/to/detector.pt --classifier /path/to/classifier.pth \
  --device cpu --tile-size 640 --overlap 0.25 --nms-iou 0.5 \
  --output-csv outputs/detection_comparison.csv --output-json outputs/detection_comparison.json
```

既存Engineの読込を使うためモデル2件の指定が必要です。分類推論は実行しません。
標準640縮小と、重なり25%の640pxタイル検出を比較します。モデル構成784を維持します。
タイル内クリップ→元画像座標復元→別タイル間NMS（IoU > 0.5）の順です。
同じタイル内の検出枠はこのNMSで削除しません。
検出枠と件数・経過時間を保存します。初回処理の影響があるため速度比較は参考値です。
密集する別細胞が抑制される可能性もあり、件数増加を精度向上とは扱いません。

## 機種・段階別の分類集計

次の例は**架空の合成データ**で、SHA256も架空値です。
`truth=1` は確認された評価対象の正例、`truth=0` は負例、空欄は未評価です。
`score` はclass 1のスコアです。医学的意味はこのツールで確認しません。

```csv
machine,stage,image_id,cell_id,truth,score
DS10,L0,synthetic-image,synthetic-1,1,90
DS10,L0,synthetic-image,synthetic-2,0,10
DS10,L0,synthetic-image,synthetic-3,,60
```

```json
{
  "schema_version": 1,
  "score_unit": "0_to_100",
  "groups": [{
    "machine": "DS10", "stage": "L0",
    "model_sha256": {"classifier": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    "input_path": "synthetic/direct224",
    "positive_class_mapping_verified": false,
    "positive_class_index": null
  }]
}
```

```bash
"$NIKON_PYTHON" -B scripts/evaluate_stages.py \
  --input-csv /path/to/measurements.csv --metadata /path/to/conditions.json \
  --classification-threshold 0.5 \
  --output-csv outputs/stages.csv --output-markdown outputs/stages.md
```

DS10/DS1000×L0〜L5を独立集計します。L0〜L5の意味は入力経路で明示し、自動で推定しません。
各組にモデルSHA256・入力経路・クラス対応確認状態を指定してください。同一組に異なる条件の測定は混在させません。
同一組・画像・セルの重複は拒否し、異なる段階の同じセルは許可します。
モデルハッシュは入力の申告値であり、集計時にモデルを再読込しません。

Precision/Recall/F1/Accuracy/Balanced Accuracy/ROC AUCとTP/FP/FN/TN、件数・除外数を保存します。
truth欠損は負例扱いせず除外。未確認クラス対応は性能指標・混同行列を「未算出」とし、ラベル付き件数だけ記録します。
確認済みの場合は `positive_class_index` に0または1を指定します。0の場合はclass 1の判定結果とAUCの向きを反転します。
片クラスではBalanced AccuracyとROC AUC、分母0では該当指標を未算出とし、理由を併記します。
AUCは同点を平均順位で扱います。CSVは未丸め、Markdown表示は小数6桁です。
検出漏れや検出枠の正解との一対一照合は別評価が必要で、この分類表には含みません。

## 検証・採用とSBOM

合成テストの実行コマンド:

```bash
"$NIKON_PYTHON" -B -m unittest tests.test_compare_preprocessing tests.test_local_measurements tests.test_evaluate_stages tests.test_source_sbom -v
```

正解データ確定後に調整用と最終評価用を分離し、機種別F1・Balanced Accuracy、見逃し・誤検出、処理時間を比較します。
学習使用区分・クラス対応が不明な状態で最終精度や改善の原因を確定しません。

SBOMは `scripts/update_source_sbom.py --sbom <既存JSON> --source <変更ソース>` を使用します。
`--source` は複数指定可能です。標準 `file_component` で明示したソースのみ更新し、変更直前のモデル・依存・対象外ソースレコードとメタデータを完全に保持します。
モデル再読込・依存環境の再収集はありません。

今回のソース更新対象（現時点ではPython環境待ちのため未実行）:

```bash
"$NIKON_PYTHON" -B scripts/update_source_sbom.py --sbom sbom/development-macos-arm64.cdx.json \
  --source app/tiled_detection.py --source scripts/compare_preprocessing.py \
  --source scripts/compare_detection.py --source scripts/evaluate_stages.py \
  --source scripts/measurement_io.py --source scripts/update_source_sbom.py
```

手法の参考: [SAHI論文](https://arxiv.org/abs/2202.06934)、[TorchvisionのResNet101](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.resnet101.html)、[しきい値の調整と評価](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html)。
一般公開モデルの前処理を、今回の提供モデルの学習条件と同一視しません。
