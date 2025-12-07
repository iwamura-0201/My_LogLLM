# LogLLM 自前ログデータセット適応ガイド

このガイドでは、LogLLMプロジェクトを自前のログデータセットに適応させるための手順を、各コードファイルごとに詳しく説明します。

---

## プロジェクト概要

LogLLMは、大規模言語モデル（LLM）を用いたログベースの異常検知システムです。主な特徴:

- **BERT**: ログメッセージの埋め込み表現を抽出
- **Llama-3**: ログシーケンスが正常か異常かを判定
- **Projector**: BERTとLlamaの埋め込み空間を接続
- **LoRA**: 効率的なファインチューニング

---

## ファイル構成と役割

### データ準備関連
- [prepareData/helper.py](file:///home/siwamura/LogLLM/prepareData/helper.py) - ログパース、ウィンドウ分割の基本機能
- [prepareData/sliding_window.py](file:///home/siwamura/LogLLM/prepareData/sliding_window.py) - BGL/Thunderbird/Liberty用データ準備
- [prepareData/session_window.py](file:///home/siwamura/LogLLM/prepareData/session_window.py) - HDFS用データ準備

### モデル関連
- [model.py](file:///home/siwamura/LogLLM/model.py) - LogLLMモデルの実装
- [customDataset.py](file:///home/siwamura/LogLLM/customDataset.py) - データセット、データローダー、前処理

### 学習・評価関連
- [train.py](file:///home/siwamura/LogLLM/train.py) - モデルの学習スクリプト
- [eval.py](file:///home/siwamura/LogLLM/eval.py) - モデルの評価スクリプト

---

## 適応手順

### ステップ1: データ形式の理解

LogLLMが期待するデータ形式は以下の通りです:

#### 最終的な学習・評価データ形式 (CSV)

学習データ（`train.csv`）と評価データ（`test.csv`）は以下のカラムを持つ必要があります:

| カラム名 | 説明 | 例 |
|---------|------|-----|
| `Content` | ログメッセージのシーケンス（`;-;`区切り） | `Error occurred ;-; Connection failed ;-; Retry attempt` |
| `Label` | 0=正常、1=異常 | `0` または `1` |
| `item_Label` | シーケンス内の各メッセージのラベルリスト | `[0, 1, 1]` |
| `session_length` | シーケンス内のメッセージ数 | `3` |

---

### ステップ2: データ準備スクリプトの選択と修正

自前のログデータセットの特性に応じて、適切なデータ準備スクリプトを選択・修正します。

#### オプション A: Fixed Size Window方式（BGL/Thunderbird/Liberty型）

**適用対象**: ブロックIDやセッションIDが無く、時系列順にログが並んでいる場合

**使用ファイル**: [prepareData/sliding_window.py](file:///home/siwamura/LogLLM/prepareData/sliding_window.py)

##### 修正箇所

```python
# ===== 必須設定 =====
data_dir = r'/path/to/your/log/directory'  # ログファイルがあるディレクトリ
log_name = "your_log_file.log"  # ログファイル名

# ===== ログ形式の定義 =====
# 自前のログ形式に合わせて log_format を定義
# 例: "2024-01-01 12:00:00 ERROR Component: Message"
log_format = '<Date> <Time> <Level> <Component>: <Content>'

# ===== ウィンドウサイズ設定 =====
window_size = 100  # 1シーケンスに含めるログメッセージ数
step_size = 100    # ウィンドウをずらすステップ数（100なら重複なし）

# ===== ログの範囲指定（オプション） =====
start_line = 0      # 読み込み開始行（大規模ファイルの場合）
end_line = None     # 読み込み終了行（Noneで全行）
```

##### ログ形式のパターン

`log_format`の各要素は`<>`で囲み、ログファイルの構造に合わせて定義します:

- `<Date>`: 日付部分
- `<Time>`: 時刻部分
- `<Level>`: ログレベル（ERROR、INFO等）
- `<Component>`: コンポーネント名
- `<Content>`: ログメッセージ本文（**必須**）
- `<Label>`: ラベル（異常の場合`-`以外、正常の場合`-`）

**重要**: 
- `<Content>`は必須です
- `<Label>`も含める必要があります（ログファイルにラベルが含まれている場合）
- ラベルが別ファイルの場合は、後でスクリプトを修正する必要があります

---

#### オプション B: Session Window方式（HDFS型）

**適用対象**: ブロックIDやセッションIDでログをグルーピングできる場合

**使用ファイル**: [prepareData/session_window.py](file:///home/siwamura/LogLLM/prepareData/session_window.py)

##### 修正箇所

```python
# ===== 必須設定 =====
data_dir = r'/path/to/your/log/directory'
log_name = "your_log_file.log"

# ===== ログ形式の定義 =====
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
```

##### セッションID抽出ロジックの修正

HDFS用スクリプトは、ログメッセージから`blk_XXX`形式のブロックIDを抽出します。自前のログで異なるID形式を使用する場合、以下の部分を修正します:

```python
# 元のコード（HDFS用）
blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])

# 修正例: セッションID "session_12345" を抽出
sessionId_list = re.findall(r'(session_\d+)', row['Content'])
```

##### ラベルファイルの準備

セッション方式では、別途ラベルファイル（`anomaly_label.csv`）が必要です:

```csv
BlockId,Label
session_001,Normal
session_002,Anomaly
session_003,Normal
```

スクリプト内のラベル読み込み部分を修正:

```python
blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
blk_df = pd.read_csv(blk_label_file)
for _, row in tqdm(blk_df.iterrows(), total=len(blk_df)):
    blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0
```

---

### ステップ3: helper.py の理解（修正不要の場合が多い）

[prepareData/helper.py](file:///home/siwamura/LogLLM/prepareData/helper.py) は基本的なユーティリティ関数を提供します。通常は修正不要ですが、理解しておくと便利です。

#### 主要関数

##### `structure_log(input_dir, output_dir, log_name, log_format, start_line, end_line)`

生ログファイルを構造化CSVに変換します。

**入力**: 生ログファイル  
**出力**: `{log_name}_structured.csv`（各行が1ログメッセージ）

##### `fixedSize_window(raw_data, window_size, step_size)`

固定サイズウィンドウでログシーケンスを作成します。

**パラメータ**:
- `raw_data`: `Content`と`Label`カラムを持つDataFrame
- `window_size`: ウィンドウサイズ（例: 100）
- `step_size`: スライドステップ（例: 100で重複なし、50で50%重複）

**出力**: ウィンドウ化されたDataFrame

##### `sliding_window(raw_data, para)`

時系列ベースのスライディングウィンドウ（時間ベース）。

**パラメータ**:
- `raw_data`: `timestamp`、`Label`、`deltaT`、`Content`カラムを持つDataFrame
- `para`: `{"window_size": 300, "step_size": 60}` (秒単位)

**注意**: 現在のコードではコメントアウトされており、Fixed Sizeが使用されています。

---

### ステップ4: データ前処理のカスタマイズ

[customDataset.py](file:///home/siwamura/LogLLM/customDataset.py) では、ログメッセージの変数部分をマスク（`<*>`に置換）します。

#### マスクパターンのカスタマイズ

自前のログに合わせて、`patterns`リストを編集します:

```python
patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|...|billion)\b',  # 数値の英単語
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',  # IPアドレス
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',  # MACアドレス
    r'[a-zA-Z0-9]*[:\.]*([\\/]+[^\\/\s\[\]]+)+[\\/]*',  # ファイルパス
    r'\b[0-9a-fA-F]{8}\b',  # 16進数（8桁）
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # 数字を含む文字列
]
```

**追加例**: タイムスタンプをマスク

```python
patterns.append(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')  # ISO 8601形式
```

#### データセット読み込みの確認

`CustomDataset`クラスは以下の形式のCSVを期待します:

```python
df = pd.read_csv(file_path)
# 必須カラム: 'Content', 'Label'
# Content: ';-;'区切りのログメッセージ
# Label: 0または1
```

**重要**: データ準備スクリプトで生成したCSVがこの形式に従っていることを確認してください。

---

### ステップ5: モデル設定の調整

[model.py](file:///home/siwamura/LogLLM/model.py) は通常、修正不要です。ただし、以下の点を理解しておきましょう:

#### LogLLMモデルのアーキテクチャ

1. **BERT**: 各ログメッセージを768次元の埋め込みに変換
2. **Projector**: 768次元 → 4096次元（Llamaの埋め込み空間）に射影
3. **Llama-3**: ログシーケンスの埋め込みを受け取り、"normal"/"anomalous"を生成

#### LoRA設定

- **BERT LoRA**: `r=4`, `lora_alpha=32`, `lora_dropout=0.01`
- **Llama LoRA**: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `target_modules=["q_proj", "v_proj"]`

**変更が必要な場合**: メモリ不足の場合、`r`を小さくするか、`load_in_4bit`を維持します。

---

### ステップ6: 学習スクリプトの設定

[train.py](file:///home/siwamura/LogLLM/train.py) で以下のパラメータを自前のデータセットに合わせて設定します。

#### 必須設定

```python
# ===== データセット設定 =====
dataset_name = 'YourDatasetName'  # データセット名
data_path = r'/path/to/your/dataset/train.csv'  # 学習データパス

# ===== モデルパス =====
Bert_path = r"/path/to/bert-base-uncased"
Llama_path = r"/path/to/Meta-Llama-3-8B"

# ===== データパラメータ =====
max_content_len = 100  # 各ログメッセージの最大トークン数
max_seq_len = 128      # シーケンス内の最大ログメッセージ数

# ===== 学習パラメータ =====
batch_size = 16
micro_batch_size = 4  # GPUメモリに応じて調整
min_less_portion = 0.3  # 少数クラス（異常）の最小割合
```

#### エポック数と学習率

LogLLMは4段階の学習プロセスを採用:

1. **Phase 1**: Llamaのみを学習（`n_epochs_1`, `lr_1`）
2. **Phase 2-1**: Projectorのみを学習（`n_epochs_2_1`, `lr_2_1`）
3. **Phase 2-2**: ProjectorとBERTを学習（`n_epochs_2_2`, `lr_2_2`）
4. **Phase 3**: 全体をファインチューニング（`n_epochs_3`, `lr_3`）

```python
n_epochs_1 = 1
n_epochs_2_1 = 1
n_epochs_2_2 = 1
n_epochs_3 = 2

lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
```

**調整のヒント**: 
- データセットが小さい場合、エポック数を増やす
- 過学習が見られる場合、学習率を下げる、またはエポック数を減らす

#### クラス不均衡の調整

`BalancedSampler`は異常と正常のバランスを調整します:

```python
min_less_portion = 0.3  # 異常ログの割合を30%に調整
```

自前データセットの異常率に応じて調整してください。

---

### ステップ7: 評価スクリプトの設定

[eval.py](file:///home/siwamura/LogLLM/eval.py) で以下を設定します。

```python
# ===== データセット設定 =====
dataset_name = 'YourDatasetName'
data_path = r'/path/to/your/dataset/test.csv'  # 評価データパス

# ===== モデルパス =====
Bert_path = r"/path/to/bert-base-uncased"
Llama_path = r"/path/to/Meta-Llama-3-8B"

# ===== ファインチューニング済みモデルパス =====
ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

# ===== パラメータ =====
max_content_len = 100
max_seq_len = 128
batch_size = 32  # 評価時は大きめでOK
```

#### 評価指標

`evalModel`関数は以下の指標を計算します:

- **Precision**: 異常と予測したもののうち、実際に異常だった割合
- **Recall**: 実際の異常のうち、正しく検出できた割合
- **F1 Score**: PrecisionとRecallの調和平均
- **Accuracy**: 全体の正解率

---

## 実行手順のまとめ

### 1. データ準備

```bash
# Option A: Fixed Sizeウィンドウ（BGL/Thunderbird/Liberty型）
python prepareData/sliding_window.py

# Option B: Sessionウィンドウ（HDFS型）
python prepareData/session_window.py
```

**出力**:
- `train.csv`: 学習データ
- `test.csv`: 評価データ
- `train_info.txt`: 学習データの統計情報
- `test_info.txt`: 評価データの統計情報

### 2. モデル学習

```bash
python train.py
```

**出力**:
- `ft_model_{dataset_name}/`: ファインチューニング済みモデル
  - `Llama_ft/`: Llamaのアダプタ
  - `Bert_ft/`: BERTのアダプタ
  - `projector.pt`: Projectorの重み

### 3. モデル評価

```bash
python eval.py
```

**出力**: コンソールに評価指標が表示されます。

---

## トラブルシューティング

### エラー: `KeyError: 'Label'`

**原因**: CSVに`Label`カラムが存在しない

**解決策**: 
1. ログファイルにラベルが含まれているか確認
2. `log_format`に`<Label>`を含める
3. または、ラベルを別途追加する後処理を実装

### エラー: `CUDA out of memory`

**原因**: GPUメモリ不足

**解決策**:
1. `micro_batch_size`を小さくする（例: 4 → 2）
2. `max_content_len`や`max_seq_len`を小さくする
3. `load_in_4bit=True`が有効か確認

### 警告: `Number of anomalous samples: 0`

**原因**: ラベルが正しく抽出されていない

**解決策**:
1. ログファイルのラベル列を確認（異常の場合`-`以外）
2. `df["Label"] = df["Label"].apply(lambda x: int(x != "-"))`が正しく動作しているか確認

### 学習時にLossが減少しない

**原因**: 学習率が不適切、またはデータの前処理に問題がある

**解決策**:
1. 学習率を調整（例: `5e-4` → `1e-4`）
2. `replace_patterns`のマスクパターンを見直す
3. データの品質を確認（ノイズが多すぎないか、ラベルが正確か）

---

## チェックリスト

### データ準備
- [ ] 自前のログファイルを準備
- [ ] ログ形式（`log_format`）を定義
- [ ] `prepareData/sliding_window.py`または`session_window.py`を修正
- [ ] データ準備スクリプトを実行
- [ ] `train.csv`と`test.csv`が正しく生成されたか確認
- [ ] `train_info.txt`でデータ統計を確認

### データ前処理
- [ ] `customDataset.py`のマスクパターンを確認・カスタマイズ
- [ ] サンプルデータで前処理をテスト

### モデル学習
- [ ] BERT、Llamaモデルをダウンロード
- [ ] `train.py`のパラメータを設定
- [ ] 学習を実行
- [ ] ファインチューニング済みモデルが保存されたか確認

### モデル評価
- [ ] `eval.py`のパラメータを設定
- [ ] 評価を実行
- [ ] 評価指標（Precision、Recall、F1）を確認

---

## 参考情報

### データセット形式の例

#### train.csv

```csv
Content,Label,item_Label,session_length
Error occurred ;-; Connection timeout ;-; Retry failed,1,"[0, 1, 1]",3
Service started ;-; Request received ;-; Response sent,0,"[0, 0, 0]",3
```

#### 重要なポイント

1. **区切り文字**: ログメッセージは` ;-; `（空白含む）で区切る
2. **ラベル**: 0=正常、1=異常
3. **item_Label**: 各メッセージのラベルリスト（文字列形式のリスト）

### 推奨データセットサイズ

- **学習データ**: 最低10,000シーケンス、推奨50,000以上
- **評価データ**: 学習データの20%程度
- **異常率**: 1%〜20%程度（BalancedSamplerで調整可能）

### GPU要件

- **推奨**: NVIDIA GPU（16GB VRAM以上）
- **最小**: NVIDIA GPU（8GB VRAM、micro_batch_size=1-2）
- **4bit量子化**: 有効（`load_in_4bit=True`）

---

## まとめ

LogLLMを自前のログデータセットに適応させる手順:

1. **ログ形式を定義**し、データ準備スクリプトを修正
2. **データ準備スクリプトを実行**して`train.csv`、`test.csv`を生成
3. **マスクパターンをカスタマイズ**（必要に応じて）
4. **学習パラメータを設定**し、`train.py`を実行
5. **評価を実行**し、性能を確認

各ステップで生成されるファイルと出力を確認しながら進めることで、問題を早期に発見できます。
