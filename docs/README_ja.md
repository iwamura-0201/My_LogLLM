# LogLLM - ログ異常検知システム

大規模言語モデル（Llama-3）とBERTを使用したログベースの異常検知システム

---

## 📚 ドキュメント

- **[環境構築ガイド](docs/environment_setup.md)** - uv仮想環境のセットアップ手順
- **[適応ガイド](docs/adaptation_guide.md)** - 自前のログデータセットへの適応方法

---

## 🚀 クイックスタート

### 1. 環境構築

```bash
# uvで仮想環境を作成
uv venv

# 仮想環境を有効化
source .venv/bin/activate

# 依存関係をインストール（詳細はdocs/environment_setup.mdを参照）
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.46.3 datasets==3.1.0 peft==0.13.2 accelerate==1.0.1 bitsandbytes==0.45.3 safetensors==0.5.3
uv pip install scikit-learn==1.3.2
```

### 2. データ準備

自前のログデータセットを準備します。詳細は[適応ガイド](docs/adaptation_guide.md)を参照してください。

```bash
# BGL/Thunderbird/Liberty型（時系列順ログ）
python prepareData/sliding_window.py

# または HDFS型（セッションID付きログ）
python prepareData/session_window.py
```

### 3. モデル学習

```bash
python train.py
```

### 4. モデル評価

```bash
python eval.py
```

---

## 📁 プロジェクト構成

```
LogLLM/
├── docs/                          # ドキュメント
│   ├── environment_setup.md       # 環境構築ガイド
│   └── adaptation_guide.md        # 適応ガイド
├── prepareData/                   # データ準備スクリプト
│   ├── helper.py                  # ログパース、ウィンドウ分割の基本機能
│   ├── sliding_window.py          # Fixed Size Window方式（BGL/Thunderbird/Liberty用）
│   └── session_window.py          # Session Window方式（HDFS用）
├── model.py                       # LogLLMモデルの実装
├── customDataset.py               # データセット、データローダー、前処理
├── train.py                       # 学習スクリプト
├── eval.py                        # 評価スクリプト
├── requirements.txt               # 依存関係（conda形式）
└── README.md                      # このファイル
```

---

## 🎯 モデルアーキテクチャ

LogLLMは3つの主要コンポーネントで構成されています:

1. **BERT (bert-base-uncased)** - 各ログメッセージを768次元の埋め込みに変換
2. **Projector (Linear Layer)** - BERTの埋め込み（768次元）をLlamaの埋め込み空間（4096次元）に射影
3. **Llama-3 (Meta-Llama-3-8B)** - ログシーケンスを受け取り、"normal"/"anomalous"を生成

### 4段階学習プロセス

1. **Phase 1**: Llamaのみを学習
2. **Phase 2-1**: Projectorのみを学習
3. **Phase 2-2**: ProjectorとBERTを学習
4. **Phase 3**: 全体をファインチューニング

### 効率的なファインチューニング

- **LoRA (Low-Rank Adaptation)**: パラメータ効率的なファインチューニング
- **4bit量子化**: メモリ使用量を削減

---

## 📊 データ形式

### 入力データ（CSV）

学習・評価データは以下の形式が必要です:

| カラム名 | 説明 | 例 |
|---------|------|-----|
| Content | `;-;`区切りのログメッセージシーケンス | `Error occurred ;-; Connection failed` |
| Label | 0=正常、1=異常 | `0` または `1` |
| item_Label | 各メッセージのラベルリスト | `[0, 1]` |
| session_length | シーケンス内のメッセージ数 | `2` |

詳細は[適応ガイド](docs/adaptation_guide.md)を参照してください。

---

## ⚙️ 設定

### train.py の主要パラメータ

```python
dataset_name = 'YourDatasetName'
data_path = r'/path/to/your/dataset/train.csv'
Bert_path = r"/path/to/bert-base-uncased"
Llama_path = r"/path/to/Meta-Llama-3-8B"

max_content_len = 100  # 各ログメッセージの最大トークン数
max_seq_len = 128      # シーケンス内の最大ログメッセージ数
batch_size = 16
micro_batch_size = 4
```

### eval.py の主要パラメータ

```python
dataset_name = 'YourDatasetName'
data_path = r'/path/to/your/dataset/test.csv'
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
```

---

## 🛠️ システム要件

- **Python**: 3.8以上
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU推奨（16GB VRAM以上）
- **OS**: Linux (Ubuntu等)

---

## 📖 引用

このプロジェクトを使用する場合は、以下を引用してください:

```
@article{logllm2024,
  title={LogLLM: Log-based Anomaly Detection Using Large Language Models},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 📝 ライセンス

このプロジェクトのライセンスについては、元のリポジトリを参照してください。

---

## 🙏 謝辞

- Hugging Face Transformers
- Meta Llama-3
- PEFT (Parameter-Efficient Fine-Tuning)
- LogHub (ログデータセット)

---

## 📧 サポート

質問や問題がある場合は、[適応ガイド](docs/adaptation_guide.md)のトラブルシューティングセクションを参照してください。
