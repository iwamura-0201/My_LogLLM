# uv仮想環境セットアップ完了

## セットアップ内容

LogLLMプロジェクト用のuv仮想環境を構築しました。

**仮想環境パス**: `/home/siwamura/LogLLM/.venv`

---

## インストール済みパッケージ

### コア機械学習フレームワーク

| パッケージ | バージョン | 説明 |
|-----------|---------|------|
| torch | 2.4.0+cu121 | PyTorch (CUDA 12.1対応) |
| torchvision | 0.19.0+cu121 | 画像処理ライブラリ |
| torchaudio | 2.4.0+cu121 | 音声処理ライブラリ |
| transformers | 4.46.3 | Hugging Face Transformers（BERT、Llama対応） |
| datasets | 3.1.0 | データセット管理ライブラリ |
| peft | 0.13.2 | Parameter-Efficient Fine-Tuning (LoRA等) |
| accelerate | 1.0.1 | 学習の高速化・分散処理 |
| bitsandbytes | 0.45.3 | 量子化ライブラリ（4bit/8bit） |

### データ処理・解析

| パッケージ | バージョン | 説明 |
|-----------|---------|------|
| numpy | 1.26.4 | 数値計算ライブラリ |
| pandas | 2.3.3 | データフレーム処理 |
| scikit-learn | 1.3.2 | 機械学習ライブラリ（評価指標等） |
| scipy | 1.16.3 | 科学計算ライブラリ |

### CUDA関連（NVIDIA）

- nvidia-cublas-cu12==12.1.3.1
- nvidia-cuda-cupti-cu12==12.1.105
- nvidia-cuda-nvrtc-cu12==12.1.105
- nvidia-cuda-runtime-cu12==12.1.105
- nvidia-cudnn-cu12==9.1.0.70
- nvidia-cufft-cu12==11.0.2.54
- nvidia-curand-cu12==10.3.2.106
- nvidia-cusolver-cu12==11.4.5.107
- nvidia-cusparse-cu12==12.1.0.106
- nvidia-nccl-cu12==2.20.5
- nvidia-nvjitlink-cu12==12.9.86
- nvidia-nvtx-cu12==12.1.105
- triton==3.0.0

### その他の依存関係

合計68パッケージがインストールされています。

---

## 仮想環境の使用方法

### 1. 仮想環境の有効化

```bash
cd /home/siwamura/LogLLM
source .venv/bin/activate
```

### 2. 有効化の確認

仮想環境が有効化されると、プロンプトに`(.venv)`が表示されます:

```
(.venv) user@host:~/LogLLM$
```

### 3. Pythonとパッケージの確認

```bash
# Pythonバージョン確認
python --version

# インストール済みパッケージ一覧
uv pip list

# 特定パッケージの確認
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4. 仮想環境の無効化

```bash
deactivate
```

---

## 学習・評価の実行

仮想環境を有効化した状態で、以下のコマンドを実行できます:

### データ準備

```bash
# Fixed Size Window方式
python prepareData/sliding_window.py

# または Session Window方式
python prepareData/session_window.py
```

### モデル学習

```bash
python train.py
```

### モデル評価

```bash
python eval.py
```

---

## トラブルシューティング

### CUDA が利用できない場合

```bash
# CUDA利用可能性チェック
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

CUDA 12.1が必要です。異なるCUDAバージョンの場合は、PyTorchを再インストールしてください。

### パッケージの追加インストール

```bash
source .venv/bin/activate
uv pip install <package-name>
```

### 仮想環境の再作成

```bash
rm -rf .venv
uv venv
source .venv/bin/activate
# 依存関係を再インストール
```

---

## システム要件

- **Python**: 3.8以上（現在の仮想環境: 3.11.13）
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU推奨（16GB VRAM以上）
- **OS**: Linux (Ubuntu等)

---

## 次のステップ

1. ✅ 仮想環境の構築完了
2. 📝 次: [adaptation_guide.md](file:///home/siwamura/.gemini/antigravity/brain/b19a3ba4-9eb4-41ef-b529-0a041aa383e1/adaptation_guide.md)を参照してデータ準備を開始
3. 🚀 自前のログデータセットでモデルを学習
