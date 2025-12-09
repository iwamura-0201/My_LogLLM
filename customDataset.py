import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re
from torch.utils.data import Sampler
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# patterns = [
#     r'[a-zA-Z0-9]*:*([/\\]+[^/\\\s]+)+[/\\]*',  # ファイルパス
#     r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # 数字を含む文字列
#     # r'[a-zA-Z0-9]+\.[a-zA-Z0-9]+',
# ]
#
# # すべてのパターンを結合
# combined_pattern = '|'.join(patterns)
#
# # 置換関数
# def replace_patterns(text):
#     return re.sub(combined_pattern, '<*>', text)

patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?', #  IP
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',   # Special MAC
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',   # MAC
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # File Path
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # word have number
]

# すべてのパターンを結合
combined_pattern = '|'.join(patterns)

# 置換関数
def replace_patterns(text):
    """
    ログメッセージの変数部分を<*>に置換する
    
    IPアドレス、ファイルパス、数値などの変数部分をマスクすることで、
    ログテンプレートの抽出を容易にする。
    
    Args:
        text: ログメッセージ文字列
    
    Returns:
        str: 変数部分が<*>に置換されたログメッセージ
    """
    text = re.sub(r'[\.]{3,}', '.. ', text)    # Replace multiple '.' with '.. '
    text = re.sub(combined_pattern, '<*>', text)
    return text


class CustomDataset(Dataset):
    """
    ログシーケンスデータセット
    
    CSVファイルからログシーケンスを読み込み、変数部分をマスク（<*>）に置換する。
    
    CSVフォーマット:
        - Content: ';-;'区切りのログメッセージシーケンス
        - Label: 0=正常、1=異常
    """
    def __init__(self, file_path, drop_duplicates=False):
        """
        データセットの初期化
        
        Args:
            file_path: CSVファイルのパス
            drop_duplicates: 重複するContentを削除するかどうか
        """
        df = pd.read_csv(file_path)
        print('Number of normal samples in original dataset: {}'.format((df['Label'].values==0).sum()))
        print('Number of anomalous samples in original dataset: {}'.format((df['Label'].values==1).sum()))
        # Contentの正規化処理。Drain済みを用いるので今回は不要（12/8時点）
        #df['Content'] = df['Content'].apply(replace_patterns)
        if drop_duplicates:
            df = df.drop_duplicates(subset='Content', keep='first')
        contents = df['Content'].values
        self.sequences = np.array([content.split(' ;-; ') for content in contents], dtype=object)
        self.labels = df['Label'].values
        if drop_duplicates:
            print('Number of normal samples after dropping duplicates: {}'.format((self.labels==0).sum()))
            print('Number of anomalous samples after dropping duplicates: {}'.format((self.labels==1).sum()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels

def merge_data(data):
    merged_data = []

    # 各サブリストの開始位置を記録
    start_positions = []

    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)

    return merged_data, start_positions


class BalancedSampler(Sampler):
    """
    クラス不均衡を調整するサンプラー
    
    少数クラス（異常）をオーバーサンプリングし、多数クラス（正常）とのバランスを調整する。
    
    Args:
        dataset: CustomDatasetインスタンス
        target_ratio: 少数クラスの目標割合（例: 0.3で30%）
        max_samples: 最大サンプル数（Noneの場合は制限なし）
        min_samples: 最小サンプル数（max_samplesがNoneの場合のみ有効）
    """
    def __init__(self, dataset, target_ratio=0.3, max_samples=None, min_samples=50000):
        """サンプラーの初期化"""
        self.labels = dataset.get_label()
        self.dataset = dataset
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples  # only if max_samples is None, min_samples can work

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomalous_indices = np.where(self.labels == 1)[0]

        self.minority_indices = (
            self.anomalous_indices if len(self.anomalous_indices) < len(self.normal_indices)
            else self.normal_indices
        )
        self.majority_indices = (
            self.normal_indices if self.minority_indices is self.anomalous_indices
            else self.anomalous_indices
        )

        self.minority_count = max(int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)), len(self.minority_indices))
        self.total_size = self.minority_count + len(self.majority_indices)

        if self.max_samples is not None:
            if self.max_samples > self.total_size:
                raise ValueError(
            f"The hyperparameter 'max_samples' should smaller than the samples in the dataset.")
            self.total_size = self.max_samples

        elif self.total_size < self.min_samples:
            self.total_size = self.min_samples


    def __iter__(self):
        oversampled_minority = np.tile(self.minority_indices, int(self.minority_count / len(self.minority_indices)))
        oversampled_minority_ = np.random.choice(
            self.minority_indices,
            self.minority_count - len(oversampled_minority),
            replace=False
        )
        combined_indices = np.concatenate([self.majority_indices, oversampled_minority, oversampled_minority_])
        if len(combined_indices) > self.total_size:
            combined_indices = np.random.choice(
                combined_indices,
                self.total_size,
                replace=False
            )
        else:
            combined_indices = np.tile(combined_indices, int(self.total_size/len(combined_indices)))
            combined_indices_ = np.random.choice(
                combined_indices,
                self.total_size-len(combined_indices),
                replace=False
            )
            combined_indices = np.concatenate([combined_indices, combined_indices_])
            np.random.shuffle(combined_indices)
        return iter(combined_indices)

    def __len__(self):
        return self.total_size


class CustomCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        sequences_, labels = zip(*batch)

        # 各シーケンスの長さを最大長で切り詰め
        sequences = [seq[:self.max_seq_len] for seq in sequences_]

        data, seq_positions = merge_data(sequences)
        seq_positions = seq_positions[1:]  # 最初の0位置を削除（後続の境界処理用）

        # 結合後のdataをエンコード
        inputs = self.tokenizer(
            data,
            return_tensors="pt",
            max_length=self.max_content_len,
            padding=True,
            truncation=True
        )

        # ラベルテンソルを構築
        # labels_tensor = torch.tensor(labels, dtype=torch.long)

        labels = np.array(labels).astype(object)

        labels[labels == 0] = 'normal'
        labels[labels == 1] = 'anomalous'

        return {
            "inputs": inputs,
            "seq_positions": torch.tensor(seq_positions, dtype=torch.long),
            # "labels": labels_tensor
            "labels": labels
        }
