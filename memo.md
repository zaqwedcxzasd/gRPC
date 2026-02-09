
1. 全体概要

目的

本システムでは

3D点群データに対する自己教師あり学習モデル（Point-MAE）

を用いて
	•	点群表現の事前学習（Pretrain）
	•	パーツセグメンテーション（Parts Segmentation）の下流タスク（Fine-tune）
	•	推論時に各点のパーツ分類を実施

を行う。

⸻

処理の位置づけ

点群データ (N×3 または N×4)
      ↓
前処理（正規化・サンプリング）
      ↓
グルーピング（FPS + KNN）
      ↓
Masked Autoencoder (Point-MAE)
      ↓
特徴量表現
      ↓
セグメンテーションヘッド（Fine-tune後）
      ↓
各点のパーツラベル推定


⸻

2. モデル基本情報

項目	内容
モデル名	Point-MAE
論文	Masked Autoencoders for Point Cloud (ECCV 2022)
フレームワーク	PyTorch
学習方式	自己教師あり（Masked Reconstruction）
下流タスク	3D点群パーツセグメンテーション
推論内容	各点のパーツラベル推定


⸻

3. 入力仕様

入力データ（Pretrain）
	•	3D点群
	•	形式：
N × 3 (xyz)

⸻

代表設定

pretrain.yaml
	•	npoints: 1024
	•	group_size: 32
	•	num_group: 64

⸻

前処理

コードから推測：
	•	FPS（Furthest Point Sampling）
	•	KNNグルーピング
	•	点群正規化
（datasets/data_transforms.py）

⸻

入力データ（Fine-tune）
	•	点群＋ラベル情報

N × 4
(x, y, z, label)

	•	label：各点のパーツラベル（点単位教師あり）

⸻

入力データ（推論時）
	•	点群データのみ

N × 3
(x, y, z)

	•	推論時は教師ラベルを使用しない

⸻

4. 出力仕様

Pretrain

出力
	•	再構成された点群

損失
	•	Chamfer Distance（cdl2）
	•	EMD（拡張）

⸻

Fine-tune

出力
	•	各点に対するパーツクラス確率

N × クラス数

（点ごとのクラス確率）

⸻

最終推論出力
	•	softmax後に argmax を適用

各点のパーツラベル


⸻

損失関数（Fine-tune時）
	•	Negative Log Likelihood Loss（NLL Loss）

F.nll_loss(pred, target)

	•	最終層で log_softmax を使用しているため
実質 Cross Entropy Loss と同等

⸻

5. 使用データセット

cfg と datasets から確認

Pretrain
	•	ShapeNet-55
cfgs/dataset_configs/ShapeNet-55.yaml

⸻

Fine-tune
	•	ModelNet40
	•	ScanObjectNN

datasets/ModelNetDataset.py
datasets/ScanObjectNNDataset.py

⸻

6. 学習設定（Pretrain）

cfgs/pretrain.yaml
	•	Optimizer: AdamW
	•	lr: 0.001
	•	weight_decay: 0.05
	•	Epoch: 300
	•	Scheduler: Cosine LR
	•	mask_ratio: 0.6
	•	depth: 12
	•	trans_dim: 384
	•	num_heads: 6

⸻

7. 学習設定（Fine-tune）

cfgs/finetune_modelnet.yaml
	•	lr: 0.0005
	•	Epoch: 300
	•	batch size: 32
	•	npoints: 1024
	•	grad clip: 10

⸻

8. モデル構成（重要）

Encoder
	•	Point grouping（FPS + KNN）
	•	Transformer encoder
	•	depth: 12

⸻

Decoder
	•	depth: 4

⸻

マスク方式
	•	mask_ratio: 0.6
	•	mask_type: rand

⸻

9. 依存モジュール

必須
	•	PyTorch
	•	CUDA

⸻

拡張

extensions/chamfer_dist
extensions/emd

※ビルド必要

⸻

10. 実行に必要なファイル
	•	cfgs/
	•	datasets/
	•	models/
	•	extensions/

⸻

11. 実行手順

事前学習

python main.py --cfg cfgs/pretrain.yaml


⸻

ファインチューニング

python main.py --cfg cfgs/finetune_modelnet.yaml


⸻

推論処理

推論の基本フロー
	1.	点群データ読み込み
	2.	学習時と同一の前処理（正規化・サンプリング）を適用
	3.	学習済み重みをロード
	4.	モデルに入力
	5.	各点のクラス確率を出力
	6.	argmax により最終ラベルを決定

⸻

推論時の処理パイプライン

point cloud
 → sampling
 → grouping
 → encoder
 → segmentation head
 → log_softmax / softmax
 → label (argmax)


⸻

推論時の注意点
	•	学習時と同じ前処理を適用する必要がある
	•	npoints を一致させる必要がある
	•	正規化方法が異なると精度が大きく低下する
	•	座標スケールが異なる場合、性能が不安定になる

⸻

12. 計算資源
	•	GPU 必須
	•	Transformer 12層
	•	点群1024

→ VRAM 12GB以上推奨

⸻

13. 既知の技術的注意点

CUDA拡張のビルドが必要
	•	chamfer
	•	emd_ext

⸻

データパス外部依存
	•	ShapeNet
	•	ModelNet

⸻

再現性
	•	seed設定はcfgに見当たらない
→ 再現性保証は弱い可能性

⸻

14. 下流タスク（Parts Segmentation）の損失関数

使用している損失
	•	Negative Log Likelihood Loss（NLL Loss）

F.nll_loss(pred, target)


⸻

モデル出力との関係
	•	最終層で log_softmax を適用しているため
Cross Entropy Loss と等価

⸻

損失の意味
	•	各点に対してパーツクラスを分類
	•	点単位の多クラス分類問題

⸻

注意点
	•	クラス重みは設定されていない
	•	出現頻度の低いパーツの精度が低下する可能性
	•	損失変更時は log_softmax との整合性に注意

⸻

15. 推論運用上の注意

精度劣化要因
	•	点群密度の変化
	•	スキャンノイズ
	•	未学習カテゴリ

⸻

実運用時のチェック項目
	•	入力点数
	•	座標スケール
	•	座標系（右手・左手）
	•	正規化範囲

⸻

