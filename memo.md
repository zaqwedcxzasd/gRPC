
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


---
---
---



Depth Anything V2 推論処理フロー

本システムでは Depth Anything V2 を用いて単眼画像から相対深度を推定する。推論処理は初期化フェーズと推論フェーズに分かれる。

⸻

1. 初期化フェーズ

1.1 モデル構築

DepthAnythingV2 は depth_anything_v2/dpt.py に定義されているクラスを生成して利用する。

# depth_anything_v2/dpt.py

class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
    ):
        super().__init__()

ここで指定する encoder によって使用する ViT の種類と中間特徴の取得位置が決定される。

⸻

1.2 エンコーダ設定

encoder には vits, vitb, vitl, vitg が存在し、本システムでは vitl を使用する。

# depth_anything_v2/dpt.py

intermediate_layer_idx = {
    'vits': [2, 5, 8, 11],
    'vitb': [2, 5, 8, 11],
    'vitl': [4, 11, 17, 23],
    'vitg': [9, 19, 29, 39]
}

vitl を指定した場合は ViT の 4 層分の中間特徴 [4, 11, 17, 23] が後段の深度推定に利用される。

⸻

1.3 バックボーン（DINOv2）の生成

初期化時に DINOv2 ベースの Vision Transformer が生成され、画像特徴抽出器として使用される。

# depth_anything_v2/dpt.py

self.pretrained = DINOv2(model_name=encoder)

このバックボーンは入力画像をパッチ分割したトークン列へ変換し、Transformer 内部の複数層の特徴を取得できる構造になっている。

⸻

1.4 深度推定ヘッド（DPTHead）の構築

Transformer の特徴から深度マップを生成するためのデコーダが初期化時に構築される。

# depth_anything_v2/dpt.py

self.depth_head = DPTHead(
    in_channels=embed_dim,
    features=features,
    out_channels=out_channels
)

このヘッドでは Transformer のトークン列を 2 次元特徴マップへ再配置し、異なる解像度の特徴を段階的に統合しながらアップサンプリングを行い、最終的に 1 チャネルの深度マップを出力する。

⸻

1.5 学習済み重みのロード

モデル生成後に外部で保存された学習済み重みを読み込むことで推論可能な状態にする。

model.load_state_dict(...)
model.eval()

eval() を設定することで Dropout や BatchNorm が推論モードで動作する。

⸻

2. 推論フェーズ

2.1 エントリポイント

推論は infer_image 関数を起点として実行される。

# depth_anything_v2/dpt.py

@torch.no_grad()
def infer_image(self, raw_image, input_size=518):

この関数は勾配計算を無効化した状態で処理を行う。

⸻

3. 入力前処理

3.1 image → tensor 変換

infer_image 内で image2tensor が呼び出され、入力画像がネットワーク入力形式へ変換される。

# depth_anything_v2/dpt.py

def image2tensor(self, raw_image, input_size=518):


⸻

3.2 処理内容

まず入力画像の元サイズが保存され、後段で出力を元解像度へ戻すために使用される。

h, w = raw_image.shape[:2]

次に OpenCV 形式の BGR 画像を RGB に変換し、画素値を 0〜1 に正規化する。

image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

その後 Resize が適用され、アスペクト比を維持したまま入力サイズを 518 基準へ調整しつつ、ViT のパッチサイズに合わせて 14 の倍数になるように整形される。

Resize(
    width=input_size,
    height=input_size,
    keep_aspect_ratio=True,
    ensure_multiple_of=14,
    resize_method="lower_bound"
)

続いて ImageNet 統計量による正規化が行われる。

NormalizeImage(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

その後テンソルへ変換され、バッチ次元が追加される。

torch.from_numpy(image).unsqueeze(0)

最後に CUDA、MPS、CPU の順で利用可能なデバイスへ転送される。

⸻

4. 推論本体（forward）

4.1 パッチサイズ計算

入力テンソルの空間サイズから ViT パッチサイズ 14 を用いてパッチ数が計算される。

patch_h = H // 14
patch_w = W // 14


⸻

4.2 DINOv2 による特徴抽出

バックボーンから指定された層の中間特徴が取得される。

# depth_anything_v2/dpt.py

features = self.pretrained.get_intermediate_layers(
    x,
    self.intermediate_layer_idx[self.encoder],
    return_class_token=True
)

ここで取得される特徴は Transformer のトークン表現であり、複数スケールの情報を含む。

⸻

4.3 DPTHead による深度推定

取得した複数層の特徴が DPTHead に入力され、2 次元特徴マップへ復元された後にマルチスケール融合を経て深度が推定される。

depth = self.depth_head(features, patch_h, patch_w)


⸻

4.4 活性化

出力された深度に ReLU が適用され、負の値が抑制される。

depth = F.relu(depth)


⸻

5. 出力後処理

5.1 元解像度へ補間

前処理で保存した元画像サイズへバイリニア補間でリサイズされる。

depth = F.interpolate(
    depth[:, None],
    (h, w),
    mode="bilinear",
    align_corners=True
)


⸻

5.2 numpy 変換

最終的な深度マップは CPU 上の numpy 配列へ変換されて返却される。

depth = depth.squeeze().cpu().numpy()


⸻

6. 最終出力

出力は入力画像と同じ解像度の 1 チャネルの相対深度マップであり、値は絶対距離ではなくスケール未定の深度表現となる。

⸻

7. 全体シーケンス

モデル生成
↓
DINOv2 backbone 構築
↓
DPTHead 構築
↓
学習済み重みロード
↓
eval 設定
↓
infer_image 呼び出し
↓
image2tensor 前処理
↓
DINOv2 中間特徴抽出
↓
DPTHead による深度生成
↓
ReLU
↓
元解像度へ補間
↓
numpy 深度マップ出力


⸻

次にどちらを作ると、引き継ぎ資料としてかなり強くなります：
	1.	「学習済み重みファイルごとの差分（vits / vitb / vitl の違い）」
	2.	「DepthAnythingV2 と ZoeDepth / MiDaS の“推論パイプライン構造差”」
	3.	「システム側の predict メソッドとの対応図（呼び出し階層図）」

実務的にはどれを README に追加予定ですか？
（大規模チームだと ③ 呼び出し階層図 が一番事故を減らします）
