# Boxing_Delay

Open AI GymのBoxingにおいて、AIに遅延を与えることで難易度調整ができるか検証する

## 使用環境（暫定であり、この環境が正しいかどうかの確証はない）

- python=3.10.*


## Gymnasiumのインストール
```
pip install gymnasium
pip install gymnasium[atari] → atariのインストール、ゲームによっては[box2D]等
pip install gymnasium[accept-rom-license] → romのインストール
```

## 学習方法
```
python boxingdqn.py
```

## 今までのあらすじ
- stable-baselines3というものを見つけた
- チュートリアルのコードをまとめた
- チュートリアル1の解析を行った
- 参考文献の辞書的なやつをチラリと見た

## 現状
- チュートリアルそれぞれの解析を行い中
- どうやって遅延を入れようかなー
- 人間がプレイできるかどうかを検証してーなー
- DQNがなんか動かないから、なんとかしてーなー（PPOだと動く）

## やることリスト
- チュートリアル4の詳細な解析
- チュートリアル5の詳細な解析
- Modelのセーブ方法、ロード方法
- Modelの動画保存方法
- DQNだとうまく動かないんご（金曜に試してみる）
- Gpuただしくつかえているか確認（金曜）→ 詳しくは辞書のInstallation

## 次やること
- チュートリアル4の詳細な解析
- Boxingじゃなくていいから、CartPoleかなんかで、必要な機能を実際に実装する

## チュートリアルから読み取れるBoxingAIに必要な機能
- Tutorial1
  - モデル選び（PPOやれDQNやら、ポリシーやら）
  - evaluate関数（どうやってモデルを評価するか）
  - Video関数（録画方法、一旦放置いずれやる）
- Tutorial2
  - モデルの保存方法、ロード方法
  - カスタム環境ラッパーの作成（行動の正規化や、時間制限機能の追加） → これを利用して遅延を入れることができるかもs
- Tutorial3　→　マルチプロセス関連なのでとりあえず必要なし
- Tutorial4
  - ハイパーパラメータの調整（実際にはやってないが、少し詳細あり）
  - コールバック1（悪い学習をしないように、セーブ機能をつけれる）→ いらないかも
  - コールバック2（パフォーマンスのリアルタイムプロット）→ いるかも
  - コールバック3（トレーニングの進捗具合、1秒あたりのタイムステップ数、学習の残り時間）→ いるかも
  - コールバック4（最適なモデルを自動保存 and 進捗具合とエピソード報酬の表示）→ いるかも
- Tutorial5
  - 新しい環境（env）を作って、stablebaselineを使えるようにする的なやつ、いらないです

## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)

## Reward
|使用モデル|使用Policy| 学習回数 | 評価（モデルのプレイ回数）| std_reward +/- mean_reward|
| ---- | ---- | ---- | ---- | ---- |
| PPO | MlpPolicy | 1 | 1 | -43.00 +/- 0.00 |
| PPO | MlpPolicy | 100 |10|~~~|
| DQN | MlpPolicy | 100 |10|~~~|

