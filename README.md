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
- チュートリアルの解析を行った
- 参考文献の辞書的なやつを見た
- モデルの保存、ロードができた
- （仮）のboxing.pyを作った
- 使用可能なモデルの整理を行った
- GPU使えてないことが判明
- 学習回数 → step数しか不可能であった

## 現状
- どうやって遅延を入れるかわからん
- 人間がプレイできるかどうかわからん（一旦放置）
- Boxingで、最適なモデルってどれやろう（PPO or A2C、Policyはどれなのか）
- そも学習がうまく行ってない気がする

## やることリスト
- コールバック2の問題解決（なくてもいいからいったん放置）
- Modelの動画保存方法（なくてもいいからいったん放置）
- Gpu使えるようにする → Nvidia Dockerの設定（月曜日）
- ハイパーパラメータの調整（一旦放置）
- 自分のx座標、y座標、敵のx座標、y座標がわかるかどうか
- 難易度設定方法

## 次やること
- 環境にどのような情報があるか（進捗報告用に）→ modelのコード確認しながら
　　→　action, rewared, obs
- Rewardをもう少し用意して、スライドに追加する（進捗報告ように）

## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)

## 使用可能モデル
- PPO
- A2C

## 使用不可能モデル
- DQN（メモリが足りない）
- SAC（行動空間がDiscreteじゃないから）
- TD3（行動空間がDiscreteじゃないから）

## モデル
|使用モデル|使用Policy|
| ---- | ---- |
| PPO | MlpPolicy |
| PPO | CnnPolicy |
| PPO | MultiInputPolicy |
| A2C | MlpPolicy |
| A2C | CnnPolicy |
| A2C | MultiInputPolicy |
| DQN | MlpPolicy |
| DQN | CnnPolicy |
| DQN | MultiInputPolicy |

## Reward
|使用モデル|使用Policy| 学習回数(step) | 評価のプレイ回数| mean_reward +/- std_reward| cpu or gpu| 遅延 |
| ---- | ---- | ---- | ---- | ---- | ---- | ----|
| PPO | MlpPolicy | 1 | 1 | -43.00 +/- 0.00 | cpu | なし |
| A2C | MlpPolicy | 100 | 10 | -30.00 +/- 0.00 | cpu | なし |
| A2C | MlpPolicy | 10000 | 10 | -41.00 | cpu | なし |
| A2C | CnnPolicy | 10000 | 10 | -41.00 | cpu | なし |
| PPO | CnnPolicy | 10000 | 10 | -3.5 | cpu | なし |
| PPO | CnnPolicy | 1000000 | 100 | -0.97 | cpu | なし |

## 遅延を入れるうえで必要なこと
- 学習の対戦相手は誰なのか
- observation, reward等の解析 → evaluateで行けるんじゃね
- mean_rewardや、std_rewardがどうなれば良いのか
- ?フレーム遅らせた位置情報をどうやって、envに渡すかどうかの方法

## チュートリアルから読み取れるBoxingAIに必要な機能
- Tutorial1
  - モデル選び（PPOやれDQNやら、ポリシーやら）
  - evaluate関数（どうやってモデルを評価するか）
  - Video関数（録画方法、一旦放置いずれやる）
- Tutorial2
  - モデルの保存方法、ロード方法
  - カスタム環境ラッパーの作成（行動の正規化や、時間制限機能の追加） → これを利用して遅延を入れることができるかも
- Tutorial3　→　マルチプロセス関連なのでとりあえず必要なし
- Tutorial4
  - ハイパーパラメータの調整（実際にはやってないが、少し詳細あり）
  - コールバック1（悪い学習をしないように、セーブ機能をつけれる）→ いらないかも SaveOnBestTrainingrewardCallback
  - コールバック2（パフォーマンスのリアルタイムプロット）→ いるかも PlottingCallback
  - コールバック3（トレーニングの進捗具合、1秒あたりのタイムステップ数、学習の残り時間）→ いるかも ProgressBarCallback
  - コールバック4（最適なモデルを自動保存 and 進捗具合とエピソード報酬の表示）→ いるかも
- Tutorial5 → 新しい環境（env）を作って、stablebaselineを使えるようにする的なやつ、いらないです

##  ミスした可能性のあるやーつ
| PPO | MlpPolicy | 100 |100|-60.16 +/- 8.75|
| PPO | CnnPolicy | 100 |100|-25 +/- 0|
| PPO | MultiInputPolicy | 100 |100|??? +/- ???|

| PPO | MlpPolicy | 1000 |10|-35.41 +/- 7.27|

| PPO | MlpPolicy | 100000 |10|-57.50 +/- 4.56|
| PPO | MlpPolicy | 100000 |100|-32.95 +/- 5.69|
| PPO | CnnPolicy | 100000 |100|-80.06 +/- 12.50|
| A2C | MlpPolicy | 100000 |100|-39.70 +/- 0.52|
