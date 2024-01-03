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

## pettingzooのインストール
```
pyenv local 3.9.16
poetry env use 3.9.16
poetry run pip install pettingzoo[atari]
poetry run pip install stable-baseline3[extra]
poetry run pip install autorom
poetry run pip install autorom[accept-rom-license]
poetry run AutoROM
poetry run AutoROM --install-dir /path/to/install
poetry run pip install --find-links dist/ --no-cache-dir AutoROM[accept-rom-license]
poetry run python multi.py 
```

## 学習方法
```
python boxingdelay.py
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
- Pettingzooというものを見つけた
- Gpu使えるようにした
- pettingzooのチュートリアルを行った
- モデルはGoogleドライブで共有することにした
- Gymnasiumのモデルが、Peggingzooでロードすることができない

## やることリスト
- 任意の対戦相手と対戦できるコード作成
- 人間とプレイ可能にするコードの作成
- 難易度設定方法

## 次やること
- Gymnasiumのモデルが、Peggingzooでロードすることができないので、格闘する

## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)
- [Envについて](https://gymnasium.farama.org/api/env/)
- [Pettingzoo.github](https://github.com/Farama-Foundation/PettingZoo)
- [Pettingzoo チュートリアル1](https://note.com/npaka/n/n9b9074b8f916)
- [Pettingzoo Documentation](https://pettingzoo.farama.org/index.html)
- [GPU1](https://zenn.dev/danchinocto/scraps/da90c7e70ec77d)
- [GPU2](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

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
| A2C | CnnPolicy | 1000000 | 100 | -100 | cpu | あり（10F） |
| A2C | CnnPolicy | 1000000 | 100 | -0.95 | gpu | あり（0F） |
| A2C | CnnPolicy | 1000000 | 100 | -1.29 | gpu | あり（3F） |


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
