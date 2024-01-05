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
poetry run pip install stable-baselines3[extra]
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
- Gymnasiumのモデルが、Pettingzooでロードすることができない
- Pettingzooで学習を行うには、stable-baselines3を使用することができない（envが違うから）
- Pettingzooで学習をすること自体はできたので、いくつか必要な機能を調べる

## Pettingzooで入れなければならない機能
- モデルのセーブ
- モデルのロード
- 行動遅延の実装
- gpuでやるべきかどうか
- どうやって強くなったと判断するか
- 対戦相手の指定コード
- 

## やることリスト
- BOxingの学習コードに使えそうなサイト、コード等をPettingzoo Documentationから見つける
- 見つけて、Windowsとlinuxで同時並行で行う。どっちでも学習が可能かどうかを調べる。
- Windowsで可能な場合進捗報告までに、デスクトップPCで学習できるところまで行いたい。

## Pettingzooで使えそうな機能
- delay_observations_v0(env, delay)(https://pettingzoo.farama.org/api/wrappers/supersuit_wrappers/)
- AgileRL(DQNの学習、セーブ、ロードが可能）→(https://pettingzoo.farama.org/tutorials/agilerl/DQN/)

## 今後の予定
1. WindowsでPoetry環境を作成する　→　無理だた（Cmake、multialeagent.pyがうまくいかない）、2をLinuxでまず頑張る
2. Pettingzooで、Boxingの遅延付き学習コードを作成する
3. 対戦相手の指定ができるコードを作成する
4. 人間と対戦できるコードを作成する
5. 

## 次やること
- Gymnasiumのモデルが、Peggingzooでロードすることができないので、格闘する
- Gymnasiumと、pegginzooの違い（envやその他）を確認し、以下に記載する

## 現状の学習パラメータ一覧
- env-id→boxing_v2
- total-timesteps→12000
- learning-rate→2.5e-4
- --num-envs
- --num-steps

## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)
- [Envについて](https://gymnasium.farama.org/api/env/)
- [Pettingzoo.github](https://github.com/Farama-Foundation/PettingZoo)
- [Pettingzoo チュートリアル1](https://note.com/npaka/n/n9b9074b8f916)
- [Pettingzoo チュートリアル2](https://note.com/npaka/n/n06d8ba36d5bc)
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
| A2C | CnnPolicy | 1000000 | 100 | -3.97 | gpu | あり（7F） |

## Reward
|使用モデル|使用Policy| 学習回数(step) | 評価のプレイ回数| mean_reward| cpu or gpu| 遅延 |
| ---- | ---- | ---- | ---- | ---- | ---- | ----|
| PPO | CnnPolicy | 1000000 | 100 | -0.97 | cpu | なし |
| A2C | CnnPolicy | 1000000 | 100 | -0.95 | gpu | あり（0F） |
| A2C | CnnPolicy | 1000000 | 100 | -1.29 | gpu | あり（3F） |
| A2C | CnnPolicy | 1000000 | 100 | -3.97 | gpu | あり（7F） |
| A2C | CnnPolicy | 1000000 | 100 | -100 | cpu | あり（10F） |


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

