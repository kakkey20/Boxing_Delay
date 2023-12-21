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
- stable-baselines3というものを見つけた。
- チュートリアルをすべて行い、どういうことができるかの整理を行った

## 現状
- やるべきことが何かをはっきりさせよう！って感じなう
- どうやって遅延を入れようかなーって感じ
- 人間がプレイできるかどうかを検証する

## やることリスト
- Boxingの学習を行って、Tutorial1のように報酬の違いを確認してみる（下にRewardとあるのでそこに記載）
- チュートリアルコードの詳細な解析
- チュートリアル1コードの詳細な解析
  - evaluate_policyというライブラリの解析
  - std_reward, mean_rewardとは
  - modelはPPO以外にどんなものがあるのか → 辞書に全部あるでーーー
  - MlpPolicyとは
- 参考文献の辞書的なやつの解析

## 次やること
- Boxingの学習を行って、Tutorial1のように報酬の違いを確認してみる（下にRewardとあるのでそこに記載）　今やっている

## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)

## Reward
| 学習回数 | 評価（モデルのプレイ回数）| std_reward +/- mean_reward|
| ---- | ---- | ---- |
| 1 | 1 | -43.00 +/- 0.00 |
| 100 |10|~~~|

