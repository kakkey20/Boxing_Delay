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

## 次やること
- Boxingの学習を行って、Tutorial1のように報酬の違いを確認してみる（下にRewardとあるのでそこに記載）　今やっている
- チュートリアルで必要な機能がどれかを考えてみる


## 参考文献
- [深層強化学習のパッケージ調査](https://qiita.com/s-inoue-git/items/edafea0bca155ce1e7a6)
- [stable-baselines3のgithub](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3のチュートリアル](https://github.com/araffin/rl-tutorial-jnrr19)
- [stable-baselines3のちゃんとした辞書的な](https://stable-baselines3.readthedocs.io/en/master/index.html)
- 

## Tutorial
- Tutorial1
  -- CartPoleの学習コード
  
- Tutorial2
  -- ああああ



## Reward
学習回数 評価（モデルのプレイ回数） →  Reward（平均報酬）
1        1               　　　　→ -43.00 +/- 0.00
100      10

