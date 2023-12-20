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

## 現状
- stable-baselines3というものを見つけた！学習は実際にできた！やったね！
- チュートリアル1を行った
- チュートリアル2を行った
- やるべきことが何かをはっきりさせよう！って感じなう

## やることリスト
- チュートリアルを試してみる
- Boxingの学習を行って、Tutorial1のように報酬の違いを理解してみる
- 人間がプレイできるかどうかを検証する

## 次やること
- チュートリアル2をやる → Chatgptで要約
- チュートリアル3をやる → Chatgptで要約
- チュートリアル4をやる → Chatgptで要約
- チュートリアル5をやる → Chatgptで要約


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
学習回数 よくわかんないやつ →  Reward
1        1               → -43.00 +/- 0.00


