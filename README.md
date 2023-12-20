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
