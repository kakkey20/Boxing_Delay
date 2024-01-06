# Boxing_Delay
Open AI GymのBoxingにおいて、AIに遅延を与えることで難易度調整ができるか検証する

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

## Pettingzooで使えそうな機能
- delay_observations_v0(env, delay)(https://pettingzoo.farama.org/api/wrappers/supersuit_wrappers/)
- AgileRL(DQNの学習、セーブ、ロードが可能）→(https://pettingzoo.farama.org/tutorials/agilerl/DQN/)

## 今後の予定
1. WindowsでPoetry環境を作成する　→　無理だた（Cmake、multialeagent.pyがうまくいかない）、2をLinuxでまず頑張る
2. Pettingzooで、Boxingの遅延付き学習コードを作成する
3. 対戦相手の指定ができるコードを作成する
4. 人間と対戦できるコードを作成する
5. 


## 参考文献
- [Pettingzoo.github](https://github.com/Farama-Foundation/PettingZoo)
- [Pettingzoo チュートリアル1](https://note.com/npaka/n/n9b9074b8f916)
- [Pettingzoo チュートリアル2](https://note.com/npaka/n/n06d8ba36d5bc)
- [Pettingzoo Documentation](https://pettingzoo.farama.org/index.html)
- [GPU1](https://zenn.dev/danchinocto/scraps/da90c7e70ec77d)
- [GPU2](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

## Reward
|使用モデル|使用Policy| 学習回数(step) | mean_reward| cpu or gpu| 遅延 |
| ---- | ---- | ---- | ---- | ---- | ----|
| PPO | CnnPolicy | 1000000  | -0.97 | cpu | なし |
| A2C | CnnPolicy | 3000000  | ??? | gpu | あり（0F） |
| A2C | CnnPolicy | 1000000  | ??? | gpu | あり（0F） | イマココ
| A2C | CnnPolicy | 1000000  | ??? | gpu | あり（0F） |
| A2C | CnnPolicy | 1000000  | -0.95 | gpu | あり（0F） |
| A2C | CnnPolicy | 1000000  | -1.29 | gpu | あり（3F） |
| A2C | CnnPolicy | 1000000  | -3.97 | gpu | あり（7F） |
| A2C | CnnPolicy | 1000000  | -100 | cpu | あり（10F） |

