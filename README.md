# Boxing_Delay
Open AI GymのBoxingにおいて、AIに遅延を与えることで難易度調整ができるか検証する

## 方向性1（Pettingzooで、行動遅延を入れた上で対戦相手を指定した学習が行えること）
- 1月26日頃を目処に実験成果を生みたい。方向性2をマシンで動かしつつ、ノートpcで以下を行う。
- Pettingzooのチュートリアルがあるので、参考にしながらコードを書いていく
### モデルのロード
- コード記載
### モデルのセーブ
- コード記載
### 行動遅延の実装
- コード記載
### 対戦相手の指定コード
- コード記載

## 方向性2（Stable-Baselines3を用いて、0-10フレームにて平均報酬がしっかり下がることを確認する）
- マシンで以下を行う。定期的にPCを確認し、効率良く学習を行えるようにする。
### 0フレームで、平均報酬が+になることを目標に学習を行う
- A2C or PPO
- CnnPolicy or Mlppolicy
- 学習回数（step）
- ハイパーパラメータ（learningn_rate等）
  
### 上を確認後、同じ条件を下に0-10フレームのモデルを作成する
- ただの作業

### 作成後、例えば0フレームのモデルが3フレームの環境だと報酬がどう変わるかなどの実験を行う
- モデル作成後相談（今は考えなくて良い）

## 今後の予定
1. WindowsでPoetry環境を作成する　→　無理だた（Cmake、multialeagent.pyがうまくいかない）、2をLinuxでまず頑張る
2. Pettingzooで、Boxingの遅延付き学習コードを作成する
3. 対戦相手の指定ができるコードを作成する
4. 人間と対戦できるコードを作成する
5. 

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

## 参考文献
- [Pettingzoo.github](https://github.com/Farama-Foundation/PettingZoo)
- [Pettingzoo チュートリアル1](https://note.com/npaka/n/n9b9074b8f916)
- [Pettingzoo チュートリアル2](https://note.com/npaka/n/n06d8ba36d5bc)
- [Pettingzoo Documentation](https://pettingzoo.farama.org/index.html)
- [GPU1](https://zenn.dev/danchinocto/scraps/da90c7e70ec77d)
- [GPU2](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)


