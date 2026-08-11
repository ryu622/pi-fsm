# 研究計画書
## PINNsによる藤村・杉原モデルの時間依存パラメータ推定
### ―サッカー選手のスプリント運動モデルの拡張と検証―

---

## 1. 研究背景

### 1.1 藤村・杉原モデルとその位置づけ

サッカーのトラッキングデータ解析において、選手の到達可能時間・到達可能領域を予測する「運動モデル」は、パス成功確率予測(Spearman et al., 2017)、スペース評価(Narizuka et al., 2021)、守備プレッシャー評価など、多くの応用研究の基盤technologyとなっている。中でも藤村・杉原モデル(Fujimura & Sugihara, 2005)は、駆動力と粘性抵抗からなる運動方程式

$$m\frac{d^2\vec{x}(t)}{dt^2} = F\vec{n} - k\frac{d\vec{x}(t)}{dt}$$

に基づき、解析解を持つシンプルな物理モデルとして広く参照されている。

### 1.2 先行研究(Narizuka et al., 2023)が発見した課題

Narizuka, Takizawa & Yamazaki (2023, *Scientific Reports*) は、J1リーグ54試合のトラッキングデータを用いて藤村・杉原モデルの妥当性を実証的に検証した。その結果:

- 到達可能領域の形状は、先行研究の批判(楕円になるはず)に反し、**モデルの予測通り円形**であることを確認
- 円の中心・半径の初速依存性も、モデルの解析解と整合
- 一方で、**モデルでは時間不変であるべきパラメータ $\alpha=k/m,\ V_{max}=F/k$ が、時間間隔 $\Delta t\lesssim1$秒の範囲でデータから逆算すると大きく変動する**、という矛盾を発見

この矛盾の原因として、①トラッキングデータの測定誤差(TRACAB系±1m)、②モデル自体の限界、の2つの可能性が挙げられたが、**いずれも検証されないまま今後の課題として残された**。同論文は解決の方向性として、時間依存パラメータを持つ拡張モデル

$$m\frac{d^2\vec{x}(t)}{dt^2} = F(t)\vec{n} - k(t)\frac{d\vec{x}(t)}{dt}$$

を提案しているが、$F(t), k(t)$の具体的な関数形や推定方法は示されていない。

### 1.3 研究の狙い

本研究は、上記の未解決課題に対し、**Physics-Informed Neural Networks (PINNs)** を用いて $F(t), k(t)$ の関数形を仮定せずデータから直接推定する枠組みを提案し、以下を明らかにすることを目的とする。

1. 時間依存パラメータの導入により、Narizuka et al. (2023) が発見した「$\Delta t$依存の不安定性」が実際に解消されるか
2. その不安定性の主因が、測定誤差とモデルの限界のいずれにあるか

---

## 2. 関連研究

| 研究 | 位置づけ |
|---|---|
| Fujimura & Sugihara (2005) | 本研究が拡張する基礎モデル |
| Narizuka et al. (2023) | 課題を発見した直接の先行研究、本研究の出発点 |
| Narizuka et al. (2021) | $z_1,z_2$によるスペース評価。藤村・杉原モデルの応用例の一つ |
| Spearman et al. (2017, 2018) | 藤村・杉原モデルとは異なる運動モデル(等加速度+制約付き最適化)によるTime-to-Intercept、パス確率・OBSOモデル |
| Raissi et al. (2019) | PINNsの提案論文。物理法則を損失関数に組み込むニューラルネット学習の基礎 |

**新規性**:これまでPINNsはサッカーのトラッキングデータ解析、特に選手運動モデルのパラメータ推定には(調査の限り)適用されていない。既存研究が「今後の課題」として言及するにとどまっていた時間依存拡張を、実際に実装・検証する点に貢献がある。

---

## 3. 提案手法

### 3.1 モデル構造

$F(t), k(t)$ を、時刻 $t$(および必要に応じて選手の現在速度など)を入力とする多層パーセプトロン(MLP)で表現する。

$$F(t) = \text{NN}_F(t;\theta_F), \qquad k(t) = \text{NN}_k(t;\theta_k)$$

推定対象は選手が加速・スプリントを開始してからの経過時間 $t\in[0, T_{max}]$($T_{max}$は予備分析(MSD解析)で確認されたモデルの妥当な適用範囲、目安10秒以内)とする。

### 3.2 損失関数

$$\mathcal{L}(\theta_F,\theta_k) = \mathcal{L}_{data} + \gamma\cdot\mathcal{L}_{physics}$$

$$\mathcal{L}_{data} = \frac{1}{N}\sum_{j=1}^N\|\hat{x}(t_j) - x_{obs}(t_j)\|^2$$

$$\mathcal{L}_{physics} = \frac{1}{M}\sum_{k=1}^M\left\|m\ddot{\hat{x}}(t_k) - F(t_k)\hat{n}(t_k) + k(t_k)\dot{\hat{x}}(t_k)\right\|^2$$

- $\mathcal{L}_{data}$:観測点(トラッキングデータのフレーム)における位置誤差
- $\mathcal{L}_{physics}$:コロケーション点(観測の有無によらず密に配置)における運動方程式の残差
- $\gamma$:物理制約の重み(ハイパーパラメータ、検証フェーズで調整)
- $\hat{x}(t)$自体もネットワーク出力とし、自動微分で$\dot{\hat{x}},\ddot{\hat{x}}$を計算する(標準的なPINNsの実装方式)

### 3.3 ベースラインとの比較対象

| モデル | 内容 |
|---|---|
| Baseline 1 | Narizuka et al. (2023) の元手法(定数 $\alpha, V_{max}$、到達円からの回帰推定) |
| Baseline 2 | 通常のニューラルネット(物理制約なし、$\mathcal{L}_{physics}$を除いた $\mathcal{L}_{data}$のみで学習した軌道回帰) |
| **提案手法** | PINNs による時間依存 $F(t), k(t)$ の推定 |

Baseline 2 との比較により、「物理制約を入れることの意義」自体も定量的に確認する。

---

## 4. 実装計画

### 4.1 開発環境・主要ライブラリ

| 用途 | ライブラリ・ツール |
|---|---|
| 深層学習フレームワーク | PyTorch(自動微分によるODE残差計算のため) |
| PINN実装の参考 | [DeepXDE](https://github.com/lululxvi/deepxde)(PINNs用ライブラリ、カスタムODEへの拡張が容易) |
| ODEソルバ(軌道の順伝播的生成、合成データ作成用) | [torchdiffeq](https://github.com/rtqichen/torchdiffeq)(PyTorchベースのニューラルODEソルバ、PINN出力を使った数値積分にも流用可) |
| トラッキングデータの読み込み | [kloppy](https://github.com/PySport/kloppy)(IDSSEデータセットの公式推奨ローダー、Hugging Face上の`pysport/idsse-data`を直接ロード可能) |
| 実験管理・ログ | Weights & Biases、または軽量にJSON/CSVでの自前ログ(4.4節参照) |
| 可視化 | matplotlib、mplsoccer(ピッチ描画) |

**成塚氏本人の実装コードについて**:Narizuka et al. (2021, 2023) に対応する著者本人の公開実装リポジトリは確認できなかった(論文中にもコード公開の記載なし)。したがって、藤村・杉原モデルの基本部分(到達円計算、ヒートマップ生成)は論文の数式(本計画書2節参照)から**自前で再実装する**。到達可能領域・ピッチコントロール系の実装パターンについては、公開されている類似実装([LaurieOnTracking/Metrica-sports](https://github.com/Metrica-sports/sample-data) の関連解析コード群など、Spearmanモデルの非公式実装コミュニティ資産)を設計の参考にする。

**データセット付属コードについて**:idsse-dataの公式リポジトリ([spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data))には`data_processing.py`が含まれるが、これはfloodlightベースでXML生データをDataFrame化し記述統計を表示するだけの薄いラッパーであり、PINN学習に必要な特徴量エンジニアリング(4.2節)は含まれない。また同スクリプトはFigshareからの生データの手動ダウンロードを前提としており、後述のkloppy経由の読み込み方針とは取得経路が異なる。今回はColab上での自動化・再現性を優先し、**kloppyベースでの実装に統一**し、このリポジトリのコードは構造理解の参考資料としてのみ位置づける。

### 4.2 データパイプライン

**データの入手について**:[spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data)(ドイツ・ブンデスリーガ1部・2部、2022/23シーズン完全試合7試合分、TRACAB光学トラッキング+公式イベントデータ、CC-BY 4.0)は、既にHugging Face上に`pysport/idsse-data`として再ホストされている。**自分でダウンロード・再アップロードする作業は不要**で、`kloppy`から直接ロードする。

```python
from kloppy import sportec
tracking = sportec.load_open_tracking_data(match_id="J03WPY")
events = sportec.load_open_event_data(match_id="J03WPY")
```

利用可能な match_id: `J03WPY`, `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WQQ`, `J03WR9`(計画時点の記載から2件訂正済み。実際にkloppyから取得できるIDで確認)

**前処理ステップ(すべて自前実装が必要)**:
1. GK除外(Narizuka et al. 2023 の前処理方針を踏襲)
2. 速度・加速度の算出(有限差分、必要に応じ低域通過フィルタ/Savitzky-Golayによる平滑化を検討)
3. 各選手・各時刻について、速度方向がx軸正方向を向くよう座標回転(元論文Fig.3の座標系)
4. 「加速・スプリント区間」の抽出:速度が持続的に増加している区間を対象に切り出し(閾値・区間長は予備分析で調整)
5. train/valid/testに試合単位で分割(データリーク防止のため、同一試合のフレームが複数splitにまたがらないようにする)
6. 前処理済みデータを軽量形式(`.npz`または`.parquet`)でキャッシュ保存(4.4節の運用フロー参照)

**データ規模に関する留意点**:idsse-dataは7試合と、Narizuka et al. (2023) が用いたJ1リーグ54試合と比べ小規模である。これは合成データ検証・クロスバリデーションの設計、および結果の解釈(サンプルサイズによる制約)に影響するため、4.5節(スケジュール)に検討時間を確保する。

### 4.3 実装フェーズ

**フェーズ1:再現実装(Baseline構築)**
- Narizuka et al. (2023) の手法(到達点ヒートマップ、到達円フィッティングによる$\alpha,V_{max}$の定数推定)をidsse-dataに対して再実装
- 論文と同様の「$\Delta t$依存の不安定性」がidsse-dataでも再現されるかを確認(研究の前提の確認)

**フェーズ2:PINN実装**
- $\text{NN}_F, \text{NN}_k$ の実装(まずは全選手共通のパラメータから開始し、段階的に選手固有の特徴量を入力に加える拡張を検討)
- 損失関数の実装、$\gamma$のハイパーパラメータ探索
- 学習の安定化(必要に応じ、$\mathcal{L}_{physics}$の重みを学習初期は小さくし徐々に増やすカリキュラム学習等を検討)

**フェーズ3:実データでの検証**(5節①②に対応)

**フェーズ4:合成データでの検証**(5節③に対応、フェーズ3の結果を踏まえ優先度を調整)

### 4.4 開発・実行環境の運用フロー

**基本方針**:ローカル(MacBook Air M5)を「非力だから使わない」のではなく、**軽量な開発・デバッグ・小規模検証のための主戦場**として積極的に活用する。PINNの学習モデル自体は$F(t),k(t)$を出力する小規模MLPであり、大規模学習を除けばM5(mpsバックエンド)でも十分実用的に動く。Colabは**大規模学習・GPU依存の処理・複数条件の一括実験にのみ限定して使う**、という分業を徹底する。

**役割分担**:

| 環境 | 役割 |
|---|---|
| ローカル(M5) | コード開発、ユニットテスト、1試合分のデータでの前処理パイプラインのデバッグ、少数エポックでの学習動作確認(CPU/mps) |
| GitHub | コードのバージョン管理、Colabからの取得元。特定のコミット/タグを指定してcheckoutすることで、結果とコードの対応関係を追跡可能にする |
| Colab | 前処理済みデータを用いた本番学習(複数$\gamma$・複数$\Delta t$設定での実験、フェーズ3・4の本実行) |
| Google Drive | 前処理済みデータのキャッシュ、学習チェックポイント、実験ログの永続化(Colabのローカルディスクは非永続なため必須) |

**典型的なワークフロー**:

```
[ローカル: M5]
  1. idsse-dataを1試合分のみkloppyでロードし、前処理パイプラインを実装・デバッグ
  2. 前処理済みデータを .npz / .parquet 形式でキャッシュ
  3. 小規模(1試合、少数エポック)でPINN学習コードの動作確認
  4. git push → GitHub(タグ付けして結果との対応を明確化)

[Colab(本番学習時)]
  5. Google Driveをマウント
  6. git clone または git checkout <tag> でリポジトリ取得
  7. pip install -r requirements.txt
  8. kloppyで全7試合をロード→前処理→Driveにキャッシュ保存(初回のみ、以降はキャッシュを再利用)
  9. 学習実行。一定エポックごとにDriveへチェックポイント保存
  10. 結果(ログ・図・モデル重み)をDriveに保存→ローカルに同期して分析
```

**セッション断・環境差異への対策(実装計画の段階で組み込む)**:

| リスク | 対策 |
|---|---|
| 無料版Colabのセッション切断(最大12時間、アイドル数十分で切断) | 一定エポックごと(例:100エポックごと)にモデル・optimizer状態をDriveへ自動チェックポイント保存する処理を、学習ループの実装当初から組み込む。再開時はチェックポイントから学習を継続できる設計にする |
| Colabのローカルディスクは非永続 | 学習に必要なデータ・出力先はすべてDriveパス経由にし、ローカルパスへの一時保存は最小限にとどめる |
| ローカル(mps)とColab(CUDA)のバックエンド差異 | `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"` の形で自動判定するコードに統一し、`requirements.txt`でライブラリバージョンを固定する |
| GPU確保の不安定さ(無料版は混雑時に割り当てられないことがある) | 学習コード自体はCPU/mpsでも(低速だが)動作する設計とし、Colabが使えない場合の代替経路を確保しておく |
| 実験条件と結果の対応が取れなくなる | 各学習実行時に、使用コミットハッシュ・ハイパーパラメータ・データバージョンをログ(JSON)としてDriveに同時保存する。W&B等の実験管理ツールの利用も検討 |

### 4.5 概算スケジュール(目安)

| フェーズ | 内容 | 目安期間 |
|---|---|---|
| 1 | データ前処理パイプライン構築(ローカルで1試合分から)、Baseline再現 | 4〜6週 |
| 2 | PINN実装(ローカルで小規模動作確認)、Colab運用フロー整備、学習の安定化 | 6〜8週 |
| 3 | 実データでの検証(ヒートマップ一致度比較、Colabで本実行) | 4週 |
| 4 | 合成データでのリカバリーテスト | 3〜4週 |
| 5 | 結果の整理・執筆 | 4週 |

---

## 5. 検証計画

### ① 学習の基本チェック
$\mathcal{L}_{data}, \mathcal{L}_{physics}$ がともに収束することを確認(train/valid双方で監視、過学習の有無をvalidで確認)。

### ② 実データでのヒートマップ一致度検証(主結果)
Narizuka et al. (2023) の検証手法(Fig.3-6相当)を、複数の $\Delta t$(例:0.2, 0.5, 1, 2, 3秒)について実施し、
- Baseline 1(定数パラメータ)
- 提案手法(時間依存パラメータ)

のそれぞれで、実測ヒートマップとの一致度(境界形状の誤差、中心・半径の予測誤差)を比較。**「$\Delta t$を変えても一貫して高い一致度が得られるか」を主要な評価軸**とする。

### ③ 合成データによるリカバリーテスト(②の後、結果に応じて実施)
1. 任意の $F_{true}(t), k_{true}(t)$ を設定し、運動方程式を数値積分して合成軌道を生成(torchdiffeqを使用)
2. idsse-dataの実測誤差水準を模したノイズ($\pm1$m相当、TRACAB由来)を付加
3. 正解を伏せた状態でPINNに推定させ、$\hat{F}(t),\hat{k}(t)$ と $F_{true}(t), k_{true}(t)$ を比較
4. ノイズ水準を振って感度分析を行うことで、②で観測される改善が測定誤差の範囲内で説明できるかを切り分ける

**優先順位**:②の結果が明確な改善を示せば③は補強的な位置づけ、②の結果が曖昧・限定的であれば③が原因切り分けの主要な手段となる。

---

## 6. 想定される課題・リスク

| リスク | 対応方針 |
|---|---|
| PINNsの学習が不安定(物理項とデータ項のバランスが取りにくい) | $\gamma$のスケジューリング、正規化、学習率の段階的調整で対応。DeepXDEの実装例を参考に既知の安定化手法を適用 |
| データ量(7試合)が先行研究(54試合)より少なく、統計的な頑健性に懸念 | 加速区間の抽出により1試合あたり多数のサンプルを確保できる可能性を検討。不足する場合は他の公開トラッキングデータ(Metrica Sports sample dataなど)の追加利用も検討 |
| $F(t),k(t)$が真に一意に定まらない(識別可能性の問題) | 合成データでのリカバリーテスト(検証③)で、そもそも手法として識別可能かを事前に確認する設計とする |
| 測定誤差とモデル限界の切り分けが、合成データのノイズモデル化の妥当性に依存する | ノイズの水準・分布を複数パターンで試し、結論の頑健性を確認する |

---

## 7. 期待される成果

1. Narizuka et al. (2023) が未解決のまま残した課題(パラメータの時間不変性の破れ)に対する、具体的な解決策の提示
2. 測定誤差とモデルの限界のどちらが主因かについての、実証的な知見
3. 学習された $F(t),k(t)$ の関数形からの、選手の加速局面における運動特性の新たな理解(駆動力・抵抗力がどのように立ち上がるか)
4. 藤村・杉原モデルは本研究群(Spearman系、Narizuka系のスペース評価、守備指標など)全体の基礎技術であるため、改良の恩恵は広範な応用に波及しうる

---

## 参考文献・リポジトリ一覧

- Fujimura, A. & Sugihara, K. (2005). Geometric analysis and quantitative evaluation of sport teamwork. *Systems and Computer in Japan*, 36, 49-58.
- Narizuka, T., Takizawa, K. & Yamazaki, Y. (2023). Validation of a motion model for soccer players' sprint by means of tracking data. *Scientific Reports*, 13, 865.
- Narizuka, T., Yamazaki, Y. & Takizawa, K. (2021). Space evaluation in football games via field weighting based on tracking data. *Scientific Reports*, 11, 5509.
- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195.
- データセット(公式リポジトリ):https://github.com/spoho-datascience/idsse-data
- データセット(Hugging Face再ホスト版、本研究での実際のロード元):https://huggingface.co/datasets/pysport/idsse-data
- kloppy:https://github.com/PySport/kloppy
- DeepXDE:https://github.com/lululxvi/deepxde
- torchdiffeq:https://github.com/rtqichen/torchdiffeq