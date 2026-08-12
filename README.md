# pi-fsm

Physics-Informed Neural Networks (PINNs) を用いて、藤村・杉原モデル
(Fujimura & Sugihara, 2005) のサッカー選手スプリント運動モデルにおける
時間依存パラメータ $F(t), k(t)$ を推定する研究プロジェクト。

Narizuka et al. (2023, *Scientific Reports*) が発見した「本来一定のはずの
運動パラメータ $\alpha=k/m, V_{max}=F/k$ が、推定に使う時間幅 $\Delta t$
によって不安定に変動する」という未解決の課題に対し、$F(t),k(t)$ を時間の
関数としてPINNで直接推定することで、①不安定性が時間依存化で解消するか、
②その原因が測定誤差かモデルの限界か、を明らかにすることを目指す。

詳細な背景・手法は [`documents/research_plan.md`](documents/research_plan.md) を参照。

## セットアップ

```bash
uv sync
```

Python 3.12、依存関係は `pyproject.toml` / `uv.lock` で管理(`uv` 必須)。

## 使い方

```bash
# データ探索
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb

# フェーズ1: Baseline再現(藤村・杉原モデルの定数パラメータ推定)
uv run python scripts/phase1_all_matches.py

# フェーズ2: PINN sanity check(合成データでの実装検証)
uv run python scripts/phase2_sanity_check.py

# フェーズ2: 実データでの汎化性検証(PINN vs Baseline1)
uv run python scripts/phase2_generalization_check.py
```

大規模な学習(複数試合・複数seed)はColabで実行する運用(`notebooks/colab_seed_sweep.ipynb`)。
ローカル/Colabの役割分担は `documents/research_plan.md` 4.4節を参照。

## 構成

```
src/pi_fsm/
  data.py          # kloppy経由でidsse-dataをロード、TRACAB座標系(m)に変換
  preprocessing.py # GK除外、速度算出、座標回転
  baseline.py      # 藤村・杉原モデルの解析解、Baseline1(定数パラメータ)推定
  segments.py       # スプリント区間抽出(PINN学習用)
  cache.py          # 前処理・区間抽出結果のparquetキャッシュ
  pinn/
    models.py        # TrajectoryNet, ForceNet, ResistanceNet
    loss.py           # データ損失・物理損失・滑らかさ正則化
    train.py           # 学習ループ(共同学習・2段階学習)
scripts/    # 各フェーズの実行スクリプト(phase1_*, phase2_*)
notebooks/  # データ探索・パラメータ調整・Colab実行用ノートブック
documents/  # 研究計画・各フェーズの結果メモ
outputs/    # 生成された図・結果
```

## 現状

- **フェーズ1**: 完了。idsse-data全7試合で$\Delta t$依存の不安定性を再現確認([`documents/phase1_baseline_results.md`](documents/phase1_baseline_results.md))
- **フェーズ2**: PINN実装・sanity check完了。実データで、時間依存モデル(PINN)が定数モデル(Baseline1)より時間軸全体で一貫して予測できることを確認。詳細・今後の課題は [`documents/phase2_pilot_results.md`](documents/phase2_pilot_results.md) を参照
