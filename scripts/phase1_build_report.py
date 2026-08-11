import base64
import json
import os

_BASE = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1_baseline")
FIGS = _BASE
OUT = os.path.join(_BASE, "report.html")


def b64(name):
    with open(f"{FIGS}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


with open(f"{FIGS}/summary.json") as f:
    summary = json.load(f)

heatmaps = b64("heatmaps.png")
reg_center = b64("regression_center.png")
reg_radius = b64("regression_radius.png")
dt_dep = b64("dt_dependence.png")

rows = ""
paper_vals = {
    "0.2": None, "0.5": None,
    "1.0": (1.23, 14.53), "2.0": (1.04, 11.19), "3.0": None,
}
for dt, v in summary.items():
    paper = paper_vals.get(dt)
    paper_str = f"{paper[0]:.2f} / {paper[1]:.2f}" if paper else "—"
    rows += f"""
    <tr>
      <td>{float(dt):.1f}</td>
      <td>{v['actual_dt']:.2f}</td>
      <td>{v['alpha']:.2f}</td>
      <td>{v['vmax']:.2f}</td>
      <td>{v['slope']:.2f}</td>
      <td>{v['rc_mean']:.2f}</td>
      <td class="paper-col">{paper_str}</td>
    </tr>"""

html = f"""<title>Baseline再現 — 藤村・杉原モデル Phase 1</title>
<style>
  :root {{
    color-scheme: light;
    --bg: #f6f7f8;
    --page-edge: #eef0f2;
    --surface: #ffffff;
    --text-primary: #14171c;
    --text-secondary: #545b66;
    --text-muted: #898781;
    --border: #dde1e6;
    --accent: #2a78d6;
    --accent-2: #eb6834;
    --code-bg: #f1f3f5;
    --table-stripe: #fafbfc;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --bg: #14161a;
      --page-edge: #101216;
      --surface: #1c1f24;
      --text-primary: #eef1f5;
      --text-secondary: #a9b0bb;
      --text-muted: #7d848f;
      --border: #2a2f37;
      --accent: #3987e5;
      --accent-2: #d95926;
      --code-bg: #23262c;
      --table-stripe: #202329;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --bg: #14161a;
    --page-edge: #101216;
    --surface: #1c1f24;
    --text-primary: #eef1f5;
    --text-secondary: #a9b0bb;
    --text-muted: #7d848f;
    --border: #2a2f37;
    --accent: #3987e5;
    --accent-2: #d95926;
    --code-bg: #23262c;
    --table-stripe: #202329;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    background: var(--page-edge);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 3.5rem 1.25rem 6rem;
  }}
  .sheet {{
    max-width: 760px;
    margin: 0 auto;
    background: var(--bg);
  }}
  .wide {{ max-width: 980px; }}

  .eyebrow {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.5rem;
  }}
  h1 {{
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 2.15rem;
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.01em;
    text-wrap: balance;
    margin: 0 0 0.6rem;
  }}
  h2 {{
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 1.4rem;
    font-weight: 600;
    text-wrap: balance;
    margin: 0 0 0.9rem;
  }}
  .subtitle {{
    color: var(--text-secondary);
    font-size: 1.02rem;
    max-width: 62ch;
    margin: 0 0 0;
  }}
  .meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.4rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 1.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--border);
  }}
  .meta b {{ color: var(--text-secondary); font-weight: 600; }}

  header.hero {{
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding-bottom: 2.4rem;
    margin-bottom: 2.6rem;
    border-bottom: 1px solid var(--border);
  }}

  section {{
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    margin: 0 auto 3.2rem;
  }}
  section.wide {{ margin-left: auto; margin-right: auto; }}

  p {{ margin: 0; max-width: 68ch; }}
  .lede {{
    font-size: 1.08rem;
    color: var(--text-primary);
  }}
  .lede strong {{ color: var(--accent); font-weight: 600; }}

  figure {{
    margin: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.1rem 0.9rem;
  }}
  figure img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
  }}
  figcaption {{
    font-size: 0.86rem;
    color: var(--text-secondary);
    margin-top: 0.75rem;
    max-width: 78ch;
  }}
  figcaption code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: var(--code-bg);
    padding: 0.05rem 0.35rem;
    border-radius: 4px;
    font-size: 0.82em;
  }}

  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.86rem;
    background: var(--surface);
  }}
  th, td {{
    text-align: right;
    padding: 0.55rem 0.9rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }}
  th {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr:nth-child(even) td {{ background: var(--table-stripe); }}
  td.paper-col {{ color: var(--text-secondary); border-left: 1px dashed var(--border); }}
  th.paper-col {{ border-left: 1px dashed var(--border); }}

  .callout {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-2);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: var(--text-secondary);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }}
  .callout b {{ color: var(--text-primary); }}
  .callout ul {{ margin: 0; padding-left: 1.2rem; }}
  .callout li {{ margin: 0.25rem 0; }}

  footer {{
    max-width: 760px;
    margin: 0 auto;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding-top: 1.6rem;
    border-top: 1px solid var(--border);
  }}
</style>

<div class="sheet">
  <header class="hero">
    <p class="eyebrow">Phase 1 — Baseline再現</p>
    <h1>藤村・杉原モデルのΔt依存不安定性は<br>idsse-dataでも再現するか</h1>
    <p class="subtitle">Narizuka, Takizawa &amp; Yamazaki (2023, <i>Sci. Rep.</i> 13:865) の到達円フィッティング手法を、Bundesliga <code>idsse-data</code> の1試合に適用した検証結果。</p>
    <div class="meta">
      <span><b>Match</b> J03WPY (Fortuna Düsseldorf – 1. FC Nürnberg)</span>
      <span><b>Source</b> idsse-data / TRACAB, 25 Hz</span>
      <span><b>Players</b> GK除外, 20名</span>
    </div>
  </header>

  <section>
    <p class="lede">全Δt帯で <strong>αとV<sub>max</sub>がΔtとともに単調に減少</strong>し、論文がJ1リーグ54試合で発見した不安定性(Fig.7)と同じ傾向が、独ブンデスリーガの別データセットでも確認された。Δt=1, 2秒での絶対値も論文の報告値と近いオーダー。</p>
  </section>

  <section class="wide">
    <p class="eyebrow">Eq. 4 · Investigation method</p>
    <h2>到達点ヒートマップと到達円</h2>
    <figure>
      <img src="data:image/png;base64,{heatmaps}" alt="Arrival point heatmaps for various v0 and delta_t, with fitted circles" />
      <figcaption>各選手の速度 <code>v(t)</code> を原点・+x方向に揃えた座標系で、<code>t+Δt</code> 後の到達点を集計(論文 Fig.3 の座標系、色は対数スケール)。オレンジの円は式(4)の解析解に、回帰で得た α, V<sub>max</sub> を代入したもの。密度雲の輪郭と円がおおむね一致しており、藤村・杉原モデルの解の形が実データと整合することを示している。</figcaption>
    </figure>
  </section>

  <section class="wide">
    <p class="eyebrow">Eq. 5, 7</p>
    <h2>円中心の v₀ 依存性 → α の推定</h2>
    <figure>
      <img src="data:image/png;base64,{reg_center}" alt="Regression of circle center x,y coordinates against v0" />
      <figcaption>y<sub>c</sub>≈0(v₀に依存しない)、x<sub>c</sub> は v₀≤6 m/s で比例。この傾きが A(α,Δt) = (1−e<sup>−αΔt</sup>)/α の推定値になり、数値求解で α を得る(論文 Fig.5 相当)。</figcaption>
    </figure>
  </section>

  <section class="wide">
    <p class="eyebrow">Eq. 6, 8</p>
    <h2>円半径の v₀ 独立性 → V<sub>max</sub> の推定</h2>
    <figure>
      <img src="data:image/png;base64,{reg_radius}" alt="Regression of circle radius against v0" />
      <figcaption>半径 r<sub>c</sub> は v₀≤6 m/s の範囲でほぼ一定。平均値が B(α, V<sub>max</sub>, Δt) の推定値になり、既知の α を使って V<sub>max</sub> を線形に解く(論文 Fig.6 相当)。v₀&gt;6 m/s でデータが疎になりバイアスが乗る点も論文と同じ挙動。</figcaption>
    </figure>
  </section>

  <section class="wide">
    <p class="eyebrow">cf. Fig. 7</p>
    <h2>$\\Delta t$ 依存性 — 不安定性の再現</h2>
    <figure>
      <img src="data:image/png;base64,{dt_dep}" alt="Alpha and Vmax as functions of delta t" />
      <figcaption>本来 Δt に依存しないはずの運動パラメータが、Δt≲1秒で急激に変化し、それ以降は緩やかに落ち着く — 論文の中心的な発見と同じ形状。</figcaption>
    </figure>
  </section>

  <section>
    <p class="eyebrow">数値の比較</p>
    <h2>推定値まとめ</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Δt (指定)</th>
            <th>Δt (実際)</th>
            <th>α [1/s]</th>
            <th>V<sub>max</sub> [m/s]</th>
            <th>slope</th>
            <th>r_c mean [m]</th>
            <th class="paper-col">論文値 (α / Vmax)</th>
          </tr>
        </thead>
        <tbody>{rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <p class="eyebrow">留意点</p>
    <div class="callout">
      <b>この結果を読む上での注意</b>
      <ul>
        <li>1試合(J03WPY)のみでの検証。7試合全体・他の試合での再現性は未確認。</li>
        <li>ヒートマップの外れ値除去(閾値 <code>c</code>)は論文の記述が曖昧なため、8近傍セル数に基づく独自実装で近似している。</li>
        <li>円中心の回帰は原点を通る比例フィット(切片なし)を採用 — 理論上妥当だが論文に明記はない。</li>
        <li>速度 $v(t)$ は論文の定義通り「1秒前との後退差分」を使用(フレーム単位の差分ではない)。</li>
      </ul>
    </div>
  </section>

  <footer>
    pi-fsm · Phase 1 baseline reproduction · src/pi_fsm/{{data,preprocessing,baseline}}.py, scripts/phase1_dt_dependence.py
  </footer>
</div>
"""

with open(OUT, "w") as f:
    f.write(html)

print("wrote", OUT, len(html), "chars")
