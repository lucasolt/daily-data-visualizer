"""
Gerador do relatório por fase — independente do app.py.
Retorna um HTML string pronto para st.components.v1.html().
"""

import math
import numpy as np
import pandas as pd

import data_prep as dp

# ---------------------------------------------------------------- helpers

def _fmt_h(h: float) -> str:
    """Horas decimais → HH:MM. NaN → '—'."""
    if pd.isna(h):
        return "—"
    sign = "-" if h < 0 else ""
    h = abs(h)
    hh = int(h)
    mm = round((h - hh) * 60)
    if mm == 60:
        hh += 1; mm = 0
    hh = hh % 24
    return f"{sign}{hh:02d}:{mm:02d}"

def _fmt_val(v, decimals=1, suffix="") -> str:
    if pd.isna(v):
        return "—"
    if decimals == 0:
        return f"{int(round(v))}{suffix}"
    return f"{v:.{decimals}f}{suffix}"

def _safe_median(s: pd.Series):
    v = s.dropna()
    return float(v.median()) if len(v) else float("nan")

def _safe_mean(s: pd.Series):
    v = s.dropna()
    return float(v.mean()) if len(v) else float("nan")

def _pct_nonzero(s: pd.Series, n_days: int) -> str:
    """Conta dias > 0 e retorna 'N (X%)'."""
    cnt = int((s.fillna(0) > 0).sum())
    if cnt == 0:
        return "—"
    pct = cnt / n_days * 100
    return f"{cnt} ({pct:.1f}%)"

def _bg_color(val, nums, direction):
    """Calcula cor de fundo rgba para heat coloring entre fases."""
    valid = [n for n in nums if not math.isnan(n)]
    if len(valid) < 2:
        return ""
    mn, mx = min(valid), max(valid)
    if mx == mn:
        return ""
    if math.isnan(val):
        return ""
    t = (val - mn) / (mx - mn)
    if direction == "lower":
        t = 1 - t
    # verde se bom, vermelho se ruim
    if t >= 0.5:
        alpha = (t - 0.5) * 2 * 0.28
        return f"rgba(29,158,117,{alpha:.3f})"
    else:
        alpha = (0.5 - t) * 2 * 0.28
        return f"rgba(216,90,48,{alpha:.3f})"


# ---------------------------------------------------------------- geração

def build_phase_report(df: pd.DataFrame) -> str:
    """Retorna string HTML do relatório de sono/comportamento por fase."""

    if "fase_label" not in df.columns or df["fase_label"].nunique() < 1:
        return "<p style='color:#9aa7b5'>Sem dados de fase disponíveis.</p>"

    # limiar de inclusão de fármaco nos pills: usado em > PILL_MIN_PCT dos dias da fase
    PILL_MIN_PCT = 0.50

    phases = dp.fase_order(df["fase_label"].unique())
    # remove "(sem fase)" se houver apenas ela
    phases = [p for p in phases if p != "(sem fase)"] or phases

    # ---- metadados por fase ----
    phase_data = {}
    for ph in phases:
        sub = df[df["fase_label"] == ph].copy()
        phase_data[ph] = sub

    def col(col_name, ph):
        return phase_data[ph].get(col_name, pd.Series(dtype=float))

    # header: datas e n dias
    def phase_header(ph):
        sub = phase_data[ph]
        n = len(sub)
        d0 = sub["date"].min().strftime("%d/%m/%y")
        d1 = sub["date"].max().strftime("%d/%m/%y")
        return ph, d0, d1, n

    headers = [phase_header(ph) for ph in phases]

    # ---- seções de dados ----
    # Cada seção: lista de linhas; cada linha: (label, [values_per_phase], [nums], direction, format_fn)
    def build_section(title, rows):
        return {"title": title, "rows": rows}

    def time_row(label, col_name, direction, use_mean_aux=True):
        """Para colunas de horário — mostra mediana + média auxiliar."""
        medians = [_safe_median(col(col_name, ph)) for ph in phases]
        means   = [_safe_mean(col(col_name, ph))   for ph in phases]
        vals    = [_fmt_h(m) for m in medians]
        aux     = [f"μ {_fmt_h(m)}" for m in means] if use_mean_aux else None
        return {"label": label, "vals": vals, "aux": aux,
                "nums": medians, "dir": direction, "type": "dual"}

    def dur_row(label, col_name, direction):
        nums = [_safe_median(col(col_name, ph)) for ph in phases]
        vals = [_fmt_h(n) for n in nums]
        return {"label": label, "vals": vals, "aux": None,
                "nums": nums, "dir": direction, "type": "single"}

    def num_row(label, col_name, direction, decimals=1, suffix="", use_mean=False):
        fn = _safe_mean if use_mean else _safe_median
        nums = [fn(col(col_name, ph)) for ph in phases]
        vals = [_fmt_val(n, decimals, suffix) for n in nums]
        return {"label": label, "vals": vals, "aux": None,
                "nums": nums, "dir": direction, "type": "single"}

    def pct_row(label, col_name, direction):
        """Percentagem calculada de colunas _h relativas."""
        rows_out = []
        for ph in phases:
            sub = phase_data[ph]
            if col_name in sub.columns:
                total = sub["sleep_duration_h"].replace(0, float("nan"))
                pct = (sub[col_name] / total * 100)
                rows_out.append(_safe_median(pct))
            else:
                rows_out.append(float("nan"))
        vals = [_fmt_val(n, 1, "%") for n in rows_out]
        return {"label": label, "vals": vals, "aux": None,
                "nums": rows_out, "dir": direction, "type": "single"}

    def ctx_row(label, col_name):
        """Dias > 0 com contagem e percentagem."""
        vals = []
        nums = []
        for ph in phases:
            sub = phase_data[ph]
            n = len(sub)
            if col_name in sub.columns:
                cnt = int((sub[col_name].fillna(0) > 0).sum())
                pct = cnt / n * 100 if n else 0
                vals.append(f"{cnt} ({pct:.1f}%)" if cnt > 0 else "—")
                nums.append(pct)
            else:
                vals.append("—")
                nums.append(float("nan"))
        return {"label": label, "vals": vals, "aux": None,
                "nums": nums, "dir": "lower", "type": "single"}

    def dual_num_row(label, col_name, direction, decimals=1, suffix="",
                     primary="mean", aux="median", min_cov=None):
        """Linha de duas estatísticas: uma principal (grande) + uma auxiliar (pequena).

        primary : "median" ou "mean" — qual estatística vira o valor principal.
        aux     : "mean", "median" ou None — qual vai na linha auxiliar (None = sem auxiliar).
                  Prefixo segue a escolha: 'μ' para média, 'md' para mediana.
        min_cov : se definido (ex. 0.20), suprime (→ NaN) a célula de uma fase cuja
                  cobertura — dias com valor não-nulo / dias da fase — for < min_cov.
                  Célula suprimida não mostra valor nem auxiliar e não entra no
                  cálculo de cor (_bg_color), como se aquela fase não existisse pra
                  esta linha.

        DEFAULT (primary="median", aux="mean") reproduz o comportamento histórico:
        mediana como principal, μ média como auxiliar. Os scores subjetivos
        sobrescrevem para primary="mean" (ver seção de scores abaixo).
        """
        stat = {"mean": _safe_mean, "median": _safe_median}
        prim_vals = [stat[primary](col(col_name, ph)) for ph in phases]
        aux_vals  = [stat[aux](col(col_name, ph)) for ph in phases] if aux else None

        if min_cov is not None:
            for i, ph in enumerate(phases):
                sub = phase_data[ph]
                n = len(sub)
                cov = (col(col_name, ph).notna().sum() / n) if n else 0.0
                if cov < min_cov:
                    prim_vals[i] = float("nan")
                    if aux_vals is not None:
                        aux_vals[i] = float("nan")

        vals = [_fmt_val(v, decimals, suffix) for v in prim_vals]
        aux_out = None
        if aux:
            pre = "μ" if aux == "mean" else "md"
            aux_out = [f"{pre} {_fmt_val(a, decimals, suffix)}" for a in aux_vals]
        return {"label": label, "vals": vals, "aux": aux_out,
                "nums": prim_vals, "dir": direction, "type": "dual"}

    # scores subjetivos: principal = MÉDIA, auxiliar = md mediana.
    # (decisão explícita — diferente das demais métricas, que usam mediana como principal.)
    # min_cov=0.20: fase com menos de 20% dos dias preenchidos não exibe score
    # (nem entra no cálculo de cor das outras fases).
    SCORE_MIN_COV = 0.20
    score_rows = []
    score_map = {
        "mood_score": ("Humor", "higher"),
        "energy_score": ("Energia", "higher"),
        "cognition_score": ("Cognição", "higher"),
        "attention_score": ("Atenção", "higher"),
        "fluencia_verbal_espontaneidade_social": ("Fluência verbal", "higher"),
        "drive_score": ("Iniciativa/Drive", "higher"),
    }
    for col_name, (label, direction) in score_map.items():
        if col_name in df.columns:
            score_rows.append(
                dual_num_row(label, col_name, direction,
                             primary="mean", aux="median", min_cov=SCORE_MIN_COV)
            )

    sections = [
        build_section("Horários — mediana (principal) · μ média", [
            time_row("Bed time",  "bed_time_h",  "lower"),
            time_row("Wake time", "wake_time_h", "lower"),
        ]),
        build_section("Duração (média HH:MM)", [
            dur_row("Sleep duration",  "sleep_duration_h", "higher"),
            dur_row("In bed",          "inbed_duration_h", None),
            dur_row("Deep sleep",      "deep_sleep_h",     "higher"),
            dur_row("Light sleep",     "light_sleep_h",    "lower"),
            dur_row("REM",             "rem_sleep_h",      "higher"),
            dur_row("Awake (sleep)",   "awake_sleep_h",    "lower"),
        ]),
        build_section("Qualidade", [
            dual_num_row("Sleep efficiency", "sleep_efficiency", "higher", 1, "%"),
            pct_row("Light %",  "light_sleep_h", "lower"),
            pct_row("REM %",    "rem_sleep_h",   "higher"),
            pct_row("Deep %",   "deep_sleep_h",  "higher"),
        ]),
        build_section("Perturbações (média)", [
            dual_num_row("Sleep latency (min) *", "sleep_latency_estimate_minutes", "lower"),
            dual_num_row("Restlessness (min)",    "restlessness_mins",   "lower"),
            dual_num_row("Interruption (min)",    "interruption_mins",   "lower"),
            dual_num_row("Full awakenings",       "full_awakenings",     "lower", 2),
        ]),
        build_section("Cardíaco (média)", [
            dual_num_row("Avg BPM", "avg_bpm", "lower", 1),
            dual_num_row("VFC",     "VFC",     "higher", 1),
        ]),
        build_section("Atividade (média)", [
            dual_num_row("Passos",          "steps",        "higher", 0),
            dual_num_row("Exercício (min)", "exercise_mins","higher", 0),
        ]),
    ]
    if score_rows:
        sections.insert(0, build_section(
            "Scores subjetivos (média principal · md mediana)", score_rows))

    ctx_section = build_section("Contexto", [
        ctx_row("Dias fumados",  "fumou"),
        ctx_row("Dias de férias","ferias"),
    ])

    # ---- regime medicamentoso por fase ----
    # colunas manhã/tarde excluídas — só totais são exibidos
    MED_EXCLUDE = {
        "venvase_morning_mg", "venvanse_evening_mg",
        "metilfenidato_mg", "metilfenidato_evening_mg",
    }
    MED_COLORS = {
        "venvanse_mg_total":      ("#B5D4F4", "#0C447C", "Lisdexanfetamina"),
        "bupropiona_mg":          ("#C0DD97", "#27500A", "Bupropiona"),
        "zolpidem_mg_total":      ("#FAC775", "#633806", "Zolpidem"),
        "fluvoxamina_mg":         ("#CECBF6", "#3C3489", "Fluvoxamina"),
        "melatonina_mg":          ("#9FE1CB", "#085041", "Melatonina"),
        "clonazepam_mg":          ("#F5C4B3", "#712B13", "Clonazepam"),
        "pregabalina_mg":         ("#D3D1C7", "#444441", "Pregabalina"),
        "tranilcipromina_mg":     ("#F5C4B3", "#712B13", "IMAO"),
        "ramelteona_mg":          ("#C0E8F0", "#074E5E", "Ramelteon"),
        "pramipexol_mg_night":    ("#E8D4F5", "#4A1873", "Pramipexol"),
        "lamotrigina_mg":         ("#F0D9A0", "#5C3D00", "Lamotrigina"),
        "aripripazole_mg":        ("#F0B8C8", "#6B0025", "Aripiprazol"),
        "metilfenidato_mg_total": ("#FFD6A5", "#7A3900", "Metilfenidato"),
        "vitamina_d_ug":          ("#FFFACD", "#5C5200", "Vit D"),
    }

    def med_pills_html(ph, min_pct: float = PILL_MIN_PCT):
        sub = phase_data[ph]
        n_phase = len(sub)
        threshold = max(1, n_phase * min_pct)
        parts = []
        for col_name, (bgc, fgc, label) in MED_COLORS.items():
            if col_name in sub.columns and (sub[col_name] > 0).sum() >= threshold:
                # mediana da dose nos dias de uso (dose>0); regime estável por fase → vira a dose real
                used = sub[col_name][sub[col_name] > 0]
                dose = used.median()
                dose_txt = ""
                if not pd.isna(dose) and dose > 0:
                    dnum = int(round(dose)) if float(dose).is_integer() or abs(dose - round(dose)) < 0.05 else round(dose, 1)
                    dose_txt = f" {dnum} mg"
                parts.append(
                    f'<span class="mt" style="background:{bgc};color:{fgc}">{label}{dose_txt}</span>'
                )
        return "<br>".join(parts) if parts else "<span style='color:#666'>—</span>"

    def med_dose_html(ph, col_name):
        """Média diária sobre TODOS os dias da fase (dias sem uso entram como 0).
        Exposição média/dia — distinto da titulação (só dias de uso) usada nas correlações."""
        sub = phase_data[ph]
        if col_name not in sub.columns:
            return "—"
        s = sub[col_name].fillna(0)
        if (s > 0).sum() == 0:
            return "—"
        return _fmt_val(s.mean(), 1, " mg")

    # ---- renderização HTML ----
    n_phases = len(phases)
    col_w = f"{100/(n_phases+1):.1f}%"

    css = """
<style>
:root{
  --bg:#0b0d10;
  --panel:#14181d;
  --panel-2:#1b2128;
  --border:#2b333d;
  --text:#e7edf5;
  --muted:#9aa7b5;
}
*{box-sizing:border-box;}
body{margin:0;padding:0;background:var(--bg);color:var(--text);
  font-family:Inter,Segoe UI,Roboto,sans-serif;}
.wrapper{background:var(--panel);border:1px solid var(--border);
  border-radius:14px;padding:18px;margin-bottom:12px;}
.scroll{overflow-x:auto;padding:.5rem 0 1rem;}
table{border-collapse:separate;border-spacing:0;font-size:13px;
  color:var(--text);min-width:760px;width:100%;}
th{font-size:11px;font-weight:600;color:var(--muted);text-align:right;
  padding:10px 10px 8px;border-bottom:1px solid var(--border);white-space:nowrap;}
th.lbl{text-align:left;width:175px;}
td{text-align:right;padding:6px 10px;
  border-bottom:1px solid rgba(255,255,255,.05);white-space:nowrap;line-height:1.4;}
td.lbl{text-align:left;color:#c7d0da;font-size:12px;font-weight:500;}
td.dose,td.ctx{color:var(--muted);}
td.dose-lbl{color:var(--muted);text-align:left;}
tr.sh>td{background:var(--panel-2);font-size:10px;font-weight:700;
  color:#b9c4d0;letter-spacing:.08em;text-transform:uppercase;padding:7px 10px;}
.ph{display:block;font-weight:700;font-size:13px;color:var(--text);}
.phn,.pdate,.aux{display:block;font-size:10px;color:var(--muted);}
.mt{display:inline-block;border-radius:999px;padding:2px 7px;
  margin:2px 0;font-size:10px;font-weight:600;}
.legend{color:var(--muted);font-size:11px;margin-top:10px;display:flex;gap:12px;flex-wrap:wrap;}
.leg-box{display:inline-block;width:12px;height:12px;border-radius:3px;
  vertical-align:middle;margin-right:4px;}
.fn{color:var(--muted);font-size:10px;margin-top:6px;}
</style>
"""

    def th_html(label, d0, d1, n):
        return (f'<th><span class="ph">Fase {label}</span>'
                f'<span class="pdate">{d0} – {d1}</span>'
                f'<span class="phn">{n} dias</span></th>')

    def render_section(sec):
        rows_html = f'<tr class="sh"><td colspan="{n_phases+1}">{sec["title"]}</td></tr>\n'
        for row in sec["rows"]:
            nums = row["nums"]
            rows_html += f'<tr><td class="lbl">{row["label"]}</td>'
            for i, ph in enumerate(phases):
                val = row["vals"][i]
                num = nums[i] if nums else float("nan")
                bg  = _bg_color(num, nums, row["dir"]) if row["dir"] else ""
                sty = f'background:{bg}' if bg else ""
                if row["type"] == "dual" and row.get("aux"):
                    aux = row["aux"][i]
                    rows_html += (f'<td style="{sty}">'
                                  f'<span>{val}</span>'
                                  f'<span class="aux">{aux}</span></td>')
                else:
                    rows_html += f'<td style="{sty}">{val}</td>'
            rows_html += '</tr>\n'
        return rows_html

    # assemble
    th_row = '<tr><th class="lbl"></th>' + "".join(th_html(*h) for h in headers) + "</tr>"

    tbody = ""
    # contexto primeiro
    tbody += render_section(ctx_section)

    # regime meds
    tbody += f'<tr class="sh"><td colspan="{n_phases+1}">Regime medicamentoso (fármacos usados em mais de {PILL_MIN_PCT*100:.0f}% dos dias)</td></tr>\n'
    tbody += '<tr><td class="lbl" style="vertical-align:top;padding:8px 10px;border-bottom:0"></td>'
    for ph in phases:
        tbody += f'<td style="text-align:left;vertical-align:top;padding:8px 10px;border-bottom:0">{med_pills_html(ph)}</td>'
    tbody += '</tr>\n'

    # doses dos principais meds
    # só fármacos de dose variável; média sobre todos os dias da fase (inclui zeros)
    DOSE_ROWS = [
        ("venvanse_mg_total",     "Lisdexanfetamina (média/dia, todos os dias)"),
        ("zolpidem_mg_total",     "Zolpidem (média/dia, todos os dias)"),
        ("metilfenidato_mg_total","Metilfenidato (média/dia, todos os dias)"),
    ]
    for col_name, label in DOSE_ROWS:
        any_present = any(
            col_name in phase_data[ph].columns and (phase_data[ph][col_name] > 0).any()
            for ph in phases
        )
        if not any_present:
            continue
        tbody += f'<tr><td class="dose-lbl">{label}</td>'
        for ph in phases:
            tbody += f'<td class="dose">{med_dose_html(ph, col_name)}</td>'
        tbody += '</tr>\n'

    # dados principais
    for sec in sections:
        tbody += render_section(sec)

    html = f"""{css}
<div class="wrapper">
  <div class="scroll">
    <table>
      <thead>{th_row}</thead>
      <tbody>{tbody}</tbody>
    </table>
  </div>
  <div class="legend">
    <span><span class="leg-box" style="background:rgba(29,158,117,.28)"></span>melhor valor relativo</span>
    <span><span class="leg-box" style="background:rgba(216,90,48,.28)"></span>pior valor relativo</span>
    <span>· intensidade proporcional ao desvio da mediana global · horários: mediana (principal), μ média · scores subjetivos: média (principal), md mediana</span>
  </div>
  <div class="fn">* Sleep latency estimada — baixa confiabilidade · doses: média dos dias em uso (dose&gt;0) · scores com &lt; 20% dos dias preenchidos na fase são omitidos</div>
</div>"""

    return html


if __name__ == "__main__":
    import data_prep as dp, pandas as pd
    df = dp.prepare(pd.read_csv("synth.csv"))
    out = build_phase_report(df)
    with open("/tmp/phase_report_test.html", "w") as f:
        f.write(out)
    print("gerado /tmp/phase_report_test.html —", len(out), "bytes")
