"""Dashboard de níveis diários — lê direto do Google Sheets publicado em CSV.

Rodar local:  streamlit run app.py
Deploy:       ver README.md

Requer: streamlit, pandas, numpy, plotly, scipy, statsmodels, requests
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sps

import data_prep as dp
import registry as rg

from phase_report import build_phase_report

st.set_page_config(page_title="Níveis diários", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- tema / paleta

# Okabe–Ito: paleta segura para as formas comuns de daltonismo (deutan/protan/tritan).
OK = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#00C896",
    "yellow": "#F0E442",
    "blue": "#4C9BE8",
    "vermillion": "#EF6351",
    "purple": "#D395C8",
    "grey": "#888888",
    "black": "#1A1A1A",
    # dark theme surfaces
    "bg":      "#0E1117",
    "surface": "#1A1D27",
    "grid":    "rgba(255,255,255,0.07)",
    "text":    "#E0E0E0",
    "subtext": "#9AA0B2",
}

# PALETTE base (constantes). Quando há registro, é estendida/substituída pela
# camada de "registro efetivo" (ver mais abaixo). Mantida aqui como fallback.
PALETTE = {
    "Energia": OK["orange"],
    "Cognição": OK["blue"],
    "Atenção": OK["purple"],
    "Humor": OK["green"],
    "Fluência/espontaneidade": OK["vermillion"],
}
SLEEP_STAGE_COLORS = {
    "deep_sleep_h": (OK["blue"], "Profundo"),
    "light_sleep_h": (OK["skyblue"], "Leve"),
    "rem_sleep_h": (OK["purple"], "REM"),
    "awake_sleep_h": ("#BFBFBF", "Acordado"),
}
DIVERGING = "RdBu_r"
SEQ = "Teal"

FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

_base = pio.templates["plotly_dark"]
_base.layout.font = dict(family=FONT, size=13, color=OK["text"])
_base.layout.title.font = dict(family=FONT, size=16, color=OK["text"])
_base.layout.paper_bgcolor = OK["bg"]
_base.layout.plot_bgcolor  = OK["surface"]
_base.layout.colorway = [
    OK["blue"], OK["orange"], OK["green"], OK["vermillion"],
    OK["purple"], OK["skyblue"], OK["yellow"], OK["grey"],
]
pio.templates.default = "plotly_dark"


def style_fig(fig, height=None, legend_top=True):
    """Aplica acabamento consistente dark: grid suave, hover legível, legenda no topo."""
    fig.update_layout(
        font=dict(family=FONT, size=13, color=OK["text"]),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=OK["surface"],
                        font_color=OK["text"], bordercolor=OK["grid"]),
        margin=dict(t=54, b=28, l=10, r=10),
        plot_bgcolor=OK["surface"],
        paper_bgcolor=OK["bg"],
    )
    if height:
        fig.update_layout(height=height)
    if legend_top:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0.0)",
            font=dict(color=OK["text"]),
        ))
    fig.update_xaxes(showgrid=True, gridcolor=OK["grid"], zeroline=False,
                     color=OK["subtext"], linecolor=OK["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=OK["grid"], zeroline=False,
                     color=OK["subtext"], linecolor=OK["grid"])
    return fig


def add_phase_vlines(fig, change_points, subplot=False, annotate=True):
    """Linha vertical tracejada em cada troca de fase. Anota o rótulo da nova fase."""
    if not change_points:
        return fig
    rc = dict(row="all", col="all") if subplot else {}
    for cp in change_points:
        fig.add_vline(
            x=cp["date"], line=dict(color="rgba(255,255,255,0.35)", width=1.2, dash="dot"),
            **rc,
        )
    if annotate:
        for cp in change_points:
            fig.add_annotation(
                x=cp["date"], y=1.0, yref="paper", yanchor="bottom",
                text=f"fase {cp['to']}", showarrow=False,
                font=dict(size=10, color=OK["subtext"]),
                bgcolor="rgba(0,0,0,0.45)",
            )
    return fig


# ---------------------------------------------------------------- prefs persistentes (Supabase)

PREFS_TABLE = "dashboard_prefs"
PREFS_ROW_ID = "default"


def _supabase_headers():
    key = st.secrets.get("supabase_key", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


@st.cache_data(ttl=10, show_spinner=False)
def _load_prefs_cached(_cache_buster: int = 0) -> dict:
    """Lê o blob de prefs salvo. Qualquer falha degrada silenciosamente para {}."""
    base_url = st.secrets.get("supabase_url", "")
    if not base_url or not st.secrets.get("supabase_key", ""):
        return {}
    try:
        r = requests.get(
            f"{base_url}/rest/v1/{PREFS_TABLE}",
            params={"id": f"eq.{PREFS_ROW_ID}", "select": "prefs"},
            headers=_supabase_headers(),
            timeout=5,
        )
        r.raise_for_status()
        rows = r.json()
        if rows:
            return rows[0].get("prefs") or {}
    except Exception:
        pass
    return {}


def save_prefs(prefs: dict) -> None:
    """Upsert fire-and-forget do blob de prefs. Nunca propaga exceção."""
    base_url = st.secrets.get("supabase_url", "")
    if not base_url or not st.secrets.get("supabase_key", ""):
        return
    try:
        requests.post(
            f"{base_url}/rest/v1/{PREFS_TABLE}",
            params={"on_conflict": "id"},
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},
            data=json.dumps({"id": PREFS_ROW_ID, "prefs": prefs}),
            timeout=5,
        )
    except Exception:
        pass
    _load_prefs_cached.clear()


if "prefs_loaded" not in st.session_state:
    st.session_state["saved_prefs"] = _load_prefs_cached()
    st.session_state["prefs_loaded"] = True

SAVED = st.session_state.get("saved_prefs", {})


def saved_or(key, fallback):
    """Valor salvo se existir e ainda for um tipo plausível, senão o fallback hardcoded."""
    return SAVED[key] if key in SAVED and SAVED[key] is not None else fallback


def filtered_default(saved_list, valid_options):
    """Filtra uma lista salva contra as opções válidas atuais."""
    if not isinstance(saved_list, list):
        return None
    out = [c for c in saved_list if c in valid_options]
    return out or None


def clamp_period(saved_start, saved_end, dmin, dmax):
    """saved_* podem ser strings ISO ou None. Retorna (start, end) válidos em [dmin, dmax]."""
    import datetime as _dt
    try:
        if isinstance(saved_start, str):
            saved_start = _dt.date.fromisoformat(saved_start)
        if isinstance(saved_end, str):
            saved_end = _dt.date.fromisoformat(saved_end)
        if saved_start is None or saved_end is None:
            return dmin, dmax
        if saved_end < dmin or saved_start > dmax:
            return dmin, dmax
        s = max(dmin, min(saved_start, dmax))
        e = max(dmin, min(saved_end, dmax))
        if s > e:
            return dmin, dmax
        return s, e
    except Exception:
        return dmin, dmax


# ---------------------------------------------------------------- registro de colunas

# O registro de schema (column_registry) é uma chave no blob de prefs.
# Vazio => app usa as constantes do data_prep (comportamento idêntico ao anterior).
COLUMN_REGISTRY = saved_or("column_registry", []) or []


def save_registry(reg_list: list) -> None:
    """Persiste o registro como mais uma chave no blob de prefs, sem apagar o resto."""
    global SAVED, COLUMN_REGISTRY
    prefs = dict(SAVED)
    prefs["column_registry"] = reg_list
    save_prefs(prefs)
    st.session_state["saved_prefs"] = prefs
    SAVED = prefs
    COLUMN_REGISTRY = reg_list


# keys simples (lidas direto do session_state, valor já serializável em JSON)
_AUTOSAVE_SIMPLE_KEYS = [
    "period_start", "period_end", "fases_sel", "roll", "show_phase_lines",
    "scores_chosen",
    "reg_y", "reg_x", "target_mode", "include_fase", "ctrl_trend", "ctrl_weekend", "ctrl_monday",
    "corr_x_col", "corr_y_col", "corr_lag", "corr_color_by_phase",
    "matrix_cols", "matrix_method",
    "emo_chosen", "emo_roll",
    "hist_var", "hist_use",
]


def autosave():
    """Lê os widgets atuais do session_state e salva o blob inteiro."""
    global SAVED
    prefs = dict(SAVED)
    for k in _AUTOSAVE_SIMPLE_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            if isinstance(v, (list, tuple)) and v and hasattr(v[0], "isoformat"):
                v = [d.isoformat() for d in v]
            elif hasattr(v, "isoformat"):
                v = v.isoformat()
            prefs[k] = v

    pred_cfg_out = {}
    for c in st.session_state.get("reg_x", []):
        lg = st.session_state.get(f"lag_{c}")
        mm = st.session_state.get(f"mm_{c}")
        if lg is not None and mm is not None:
            pred_cfg_out[c] = [int(lg), int(mm)]
    prefs["pred_cfg"] = pred_cfg_out

    # preserva o registro já salvo (autosave de outras abas não pode apagá-lo)
    if "column_registry" in SAVED:
        prefs["column_registry"] = SAVED["column_registry"]

    save_prefs(prefs)
    st.session_state["saved_prefs"] = prefs
    SAVED = prefs


# ---------------------------------------------------------------- fonte de dados

st.sidebar.title("Níveis diários")

CACHE_TTL = 300  # segundos


@st.cache_data(ttl=CACHE_TTL, show_spinner="Carregando planilha...")
def fetch(url: str, _registry_key: str = "") -> pd.DataFrame:
    """_registry_key entra na assinatura só pra invalidar o cache quando o
    registro muda (parsing depende dele). O valor em si não é usado aqui."""
    return dp.prepare(dp.load_csv(url), registry=COLUMN_REGISTRY or None)


# chave de cache derivada do registro: muda => refaz o prepare
_reg_cache_key = json.dumps(COLUMN_REGISTRY, sort_keys=True) if COLUMN_REGISTRY else ""

default_url = st.secrets.get("sheet_csv_url", "")
url = st.sidebar.text_input(
    "URL CSV da planilha",
    value=default_url,
    help="URL de exportação CSV do Google Sheets (ver README). "
    "Cache de 5 min — edições na planilha aparecem no próximo reload.",
)
uploaded = st.sidebar.file_uploader("…ou subir CSV manualmente", type="csv")

if st.sidebar.button("🔄 Recarregar agora"):
    st.cache_data.clear()
    st.rerun()

# guarda o CSV cru (strings) pra varredura do registro, sem reprocessar
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_raw(url: str) -> pd.DataFrame:
    return dp.load_csv(url)


df = None
df_raw = None
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    df = dp.prepare(df_raw.copy(), registry=COLUMN_REGISTRY or None)
elif url.strip():
    try:
        df = fetch(url.strip(), _reg_cache_key)
        df_raw = fetch_raw(url.strip())
    except Exception as e:
        st.error(f"Falha ao carregar a URL: {e}")

if df is None or df.empty:
    st.info(
        "Aponte a URL CSV da planilha na barra lateral, ou suba um CSV. "
        "Instruções de publicação da planilha no README."
    )
    st.stop()

# ---------------------------------------------------------------- registro efetivo
# Resolve, UMA vez, as estruturas que o resto do app usa: scores, labels, paleta,
# meds. Com registro => deriva dele. Sem registro => constantes (idêntico ao atual).

def _build_effective(df, registry):
    rmap = rg.registry_to_map(registry) if registry else {}
    if rmap and rg.scores(rmap):
        sc = [c for c in rg.scores(rmap) if c in df.columns]
        labels = dict(dp.SCORE_LABELS)
        labels.update(rg.score_labels(rmap))
        dirs = rg.score_directions(rmap)
        palette = dict(PALETTE)  # mantém base; estende/sobrescreve por label
        for i, c in enumerate(sc):
            lbl = labels.get(c, c)
            palette[lbl] = rg.color_for(rmap, c, i)
        med_cols = [c for c in rg.meds(rmap) if c in df.columns]
        return sc, labels, dirs, palette, med_cols
    # fallback: constantes
    sc = [c for c in dp.SCORE_COLS if c in df.columns]
    labels = dict(dp.SCORE_LABELS)
    dirs = {c: "higher" for c in dp.SCORE_COLS}
    return sc, labels, dirs, dict(PALETTE), None


SCORE_COLS_EFF, SCORE_LABELS_EFF, SCORE_DIRS_EFF, PALETTE_EFF, MED_COLS_EFF = \
    _build_effective(df, COLUMN_REGISTRY)


def score_label(c):
    return SCORE_LABELS_EFF.get(c, c)


def palette_for(label):
    return PALETTE_EFF.get(label, OK["grey"])


# ------------------------------------------------ filtro de período (slider)
dmin, dmax = df["date"].min().date(), df["date"].max().date()
if dmin == dmax:
    period = (dmin, dmax)
    st.sidebar.caption(f"Único dia: {dmin:%d/%m/%y}")
else:
    _default_start, _default_end = clamp_period(
        SAVED.get("period_start"), SAVED.get("period_end"), dmin, dmax
    )
    period = st.sidebar.slider(
        "Período",
        min_value=dmin, max_value=dmax,
        value=(_default_start, _default_end),
        format="DD/MM/YY",
        help="Arraste as pontas para recortar o intervalo de datas.",
        key="_period_widget",
        on_change=lambda: (
            st.session_state.__setitem__("period_start", st.session_state["_period_widget"][0]),
            st.session_state.__setitem__("period_end", st.session_state["_period_widget"][1]),
            autosave(),
        ),
    )
    st.session_state.setdefault("period_start", period[0])
    st.session_state.setdefault("period_end", period[1])
df = df[(df["date"].dt.date >= period[0]) & (df["date"].dt.date <= period[1])]

# ------------------------------------------------ filtro de fase
has_fase = "fase_label" in df.columns and df["fase_label"].nunique() > 1
if has_fase:
    fase_opts = dp.fase_order(df["fase_label"].unique())
    _fases_default = filtered_default(SAVED.get("fases_sel"), fase_opts) or fase_opts
    fases_sel = st.sidebar.multiselect(
        "Fase", fase_opts, default=_fases_default,
        help="Vazio = todas. Filtrar fases pode deixar o período não-contíguo.",
        key="fases_sel", on_change=autosave,
    )
    if fases_sel and len(fases_sel) < len(fase_opts):
        df = df[df["fase_label"].isin(fases_sel)]

if df.empty:
    st.warning("Nenhum dia atende aos filtros atuais.")
    st.stop()

roll = st.sidebar.slider(
    "Janela média móvel (dias)", 3, 28, int(saved_or("roll", 7)),
    key="roll", on_change=autosave,
)

show_phase_lines = st.sidebar.checkbox(
    "Marcar trocas de fase nos gráficos",
    value=bool(saved_or("show_phase_lines", has_fase)), disabled=not has_fase,
    key="show_phase_lines", on_change=autosave,
)

# score_cols e meds agora vêm do registro efetivo
score_cols = [c for c in SCORE_COLS_EFF if c in df.columns]
meds = dp.active_meds(df, med_cols=MED_COLS_EFF)
change_points = dp.phase_change_points(df) if (has_fase and show_phase_lines) else []

st.sidebar.caption(
    f"{len(df)} dias · {period[0]:%d/%m/%y} → {period[1]:%d/%m/%y} · cache {CACHE_TTL//60} min"
)

# ---------------------------------------------------------------- helpers de plot


def line_with_roll(fig, x, y, name, color, window, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name, mode="markers",
            marker=dict(color=color, size=6, opacity=0.4,
                        line=dict(width=0.5, color="white")),
            legendgroup=name, showlegend=False,
            hovertemplate="%{y:.1f}<extra>" + name + "</extra>",
        ),
        **kw,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y.rolling(window, min_periods=max(2, window // 3)).mean(),
            name=name, mode="lines", line=dict(color=color, width=2.6, shape="spline"),
            legendgroup=name,
            hovertemplate="%{y:.1f}<extra>" + name + " (média móvel)</extra>",
        ),
        **kw,
    )


def med_label(m: str) -> str:
    """Rótulo de med: usa o do registro efetivo se houver, senão deriva do nome."""
    lbl = SCORE_LABELS_EFF.get(m)  # SCORE_LABELS_EFF só tem scores; meds não entram aqui
    rmap = rg.registry_to_map(COLUMN_REGISTRY) if COLUMN_REGISTRY else {}
    e = rmap.get(m)
    if e and e.get("kind") == "med" and e.get("label"):
        return e["label"]
    return m.replace("_mg_total", "").replace("_mg", "").replace("_", " ")


# ---------------------------------------------------------------- tabs

(tab_vis, tab_scores, tab_sono, tab_med, tab_atv, tab_corr, tab_emo,
 tab_fases, tab_dados, tab_reg) = st.tabs(
    ["Visão geral", "Scores", "Sono", "Medicações", "Atividade", "Correlações",
     "Emoções", "Fases", "Dados", "Registro"]
)

# ------------------------------------------------ visão geral
with tab_vis:
    last7 = df.tail(7)
    prev7 = df.iloc[-14:-7] if len(df) >= 14 else pd.DataFrame(columns=df.columns)

    def kpi(col_st, label, series_now, series_prev, fmt="{:.1f}"):
        now = series_now.mean()
        prev = series_prev.mean() if len(series_prev) else np.nan
        delta = None if (pd.isna(now) or pd.isna(prev)) else fmt.format(now - prev)
        col_st.metric(label, "—" if pd.isna(now) else fmt.format(now), delta)

    c = st.columns(5)
    kpi(c[0], "Humor (média 7d)", last7.get("mood_score", pd.Series(dtype=float)), prev7.get("mood_score", pd.Series(dtype=float)))
    kpi(c[1], "Energia (7d)", last7.get("energy_score", pd.Series(dtype=float)), prev7.get("energy_score", pd.Series(dtype=float)))
    kpi(c[2], "Sono h (7d)", last7.get("sleep_duration_h", pd.Series(dtype=float)), prev7.get("sleep_duration_h", pd.Series(dtype=float)))
    kpi(c[3], "Eficiência sono (7d)", last7.get("sleep_efficiency", pd.Series(dtype=float)), prev7.get("sleep_efficiency", pd.Series(dtype=float)), "{:.0f}")
    kpi(c[4], "Passos (7d)", last7.get("steps", pd.Series(dtype=float)), prev7.get("steps", pd.Series(dtype=float)), "{:.0f}")
    st.caption("Delta = média dos últimos 7 dias vs 7 dias anteriores.")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("Humor e energia", "Duração do sono (h)"),
    )
    for col_name in ["mood_score", "energy_score"]:
        if col_name in df.columns:
            label = score_label(col_name)
            line_with_roll(fig, df["date"], df[col_name], label, palette_for(label), roll, row=1, col=1)
    if "sleep_duration_h" in df.columns:
        fig.add_trace(
            go.Bar(x=df["date"], y=df["sleep_duration_h"], name="Sono (h)",
                   marker_color=OK["skyblue"], opacity=0.8, showlegend=False,
                   hovertemplate="%{y:.1f} h<extra>sono</extra>"),
            row=2, col=1,
        )
    add_phase_vlines(fig, change_points, subplot=True)
    style_fig(fig, height=580)
    fig.update_layout(hovermode="x unified", barmode="overlay")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ scores
with tab_scores:
    _scores_default = filtered_default(SAVED.get("scores_chosen"), score_cols) or score_cols
    chosen = st.multiselect(
        "Séries", score_cols, default=_scores_default,
        format_func=lambda c: score_label(c),
        key="scores_chosen", on_change=autosave,
    )
    fig = go.Figure()
    for col_name in chosen:
        label = score_label(col_name)
        line_with_roll(fig, df["date"], df[col_name], label, palette_for(label), roll)
    if chosen:
        top = df[chosen].max().max() * 1.02
        fig.add_trace(go.Scatter(
            x=df["date"], y=[top] * len(df), mode="markers",
            marker=dict(symbol="line-ns", size=10, color="rgba(0,0,0,0.12)"),
            text=df["obs_all"], hovertemplate="%{text}<extra>obs</extra>",
            showlegend=False,
        ))
    add_phase_vlines(fig, change_points)
    style_fig(fig, height=540)
    fig.update_layout(hovermode="x unified", yaxis_title="Score")
    st.plotly_chart(fig, width="stretch")
    st.caption("Pontos = valor diário; linha = média móvel. Marcas cinzas no topo carregam as observações do dia (hover).")

# ------------------------------------------------ sono
with tab_sono:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{}], [{}]],
        subplot_titles=("Duração × eficiência", "Estágios (h)", "Horário de dormir/acordar"),
    )
    if "sleep_duration_h" in df.columns:
        fig.add_trace(
            go.Bar(x=df["date"], y=df["sleep_duration_h"], name="Duração (h)",
                   marker_color=OK["skyblue"], opacity=0.8,
                   hovertemplate="%{y:.1f} h<extra>duração</extra>"),
            row=1, col=1, secondary_y=False,
        )
    if "sleep_efficiency" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["sleep_efficiency"], name="Eficiência (%)",
                       line=dict(color=OK["orange"], width=2.4),
                       hovertemplate="%{y:.0f}%<extra>eficiência</extra>"),
            row=1, col=1, secondary_y=True,
        )
    for stage, (color, label) in SLEEP_STAGE_COLORS.items():
        if stage in df.columns:
            fig.add_trace(
                go.Bar(x=df["date"], y=df[stage], name=label, marker_color=color,
                       hovertemplate="%{y:.1f} h<extra>" + label + "</extra>"),
                row=2, col=1,
            )
    bed = df.get("bed_time_h")
    wake = df.get("wake_time_h")
    if bed is not None:
        fig.add_trace(
            go.Scatter(x=df["date"], y=bed, name="Foi pra cama",
                       mode="markers+lines", line=dict(color=OK["blue"], width=1.4),
                       marker=dict(size=5),
                       text=bed.map(dp.hours_to_label),
                       hovertemplate="%{text}<extra>cama</extra>"),
            row=3, col=1,
        )
    if wake is not None:
        fig.add_trace(
            go.Scatter(x=df["date"], y=wake, name="Acordou",
                       mode="markers+lines", line=dict(color=OK["orange"], width=1.4),
                       marker=dict(size=5),
                       text=wake.map(dp.hours_to_label),
                       hovertemplate="%{text}<extra>acordou</extra>"),
            row=3, col=1,
        )
    tickvals = list(range(0, 34, 2))
    fig.update_yaxes(tickvals=tickvals,
                     ticktext=[dp.hours_to_label(v) for v in tickvals], row=3, col=1)
    add_phase_vlines(fig, change_points, subplot=True)
    style_fig(fig, height=840)
    fig.update_layout(barmode="stack", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    extras = {
        "sleep_latency_estimate_minutes": "Latência (min)",
        "full_awakenings": "Despertares completos",
        "restlessness_mins": "Inquietação (min)",
    }
    for (col_name, label), col_st in zip(extras.items(), [c1, c2, c3]):
        if col_name in df.columns:
            mini = go.Figure(go.Bar(x=df["date"], y=df[col_name], marker_color=OK["purple"]))
            style_fig(mini, height=210, legend_top=False)
            mini.update_layout(title=label, margin=dict(t=42, b=10, l=10, r=10))
            col_st.plotly_chart(mini, width="stretch")

# ------------------------------------------------ medicações
with tab_med:
    if not meds:
        st.info("Nenhuma medicação com registro no período.")
    else:
        sub = df.set_index("date")[meds]
        norm = sub.div(sub.max().replace(0, np.nan))
        fig = go.Figure(
            go.Heatmap(
                z=norm.T.values,
                x=sub.index, y=[med_label(m) for m in meds],
                customdata=sub.T.values,
                hovertemplate="%{x|%d/%m}: %{customdata:.1f} mg<extra>%{y}</extra>",
                colorscale=SEQ, showscale=False, ygap=3,
            )
        )
        style_fig(fig, height=90 + 44 * len(meds), legend_top=False)
        fig.update_layout(title="Doses por dia (intensidade relativa ao máximo de cada fármaco)")
        for cp in change_points:
            fig.add_vline(x=cp["date"], line=dict(color="rgba(0,0,0,0.5)", width=1.2, dash="dot"))
        st.plotly_chart(fig, width="stretch")

        med_sel = st.selectbox("Detalhe de um fármaco", meds, format_func=med_label)
        fig2 = go.Figure(
            go.Scatter(x=df["date"], y=df[med_sel], mode="lines+markers",
                       line_shape="hv", line=dict(color=OK["green"], width=2.4),
                       marker=dict(size=5),
                       hovertemplate="%{y:.1f} mg<extra></extra>")
        )
        add_phase_vlines(fig2, change_points)
        style_fig(fig2, height=280, legend_top=False)
        fig2.update_layout(yaxis_title="mg / unidades")
        st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------ atividade
with tab_atv:
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=list(dp.ACTIVITY_LABELS.values()))
    pos = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = [OK["blue"], OK["green"], OK["vermillion"], OK["purple"]]
    for (col_name, _), (r, c_), color in zip(dp.ACTIVITY_LABELS.items(), pos, colors):
        if col_name in df.columns:
            if col_name in ("avg_bpm", "VFC"):
                line_with_roll(fig, df["date"], df[col_name], col_name, color, roll, row=r, col=c_)
            else:
                fig.add_trace(
                    go.Bar(x=df["date"], y=df[col_name], marker_color=color, opacity=0.85,
                           showlegend=False),
                    row=r, col=c_,
                )
    add_phase_vlines(fig, change_points, subplot=True, annotate=False)
    style_fig(fig, height=640, legend_top=False)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ correlações
with tab_corr:
    num_cols = sorted(
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= 5
        and c not in ("weekday", "fase")
    )
    med_set = set(meds)

    st.markdown("##### Dispersão com lag")
    c1, c2, c3 = st.columns([2, 2, 1])
    _saved_x_col = saved_or("corr_x_col", None)
    _x_idx = (num_cols.index(_saved_x_col) if _saved_x_col in num_cols
              else (num_cols.index("sleep_duration_h") if "sleep_duration_h" in num_cols else 0))
    x_col = c1.selectbox("X (preditor)", num_cols, index=_x_idx,
                         key="corr_x_col", on_change=autosave)
    _saved_y_col = saved_or("corr_y_col", None)
    _y_idx = (num_cols.index(_saved_y_col) if _saved_y_col in num_cols
              else (num_cols.index("mood_score") if "mood_score" in num_cols else 0))
    y_col = c2.selectbox("Y (desfecho)", num_cols, index=_y_idx,
                         key="corr_y_col", on_change=autosave)
    lag = c3.number_input("Lag (dias)", 0, 7, int(saved_or("corr_lag", 0)),
                          help="X de k dias atrás vs Y de hoje.",
                          key="corr_lag", on_change=autosave)

    x_is_med = x_col in med_set
    med_mode = None
    if x_is_med:
        opts = ["Titulação (só dias em uso)", "Uso vs não-uso (todos os dias)"]
        if not dp.med_varies_within_use(df, x_col):
            opts = opts[::-1]
        med_mode = st.radio(
            f"Modo para **{med_label(x_col)}**", opts, horizontal=True,
            help="Titulação isola dose-resposta. Uso vs não-uso compara desfecho com/sem o fármaco.",
        )

    if x_is_med and med_mode and med_mode.startswith("Uso"):
        used = (df[x_col].fillna(0) > 0).astype(int).shift(lag)
        comp = pd.DataFrame({"used": used, "y": df[y_col]}).dropna()
        g0 = comp.loc[comp["used"] == 0, "y"]
        g1 = comp.loc[comp["used"] == 1, "y"]
        if len(g0) >= 3 and len(g1) >= 3:
            fig = go.Figure()
            fig.add_trace(go.Box(y=g0, name="sem", marker_color=OK["grey"],
                                 boxpoints="all", jitter=0.4, pointpos=0))
            fig.add_trace(go.Box(y=g1, name="com", marker_color=OK["green"],
                                 boxpoints="all", jitter=0.4, pointpos=0))
            style_fig(fig, height=440, legend_top=False)
            fig.update_layout(yaxis_title=y_col,
                              title=f"{y_col} em dias com vs sem {med_label(x_col)}")
            st.plotly_chart(fig, width="stretch")
            u, pu = sps.mannwhitneyu(g1, g0, alternative="two-sided")
            d_md = g1.median() - g0.median()
            st.caption(
                f"n_com = {len(g1)} · n_sem = {len(g0)} · "
                f"Δmediana = {d_md:+.2f} · Mann–Whitney U = {u:.0f} (p = {pu:.3f}). "
                "Teste não-paramétrico, sem ajuste para confundidores."
            )
        else:
            st.warning("Poucos dias em algum dos grupos (com/sem) pra comparar.")
    else:
        x_raw = df[x_col].shift(lag)
        pair = pd.DataFrame({
            "x": x_raw.values, "y": df[y_col].values,
            "date": df["date"].dt.strftime("%d/%m"),
            "fase": df.get("fase_label", pd.Series(["—"] * len(df))).values,
        })
        note = ""
        if x_is_med:
            pair = pair[pair["x"] > 0]
            note = " · só dias em uso"
        pair = pair.dropna(subset=["x", "y"])

        if len(pair) >= 3:
            r_p, p_p = sps.pearsonr(pair["x"], pair["y"])
            r_s, p_s = sps.spearmanr(pair["x"], pair["y"])
            slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
            xs = np.linspace(pair["x"].min(), pair["x"].max(), 50)

            color_by_phase = has_fase and pair["fase"].nunique() > 1 and st.checkbox(
                "Colorir pontos por fase", value=bool(saved_or("corr_color_by_phase", False)),
                key="corr_color_by_phase", on_change=autosave)
            fig = go.Figure()
            if color_by_phase:
                cmap = [OK["blue"], OK["orange"], OK["green"], OK["vermillion"],
                        OK["purple"], OK["skyblue"]]
                for i, f in enumerate(dp.fase_order(pair["fase"].unique())):
                    sl = pair[pair["fase"] == f]
                    fig.add_trace(go.Scatter(
                        x=sl["x"], y=sl["y"], mode="markers", name=f"fase {f}",
                        text=sl["date"], marker=dict(size=8, opacity=0.75,
                                                     color=cmap[i % len(cmap)]),
                        hovertemplate="%{text}: (%{x:.1f}, %{y:.1f})<extra>fase " + f + "</extra>",
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=pair["x"], y=pair["y"], mode="markers", text=pair["date"],
                    marker=dict(color=OK["blue"], size=8, opacity=0.7),
                    hovertemplate="%{text}: (%{x:.1f}, %{y:.1f})<extra></extra>", name="dias",
                ))
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                     line=dict(color=OK["vermillion"], dash="dash", width=2),
                                     name="OLS"))
            style_fig(fig, height=460, legend_top=color_by_phase)
            fig.update_layout(
                xaxis_title=f"{x_col}" + (f" (lag {lag}d)" if lag else "") + note,
                yaxis_title=y_col,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"n = {len(pair)}{note} · Pearson r = {r_p:.2f} (p = {p_p:.3f}) · "
                f"Spearman ρ = {r_s:.2f} (p = {p_s:.3f})"
            )
        else:
            st.warning("Poucos pares válidos pra esse cruzamento.")

    st.markdown("---")
    st.markdown("##### Regressão (OLS) com transformações")
    st.caption(
        "Modelo descritivo, não causal. Séries diárias são autocorrelacionadas — "
        "use AR(1) ou Δ pra não inflar a significância. Leia os n por fase antes de interpretar."
    )

    _saved_reg_y = saved_or("reg_y", None)
    _reg_y_default_idx = (
        num_cols.index(_saved_reg_y) if _saved_reg_y in num_cols
        else (num_cols.index("mood_score") if "mood_score" in num_cols else 0)
    )
    reg_y = st.selectbox(
        "Desfecho (Y)", num_cols, index=_reg_y_default_idx,
        key="reg_y", on_change=autosave,
    )

    _mode_opts = ["Nível (cru)", "AR(1): incluir Y(t-1)", "Δ alvo (primeira diferença)"]
    _saved_mode = saved_or("target_mode", "Nível (cru)")
    _mode_idx = _mode_opts.index(_saved_mode) if _saved_mode in _mode_opts else 0
    target_mode = st.radio(
        "Transformação do alvo / autocorrelação",
        _mode_opts, index=_mode_idx, horizontal=True,
        key="target_mode", on_change=autosave,
        help="Nível = Y bruto. AR(1) adiciona Y de ontem. Δ modela a variação diária. "
        "AR(1) e Δ são mutuamente exclusivos.",
    )

    _candidate_x = [c for c in num_cols if c != reg_y]
    _saved_reg_x = filtered_default(saved_or("reg_x", None), _candidate_x)
    pred_default = _saved_reg_x if _saved_reg_x is not None else (
        [c for c in ["sleep_duration_h"] if c in num_cols and c != reg_y]
    )
    reg_x = st.multiselect(
        "Preditores", _candidate_x, default=pred_default,
        key="reg_x", on_change=autosave,
    )

    _saved_pred_cfg = saved_or("pred_cfg", {}) or {}
    pred_cfg = {}
    if reg_x:
        st.caption("Transformação por preditor (lag = dias atrás; MM = média móvel de N dias):")
        for c in reg_x:
            cc1, cc2, cc3 = st.columns([3, 1, 1])
            cc1.markdown(f"&nbsp;**{med_label(c)}**", unsafe_allow_html=True)
            _saved_lg, _saved_mm = _saved_pred_cfg.get(c, [0, 1]) if isinstance(
                _saved_pred_cfg.get(c), list) and len(_saved_pred_cfg.get(c, [])) == 2 else (0, 1)
            lg = cc2.number_input("lag", 0, 14, int(_saved_lg), key=f"lag_{c}",
                                  on_change=autosave)
            mm = cc3.number_input("MM", 1, 28, int(_saved_mm), key=f"mm_{c}",
                                  help="1 = sem média móvel", on_change=autosave)
            pred_cfg[c] = (int(lg), int(mm))

    cctrl1, cctrl2 = st.columns(2)
    include_fase = cctrl1.checkbox(
        "Incluir fase  C(fase)", value=bool(saved_or("include_fase", False)),
        disabled=not has_fase, key="include_fase", on_change=autosave,
        help="Deslocamentos de nível entre fases.",
    )
    ctrl_trend = cctrl1.checkbox(
        "Tendência linear (trend)", value=bool(saved_or("ctrl_trend", False)),
        key="ctrl_trend", on_change=autosave,
        help="Dia-índice como preditor.",
    )
    ctrl_weekend = cctrl2.checkbox(
        "Dummy fim de semana", value=bool(saved_or("ctrl_weekend", False)),
        key="ctrl_weekend", on_change=autosave,
    )
    ctrl_monday = cctrl2.checkbox(
        "Dummy segunda-feira", value=bool(saved_or("ctrl_monday", False)),
        key="ctrl_monday", on_change=autosave,
    )

    if st.button("Rodar regressão"):
        try:
            import statsmodels.api as sm
        except ImportError:
            st.error("statsmodels não instalado. Adicione `statsmodels` ao requirements.txt.")
            st.stop()

        if not reg_x and not include_fase and target_mode == "Nível (cru)" and not (
            ctrl_trend or ctrl_weekend or ctrl_monday):
            st.warning("Escolha ao menos um preditor ou uma transformação.")
        else:
            d = df.sort_values("date").reset_index(drop=True).copy()

            y = d[reg_y].astype(float)
            y_name = reg_y
            if target_mode.startswith("Δ"):
                y = y.diff()
                y_name = f"Δ{reg_y}"

            reg = pd.DataFrame({y_name: y})
            built, collinear = [], []

            if target_mode.startswith("AR"):
                ar_name = f"{reg_y}__lag1"
                reg[ar_name] = d[reg_y].astype(float).shift(1)
                built.append(ar_name)

            for c in reg_x:
                lg, mm = pred_cfg[c]
                s = d[c].astype(float)
                if mm > 1:
                    s = s.rolling(mm, min_periods=max(2, mm // 2)).mean()
                if lg > 0:
                    s = s.shift(lg)
                nm = c + (f"_mm{mm}" if mm > 1 else "") + (f"_lag{lg}" if lg > 0 else "")
                reg[nm] = s
                built.append(nm)
                if include_fase:
                    g = d.assign(_v=s).groupby("fase_label")["_v"].nunique(dropna=True)
                    if (g.fillna(0) <= 1).all():
                        collinear.append(med_label(c))

            if ctrl_trend:
                reg["trend"] = np.arange(len(d), dtype=float)
                built.append("trend")
            if ctrl_weekend:
                reg["weekend"] = (d["date"].dt.weekday >= 5).astype(float)
                built.append("weekend")
            if ctrl_monday:
                reg["monday"] = (d["date"].dt.weekday == 0).astype(float)
                built.append("monday")

            fase_cols = []
            if include_fase:
                dummies = pd.get_dummies(d["fase_label"], prefix="fase", drop_first=True, dtype=float)
                for col in dummies.columns:
                    reg[col] = dummies[col].values
                fase_cols = list(dummies.columns)
                built += fase_cols

            if collinear:
                st.warning(
                    "⚠️ Colinear com a fase (constante dentro de cada fase): "
                    + ", ".join(collinear)
                    + ". O efeito desses preditores não é separável do da fase. "
                    "Remova-os OU tire a fase."
                )

            reg = reg.dropna()
            k = len(built)
            if len(reg) < k + 2:
                st.warning(f"Poucas observações completas (n = {len(reg)}) pra {k} termos.")
            else:
                X = sm.add_constant(reg[built], has_constant="add")
                res = sm.OLS(reg[y_name], X).fit()
                coefs = pd.DataFrame({
                    "coef": res.params, "EP": res.bse,
                    "t": res.tvalues, "p": res.pvalues,
                    "IC 2.5%": res.conf_int()[0], "IC 97.5%": res.conf_int()[1],
                }).round(3)
                st.dataframe(coefs, width="stretch")
                terms_txt = " + ".join(built) if built else "const"
                st.caption(
                    f"`{y_name} ~ {terms_txt}` · n = {int(res.nobs)} · R² = {res.rsquared:.3f} · "
                    f"R² aj. = {res.rsquared_adj:.3f} · F p = {res.f_pvalue:.3g} · "
                    f"nº de condição = {res.condition_number:.0f}"
                    + ("  (alto → multicolinearidade)" if res.condition_number > 100 else "")
                )
                if target_mode.startswith("AR"):
                    st.caption("AR(1): o coeficiente de Y(t-1) capta a inércia.")
                if include_fase:
                    npf = d.groupby("fase_label")[reg_y].size()
                    st.caption("n por fase: " + " · ".join(f"{k_}: {v}" for k_, v in npf.items()))

    st.markdown("---")
    st.markdown("##### Matriz de correlação")
    default_matrix = [c for c in (
        score_cols + ["sleep_duration_h", "sleep_efficiency", "deep_sleep_h",
                      "rem_sleep_h", "steps", "avg_bpm", "VFC"]
    ) if c in num_cols]
    _matrix_default = filtered_default(saved_or("matrix_cols", None), num_cols) or default_matrix
    matrix_cols = st.multiselect("Variáveis", num_cols, default=_matrix_default,
                                 key="matrix_cols", on_change=autosave)
    if len(matrix_cols) >= 2:
        _method_opts = ["spearman", "pearson"]
        _saved_method = saved_or("matrix_method", "spearman")
        _method_idx = _method_opts.index(_saved_method) if _saved_method in _method_opts else 0
        method = st.radio("Método", _method_opts, index=_method_idx, horizontal=True,
                          key="matrix_method", on_change=autosave)
        corr = df[matrix_cols].corr(method=method)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            zmin=-1, zmax=1, colorscale=DIVERGING,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=11, color=OK["text"]),
            colorbar=dict(title=dict(text="ρ / r", font=dict(color=OK["subtext"])),
                          tickfont=dict(color=OK["subtext"])),
        ))
        style_fig(fig, height=140 + 42 * len(matrix_cols), legend_top=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Selecione ao menos duas variáveis.")

# ------------------------------------------------ emoções
with tab_emo:
    vocab_all = dp.emotion_vocabulary(df)
    if not vocab_all:
        st.info("Sem coluna `emotion_keywords` com dados no período.")
    else:
        st.caption(
            "Frequência de cada emoção ao longo do tempo. A linha é a média móvel — "
            "proporção de dias com a emoção numa janela."
        )
        c1, c2 = st.columns([3, 1])
        default_top = vocab_all[:min(6, len(vocab_all))]
        _emo_default = filtered_default(saved_or("emo_chosen", None), vocab_all) or default_top
        chosen_emo = c1.multiselect(
            "Emoções", vocab_all, default=_emo_default,
            format_func=lambda e: f"{e}",
            help="Ordenadas por frequência.",
            key="emo_chosen", on_change=autosave,
        )
        emo_roll = c2.number_input(
            "Janela MM (dias)", 3, 28, int(saved_or("emo_roll", max(7, roll))),
            key="emo_roll", on_change=autosave,
        )

        if chosen_emo:
            pf = dp.emotion_presence_frame(df, chosen_emo)
            cmap = [OK["blue"], OK["orange"], OK["green"], OK["vermillion"],
                    OK["purple"], OK["skyblue"], OK["yellow"], OK["grey"]]
            fig = go.Figure()
            for i, e in enumerate(chosen_emo):
                color = cmap[i % len(cmap)]
                roll_series = pf[e].rolling(emo_roll, min_periods=max(2, emo_roll // 3)).mean() * 100
                fig.add_trace(go.Scatter(
                    x=df["date"], y=pf[e] * 100, mode="markers",
                    marker=dict(color=color, size=4, opacity=0.18),
                    legendgroup=e, showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=roll_series, mode="lines",
                    line=dict(color=color, width=2.6, shape="spline"),
                    name=e, legendgroup=e,
                    hovertemplate="%{y:.0f}% dos dias<extra>" + e + "</extra>",
                ))
            add_phase_vlines(fig, change_points)
            style_fig(fig, height=460)
            fig.update_layout(hovermode="x unified",
                              yaxis_title=f"% de dias com a emoção (MM {emo_roll}d)",
                              yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

            freq = pd.DataFrame({
                "emoção": chosen_emo,
                "dias": [int(pf[e].sum()) for e in chosen_emo],
                "% do período": [f"{pf[e].mean()*100:.1f}%" for e in chosen_emo],
            })
            st.caption(f"Vocabulário completo: {len(vocab_all)} emoções distintas.")
            st.dataframe(freq, width="stretch", hide_index=True)
        else:
            st.info("Selecione ao menos uma emoção.")

# ------------------------------------------------ fases
with tab_fases:
    if not has_fase or df["fase_label"].nunique() < 2:
        st.info("Nenhuma fase detectada nos dados do período selecionado.")
    else:
        report_html = build_phase_report(df, registry=COLUMN_REGISTRY or None)
        st.components.v1.html(report_html, height=1200, scrolling=True)
        st.download_button(
            "⬇ Baixar relatório HTML",
            report_html.encode("utf-8"),
            "relatorio_fases.html",
            "text/html",
        )

# ------------------------------------------------ dados
with tab_dados:
    st.dataframe(df, width="stretch", height=520)
    st.download_button(
        "Baixar CSV processado",
        df.to_csv(index=False).encode("utf-8"),
        "niveis_diarios_processado.csv",
        "text/csv",
    )

    with st.expander("Distribuição de uma variável"):
        num_cols_d = sorted(
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 5
            and c not in ("weekday", "fase")
        )
        _saved_hist_var = saved_or("hist_var", None)
        _hist_idx = (num_cols_d.index(_saved_hist_var) if _saved_hist_var in num_cols_d
                     else (num_cols_d.index("mood_score") if "mood_score" in num_cols_d else 0))
        var = st.selectbox("Variável", num_cols_d, index=_hist_idx,
                           key="hist_var", on_change=autosave)
        s = df[var].astype(float)
        only_use = False
        if var in set(meds):
            only_use = st.checkbox(
                "Só dias em uso (dose > 0)", value=bool(saved_or("hist_use", True)),
                key="hist_use", on_change=autosave,
            )
            if only_use:
                s = s[s > 0]
        s = s.dropna()
        if len(s) >= 3:
            nbins = st.slider("Bins", 5, 60, min(30, max(5, len(s) // 3)), key="hist_bins")
            fig = go.Figure(go.Histogram(x=s, nbinsx=nbins, marker_color=OK["blue"],
                                         marker_line=dict(width=0.5, color=OK["surface"])))
            fig.add_vline(x=s.mean(), line=dict(color=OK["orange"], width=2),
                          annotation_text=f"μ {s.mean():.1f}", annotation_position="top")
            fig.add_vline(x=s.median(), line=dict(color=OK["green"], width=2, dash="dash"),
                          annotation_text=f"md {s.median():.1f}", annotation_position="bottom")
            style_fig(fig, height=340, legend_top=False)
            fig.update_layout(xaxis_title=var + (" · só dias em uso" if only_use else ""),
                              yaxis_title="dias", bargap=0.05)
            st.plotly_chart(fig, width="stretch")
            desc = s.describe()
            st.caption(
                f"n = {int(desc['count'])} · μ = {desc['mean']:.2f} · md = {s.median():.2f} · "
                f"dp = {desc['std']:.2f} · min = {desc['min']:.2f} · "
                f"p25 = {desc['25%']:.2f} · p75 = {desc['75%']:.2f} · máx = {desc['max']:.2f}"
            )
        else:
            st.info("Poucos valores pra histograma.")

# ------------------------------------------------ registro de colunas
with tab_reg:
    st.markdown("##### Registro de colunas")
    st.caption(
        "Classifique cada coluna da planilha. Colunas marcadas como **score**, "
        "**numeric**, **med** ou **bool** passam a ser tratadas automaticamente "
        "(parsing, gráficos genéricos, regressão) sem editar código. "
        "**duration**/**clock** e visões especiais seguem no código. "
        "O palpite vem pré-preenchido — revise e salve."
    )

    if df_raw is None:
        st.warning("Sem CSV cru disponível pra varredura. Recarregue a planilha.")
    else:
        # monta a grade: merge varredura + registro salvo
        merged = rg.merge_scan_with_registry(df_raw, COLUMN_REGISTRY)

        # separa o que é editável (in_registry) do que é read-only (duration/clock/estrutural)
        editable = [m for m in merged if m.get("in_registry", True)]
        locked = [m for m in merged if not m.get("in_registry", True)]

        # DataFrame pra data_editor
        grid = pd.DataFrame([{
            "coluna": m["col"],
            "label": m.get("label", m["col"]),
            "kind": m.get("kind", "text"),
            "direction": m.get("direction", "none"),
            "color": m.get("color", ""),
            "enabled": bool(m.get("enabled", True)),
            "palpite": m.get("_guess") or "",
            "cobertura": f"{m.get('coverage', 0)*100:.0f}%",
            "amostra": ", ".join(str(x) for x in (m.get("sample") or [])[:4]),
            "ausente": m.get("missing", False),
        } for m in editable])

        edited = st.data_editor(
            grid,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "coluna": st.column_config.TextColumn("coluna", disabled=True, width="medium"),
                "label": st.column_config.TextColumn("label", width="medium"),
                "kind": st.column_config.SelectboxColumn(
                    "kind", options=rg.KINDS, width="small", required=True),
                "direction": st.column_config.SelectboxColumn(
                    "direction", options=rg.DIRECTIONS, width="small",
                    help="Só aplica a score/numeric."),
                "color": st.column_config.TextColumn("color (hex)", width="small"),
                "enabled": st.column_config.CheckboxColumn("on", width="small"),
                "palpite": st.column_config.TextColumn("palpite", disabled=True, width="small"),
                "cobertura": st.column_config.TextColumn("cob.", disabled=True, width="small"),
                "amostra": st.column_config.TextColumn("amostra", disabled=True, width="medium"),
                "ausente": st.column_config.CheckboxColumn("ausente?", disabled=True, width="small"),
            },
            key="registry_editor",
        )

        cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 2])
        if cbtn1.button("💾 Salvar registro", type="primary"):
            new_reg = []
            for _, row in edited.iterrows():
                new_reg.append(rg.normalize_entry({
                    "col": row["coluna"],
                    "label": row["label"],
                    "kind": row["kind"],
                    "direction": row["direction"],
                    "color": row["color"],
                    "enabled": bool(row["enabled"]),
                }))
            save_registry(new_reg)
            st.success(f"Registro salvo ({len(new_reg)} colunas). Recarregando…")
            st.cache_data.clear()
            st.rerun()

        if cbtn2.button("🌱 Semear das constantes"):
            seeded = rg.seed_from_constants()
            save_registry(seeded)
            st.success(f"Registro semeado das constantes ({len(seeded)} colunas). Recarregando…")
            st.cache_data.clear()
            st.rerun()

        if locked:
            st.markdown("###### Colunas fora do registro (tratadas no código)")
            st.caption(
                "duration/clock/estruturais — aparecem aqui só pra referência; "
                "seu comportamento está hardcoded (estágios de sono, wrap de madrugada, etc.)."
            )
            lock_df = pd.DataFrame([{
                "coluna": m["col"], "tipo": m.get("kind", ""),
                "motivo": m.get("reason", ""),
            } for m in locked])
            st.dataframe(lock_df, width="stretch", hide_index=True)
