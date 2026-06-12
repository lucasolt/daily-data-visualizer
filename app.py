"""Dashboard de níveis diários — lê direto do Google Sheets publicado em CSV.

Rodar local:  streamlit run app.py
Deploy:       ver README.md
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sps

import data_prep as dp

st.set_page_config(page_title="Níveis diários", page_icon="📈", layout="wide")

PALETTE = {
    "Energia": "#E8A33D",
    "Cognição": "#5B8DEF",
    "Atenção": "#9B6BD3",
    "Humor": "#3DBE8B",
    "Fluência/espontaneidade": "#E36AA0",
}
SLEEP_STAGE_COLORS = {
    "deep_sleep_h": ("#2D4A8A", "Profundo"),
    "light_sleep_h": ("#7EA6E0", "Leve"),
    "rem_sleep_h": ("#B79CE4", "REM"),
    "awake_sleep_h": ("#D9D9D9", "Acordado"),
}

# ---------------------------------------------------------------- fonte de dados

st.sidebar.title("Níveis diários")

CACHE_TTL = 300  # segundos


@st.cache_data(ttl=CACHE_TTL, show_spinner="Carregando planilha...")
def fetch(url: str) -> pd.DataFrame:
    return dp.prepare(dp.load_csv(url))


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

df = None
if uploaded is not None:
    df = dp.prepare(pd.read_csv(uploaded))
elif url.strip():
    try:
        df = fetch(url.strip())
    except Exception as e:
        st.error(f"Falha ao carregar a URL: {e}")

if df is None or df.empty:
    st.info(
        "Aponte a URL CSV da planilha na barra lateral, ou suba um CSV. "
        "Instruções de publicação da planilha no README."
    )
    st.stop()

# filtro de período
dmin, dmax = df["date"].min().date(), df["date"].max().date()
period = st.sidebar.date_input(
    "Período", value=(dmin, dmax), min_value=dmin, max_value=dmax
)
if isinstance(period, tuple) and len(period) == 2:
    df = df[(df["date"].dt.date >= period[0]) & (df["date"].dt.date <= period[1])]

# filtro de fase
if "fase" in df.columns:
    def _fase_label(v):
        if pd.isna(v):
            return "(sem fase)"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()

    def _fase_sort_key(label):
        try:
            return (0, float(label), "")
        except ValueError:
            return (1, 0.0, label)  # "(sem fase)" e rótulos não numéricos no fim

    fase_labels = df["fase"].map(_fase_label)
    fase_opts = sorted(fase_labels.unique(), key=_fase_sort_key)
    if len(fase_opts) > 1:
        fases_sel = st.sidebar.multiselect(
            "Fase", fase_opts, default=fase_opts,
            help="Vazio = todas. Filtrar fases pode deixar o período não-contíguo — "
            "a média móvel atravessa os buracos.",
        )
        if fases_sel and len(fases_sel) < len(fase_opts):
            df = df[fase_labels.isin(fases_sel)]

if df.empty:
    st.warning("Nenhum dia atende aos filtros atuais.")
    st.stop()

roll = st.sidebar.slider("Janela média móvel (dias)", 3, 28, 7)

score_cols = [c for c in dp.SCORE_COLS if c in df.columns]
meds = dp.active_meds(df)

st.sidebar.caption(
    f"{len(df)} dias · {dmin:%d/%m/%y} → {dmax:%d/%m/%y} · cache {CACHE_TTL//60} min"
)

# ---------------------------------------------------------------- helpers de plot


def line_with_roll(fig, x, y, name, color, window, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name, mode="markers",
            marker=dict(color=color, size=5, opacity=0.45),
            legendgroup=name, showlegend=False, hovertemplate="%{y}<extra>" + name + "</extra>",
        ),
        **kw,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y.rolling(window, min_periods=max(2, window // 3)).mean(),
            name=name, mode="lines", line=dict(color=color, width=2.4),
            legendgroup=name,
        ),
        **kw,
    )


# ---------------------------------------------------------------- tabs

tab_vis, tab_scores, tab_sono, tab_med, tab_atv, tab_corr, tab_dados = st.tabs(
    ["Visão geral", "Scores", "Sono", "Medicações", "Atividade", "Correlações", "Dados"]
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
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Humor e energia", "Duração do sono (h)"),
    )
    for col_name in ["mood_score", "energy_score"]:
        if col_name in df.columns:
            label = dp.SCORE_LABELS[col_name]
            line_with_roll(fig, df["date"], df[col_name], label, PALETTE[label], roll, row=1, col=1)
    if "sleep_duration_h" in df.columns:
        fig.add_trace(
            go.Bar(x=df["date"], y=df["sleep_duration_h"], name="Sono (h)",
                   marker_color="#7EA6E0", opacity=0.7, showlegend=False),
            row=2, col=1,
        )
    fig.update_layout(height=560, hovermode="x unified", margin=dict(t=50, b=20))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ scores
with tab_scores:
    chosen = st.multiselect(
        "Séries", score_cols, default=score_cols,
        format_func=lambda c: dp.SCORE_LABELS.get(c, c),
    )
    fig = go.Figure()
    for col_name in chosen:
        label = dp.SCORE_LABELS.get(col_name, col_name)
        line_with_roll(fig, df["date"], df[col_name], label, PALETTE.get(label, "#888"), roll)
    # anotações de observações no hover
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=[df[chosen].max().max() * 1.02 if chosen else 1] * len(df),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, color="rgba(0,0,0,0.12)"),
            text=df["obs_all"], hovertemplate="%{text}<extra>obs</extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=520, hovermode="x unified",
        yaxis_title="Score", margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Pontos = valor diário; linha = média móvel. Marcas cinzas no topo carregam as observações do dia (hover).")

# ------------------------------------------------ sono
with tab_sono:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        specs=[[{"secondary_y": True}], [{}], [{}]],
        subplot_titles=("Duração × eficiência", "Estágios (h)", "Horário de dormir/acordar"),
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["sleep_duration_h"], name="Duração (h)",
               marker_color="#7EA6E0", opacity=0.75),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["sleep_efficiency"], name="Eficiência (%)",
                   line=dict(color="#E8A33D", width=2)),
        row=1, col=1, secondary_y=True,
    )
    for stage, (color, label) in SLEEP_STAGE_COLORS.items():
        if stage in df.columns:
            fig.add_trace(
                go.Bar(x=df["date"], y=df[stage], name=label, marker_color=color),
                row=2, col=1,
            )
    bed = df.get("bed_time_h")
    wake = df.get("wake_time_h")
    if bed is not None:
        fig.add_trace(
            go.Scatter(x=df["date"], y=bed, name="Foi pra cama",
                       mode="markers+lines", line=dict(color="#2D4A8A", width=1),
                       text=bed.map(dp.hours_to_label),
                       hovertemplate="%{text}<extra>cama</extra>"),
            row=3, col=1,
        )
    if wake is not None:
        fig.add_trace(
            go.Scatter(x=df["date"], y=wake, name="Acordou",
                       mode="markers+lines", line=dict(color="#E8A33D", width=1),
                       text=wake.map(dp.hours_to_label),
                       hovertemplate="%{text}<extra>acordou</extra>"),
            row=3, col=1,
        )
    tickvals = list(range(0, 34, 2))
    fig.update_yaxes(
        tickvals=tickvals, ticktext=[dp.hours_to_label(v) for v in tickvals],
        row=3, col=1,
    )
    fig.update_layout(height=820, barmode="stack", hovermode="x unified",
                      margin=dict(t=50, b=20))
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    extras = {
        "sleep_latency_estimate_minutes": "Latência (min)",
        "full_awakenings": "Despertares completos",
        "restlessness_mins": "Inquietação (min)",
    }
    for (col_name, label), col_st in zip(extras.items(), [c1, c2, c3]):
        if col_name in df.columns:
            mini = go.Figure(go.Bar(x=df["date"], y=df[col_name], marker_color="#9B6BD3"))
            mini.update_layout(height=200, title=label, margin=dict(t=40, b=10, l=10, r=10))
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
                x=sub.index, y=[m.replace("_mg", "").replace("_", " ") for m in meds],
                customdata=sub.T.values,
                hovertemplate="%{x|%d/%m}: %{customdata} mg/dose<extra>%{y}</extra>",
                colorscale="Teal", showscale=False, ygap=3,
            )
        )
        fig.update_layout(height=80 + 42 * len(meds), margin=dict(t=20, b=20),
                          title="Doses por dia (intensidade relativa ao máximo de cada fármaco)")
        st.plotly_chart(fig, width="stretch")

        med_sel = st.selectbox(
            "Detalhe de um fármaco", meds,
            format_func=lambda m: m.replace("_", " "),
        )
        fig2 = go.Figure(
            go.Scatter(x=df["date"], y=df[med_sel], mode="lines+markers",
                       line_shape="hv", line=dict(color="#3DBE8B", width=2))
        )
        fig2.update_layout(height=260, yaxis_title="mg / unidades",
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------ atividade
with tab_atv:
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=list(dp.ACTIVITY_LABELS.values()))
    pos = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ["#5B8DEF", "#3DBE8B", "#E36AA0", "#9B6BD3"]
    for (col_name, _), (r, c_), color in zip(dp.ACTIVITY_LABELS.items(), pos, colors):
        if col_name in df.columns:
            if col_name in ("avg_bpm", "VFC"):
                line_with_roll(fig, df["date"], df[col_name], col_name, color, roll, row=r, col=c_)
            else:
                fig.add_trace(
                    go.Bar(x=df["date"], y=df[col_name], marker_color=color, showlegend=False),
                    row=r, col=c_,
                )
    fig.update_layout(height=620, showlegend=False, margin=dict(t=50, b=20))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ correlações
with tab_corr:
    num_cols = sorted(
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= 5
        and c not in ("weekday", "fase")
    )

    st.markdown("##### Dispersão com lag")
    c1, c2, c3 = st.columns([2, 2, 1])
    x_col = c1.selectbox("X (preditor)", num_cols,
                         index=num_cols.index("sleep_duration_h") if "sleep_duration_h" in num_cols else 0)
    y_col = c2.selectbox("Y (desfecho)", num_cols,
                         index=num_cols.index("mood_score") if "mood_score" in num_cols else 0)
    lag = c3.number_input("Lag (dias)", 0, 7, 0,
                          help="X de k dias atrás vs Y de hoje. Ex.: sono de ontem → humor de hoje = lag 1.")

    pair = pd.DataFrame({
        "x": df[x_col].shift(lag).values,
        "y": df[y_col].values,
        "date": df["date"].dt.strftime("%d/%m"),
    }).dropna(subset=["x", "y"])

    if len(pair) >= 3:
        r_p, p_p = sps.pearsonr(pair["x"], pair["y"])
        r_s, p_s = sps.spearmanr(pair["x"], pair["y"])
        slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
        xs = np.linspace(pair["x"].min(), pair["x"].max(), 50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pair["x"], y=pair["y"], mode="markers", text=pair["date"],
            marker=dict(color="#5B8DEF", size=8, opacity=0.7),
            hovertemplate="%{text}: (%{x}, %{y})<extra></extra>", name="dias",
        ))
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                 line=dict(color="#E8A33D", dash="dash"), name="OLS"))
        fig.update_layout(
            height=440, xaxis_title=f"{x_col}" + (f" (lag {lag}d)" if lag else ""),
            yaxis_title=y_col, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"n = {len(pair)} · Pearson r = {r_p:.2f} (p = {p_p:.3f}) · "
            f"Spearman ρ = {r_s:.2f} (p = {p_s:.3f})"
        )
    else:
        st.warning("Poucos pares válidos pra esse cruzamento.")

    st.markdown("##### Matriz de correlação")
    default_matrix = [c for c in (
        score_cols + ["sleep_duration_h", "sleep_efficiency", "deep_sleep_h",
                      "rem_sleep_h", "steps", "avg_bpm", "VFC"]
    ) if c in num_cols]
    matrix_cols = st.multiselect("Variáveis", num_cols, default=default_matrix)
    if len(matrix_cols) >= 2:
        method = st.radio("Método", ["spearman", "pearson"], horizontal=True)
        corr = df[matrix_cols].corr(method=method)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            zmin=-1, zmax=1, colorscale="RdBu_r",
            text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig.update_layout(height=120 + 40 * len(matrix_cols), margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ dados
with tab_dados:
    st.dataframe(df, width="stretch", height=520)
    st.download_button(
        "Baixar CSV processado",
        df.to_csv(index=False).encode("utf-8"),
        "niveis_diarios_processado.csv",
        "text/csv",
    )
