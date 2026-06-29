"""Parsing e preparação da planilha de níveis diários.

Lida com: decimal em vírgula, datas dd/mm/yyyy, durações HH:MM,
horários de cama/acordar com wrap pós-meia-noite, colunas de medicação.
"""

import io

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- grupos de colunas

SCORE_COLS = [
    "energy_score",
    "cognition_score",
    "attention_score",
    "mood_score",
    "drive_score",
    "fluencia_verbal_espontaneidade_social",
]

SCORE_LABELS = {
    "energy_score": "Energia",
    "cognition_score": "Cognição",
    "attention_score": "Atenção",
    "mood_score": "Humor",
    "drive_score": "Iniciativa/Drive",
    "fluencia_verbal_espontaneidade_social": "Fluência/espontaneidade",
}

MED_COLS = [
    "bupropiona_mg",
    "venvase_morning_mg",
    "venvanse_evening_mg",
    "venvanse_mg_total",
    "zolpidem_mg_total",
    "metilfenidato_mg_total",
    "ramelteona_mg",
    "fluvoxamina_mg",
    "melatonina_mg",
    "clonazepam_mg",
    "pregabalina_mg",
    "vitamina_d_ug",
    "tranilcipromina_mg",
    "pramipexol_mg_night",
    "lamotrigina_mg",
    "aripripazole_mg",
]

DURATION_COLS = [
    "sleep_duration",
    "inbed_duration",
    "deep_sleep",
    "light_sleep",
    "rem_sleep",
    "awake_sleep",
]

CLOCK_COLS = ["bed_time", "sleep_time", "wake_time"]

NUMERIC_COMMA_COLS = SCORE_COLS + MED_COLS + [
    "sleep_efficiency",
    "light_sleep_perc",
    "rem_sleep_perc",
    "deep_sleep_perc",
    "sleep_latency_estimate_minutes",
    "restlessness_mins",
    "interruption_mins",
    "full_awakenings",
    "sound_sleep_mins",
    "time_to_sound_sleep_mins",
    "steps",
    "avg_bpm",
    "VFC",
    "exercise_mins",
    "zolpidem_x_5mg_day",
    "zolpidem_5mg",
    "metilfenidato_mg",
    "metilfenidato_evening_mg",
]

OBS_COLS = ["observações", "obs2", "obs3", "obs4", "observation", "emotion_keywords"]

ACTIVITY_LABELS = {
    "steps": "Passos",
    "exercise_mins": "Exercício (min)",
    "avg_bpm": "BPM médio",
    "VFC": "VFC",
}


# ---------------------------------------------------------------- helpers numéricos

def _to_num(series: pd.Series) -> pd.Series:
    """Converte série com decimal em vírgula pra float."""
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _hhmm_to_hours(series: pd.Series) -> pd.Series:
    """'HH:MM' -> horas (float)."""
    def conv(v):
        if not isinstance(v, str) or ":" not in v:
            return np.nan
        try:
            h, m = v.strip().split(":")[:2]
            return int(h) + int(m) / 60.0
        except (ValueError, TypeError):
            return np.nan
    return series.map(conv)


def _clock_to_hours(series: pd.Series, wrap_before: float = 18.0) -> pd.Series:
    """Horário do relógio -> horas contínuas; antes de `wrap_before` vira +24h
    (01:36 -> 25.6), pra plotar madrugada acima da meia-noite."""
    h = _hhmm_to_hours(series)
    return h.where(h >= wrap_before, h + 24)


def hours_to_label(x: float) -> str:
    if pd.isna(x):
        return ""
    x = x % 24
    return f"{int(x):02d}:{int(round((x - int(x)) * 60)):02d}"


# ---------------------------------------------------------------- fases

def fase_label(v) -> str:
    """Rótulo canônico de uma célula de fase."""
    if pd.isna(v):
        return "(sem fase)"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def fase_sort_key(label: str):
    """Ordena fases numericamente; rótulos não-numéricos e '(sem fase)' no fim."""
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def fase_order(labels) -> list[str]:
    return sorted(set(labels), key=fase_sort_key)


def phase_change_points(df: pd.DataFrame, fase_col: str = "fase_label"):
    """Datas em que a fase muda, no df já ordenado por data.

    Retorna lista de dicts: {date, from, to}. Ignora a primeira linha
    (não há transição antes dela). Robusto a período não-contíguo:
    detecta a mudança no ponto em que o rótulo difere do anterior.
    """
    if fase_col not in df.columns or df.empty:
        return []
    out = []
    prev = None
    for d, lab in zip(df["date"], df[fase_col]):
        if prev is not None and lab != prev:
            out.append({"date": d, "from": prev, "to": lab})
        prev = lab
    return out


# ---------------------------------------------------------------- medicação

def med_dose_used(df: pd.DataFrame, med: str):
    """Retorna (dose, used): série de mg (zeros = não-uso) e indicador binário."""
    dose = df[med].fillna(0.0)
    used = (dose > 0).astype(int)
    return dose, used


def med_varies_within_use(df: pd.DataFrame, med: str, min_used: int = 4) -> bool:
    """True se o fármaco tem dose variável (titulado) entre os dias de uso.
    Usado pra decidir se 'titulação' é um modo informativo pra esse fármaco."""
    dose, _ = med_dose_used(df, med)
    used = dose[dose > 0]
    return used.nunique() >= 2 and len(used) >= min_used


def active_meds(df: pd.DataFrame, min_days: int = 1, med_cols=None) -> list[str]:
    """Medicações com uso (>0) em pelo menos `min_days` dias.
    med_cols: lista de colunas de medicação a considerar; None -> constante MED_COLS."""
    cols = med_cols if med_cols is not None else MED_COLS
    out = []
    for col in cols:
        if col in df.columns and (df[col].fillna(0) > 0).sum() >= min_days:
            out.append(col)
    return out


# ---------------------------------------------------------------- carga

def load_csv(source) -> pd.DataFrame:
    """source: URL (str) ou file-like (upload)."""
    if isinstance(source, str):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
    return pd.read_csv(source)


EMOTION_COL = "emotion_keywords"


def _split_emotions(cell) -> list[str]:
    """Célula 'Irritabilidade, Depressão, Anedonia' -> lista normalizada (Title Case, sem vazios)."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    for tok in cell.split(","):
        t = tok.strip()
        if t:
            out.append(t[:1].upper() + t[1:].lower() if t.isupper() or t.islower() else t)
    return out


def emotion_vocabulary(df: pd.DataFrame, top_n: int | None = None) -> list[str]:
    """Emoções distintas ordenadas por frequência (nº de dias em que aparecem)."""
    if EMOTION_COL not in df.columns:
        return []
    from collections import Counter
    c = Counter()
    for cell in df[EMOTION_COL]:
        for e in set(_split_emotions(cell)):  # set: conta 1x por dia
            c[e] += 1
    ordered = [e for e, _ in c.most_common()]
    return ordered[:top_n] if top_n else ordered


def emotion_presence_frame(df: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    """DataFrame indexado por date, 1 coluna binária por emoção (1 = presente no dia)."""
    rows = []
    for cell in df[EMOTION_COL] if EMOTION_COL in df.columns else []:
        present = set(_split_emotions(cell))
        rows.append({e: (1 if e in present else 0) for e in emotions})
    out = pd.DataFrame(rows, index=df["date"].values)
    for e in emotions:
        if e not in out.columns:
            out[e] = 0
    return out[emotions] if emotions else out


def _groups_from_registry(registry):
    """(numeric_comma_cols, med_cols) derivados do registro.
    registry None/vazio -> (None, None), sinalizando ao prepare que use as constantes.
    """
    if not registry:
        return None, None
    import registry as rg
    rmap = rg.registry_to_map(registry)
    return rg.numeric_comma_cols(rmap), rg.meds(rmap)


def prepare(df: pd.DataFrame, registry=None) -> pd.DataFrame:
    """registry: lista de entradas de schema (ver registry.py). None/vazio =>
    usa as constantes hardcoded (comportamento idêntico à versão anterior).
    Quando presente, score/numeric/med do registro definem o parsing genérico
    (vírgula->float e fillna-0 em med). duration/clock seguem nas constantes."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # grupos de parsing: do registro se houver, senão constantes
    _num_cols, _med_cols = _groups_from_registry(registry)
    if _num_cols is None:
        _num_cols, _med_cols = NUMERIC_COMMA_COLS, MED_COLS

    for col in _num_cols:
        if col in df.columns:
            df[col] = _to_num(df[col])

    # decisão explícita: 0 e missing são equivalentes (= não-uso) para fármacos.
    for col in _med_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in DURATION_COLS:
        if col in df.columns:
            df[col + "_h"] = _hhmm_to_hours(df[col])

    for col in ["bed_time", "sleep_time"]:
        if col in df.columns:
            df[col + "_h"] = _clock_to_hours(df[col])
    if "wake_time" in df.columns:
        df["wake_time_h"] = _hhmm_to_hours(df["wake_time"])

    # rótulo de fase canônico (sempre presente; "(sem fase)" se ausente/NaN)
    if "fase" in df.columns:
        df["fase_label"] = df["fase"].map(fase_label)
    else:
        df["fase_label"] = "(sem fase)"

    # texto consolidado pra hover
    obs_present = [c for c in OBS_COLS if c in df.columns]

    def join_obs(row):
        parts = [str(row[c]) for c in obs_present
                 if pd.notna(row[c]) and str(row[c]).strip()]
        return " | ".join(parts)

    df["obs_all"] = df.apply(join_obs, axis=1) if obs_present else ""

    return df
