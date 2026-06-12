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
    "fluencia_verbal_espontaneidade_social",
]

SCORE_LABELS = {
    "energy_score": "Energia",
    "cognition_score": "Cognição",
    "attention_score": "Atenção",
    "mood_score": "Humor",
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


# ---------------------------------------------------------------- helpers

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


# ---------------------------------------------------------------- carga

def load_csv(source) -> pd.DataFrame:
    """source: URL (str) ou file-like (upload)."""
    if isinstance(source, str):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
    return pd.read_csv(source)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in NUMERIC_COMMA_COLS:
        if col in df.columns:
            df[col] = _to_num(df[col])

    for col in DURATION_COLS:
        if col in df.columns:
            df[col + "_h"] = _hhmm_to_hours(df[col])

    for col in ["bed_time", "sleep_time"]:
        if col in df.columns:
            df[col + "_h"] = _clock_to_hours(df[col])
    if "wake_time" in df.columns:
        df["wake_time_h"] = _hhmm_to_hours(df["wake_time"])

    # texto consolidado pra hover
    obs_present = [c for c in OBS_COLS if c in df.columns]

    def join_obs(row):
        parts = [str(row[c]) for c in obs_present
                 if pd.notna(row[c]) and str(row[c]).strip()]
        return " | ".join(parts)

    df["obs_all"] = df.apply(join_obs, axis=1) if obs_present else ""

    return df


def active_meds(df: pd.DataFrame, min_days: int = 1) -> list[str]:
    """Medicações com uso (>0) em pelo menos `min_days` dias."""
    out = []
    for col in MED_COLS:
        if col in df.columns and (df[col].fillna(0) > 0).sum() >= min_days:
            out.append(col)
    return out
