"""Registro de schema de colunas, editável em runtime.

Permite classificar colunas novas da planilha (kind, label, direction, cor)
sem tocar nas constantes do data_prep. Persiste como a chave `column_registry`
no blob `dashboard_prefs` do Supabase (gerido pelo app.py).

Princípios:
- O registro é a FONTE ÚNICA para score/numeric/med/bool no que toca
  classificação e parsing genérico (virgula->float, fillna-0 em med).
- duration/clock NÃO entram no registro — seguem nas constantes do data_prep,
  porque arrastam comportamento especial (_h, wrap, estagios empilhados).
- A inferência (scan) só SUGERE um kind inicial para a UI; nunca decide nada.
  O pipeline consome apenas o registro SALVO, jamais o palpite.
- registry vazio/None => comportamento idêntico ao app atual (constantes).
"""

import re
import numpy as np
import pandas as pd

import data_prep as dp

# ---------------------------------------------------------------- schema

# kinds que vivem no registro e o que cada um dispara
KINDS = ["score", "numeric", "med", "bool", "text", "ignore"]
DIRECTIONS = ["higher", "lower", "none"]

# colunas estruturais e as cobertas por constante hardcoded — ficam FORA do registro
STRUCTURAL = {"date", "fase", "fase_label", "obs_all"}
# nomes reconhecidos como duration/clock (espelham as constantes do data_prep)
OUT_OF_REGISTRY_NAMES = set(dp.DURATION_COLS) | set(dp.CLOCK_COLS) | STRUCTURAL

# paleta Okabe–Ito pra default de cor (mesma do app)
_PALETTE_CYCLE = ["#E69F00","#56B4E9","#00C896","#F0E442","#4C9BE8","#EF6351","#D395C8","#888888"]

# rótulos canônicos já conhecidos (das constantes / phase_report). Usados para
# pré-preencher o label de colunas CONHECIDAS no merge — não é "adivinhar", é
# recuperar info que já existe e é verdade. Coluna genuinamente nova segue com
# label = nome cru, que o usuário edita.
_KNOWN_MED_LABELS = {
    "venvanse_mg_total": "Lisdexanfetamina", "bupropiona_mg": "Bupropiona",
    "zolpidem_mg_total": "Zolpidem", "fluvoxamina_mg": "Fluvoxamina",
    "melatonina_mg": "Melatonina", "clonazepam_mg": "Clonazepam",
    "pregabalina_mg": "Pregabalina", "tranilcipromina_mg": "IMAO",
    "ramelteona_mg": "Ramelteon", "pramipexol_mg_night": "Pramipexol",
    "lamotrigina_mg": "Lamotrigina", "aripripazole_mg": "Aripiprazol",
    "metilfenidato_mg_total": "Metilfenidato", "vitamina_d_ug": "Vit D",
}


def known_label(col: str) -> str | None:
    """Rótulo canônico conhecido pra uma coluna, das constantes do data_prep
    (scores, atividade) ou dos meds. None se não há rótulo conhecido."""
    if col in dp.SCORE_LABELS:
        return dp.SCORE_LABELS[col]
    if col in dp.ACTIVITY_LABELS:
        return dp.ACTIVITY_LABELS[col]
    if col in _KNOWN_MED_LABELS:
        return _KNOWN_MED_LABELS[col]
    return None


def blank_entry(col: str) -> dict:
    """Entrada padrão pra uma coluna ainda não classificada."""
    return {"col": col, "label": col, "kind": "text",
            "direction": "none", "color": "", "enabled": True}


def normalize_entry(e: dict, idx: int = 0) -> dict:
    """Garante todos os campos presentes e válidos; preenche defaults faltantes."""
    col = e.get("col", "")
    kind = e.get("kind", "text")
    if kind not in KINDS:
        kind = "text"
    direction = e.get("direction", "none")
    if direction not in DIRECTIONS:
        direction = "none"
    # direction só faz sentido pra score/numeric; força none nos demais
    if kind not in ("score", "numeric"):
        direction = "none"
    color = e.get("color") or ""
    label = e.get("label") or col
    enabled = bool(e.get("enabled", True))
    return {"col": col, "label": label, "kind": kind,
            "direction": direction, "color": color, "enabled": enabled}


def registry_to_map(registry: list[dict]) -> dict[str, dict]:
    """Lista de entradas -> dict col->entrada normalizada."""
    out = {}
    for i, e in enumerate(registry or []):
        ne = normalize_entry(e, i)
        if ne["col"]:
            out[ne["col"]] = ne
    return out


# ---------------------------------------------------------------- inferência (palpite p/ UI)

_HHMM_RE = re.compile(r"^\d{1,2}:\d{2}$")
_BOOL_TOKENS = {"true", "false"}


def _nonempty(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna())


def _as_float_comma(tok):
    try:
        return float(str(tok).replace(",", "."))
    except (ValueError, AttributeError):
        return np.nan


def infer_kind(name: str, raw: pd.Series, n_rows: int) -> dict:
    """Palpite de kind + sinais, a partir da coluna CRUA (string).
    NUNCA infere direction (polaridade não é detectável pelo formato).
    Retorna in_registry=False pras colunas que pertencem a constante hardcoded."""
    vals = _nonempty(raw)
    coverage = round(len(vals) / n_rows, 3) if n_rows else 0.0
    nunique = int(vals.nunique())
    base = {"col": name, "coverage": coverage, "n_unique": nunique,
            "in_registry": True, "kind": "text", "reason": "",
            "sample": list(vals.unique()[:6])}

    if name in OUT_OF_REGISTRY_NAMES:
        if name in STRUCTURAL:
            base.update(in_registry=False, kind="ignore", reason="estrutural")
        elif name in dp.CLOCK_COLS:
            base.update(in_registry=False, kind="clock", reason="horário (constante)")
        else:
            base.update(in_registry=False, kind="duration", reason="duração (constante)")
        return base

    if len(vals) == 0:
        base.update(kind="ignore", reason="vazia")
        return base

    low = vals.str.lower()
    if float(np.mean([v in _BOOL_TOKENS for v in low.values])) >= 0.95:
        base.update(kind="bool", reason="TRUE/FALSE")
        return base

    floats = vals.map(_as_float_comma)
    frac_num = float(floats.notna().mean())
    distinct = set(floats.dropna().unique())
    if frac_num >= 0.95 and distinct.issubset({0.0, 1.0}) and len(distinct) <= 2:
        base.update(kind="bool", reason="binário 0/1")
        return base

    if float(np.mean([bool(_HHMM_RE.match(v)) for v in vals.values])) >= 0.8:
        base.update(in_registry=False, kind="duration",
                    reason="formato HH:MM (tratar como duration/clock no código)")
        return base

    if frac_num >= 0.9:
        nz = floats.dropna()
        frac_zero = float((nz == 0).mean()) if len(nz) else 0.0
        vmax = float(nz.max()) if len(nz) else np.nan
        vmin = float(nz.min()) if len(nz) else np.nan
        name_l = name.lower()
        if re.search(r"(_mg|_ug|_mcg)(_|$)|dose", name_l):
            base.update(kind="med", reason=f"nome de dose · {frac_zero:.0%} zeros")
            return base
        looks_score = name_l.endswith("_score") or "fluencia" in name_l
        small = (not np.isnan(vmax)) and vmax <= 10.0 and (np.isnan(vmin) or vmin >= 0)
        if looks_score or (small and nunique <= 21 and frac_zero < 0.2):
            base.update(kind="score",
                        reason="nome _score" if looks_score else f"escala 0..{vmax:.0f}")
            return base
        base.update(kind="numeric", reason=f"contínuo {vmin:.0f}..{vmax:.0f}")
        return base

    base.update(kind="text", reason=f"texto · {nunique} distintos")
    return base


def scan(df_raw: pd.DataFrame) -> list[dict]:
    """Varre o df CRU (strings) e devolve um palpite por coluna, na ordem da planilha."""
    n = len(df_raw)
    return [infer_kind(c, df_raw[c], n) for c in df_raw.columns]


def merge_scan_with_registry(df_raw: pd.DataFrame, registry: list[dict]) -> list[dict]:
    """Combina varredura atual com o registro salvo, para a UI:
    - coluna já no registro => usa a entrada SALVA (kind/label/dir do usuário), anexa sinais da varredura
    - coluna nova => entrada em branco com kind=PALPITE, anexa sinais
    - entrada salva cuja coluna sumiu da planilha => marcada missing=True (não apaga)
    """
    saved = registry_to_map(registry)
    guesses = {g["col"]: g for g in scan(df_raw)}
    rows = []
    seen = set()
    for col in df_raw.columns:
        g = guesses[col]
        seen.add(col)
        if col in saved:
            e = dict(saved[col])
            e.update(_guess=g["kind"], coverage=g["coverage"],
                     n_unique=g["n_unique"], reason=g["reason"],
                     in_registry=g["in_registry"], sample=g["sample"], missing=False)
        else:
            e = blank_entry(col)
            e["kind"] = g["kind"]  # palpite como valor inicial
            kl = known_label(col)  # recupera label canônico se conhecido
            if kl:
                e["label"] = kl
            e.update(_guess=g["kind"], coverage=g["coverage"],
                     n_unique=g["n_unique"], reason=g["reason"],
                     in_registry=g["in_registry"], sample=g["sample"], missing=False)
        rows.append(e)
    # entradas salvas órfãs (coluna sumiu)
    for col, e in saved.items():
        if col not in seen:
            ee = dict(e)
            ee.update(_guess=None, coverage=0.0, n_unique=0, reason="coluna ausente na planilha",
                      in_registry=True, sample=[], missing=True)
            rows.append(ee)
    return rows


# ---------------------------------------------------------------- seeding inicial


def seed_from_constants() -> list[dict]:
    """Gera o registro inicial a partir das constantes atuais do data_prep,
    pra UI não começar com tudo em branco. Idempotente (uma entrada por coluna)."""
    out = []
    seen = set()

    def add(col, label, kind, direction="none", color=""):
        if col in seen:
            return
        seen.add(col)
        out.append(normalize_entry(
            {"col": col, "label": label, "kind": kind,
             "direction": direction, "color": color, "enabled": True}))

    # scores (todos higher no estado atual; usuário muda ansiedade etc. se surgir)
    for c in dp.SCORE_COLS:
        add(c, dp.SCORE_LABELS.get(c, c), "score", "higher")
    # meds
    for c in dp.MED_COLS:
        add(c, _KNOWN_MED_LABELS.get(c, c), "med", "none")
    # numéricos simples = NUMERIC_COMMA_COLS que não são score nem med
    num_simple = [c for c in dp.NUMERIC_COMMA_COLS
                  if c not in set(dp.SCORE_COLS) and c not in set(dp.MED_COLS)]
    for c in num_simple:
        lbl = dp.ACTIVITY_LABELS.get(c, c)
        add(c, lbl, "numeric", "none")
    return out


# ---------------------------------------------------------------- derivadores
# Estes substituem as constantes quando há registro. Cada um tem fallback embutido
# no data_prep (registry=None -> constante), então aqui assumimos registry não-vazio.

def _enabled_of_kind(rmap: dict, kind: str) -> list[str]:
    return [c for c, e in rmap.items() if e["kind"] == kind and e["enabled"]]


def scores(rmap: dict) -> list[str]:
    return _enabled_of_kind(rmap, "score")


def score_labels(rmap: dict) -> dict:
    return {c: e["label"] for c, e in rmap.items() if e["kind"] == "score" and e["enabled"]}


def score_directions(rmap: dict) -> dict:
    return {c: e["direction"] for c, e in rmap.items() if e["kind"] == "score" and e["enabled"]}


def meds(rmap: dict) -> list[str]:
    return _enabled_of_kind(rmap, "med")


def med_labels(rmap: dict) -> dict:
    return {c: e["label"] for c, e in rmap.items() if e["kind"] == "med" and e["enabled"]}


def numerics(rmap: dict) -> list[str]:
    return _enabled_of_kind(rmap, "numeric")


def bools(rmap: dict) -> list[str]:
    return _enabled_of_kind(rmap, "bool")


def numeric_comma_cols(rmap: dict) -> list[str]:
    """Todas as colunas que levam vírgula->float: score + numeric + med."""
    return (_enabled_of_kind(rmap, "score")
            + _enabled_of_kind(rmap, "numeric")
            + _enabled_of_kind(rmap, "med"))


def color_for(rmap: dict, col: str, idx: int = 0) -> str:
    e = rmap.get(col)
    if e and e.get("color"):
        return e["color"]
    return _PALETTE_CYCLE[idx % len(_PALETTE_CYCLE)]
