# Dashboard de níveis diários

Streamlit app que lê a planilha do Google Sheets via URL CSV e plota scores,
sono, medicações, atividade e correlações com lag. Cache de 5 min — edições na
planilha aparecem no próximo reload (ou no botão "Recarregar agora").

## Estrutura

```
app.py            # interface
data_prep.py      # parsing (decimal vírgula, dd/mm/yyyy, HH:MM, wrap madrugada)
requirements.txt
```

## 1. Obter a URL CSV da planilha

Duas opções, da menos pra mais exposta:

**A — Compartilhar por link (recomendada):**
1. Na planilha: Compartilhar → "Qualquer pessoa com o link" → Leitor.
2. A URL é:
   ```
   https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/export?format=csv&gid=<GID>
   ```
   - `SPREADSHEET_ID`: o trecho longo da URL da planilha.
   - `GID`: número que aparece em `#gid=` quando a aba "Página5" está aberta.

**B — Arquivo → Compartilhar → Publicar na web** (aba Página5, formato CSV).
Gera URL direta, mas o conteúdo fica indexável/cacheável pelo Google além do link.

⚠️ **Privacidade:** nas duas opções, qualquer pessoa que tenha a URL acessa os
dados — e isso inclui medicação psiquiátrica e notas pessoais. O risco prático é
baixo (URL não-adivinhável), mas não é zero: a URL vai parar no histórico do
navegador, nos logs do Streamlit Cloud, e em qualquer lugar onde tu colar ela.
Se isso incomodar, a alternativa fechada é service account do Google
(`gspread` + secret JSON no Streamlit Cloud) — me pede que eu adapto o
`load_csv` pra isso, são ~15 linhas.

## 2. Rodar local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Cola a URL na barra lateral, ou cria `.streamlit/secrets.toml`:

```toml
sheet_csv_url = "https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=..."
```

## 3. Deploy no Streamlit Community Cloud (grátis)

1. Sobe os 3 arquivos num repo do GitHub (**privado serve** — o Streamlit Cloud
   acessa repos privados).
2. share.streamlit.io → New app → aponta pro repo, branch, `app.py`.
3. Em Settings → Secrets do app, cola o `sheet_csv_url` (assim a URL não fica
   no código do repo).
4. Opcional: Settings → Sharing → restringir o acesso ao app por e-mail
   (viewer auth), já que o dashboard em si também expõe os dados.

## Notas de parsing

- Datas `dd/mm/yyyy`; decimais com vírgula convertidos automaticamente.
- Durações `HH:MM` viram horas decimais (`sleep_duration_h` etc.).
- Horário de cama antes das 18h é tratado como pós-meia-noite (01:36 → 25,6h)
  pra série ficar contínua no gráfico.
- Medicações: só aparecem as com pelo menos 1 dia de uso > 0 no período.
- Correlações: lag k = X de k dias atrás vs Y de hoje (ex.: sono de ontem →
  humor de hoje). Pearson e Spearman com p-valor e n.
