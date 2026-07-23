# Round Advisor — Adaptyv Bio Take-Home Project

## Obiettivo

Dimostrare come un layer di active learning + agente orchestratore sopra i dati sperimentali
generati dal laboratorio Adaptyv possa ridurre il numero di design da testare per trovare i
binder migliori, e come questo si integrerebbe nel loro workflow API (upload designs → run
esperimenti → risultati → nuovo round).

Non è un modello "production-grade": è un POC che quantifica il valore di una strategia di
selezione informata rispetto a una selezione casuale, usando dati reali pubblicati da Adaptyv
(competizione EGFR, round 1 e 2).

## Componenti

1. **Data layer** — ingestion e pulizia dei dati delle competizioni EGFR (round 1 e round 2).
2. **Feature engineering** — feature fisico-chimiche da sequenza (Biopython ProtParam),
   eventualmente embedding ESM2 small come stretch goal.
3. **Modello predittivo** — regressione su log(KD), cross-validated, metrica principale
   Spearman correlation (non R², dataset piccolo e con outlier).
4. **Backtest active learning** — usa il round 1 come storico, simula la selezione dei
   candidati per il round 2 con un'acquisition function (UCB su predizione + incertezza da
   ensemble bootstrap), confronta contro selezione random. Output: "con lo stesso budget
   sperimentale, la strategia informata trova X% dei top binder reali in N test, contro Y%
   della selezione casuale".
5. **Mock API Adaptyv** — schema REST plausibile ispirato al flusso pubblico del prodotto
   (`POST /designs`, `GET /designs/{id}/results`), documentato in OpenAPI/Pydantic. Va
   dichiarato esplicitamente come ipotesi, non lo schema reale.
6. **Agente con tool-calling** (Claude API) — 3-4 tool al massimo:
   - `predict_affinity(sequences)`
   - `rank_candidates(sequences, budget)`
   - `submit_batch(sequences)` → mock API
   - `summarize_round(results)`
   Riceve richieste in linguaggio naturale ("ho budget per 20 test, dammi la selezione
   migliore e spiega il trade-off") e orchestra le chiamate.
7. **UI Streamlit** — interfaccia demo che unisce modello + agente, deployabile su Streamlit
   Community Cloud.

## Struttura repo proposta

```
round-advisor/
├── README.md                  # rationale, come girare il progetto, risultati chiave
├── PLAN.md                    # questo file
├── requirements.txt
├── .env.example                # ANTHROPIC_API_KEY placeholder
├── data/
│   ├── raw/                    # dati scaricati as-is (vedi sezione "Dove prendere i dati")
│   │   ├── egfr_round1/
│   │   └── egfr_round2/
│   └── processed/               # output di cleaning/feature engineering
├── src/
│   ├── data_loader.py           # parsing e pulizia dei dati grezzi
│   ├── features.py              # feature engineering (ProtParam, opz. ESM2)
│   ├── model.py                 # training, CV, metriche
│   ├── active_learning.py       # acquisition function + backtest round1→round2
│   ├── mock_api.py              # schema Pydantic + endpoint FastAPI/mock
│   ├── agent.py                 # agente con tool-calling (Claude API)
│   └── app.py                   # entry point Streamlit
├── notebooks/
│   └── 01_eda.ipynb             # esplorazione dati, grafici per il Loom
└── tests/
    └── test_active_learning.py  # sanity check sul backtest
```

## Piano a 2 giorni

**Giorno 1 — dati e modello**
- [ ] Setup repo, ambiente virtuale, requirements.txt
- [ ] Download e parsing dati round 1 e round 2 (`data_loader.py`)
- [ ] EDA: distribuzione KD, sequenze fallite/non espresse, cosa distingue i binder forti
- [ ] Feature engineering (fisico-chimiche baseline)
- [ ] Modello baseline + cross-validation, report Spearman
- [ ] Backtest active learning round1→round2, fissare il numero "valore aggiunto" da citare nel Loom

**Giorno 2 — agente e presentazione**
- [ ] Mock API con schema OpenAPI/Pydantic
- [ ] Agente con tool-calling, system prompt allineato al dominio
- [ ] App Streamlit che unisce modello + agente
- [ ] README con rationale esplicito legato al business Adaptyv, citazione fonte dati e licenza ODC-BY
- [ ] Deploy (Streamlit Cloud) o screen recording locale
- [ ] Registrazione Loom (~5 min): problema, dati, risultato del backtest, demo agente end-to-end, nota onestà su limiti del modello e sullo schema API ipotetico

## Cose da dire esplicitamente nel Loom (rischi/limiti)

- Il dataset è piccolo (centinaia di punti): il modello è illustrativo, non da produzione — non
  sopravvalutare l'accuratezza predittiva davanti a un team che questi dati li genera per mestiere.
- Lo schema dell'API è un'ipotesi basata sulla documentazione pubblica, non quello reale —
  inquadrarlo come punto di partenza per allineamento futuro.
- Citare fonte dati e licenza ODC-BY nel README.
- Tenere il tool-calling dell'agente a 3-4 funzioni: un agente semplice che funziona bene batte
  un agente ambizioso che si rompe in demo.

## Dove prendere i dati

Repo GitHub ufficiali Adaptyv (licenza dati ODC-BY, codice Apache 2):

- Round 1 (binding affinity EGFR): https://github.com/adaptyvbio/egfr_competition_1
  - Dati grezzi + curve cinetiche: https://api.adaptyvbio.com/storage/v1/object/public/egfr_design_competition/package.zip
  - Dati processati (characterization data): cartella `results/` nel repo
- Round 2 (binding + neutralizzazione EGFR): https://github.com/adaptyvbio/egfr_competition_2
  - Dati processati (affinità + similarità di sequenza + neutralizzazione):
    https://api.adaptyvbio.com/storage/v1/object/public/egfr_design_competition_2/package_neutralisation.zip
- Hub generale con altri dataset futuri: https://proteinbase.com

Scarica entrambi gli zip in `data/raw/egfr_round1/` e `data/raw/egfr_round2/`, mantenendo la
struttura interna originale (serve per capire il formato delle curve SPR grezze se decidi di
usarle oltre ai dati processati).
