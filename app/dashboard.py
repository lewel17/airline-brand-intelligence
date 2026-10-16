"""Brand Intelligence Dashboard - Ryanair vs EasyJet.

Avvio (dalla cartella principale del progetto):
    streamlit run app/dashboard.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"

st.set_page_config(page_title="Brand Intelligence — Ryanair vs EasyJet", layout="wide")

RYANAIR = "#3B82C4"
EASYJET = "#E8843C"
POS = "#3AA76D"
NEG = "#D94F4F"
NEUTRO = "#8B95A5"
ACCENT = "#3B82C4"
GRID = "rgba(128,128,128,0.2)"
FONT = dict(size=13, color="#7A7A7A")

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2.5rem; max-width: 1200px; }}
    #MainMenu, footer {{ visibility: hidden; }}

    .hero {{ margin-bottom: 30px; border-bottom: 2px solid {ACCENT}; padding-bottom: 20px; }}
    .hero-title {{ font-size: 38px !important; font-weight: 700; margin: 0; letter-spacing: -1px;
                   color: {ACCENT}; line-height: 1.2; }}
    .hero-sub {{ font-size: 16px; margin-top: 10px; opacity: 0.7; }}

    .card {{ border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 20px 22px; }}
    .card-label {{ font-size: 12px; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px;
                   margin-bottom: 8px; }}
    .card-value {{ font-size: 30px; font-weight: 600; color: {ACCENT}; line-height: 1; }}

    .sec {{ font-size: 17px; font-weight: 600; margin: 30px 0 4px; }}
    .sec-note {{ font-size: 13px; opacity: 0.6; margin-bottom: 8px; }}

    .insight {{ border: 1px solid rgba(128,128,128,0.25); border-left: 3px solid {ACCENT};
                border-radius: 10px; padding: 20px 24px; margin-top: 12px; }}
    .insight-t {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: {ACCENT}; }}
    .insight-x {{ font-size: 14px; line-height: 1.75; opacity: 0.85; }}

    .foot {{ font-size: 12px; opacity: 0.55; margin-top: 36px; line-height: 1.6; }}

    div[role="radiogroup"] {{ flex-direction: row; gap: 8px; }}
    div[role="radiogroup"] label {{ border: 1px solid rgba(128,128,128,0.3); border-radius: 8px;
                                    padding: 8px 20px; margin: 0; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def carica_dati():
    recensioni = pd.read_csv(DATA / "dashboard_data.csv")
    temi = pd.read_csv(DATA / "results" / "temi_principali.csv")
    return recensioni, temi


df, temi = carica_dati()

st.markdown(f"""
<div class="hero">
    <p class="hero-title">Brand Intelligence Dashboard</p>
    <p class="hero-sub">Comparative brand reputation analysis in low-cost air travel ·
    {len(df):,} Skytrax reviews · distilBERT sentiment + BERTopic themes</p>
</div>
""", unsafe_allow_html=True)

scelta = st.radio("Airline", ["Both", "Ryanair", "EasyJet"], horizontal=True,
                  label_visibility="collapsed")
dati = df if scelta == "Both" else df[df['AirlineName'] == scelta]


def perc_pos(sub):
    return round((sub['sentiment'] == 'POSITIVE').mean() * 100) if len(sub) else 0


# --- KPI ---
st.markdown("<br>", unsafe_allow_html=True)
metriche = [
    ("Reviews", f"{len(dati):,}"),
    ("Positive sentiment", f"{perc_pos(dati)}%"),
    ("Average score", f"{dati['OverallScore'].mean():.1f}/10"),
    ("Would recommend", f"{round((dati['Recommended'] == 'yes').mean() * 100)}%"),
]
for col, (lab, val) in zip(st.columns(4), metriche):
    col.markdown(f'<div class="card"><div class="card-label">{lab}</div>'
                 f'<div class="card-value">{val}</div></div>', unsafe_allow_html=True)

ry = df[df['AirlineName'] == 'Ryanair']
ej = df[df['AirlineName'] == 'EasyJet']
LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=FONT,
              margin=dict(t=10, b=10, l=10, r=10))

cA, cB = st.columns(2)

with cA:
    st.markdown('<p class="sec">Sentiment comparison</p>', unsafe_allow_html=True)
    rp, ep = perc_pos(ry), perc_pos(ej)
    f1 = go.Figure()
    f1.add_trace(go.Bar(name='Positive', x=['Ryanair', 'EasyJet'], y=[rp, ep], marker_color=POS,
                        text=[f"{rp}%", f"{ep}%"], textposition="inside"))
    f1.add_trace(go.Bar(name='Negative', x=['Ryanair', 'EasyJet'], y=[100 - rp, 100 - ep],
                        marker_color=NEG, text=[f"{100 - rp}%", f"{100 - ep}%"],
                        textposition="inside"))
    f1.update_layout(barmode='stack', height=330, yaxis=dict(title="% of reviews", gridcolor=GRID),
                     legend=dict(orientation="h", y=1.15), **LAYOUT)
    st.plotly_chart(f1, use_container_width=True)

with cB:
    st.markdown('<p class="sec">Score distribution</p>', unsafe_allow_html=True)
    voti = dati['OverallScore'].dropna().astype(int).value_counts().sort_index()
    colore = RYANAIR if scelta == 'Ryanair' else EASYJET if scelta == 'EasyJet' else NEUTRO
    f2 = go.Figure(go.Bar(x=voti.index, y=voti.values, marker_color=colore))
    f2.update_layout(height=330, xaxis=dict(title="Overall score (1-10)", dtick=1),
                     yaxis=dict(title="Reviews", gridcolor=GRID), **LAYOUT)
    st.plotly_chart(f2, use_container_width=True)

# --- Temi critici ---
st.markdown('<p class="sec">Most critical themes — % negative reviews</p>', unsafe_allow_html=True)
st.markdown('<p class="sec-note">Themes extracted with BERTopic, separately for each airline. '
            '“Not emerged” means the theme did not form a cluster for that airline '
            '(it is absent, not 0%).</p>', unsafe_allow_html=True)

ordine = list(dict.fromkeys(temi['tema']))
f3 = go.Figure()
for compagnia, colore in [("Ryanair", RYANAIR), ("EasyJet", EASYJET)]:
    sub = temi[temi['compagnia'] == compagnia].set_index('tema').reindex(ordine)
    valori = sub['perc_negative']
    f3.add_trace(go.Bar(
        name=compagnia, y=ordine, x=valori.fillna(0), orientation='h', marker_color=colore,
        text=[f"{v:.0f}%" if pd.notna(v) else "not emerged" for v in valori],
        textposition="outside"))
f3.update_layout(barmode='group', height=420,
                 xaxis=dict(title="% negative", gridcolor=GRID, range=[0, 112]),
                 yaxis=dict(autorange="reversed"),
                 legend=dict(orientation="h", y=1.08), **LAYOUT)
st.plotly_chart(f3, use_container_width=True)

# --- Insight ---
st.markdown("""
<div class="insight">
    <div class="insight-t">Strategic insights</div>
    <div class="insight-x">
    <b>Ryanair</b> has a distinctive pain point in online check-in and boarding passes
    (up to 97% negative sentiment), a theme that does not emerge at all for EasyJet.
    Priority action: simplify the check-in process.<br><br>
    <b>EasyJet</b> keeps an advantage in onboard service, with recurring positive mentions of
    crew and assistance: a lever for competitive differentiation.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="foot">
Data: Skytrax reviews (Kaggle “AirlineReviews” dataset), Ryanair and EasyJet, after cleaning.
Sentiment: distilbert-base-uncased-finetuned-sst-2-english. Themes: BERTopic + HDBSCAN.<br>
BSc thesis, Università degli Studi di Milano-Bicocca · Leonardo Palma
</div>
""", unsafe_allow_html=True)
