"""Interface Streamlit do Epidemiological Forecaster."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


st.set_page_config(
    page_title="Epidemiological Forecaster",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "").strip().rstrip("/")
GEOJSON_PATH = Path(__file__).resolve().parent / "assets" / "limites_bairros_recife_2023.geojson"

if not API_URL:
    st.error("API_URL não foi configurada. Informe a URL pública da FastAPI no ambiente do Streamlit.")
    st.stop()
RISK_ORDER = ["Baixo", "Médio", "Alto", "Crítico"]
RISK_COLORS = {
    "Baixo": "#20c665",
    "Médio": "#ffd448",
    "Alto": "#ff8618",
    "Crítico": "#f1283c",
}
RISK_ACTIONS = {
    "Baixo": [
        "Manter o monitoramento epidemiológico de rotina.",
        "Preservar as ações regulares de eliminação de criadouros.",
    ],
    "Médio": [
        "Reforçar a vigilância de novos casos e sintomas.",
        "Ampliar a comunicação preventiva com a população.",
        "Priorizar inspeções em locais com recorrência de focos.",
    ],
    "Alto": [
        "Intensificar inspeções domiciliares e controle vetorial.",
        "Mobilizar equipes de vigilância para resposta antecipada.",
        "Acompanhar diariamente a evolução das notificações.",
    ],
    "Crítico": [
        "Acionar resposta epidemiológica prioritária.",
        "Concentrar ações de controle vetorial no bairro.",
        "Reforçar comunicação de risco e capacidade assistencial.",
        "Reavaliar o cenário assim que novos dados forem recebidos.",
    ],
}


st.markdown(
    """
    <style>
    :root { --navy:#06243d; --blue:#1683ff; --border:#dfe8f2; --muted:#64748b; }
    .stApp { background:#f5f8fc; color:#092b4c; }
    .block-container { padding-top:1rem; padding-bottom:2rem; max-width:1500px; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#082f50,#031a2d); }
    [data-testid="stSidebar"] * { color:white; }
    .brand { text-align:center; padding:12px 0 22px; }
    .brand-icon { font-size:54px; }
    .brand-title { font-size:30px; font-weight:900; letter-spacing:.04em; }
    .brand-sub { font-size:13px; opacity:.85; }
    .hero { background:white; border:1px solid var(--border); border-radius:16px; padding:18px 22px;
            box-shadow:0 8px 24px rgba(25,52,82,.06); margin-bottom:14px; }
    .hero h1 { margin:0; font-size:28px; color:#092b4c; }
    .hero p { margin:6px 0 0; color:var(--muted); }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--border); border-radius:16px;
            padding:16px 18px; box-shadow:0 7px 20px rgba(25,52,82,.06); min-height:118px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        font-weight:800 !important; color:#23415f !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color:#092b4c !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:white; border-color:var(--border);
            border-radius:16px; box-shadow:0 7px 20px rgba(25,52,82,.05); }
    .risk-card { border-radius:14px; color:white; padding:16px; min-height:135px;
            box-shadow:0 8px 18px rgba(15,36,60,.10); }
    .risk-card .horizon { font-size:13px; font-weight:800; opacity:.9; }
    .risk-card .level { font-size:24px; font-weight:950; margin:8px 0 3px; }
    .risk-card .confidence { font-size:14px; font-weight:750; }
    .risk-card .target { font-size:12px; opacity:.9; margin-top:8px; }
    .definition-table { width:100%; border-collapse:collapse; font-size:14px; }
    .definition-table th,.definition-table td { padding:10px; border-bottom:1px solid #e6edf5; text-align:left; }
    .definition-table th { color:#23415f; background:#f7faff; }
    .notice { background:#eef6ff; border:1px solid #cfe4ff; border-radius:12px; padding:12px 14px;
              color:#21476c; font-size:13px; }
    .update-row { padding:9px 0; border-bottom:1px solid #e6edf5; }
    .update-row:last-child { border:0; }
    .update-label { color:var(--muted); font-size:12px; }
    .update-value { color:#092b4c; font-weight:800; }
    .action-list li { margin:8px 0; color:#23415f; }
    div[data-testid="stDownloadButton"] button {
        background:#0b6ffb !important; color:#ffffff !important; border:1px solid #0b6ffb !important;
        font-weight:800 !important;
    }
    div[data-testid="stDownloadButton"] button * { color:#ffffff !important; }
    div[data-testid="stDownloadButton"] button:hover {
        background:#075dcc !important; border-color:#075dcc !important; color:#ffffff !important;
    }
    h2,h3 { color:#092b4c !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def br_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def br_float(value: float | int, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%".replace(".", ",")


@st.cache_data(show_spinner=False)
def load_neighborhood_geojson() -> dict:
    """Carrega os limites de 2023 e padroniza a chave usada pelo modelo."""
    with GEOJSON_PATH.open("r", encoding="utf-8-sig") as source:
        geojson = json.load(source)
    name_aliases = {"SITIO DOS PINTOS": "SITIO DOS PINTOS SAO BRAS"}
    for feature in geojson["features"]:
        properties = feature.setdefault("properties", {})
        source_name = str(properties.get("EBAIRRNOME", "")).strip().upper()
        properties["bairro_norm"] = name_aliases.get(source_name, source_name)
    return geojson


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict:
    # O plano gratuito do Render pode precisar de cerca de um minuto para
    # reativar a API após um período sem acessos.
    response = requests.get(f"{API_URL}{path}", params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def load_or_stop(path: str, params: dict | None = None) -> dict:
    try:
        return api_get(path, params)
    except requests.RequestException as exc:
        st.error(
            "Não foi possível consultar a API do Epidemiological Forecaster. Se este for o primeiro "
            "acesso após um período sem uso, aguarde um minuto e recarregue a página. Confira também "
            "a variável `API_URL` no ambiente do Streamlit."
        )
        st.caption(str(exc))
        st.stop()


def panel(title: str):
    container = st.container(border=True)
    container.markdown(f"### {title}")
    return container


def probability_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Categoria": RISK_ORDER,
            "Probabilidade": [row["prob_baixo"], row["prob_medio"], row["prob_alto"], row["prob_critico"]],
        }
    )


def probability_chart(row: pd.Series) -> go.Figure:
    frame = probability_frame(row)
    figure = px.bar(
        frame,
        x="Categoria",
        y="Probabilidade",
        color="Categoria",
        color_discrete_map=RISK_COLORS,
        text=frame["Probabilidade"].map(pct),
        category_orders={"Categoria": RISK_ORDER},
    )
    figure.update_traces(textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%", title=None)
    figure.update_xaxes(title=None)
    figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=385)
    return figure


def threshold_table() -> str:
    rows = [
        ("Baixo", "Menor que 100 por 100 mil habitantes"),
        ("Médio", "De 100 até menos de 300 por 100 mil"),
        ("Alto", "De 300 até menos de 500 por 100 mil"),
        ("Crítico", "Igual ou superior a 500 por 100 mil"),
    ]
    body = "".join(
        f'<tr><td><span style="color:{RISK_COLORS[level]};font-weight:900">● {level}</span></td><td>{definition}</td></tr>'
        for level, definition in rows
    )
    return (
        '<table class="definition-table"><thead><tr><th>Nível</th>'
        '<th>Incidência acumulada em 4 semanas</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def horizon_cards(forecasts: pd.DataFrame, scope: str) -> None:
    columns = st.columns(4)
    for column, (_, row) in zip(columns, forecasts.sort_values("horizonte_semanas").iterrows()):
        horizon = int(row["horizonte_semanas"])
        with column:
            if scope == "bairro":
                level = row["categoria_prevista"]
                color = RISK_COLORS[level]
                main = level
                detail = f"Confiança: {pct(row['confianca_modelo'])}"
            else:
                color = "#1683ff"
                main = f"{int(row['bairros_alto_critico'])} bairros"
                detail = f"{int(row['bairros_criticos'])} críticos · confiança média {pct(row['confianca_modelo'])}"
            st.markdown(
                f"""
                <div class="risk-card" style="background:{color}">
                  <div class="horizon">S+{horizon}</div>
                  <div class="level">{main}</div>
                  <div class="confidence">{detail}</div>
                  <div class="target">Semana-alvo: {row['semana_alvo']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">🦟</div>'
        '<div class="brand-title">Epidemiological<br>Forecaster</div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navegação", ["Dashboard", "Bairros"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("XGBoost · Criticidade v2")
    st.caption("Produção operacional + validação")


options = load_or_stop("/api/options")
origin_labels = [item["label"] for item in options["origens"]]

st.markdown(
    """
    <div class="hero">
      <h1>Epidemiological Forecaster</h1>
      <p>Classificação da criticidade epidemiológica em Recife com horizontes de uma a quatro semanas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Dashboard":
    filter_week, filter_horizon, filter_level = st.columns([1.4, 1, 1])
    with filter_week:
        origin_label = st.selectbox("Semana epidemiológica de origem", origin_labels, index=0)
    with filter_horizon:
        horizon = st.selectbox("Horizonte", [1, 2, 3, 4], format_func=lambda value: f"S+{value}")
    with filter_level:
        level = st.selectbox("Nível de criticidade", ["Todos", *RISK_ORDER])
    origin_year, origin_week = [int(value) for value in origin_label.split("-")]

    dashboard = load_or_stop(
        "/api/dashboard",
        {
            "origin_year": origin_year,
            "origin_week": origin_week,
            "horizon": horizon,
            "level": None if level == "Todos" else level,
        },
    )
    detail = load_or_stop("/api/bairro", {"origin_year": origin_year, "origin_week": origin_week})
    performance = load_or_stop("/api/performance")
    kpis = dashboard["kpis"]

    st.caption(
        f"Previsão feita a partir da SE {origin_label} para a SE {dashboard['filtros']['semana_alvo']}."
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Bairros monitorados", br_int(kpis["bairros_monitorados"]))
    k2.metric("Bairros altos", br_int(kpis["bairros_altos"]))
    k3.metric("Bairros críticos", br_int(kpis["bairros_criticos"]))
    k4.metric("Incidência recente observada", br_float(kpis["incidencia_recente_observada_4s_100k"]), help="Acumulado de 4 semanas por 100 mil habitantes.")
    k5.metric("Confiança média", pct(kpis["confianca_media"]))

    left, right = st.columns([1.25, 1])
    with left:
        with panel("Painel de criticidade dos bairros"):
            neighborhoods = pd.DataFrame(dashboard["bairros"])
            if neighborhoods.empty:
                st.info("Nenhum bairro corresponde ao nível selecionado.")
            else:
                geojson = load_neighborhood_geojson()
                figure = px.choropleth_map(
                    neighborhoods,
                    geojson=geojson,
                    locations="bairro_norm",
                    featureidkey="properties.bairro_norm",
                    color="categoria_prevista",
                    color_discrete_map=RISK_COLORS,
                    category_orders={"categoria_prevista": RISK_ORDER},
                    labels={"categoria_prevista": "Criticidade"},
                    hover_name="bairro_norm",
                    custom_data=["semana_alvo", "confianca_modelo", "prob_alto", "prob_critico"],
                    map_style="carto-positron",
                    center={"lat": -8.055, "lon": -34.91},
                    zoom=10.2,
                    opacity=0.78,
                )
                figure.update_traces(
                    hovertemplate=(
                        "<b>%{location}</b><br>Semana-alvo: %{customdata[0]}<br>"
                        "Confiança: %{customdata[1]:.1%}<br>P(Alto): %{customdata[2]:.1%}<br>"
                        "P(Crítico): %{customdata[3]:.1%}<extra></extra>"
                    ),
                    marker_line_width=1,
                    marker_line_color="white",
                )
                figure.update_layout(
                    margin=dict(l=4, r=4, t=4, b=4),
                    height=450,
                    legend=dict(title="Criticidade", orientation="h", y=0.01, x=0.01),
                )
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        with panel("Bairros com maior risco"):
            ranking = pd.DataFrame(dashboard["ranking"])
            ranking = ranking[["bairro_norm", "categoria_prevista", "confianca_modelo", "prob_alto", "prob_critico"]]
            ranking.columns = ["Bairro", "Criticidade", "Confiança", "P(Alto)", "P(Crítico)"]
            st.dataframe(
                ranking,
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "Confiança": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                    "P(Alto)": st.column_config.NumberColumn(format="percent"),
                    "P(Crítico)": st.column_config.NumberColumn(format="percent"),
                },
            )

    distribution_col, horizons_col = st.columns([0.8, 1.4])
    with distribution_col:
        with panel("Distribuição da criticidade"):
            distribution = pd.DataFrame(dashboard["distribuicao"])
            figure = px.pie(
                distribution,
                names="categoria",
                values="bairros",
                color="categoria",
                color_discrete_map=RISK_COLORS,
                hole=0.58,
                category_orders={"categoria": RISK_ORDER},
            )
            figure.update_traces(
                textinfo="label+value+percent",
                hovertemplate="%{label}: %{value} bairros (%{percent})<extra></extra>",
            )
            figure.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=330, showlegend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with horizons_col:
        with panel("Evolução da criticidade por horizonte"):
            horizon_distribution = pd.DataFrame(dashboard["distribuicao_horizontes"])
            long = horizon_distribution.melt(
                id_vars="horizonte_semanas",
                value_vars=RISK_ORDER,
                var_name="Categoria",
                value_name="Bairros",
            )
            long["Horizonte"] = long["horizonte_semanas"].map(lambda value: f"S+{int(value)}")
            figure = px.bar(
                long,
                x="Horizonte",
                y="Bairros",
                color="Categoria",
                color_discrete_map=RISK_COLORS,
                category_orders={"Categoria": RISK_ORDER},
                text="Bairros",
            )
            figure.update_layout(barmode="stack", margin=dict(l=5, r=5, t=10, b=5), height=330)
            figure.update_xaxes(title=None)
            figure.update_yaxes(title="Número de bairros")
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    climate_col, performance_col, update_col = st.columns([1.2, 1.1, 0.8])
    with climate_col:
        with panel("Condições climáticas recentes"):
            climate = pd.DataFrame(detail["clima"])
            figure = go.Figure()
            figure.add_bar(x=climate["semana"], y=climate["precipitacao_total"], name="Chuva (mm)", marker_color="#1683ff")
            figure.add_scatter(x=climate["semana"], y=climate["temp_max_media"], name="Temperatura máxima (°C)", yaxis="y2", line=dict(color="#f1283c", width=3))
            figure.update_layout(
                height=330,
                margin=dict(l=5, r=5, t=20, b=5),
                legend=dict(orientation="h", y=1.12),
                yaxis=dict(title="mm"),
                yaxis2=dict(title="°C", overlaying="y", side="right"),
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with performance_col:
        with panel("Desempenho do XGBoost"):
            metrics = pd.DataFrame(performance["metricas"])
            metric_long = metrics.melt(
                id_vars="horizonte_semanas",
                value_vars=["acuracia_balanceada", "f1_macro", "f2_macro"],
                var_name="Métrica",
                value_name="Valor",
            )
            metric_long["Horizonte"] = metric_long["horizonte_semanas"].map(lambda value: f"S+{int(value)}")
            metric_long["Métrica"] = metric_long["Métrica"].map(
                {"acuracia_balanceada": "Acurácia balanceada", "f1_macro": "F1 macro", "f2_macro": "F2 macro"}
            )
            figure = px.line(metric_long, x="Horizonte", y="Valor", color="Métrica", markers=True)
            figure.update_yaxes(range=[0, 1], tickformat=".0%", title=None)
            figure.update_xaxes(title=None)
            figure.update_layout(height=330, margin=dict(l=5, r=5, t=20, b=5), legend=dict(orientation="h", y=1.16))
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
            if performance.get("fonte") == "producao":
                st.caption(
                    f"Produção: {performance.get('semanas_avaliadas')} semanas consolidadas "
                    f"({performance.get('periodo_inicio')} a {performance.get('periodo_fim')})."
                )
            else:
                st.caption("Referência atual: avaliação temporal independente com dados de 2021.")
    with update_col:
        with panel("Atualização"):
            update = dashboard["atualizacao"]
            mode_label = "Produção" if update["modo"] == "producao" else "Validação histórica"
            st.markdown(
                f"""
                <div class="update-row"><div class="update-label">Modelo</div><div class="update-value">{update['modelo']}</div></div>
                <div class="update-row"><div class="update-label">Versão</div><div class="update-value">{update['versao']}</div></div>
                <div class="update-row"><div class="update-label">Dados processados até</div><div class="update-value">SE {update['dados_processados_ate']}</div></div>
                <div class="update-row"><div class="update-label">Modo</div><div class="update-value">{mode_label}</div></div>
                <div class="update-row"><div class="update-label">Previsões geradas em</div><div class="update-value">{update['previsoes_geradas_em']}</div></div>
                """,
                unsafe_allow_html=True,
            )

    with panel("Como interpretar os níveis de criticidade"):
        st.markdown(threshold_table(), unsafe_allow_html=True)
        st.caption("Os limites são aplicados à incidência acumulada na janela de quatro semanas terminada na semana-alvo.")


else:
    filter_week, filter_neighborhood, filter_horizon = st.columns([1.2, 1.6, 0.8])
    with filter_week:
        origin_label = st.selectbox("Semana epidemiológica de origem", origin_labels, index=0)
    with filter_neighborhood:
        neighborhood_label = st.selectbox(
            "Bairro",
            ["Todos os bairros", *[name.title() for name in options["bairros"]]],
        )
    with filter_horizon:
        selected_horizon = st.selectbox("Detalhar horizonte", [1, 2, 3, 4], format_func=lambda value: f"S+{value}")
    origin_year, origin_week = [int(value) for value in origin_label.split("-")]
    neighborhood = None if neighborhood_label == "Todos os bairros" else neighborhood_label.upper()
    detail = load_or_stop(
        "/api/bairro",
        {"origin_year": origin_year, "origin_week": origin_week, "bairro": neighborhood},
    )
    forecasts = pd.DataFrame(detail["previsoes"]).sort_values("horizonte_semanas")
    if "status_avaliacao" not in forecasts:
        forecasts["status_avaliacao"] = "Consolidado"
    if "data_avaliacao" not in forecasts:
        forecasts["data_avaliacao"] = pd.NA
    selected = forecasts.loc[forecasts["horizonte_semanas"].astype(int).eq(selected_horizon)].iloc[0]

    st.subheader(f"{detail['nome']} — previsões de criticidade")
    mode = str(selected.get("modo_resultado", "validacao_historica_2021"))
    mode_text = (
        "As previsões são operacionais e serão avaliadas quando os dados observados estiverem disponíveis."
        if mode == "producao"
        else "Os resultados são previsões históricas produzidas sem utilizar os valores futuros."
    )
    st.caption(f"Informações disponíveis até a SE {detail['semana_origem']}. {mode_text}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("População", br_int(detail["populacao"]))
    k2.metric("Casos observados — 4 semanas", br_int(detail["casos_observados_recentes_4s"]))
    k3.metric("Incidência observada — 4 semanas", br_float(detail["incidencia_recente_4s_100k"]))
    k4.metric("Criticidade observada recente", detail["categoria_recente_observada"])
    if detail["escopo"] == "bairro":
        k5.metric(f"Previsão S+{selected_horizon}", selected["categoria_prevista"])
    else:
        k5.metric(f"Alto/crítico em S+{selected_horizon}", br_int(selected["bairros_alto_critico"]))

    st.markdown("### Horizontes de previsão")
    horizon_cards(forecasts, detail["escopo"])
    st.caption("Cada horizonte classifica a incidência acumulada em quatro semanas terminada na respectiva semana-alvo.")

    probability_col, trajectory_col = st.columns([0.85, 1.35])
    with probability_col:
        with panel(f"Probabilidade estimada — S+{selected_horizon}"):
            st.plotly_chart(probability_chart(selected), width="stretch", config={"displayModeBar": False})
    with trajectory_col:
        with panel("Trajetória das probabilidades nos quatro horizontes"):
            trajectory = forecasts.melt(
                id_vars=["horizonte_semanas", "semana_alvo"],
                value_vars=["prob_baixo", "prob_medio", "prob_alto", "prob_critico"],
                var_name="Categoria",
                value_name="Probabilidade",
            )
            trajectory["Categoria"] = trajectory["Categoria"].map(
                {"prob_baixo": "Baixo", "prob_medio": "Médio", "prob_alto": "Alto", "prob_critico": "Crítico"}
            )
            trajectory["Horizonte"] = trajectory["horizonte_semanas"].map(lambda value: f"S+{int(value)}")
            figure = px.line(
                trajectory,
                x="Horizonte",
                y="Probabilidade",
                color="Categoria",
                markers=True,
                color_discrete_map=RISK_COLORS,
                category_orders={"Categoria": RISK_ORDER},
                hover_data=["semana_alvo"],
            )
            figure.update_yaxes(range=[0, 1.08], tickformat=".0%", title=None)
            figure.update_xaxes(title=None)
            figure.update_layout(
                height=385,
                margin=dict(l=5, r=5, t=52, b=5),
                legend=dict(orientation="h", y=1.18, x=0),
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    history_col, climate_col = st.columns([1.3, 1])
    with history_col:
        with panel("Histórico observado até a semana de origem"):
            history = pd.DataFrame(detail["historico"])
            figure = go.Figure()
            figure.add_bar(x=history["semana"], y=history["casos_totais"], name="Casos na semana", marker_color="#1683ff")
            figure.add_scatter(x=history["semana"], y=history["casos_acumulados_4s"], name="Acumulado de 4 semanas", line=dict(color="#13bfcf", width=3))
            figure.update_layout(height=350, margin=dict(l=5, r=5, t=20, b=5), legend=dict(orientation="h", y=1.12))
            figure.update_yaxes(title="Casos observados")
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with climate_col:
        with panel("Condições climáticas recentes"):
            climate = pd.DataFrame(detail["clima"])
            figure = go.Figure()
            figure.add_bar(x=climate["semana"], y=climate["precipitacao_total"], name="Chuva (mm)", marker_color="#1683ff")
            figure.add_scatter(x=climate["semana"], y=climate["temp_max_media"], name="Temperatura máxima (°C)", yaxis="y2", line=dict(color="#f1283c", width=3))
            figure.update_layout(
                height=350,
                margin=dict(l=5, r=5, t=20, b=5),
                legend=dict(orientation="h", y=1.12),
                yaxis=dict(title="mm"),
                yaxis2=dict(title="°C", overlaying="y", side="right"),
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    definition_col, actions_col = st.columns([1.1, 0.9])
    with definition_col:
        with panel("Como interpretar os níveis"):
            st.markdown(threshold_table(), unsafe_allow_html=True)
    with actions_col:
        with panel("Ações recomendadas"):
            if detail["escopo"] == "bairro":
                level = selected["categoria_prevista"]
                actions = RISK_ACTIONS[level]
                st.markdown(f"**Nível previsto em S+{selected_horizon}: {level}**")
            else:
                actions = [
                    "Priorizar os bairros classificados como Alto ou Crítico no ranking.",
                    "Reavaliar os horizontes com crescimento da proporção de categorias graves.",
                    "Cruzar os alertas com notificações e condições climáticas recentes.",
                ]
                st.markdown(f"**Resumo municipal em S+{selected_horizon}**")
            st.markdown('<ul class="action-list">' + "".join(f"<li>{action}</li>" for action in actions) + "</ul>", unsafe_allow_html=True)

    with panel("Detalhes das previsões"):
        if detail["escopo"] == "bairro":
            table = forecasts[[
                "semana_alvo", "horizonte_semanas", "categoria_prevista", "confianca_modelo",
                "prob_baixo", "prob_medio", "prob_alto", "prob_critico",
                "categoria_observada_alvo", "status_avaliacao", "previsao_correta",
            ]].copy()
            table.columns = [
                "Semana-alvo", "Horizonte", "Categoria prevista", "Confiança",
                "P(Baixo)", "P(Médio)", "P(Alto)", "P(Crítico)",
                "Categoria observada", "Situação", "Acerto",
            ]
            table["Categoria observada"] = table["Categoria observada"].fillna("Aguardando dados")

            def result_label(row: pd.Series) -> str:
                if row["Situação"] == "Pendente" or pd.isna(row["Acerto"]):
                    return "Aguardando observação"
                result = "Acertou" if bool(row["Acerto"]) else "Errou"
                return f"{result} (provisório)" if row["Situação"] == "Provisório" else result

            table["Resultado"] = table.apply(result_label, axis=1)
            table = table.drop(columns="Acerto")
        else:
            table = forecasts[[
                "semana_alvo", "horizonte_semanas", "bairros_alto_critico", "bairros_criticos",
                "confianca_modelo", "prob_baixo", "prob_medio", "prob_alto", "prob_critico",
            ]].copy()
            table.columns = [
                "Semana-alvo", "Horizonte", "Bairros alto/crítico", "Bairros críticos",
                "Confiança média", "P(Baixo)", "P(Médio)", "P(Alto)", "P(Crítico)",
            ]
        table["Horizonte"] = table["Horizonte"].map(lambda value: f"S+{int(value)}")
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Baixar dados filtrados (CSV)",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"epidemiological_forecaster_{detail['nome'].lower().replace(' ', '_')}_{origin_label}.csv",
            mime="text/csv",
        )
