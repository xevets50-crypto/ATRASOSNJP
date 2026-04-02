import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# =============================
# CONFIG
# =============================
PASTA_DADOS = "dados"

st.set_page_config(layout="wide")
st.title("🚚 Torre de Controle Logística")

# =============================
# ARQUIVOS
# =============================
def listar_arquivos(pasta):
    if not os.path.exists(pasta):
        return []
    return [f for f in os.listdir(pasta) if f.endswith(".xlsx")]

arquivos = listar_arquivos(PASTA_DADOS)

st.sidebar.header("📁 Fonte de Dados")
upload = st.sidebar.file_uploader("Upload manual", type=["xlsx"])

if arquivos:
    arq = st.sidebar.selectbox("Arquivos GitHub", arquivos)
    caminho = os.path.join(PASTA_DADOS, arq)
elif upload:
    caminho = upload
else:
    st.stop()

# =============================
# LEITURA
# =============================
df = pd.read_excel(caminho, header=1)

# =============================
# LIMPEZA
# =============================
df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.str.contains("Unnamed", na=False)]
df.dropna(how="all", inplace=True)

# =============================
# CLIENTE (COLUNA P)
# =============================
col_cliente = df.columns[15]

df["cliente"] = (
    df[col_cliente]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

df["cliente"] = df["cliente"].str.split(" - ", n=1).str[-1]

# =============================
# COLUNAS
# =============================
def achar(nomes):
    for nome in nomes:
        for col in df.columns:
            if nome.lower() in col.lower():
                return col
    return None

col_unidade = achar(["receptora","filial"])
col_cidade = achar(["cidade"])
col_dias = achar(["dias"])
col_local = achar(["localizacao","localização","local atual","status"])
col_emissao = achar(["emissao"])

# =============================
# TRATAMENTO
# =============================
df["dias"] = pd.to_numeric(df[col_dias], errors="coerce").fillna(0) if col_dias else 0

# Última ocorrência
col_desc_ultima = achar(["descricao da ultima ocorrencia","descrição da ultima ocorrência"])
if col_desc_ultima:
    df["ultima_ocorrencia_desc"] = df[col_desc_ultima].astype(str)
else:
    df["ultima_ocorrencia_desc"] = ""

# Entregue
if col_local:
    df["entregue"] = df[col_local].astype(str).str.contains(
        "CTRC ENTREGUE|BAIXADO", case=False, na=False
    )
else:
    df["entregue"] = False

df["atraso"] = (df["dias"] > 0) & (~df["entregue"])

def status(d,a):
    if not a: return "No Prazo"
    elif d <= 3: return "Leve"
    elif d <= 7: return "Atrasado"
    else: return "Extremo"

df["status"] = df.apply(lambda x: status(x["dias"], x["atraso"]), axis=1)

# =============================
# FILTROS
# =============================
st.sidebar.header("🎯 Filtros")

f_cliente = st.sidebar.multiselect("Cliente", sorted(df["cliente"].dropna().unique()))
if f_cliente:
    df = df[df["cliente"].isin(f_cliente)]

if col_unidade:
    f_unidade = st.sidebar.multiselect(
        "Unidade Receptora",
        sorted(df[col_unidade].dropna().astype(str).unique())
    )
    if f_unidade:
        df = df[df[col_unidade].astype(str).isin(f_unidade)]

if col_cidade:
    f_cidade = st.sidebar.multiselect(
        "Cidade",
        sorted(df[col_cidade].dropna().astype(str).unique())
    )
    if f_cidade:
        df = df[df[col_cidade].astype(str).isin(f_cidade)]

# =============================
# IA PREVISÃO
# =============================
df_modelo = df.copy()
df_modelo["target"] = (df_modelo["dias"] >= 5).astype(int)

X = df_modelo[["dias"]]
y = df_modelo["target"]

if len(df_modelo) > 10 and df_modelo["target"].nunique() > 1:
    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X, y)
    df["prob"] = modelo.predict_proba(X)[:,1]
else:
    df["prob"] = 0

df["prob"] = df["prob"].fillna(0).clip(0,1)

# =============================
# PRIORIDADE
# =============================
df["score"] = df["dias"]*2 + df["prob"]*10 + (~df["entregue"])*5
df["prioridade"] = df["score"].apply(
    lambda s: "CRÍTICA" if s>=20 else "ALTA" if s>=12 else "MÉDIA" if s>=6 else "BAIXA"
)

# =============================
# KPI
# =============================
c1,c2,c3 = st.columns(3)
c1.metric("Total", len(df))
c2.metric("Atrasos", int(df["atraso"].sum()))

sla_valor = ((~df['atraso']).sum()/len(df)*100) if len(df) > 0 else 0
c3.metric("SLA", f"{sla_valor:.1f}%")

st.divider()

# =============================
# DASHBOARD FILIAL
# =============================
st.subheader("🏢 Performance por Filial")

if col_unidade:
    base = df.copy()
    base[col_unidade] = base[col_unidade].astype(str)

    sla = (
        base.groupby(col_unidade)
        .agg(total=("atraso","count"), atrasos=("atraso","sum"))
        .reset_index()
    )

    sla["SLA"] = (1 - sla["atrasos"]/sla["total"]) * 100

    def faixa(s):
        if s < 70: return "Crítico"
        elif s < 85: return "Ruim"
        elif s < 95: return "Atenção"
        else: return "Bom"

    sla["faixa"] = sla["SLA"].apply(faixa)

    cores = {
        "Crítico": "#E74C3C",
        "Ruim": "#E67E22",
        "Atenção": "#F1C40F",
        "Bom": "#2ECC71"
    }

    grafico = alt.Chart(sla).mark_bar().encode(
        x=alt.X(f"{col_unidade}:N"),
        y=alt.Y("SLA:Q"),
        color=alt.Color(
            "faixa:N",
            scale=alt.Scale(domain=list(cores.keys()), range=list(cores.values())),
            legend=None
        )
    )

    st.altair_chart(grafico, width='stretch')

# =============================
# TENDÊNCIA
# =============================
st.subheader("📈 Tendência de Atrasos")

if col_emissao:
    df["data"] = pd.to_datetime(df[col_emissao], errors="coerce")

    tendencia = (
        df[df["atraso"]]
        .groupby(df["data"].dt.date)
        .size()
        .reset_index(name="qtd")
    )

    grafico = alt.Chart(tendencia).mark_line(point=True).encode(
        x="data:T",
        y="qtd:Q"
    ).mark_line(color="#E74C3C")

    st.altair_chart(grafico, width='stretch')

# =============================
# RANKING
# =============================
st.subheader("🏆 Ranking de Remetentes")

ranking = (
    df[df["atraso"]]
    .groupby("cliente")
    .size()
    .reset_index(name="Atrasos")
    .sort_values("Atrasos", ascending=False)
    .head(10)
)

grafico = alt.Chart(ranking).mark_bar().encode(
    x="Atrasos:Q",
    y=alt.Y("cliente:N", sort="-x"),
    color=alt.value("#E74C3C")
)

st.altair_chart(grafico, width='stretch')

# =============================
# TABELA DE ACOMPANHAMENTO
# =============================
st.subheader("📋 Acompanhamento")

if col_local:
    df_acomp = df[
        ~df[col_local].astype(str).str.contains(
            "CTRC ENTREGUE|BAIXADO", case=False, na=False
        )
    ].copy()
else:
    df_acomp = df.copy()

df_acomp = df_acomp.sort_values(["score", "dias"], ascending=[False, False])

tabela = pd.DataFrame()
tabela["Remetente"] = df_acomp["cliente"]

col_nota = achar(["numero da nota fiscal","nota fiscal"])
if col_nota:
    tabela["Número da Nota Fiscal"] = df_acomp[col_nota].apply(
        lambda x: str(int(float(x))) if pd.notnull(x) and str(x).replace('.','',1).isdigit() else ""
    )

tabela["Dias de Atraso"] = df_acomp["dias"].astype(int)
tabela["Local Atual"] = df_acomp[col_local] if col_local else ""
tabela["Última Ocorrência"] = df_acomp["ultima_ocorrencia_desc"]

if col_unidade:
    tabela["Unidade Receptora"] = df_acomp[col_unidade]

col_setor = achar(["setor destino","destino","departamento"])
if col_setor:
    tabela["Setor de Destino"] = df_acomp[col_setor]

tabela["Prioridade"] = df_acomp["prioridade"]
tabela["Probabilidade (%)"] = (df_acomp["prob"]*100).round(1)

def destacar_dias(s):
    cores = []
    for i, row in tabela.iterrows():
        if row["Prioridade"] == "CRÍTICA":
            cores.append("background-color: #E74C3C; color: white")
        elif row["Prioridade"] == "ALTA":
            cores.append("background-color: #E67E22; color: white")
        elif row["Prioridade"] == "MÉDIA":
            cores.append("background-color: #F1C40F; color: black")
        elif row["Prioridade"] == "BAIXA":
            cores.append("background-color: #2ECC71; color: white")
        else:
            cores.append("")
    return pd.Series(cores, index=s.index)

tabela_colorida = tabela.style.apply(destacar_dias, subset=["Dias de Atraso"])
st.dataframe(tabela_colorida, width='stretch')

# =============================
# DOWNLOAD
# =============================
st.download_button(
    "⬇ Baixar relatório",
    tabela.to_csv(index=False, sep=";", encoding="utf-8-sig"),
    "relatorio.csv"
)

st.caption(f"Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
