
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from io import BytesIO
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Análisis de procesos fermentativos",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_COLUMNAS = 50
VARIABLE_RESPUESTA = "Max FPU/ml"

VARIABLES_NUMERICAS = [
    "Avg_temp",
    "Avg_pH",
    "Min_DO %",
    "Inoculum size (mL)",
    "Max_RPM",
    "Total feed of Lactose (mL)",
    "Total culture duration (days)",
    "Total media volume (mL)",
    "Max FPU/ml"
]

VARIABLES_BOX = [
    "Contamination (Y/N)",
    "Avg_pH",
    "Avg_temp",
    "Total culture duration (days)",
    "Inoculum size (mL)",
    "Max_RPM",
    "Min_DO %",
    "Total feed of Lactose (mL)",
    "Total media volume (mL)"
]

VARIABLES_DISPERSION = [
    "Avg_temp",
    "Avg_pH",
    "Min_DO %",
    "Inoculum size (mL)",
    "Max_RPM",
    "Total feed of Lactose (mL)",
    "Total culture duration (days)",
    "Total media volume (mL)",
    "Contamination (Y/N)"
]


# =========================================================
# 2. DISEÑO VISUAL
# =========================================================

st.markdown("""
<style>
/* Colores adaptables al tema claro u oscuro de Streamlit */
:root {
    --panel-bg: #FFFFFF;
    --panel-border: color-mix(in srgb, var(--text-color) 18%, transparent);
    --panel-title: #17365D;
    --panel-text: #56708D;
    --accent: #79B8E8;
    --source-bg: #D8E9F8;
}

/* Fondo y texto principal */
.stApp {
    background-color: var(--background-color, #EFF5FC);
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17365D 0%, #244A73 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] h2 {
