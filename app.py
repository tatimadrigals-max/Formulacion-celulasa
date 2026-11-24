
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
    font-size: 1.55rem !important;
    line-height: 1.3 !important;
    font-weight: 750 !important;
    letter-spacing: 0.1px;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #D8E9F8 !important;
}

/* Carga de archivos */
[data-testid="stFileUploader"] section {
    background-color: rgba(121, 184, 232, 0.12);
    border: 1px dashed #79B8E8;
}

/* Encabezados: respetan el color de texto del tema */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}

/* Métricas */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    padding: 17px;
    border-radius: 12px;
    border: 1px solid var(--panel-border);
}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
}

/* Tarjetas informativas */
.info-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 18px;
    min-height: 112px;
}
.info-card-title {
    color: var(--text-color);
    font-weight: 700;
    font-size: 15px;
}
.info-card-text {
    color: var(--text-color);
    opacity: 0.82;
    font-size: 13px;
}

/* Fuente de datos */
.source-card {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border-left: 4px solid #79B8E8;
    border-radius: 8px;
    padding: 14px 18px;
}
.source-card a {
    color: var(--text-color) !important;
    text-decoration: underline;
}

/* Pestañas */
.stTabs [data-baseweb="tab"] {
    font-weight: 650;
    color: var(--text-color);
}
.stTabs [aria-selected="true"] {
    color: var(--text-color) !important;
    border-bottom-color: #79B8E8 !important;
}

/* Tablas */
[data-testid="stDataFrame"] {
    border: 1px solid var(--panel-border);
    border-radius: 10px;
}

/* Botones */
.stButton > button {
    background-color: #244A73;
    color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #79B8E8;
}
.stButton > button:hover {
    background-color: #79B8E8;
    color: #17365D;
}

/* Avisos y contenedores: conservar contraste del tema */
[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. FUNCIONES DE PROCESAMIENTO
# =========================================================

def leer_archivo(archivo, hoja=None):

    extension = Path(archivo.name).suffix.lower()
    archivo.seek(0)

    if extension == ".csv":

        try:
            return pd.read_csv(
                archivo,
                sep=None,
                engine="python"
            )

        except UnicodeDecodeError:

            archivo.seek(0)

            return pd.read_csv(
                archivo,
                sep=None,
                engine="python",
                encoding="latin-1"
            )

    elif extension in [".xlsx", ".xls"]:

        return pd.read_excel(
            archivo,
            sheet_name=hoja
        )

    raise ValueError("Formato no compatible.")


def limpiar_datos(df):

    df = df.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Evitar nombres de columnas duplicados
    columnas_nuevas = []
    contador = {}

    for col in df.columns:

        if col not in contador:
            contador[col] = 0
            columnas_nuevas.append(col)

        else:
            contador[col] += 1
            columnas_nuevas.append(
                f"{col}_{contador[col]}"
            )

    df.columns = columnas_nuevas

    valores_nulos = [
        "N/A", "NA", "n/a", "na",
        "NULL", "null", "None", "none",
        "-", ""
    ]

    for col in df.columns:

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):

            df[col] = df[col].replace(
                valores_nulos,
                np.nan
            )

    # Conversión de columnas numéricas
    for col in df.columns:

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):

            original = df[col]
            texto = original.astype("string").str.strip()

            numerico = pd.to_numeric(
                texto.str.replace(",", ".", regex=False),
                errors="coerce"
            )

            cantidad_validos = original.notna().sum()

            if cantidad_validos > 0:

                proporcion = (
                    numerico.notna().sum()
                    / cantidad_validos
                )

                if proporcion >= 0.80:
                    df[col] = numerico

    # Unificar categorías
    for col in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        df[col] = df[col].replace({
            "Medium yeild": "Medium Yeild"
        })

    return df


def preparar_dataframe_visual(df):

    visual = df.copy()

    for col in visual.columns:

        if (
            pd.api.types.is_object_dtype(visual[col])
            or pd.api.types.is_string_dtype(visual[col])
        ):

            visual[col] = visual[col].map(
                lambda valor:
                None if pd.isna(valor)
                else str(valor)
            )

    return visual


def letra_excel(numero):

    resultado = ""

    while numero > 0:

        numero, residuo = divmod(numero - 1, 26)

        resultado = (
            chr(65 + residuo) + resultado
        )

    return resultado


def identificar_tipo(serie):

    if pd.api.types.is_bool_dtype(serie):
        return "Booleano"

    if pd.api.types.is_numeric_dtype(serie):
        return "Numérico"

    if pd.api.types.is_datetime64_any_dtype(serie):
        return "Fecha/hora"

    valores = serie.dropna()

    if len(valores) > 0:

        muestra = valores.astype(str).head(100)

        fechas = pd.to_datetime(
            muestra,
            errors="coerce",
            format="mixed"
        )

        proporcion_fechas = fechas.notna().mean()

        tiene_formato_fecha = muestra.str.contains(
            r"[-/:T]",
            regex=True
        ).mean()

        if (
            proporcion_fechas >= 0.90
            and tiene_formato_fecha >= 0.80
        ):
            return "Fecha/hora"

    return "Categórico"


def crear_tabla_eda(df):

    filas = []

    for posicion, columna in enumerate(
        df.columns[:MAX_COLUMNAS],
        start=1
    ):

        serie = df[columna]
        tipo = identificar_tipo(serie)

        fila = {
            "Variable": columna,
            "Tipo de dato": tipo,
            "Columna Excel": letra_excel(posicion),
            "Cantidad de registros": len(df),
            "Valores nulos": int(serie.isna().sum()),
            "Valores únicos": int(serie.nunique(dropna=True)),
            "Mínimo": np.nan,
            "Máximo": np.nan,
            "Media": np.nan,
            "Desviación estándar": np.nan,
            "Mediana": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
            "Niveles categóricos": ""
        }

        if tipo == "Numérico":

            valores = pd.to_numeric(
                serie,
                errors="coerce"
            )

            fila["Mínimo"] = valores.min()
            fila["Máximo"] = valores.max()
            fila["Media"] = valores.mean()
            fila["Desviación estándar"] = valores.std()
            fila["Mediana"] = valores.median()
            fila["Q1"] = valores.quantile(0.25)
            fila["Q3"] = valores.quantile(0.75)

        elif tipo in ["Categórico", "Booleano"]:

            categorias = serie.dropna().unique()

            if len(categorias) <= 5:

                fila["Niveles categóricos"] = ", ".join(
                    sorted(str(x) for x in categorias)
                )

            else:

                fila["Niveles categóricos"] = (
                    f"Más de 5 niveles ({len(categorias)})"
                )

        filas.append(fila)

    return pd.DataFrame(filas)


def exportar_excel(df, tabla_eda):

    salida = BytesIO()

    with pd.ExcelWriter(
        salida,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Datos",
            index=False
        )

        tabla_eda.to_excel(
            writer,
            sheet_name="EDA",
            index=False
        )

    salida.seek(0)

    return salida.getvalue()


def aplicar_estilo_figura(figura):

    # plotly_dark permite que los gráficos mantengan contraste en modo oscuro.
    # El tema de Streamlit se detecta desde la configuración del usuario.
    tema_streamlit = st.get_option("theme.base")
    plantilla = "plotly_dark" if tema_streamlit == "dark" else "plotly_white"

    color_texto = "#E6EDF5" if plantilla == "plotly_dark" else "#29483D"
    color_fondo = "#0E1117" if plantilla == "plotly_dark" else "#FFFFFF"
    color_trazado = "#161B22" if plantilla == "plotly_dark" else "#FFFFFF"
    color_rejilla = "#344255" if plantilla == "plotly_dark" else "#E6EDF5"

    figura.update_layout(
        template=plantilla,
        title_x=0.02,
        font=dict(color=color_texto),
        paper_bgcolor=color_fondo,
        plot_bgcolor=color_trazado,
        margin=dict(l=25, r=25, t=65, b=35),
        xaxis=dict(gridcolor=color_rejilla, zerolinecolor=color_rejilla),
        yaxis=dict(gridcolor=color_rejilla, zerolinecolor=color_rejilla)
    )

    return figura


# =========================================================
# 4. FUNCIONES DE VISUALIZACIÓN
# =========================================================

def crear_histograma(df, variable, bins):

    datos = df[[variable]].copy()

    datos[variable] = pd.to_numeric(
        datos[variable],
        errors="coerce"
    )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if datos.empty:

        st.warning(
            "No hay valores numéricos disponibles."
        )

        return

    figura = px.histogram(
        datos,
        x=variable,
        nbins=bins,
        title=f"Distribución de {variable}",
        labels={variable: variable},
        color_discrete_sequence=["#39856B"]
    )

    figura.update_layout(
        xaxis_title=variable,
        yaxis_title="Frecuencia",
        bargap=0.08
    )

    aplicar_estilo_figura(figura)

    st.plotly_chart(
        figura,
        width="stretch",
        key=f"hist_{variable}"
    )

    st.caption(
        "El histograma permite observar la frecuencia "
        "y la forma de distribución de la variable."
    )


def crear_boxplot(df, variable_grupo):

    if VARIABLE_RESPUESTA not in df.columns:

        st.error(
            f"No se encontró {VARIABLE_RESPUESTA}."
        )

        return

    datos = df[
        [VARIABLE_RESPUESTA, variable_grupo]
    ].copy()

    datos[VARIABLE_RESPUESTA] = pd.to_numeric(
        datos[VARIABLE_RESPUESTA],
        errors="coerce"
    )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan
    )

    datos = datos.dropna(
        subset=[VARIABLE_RESPUESTA, variable_grupo]
    )

    if datos.empty:

        st.warning(
            "No hay datos válidos para generar el boxplot."
        )

        return

    if pd.api.types.is_numeric_dtype(
        datos[variable_grupo]
    ):

        cantidad_unicos = datos[
            variable_grupo
        ].nunique()

        if cantidad_unicos < 2:

            st.warning(
                "La variable no tiene suficientes valores diferentes."
            )

            return

        if cantidad_unicos > 8:

            cantidad_grupos = min(5, cantidad_unicos)

            try:

                datos["Grupo"] = pd.qcut(
                    datos[variable_grupo],
                    q=cantidad_grupos,
                    duplicates="drop"
                )

            except ValueError:

                st.warning(
                    "No fue posible crear intervalos."
                )

                return

            datos["Grupo"] = datos["Grupo"].astype(str)

            titulo_grupo = f"{variable_grupo} (intervalos)"

        else:

            datos["Grupo"] = datos[
                variable_grupo
            ].astype(str)

            titulo_grupo = variable_grupo

    else:

        datos["Grupo"] = datos[
            variable_grupo
        ].astype(str)

        titulo_grupo = variable_grupo

    figura = px.box(
        datos,
        x="Grupo",
        y=VARIABLE_RESPUESTA,
        points="outliers",
        color="Grupo",
        title=f"{VARIABLE_RESPUESTA} según {titulo_grupo}",
        labels={
            "Grupo": titulo_grupo,
            VARIABLE_RESPUESTA: VARIABLE_RESPUESTA
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    figura.update_layout(
        showlegend=False,
        xaxis_title=titulo_grupo,
        yaxis_title=VARIABLE_RESPUESTA
    )

    aplicar_estilo_figura(figura)

    st.plotly_chart(
        figura,
        width="stretch",
        key=f"box_{variable_grupo}"
    )

    st.caption(
        "El boxplot permite comparar la mediana, dispersión "
        "y posibles valores atípicos de FPU/ml entre grupos."
    )


def crear_dispersion(df, variable_x):

    datos = df[
        [variable_x, VARIABLE_RESPUESTA]
    ].copy()

    if variable_x == "Contamination (Y/N)":

        datos[variable_x] = (
            datos[variable_x]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "Y": 1,
                "YES": 1,
                "N": 0,
                "NO": 0
            })
        )

    else:

        datos[variable_x] = pd.to_numeric(
            datos[variable_x],
            errors="coerce"
        )

    datos[VARIABLE_RESPUESTA] = pd.to_numeric(
        datos[VARIABLE_RESPUESTA],
        errors="coerce"
    )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(datos) < 2:

        st.warning(
            "No hay suficientes datos para generar el gráfico."
        )

        return

    fig = px.scatter(
        datos,
        x=variable_x,
        y=VARIABLE_RESPUESTA,
        title=f"{VARIABLE_RESPUESTA} según {variable_x}",
        opacity=0.65,
        labels={
            variable_x: variable_x,
            VARIABLE_RESPUESTA: VARIABLE_RESPUESTA
        }
    )

    fig.update_traces(
        marker=dict(size=8)
    )

    if datos[variable_x].nunique() > 1:

        coeficientes = np.polyfit(
            datos[variable_x],
            datos[VARIABLE_RESPUESTA],
            1
        )

        x_linea = np.linspace(
            datos[variable_x].min(),
            datos[variable_x].max(),
            100
        )

        y_linea = np.polyval(
            coeficientes,
            x_linea
        )

        fig.add_scatter(
            x=x_linea,
            y=y_linea,
            mode="lines",
            name="Tendencia lineal",
            line=dict(width=3)
        )

    fig.update_layout(
        height=520,
        xaxis_title=variable_x,
        yaxis_title=VARIABLE_RESPUESTA,
        hovermode="closest"
    )

    aplicar_estilo_figura(fig)

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"dispersion_{variable_x}"
    )

    st.markdown("**¿Cómo interpretar el gráfico?**")
    st.markdown(
        "- **Cada punto** representa un registro o corrida de fermentación. "
        "El eje X muestra la variable seleccionada y el eje Y la actividad "
        "máxima de celulasa (FPU/mL).\n"
        "- **La línea de tendencia** resume la dirección general de la "
        "relación entre ambas variables: ascendente, descendente o sin "
        "una tendencia lineal clara.\n"
        "- **La separación de los puntos** permite observar qué tan "
        "variables son los resultados para valores similares de la "
        "variable seleccionada."
    )
    st.caption(
        "La tendencia es descriptiva: no demuestra causalidad ni "
        "significancia estadística. Si los puntos están muy dispersos, "
        "la variable seleccionada por sí sola puede no explicar bien "
        "las diferencias en FPU/mL."
    )


# =========================================================
# 5. PREPARACIÓN DEL MODELO
# =========================================================

def preparar_datos_modelo(df):

    variables = [
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

    variables = [
        v for v in variables
        if v in df.columns
    ]

    if VARIABLE_RESPUESTA not in df.columns:
        return None, None, None

    if len(variables) < 2:
        return None, None, None

    datos = df[
        variables + [VARIABLE_RESPUESTA]
    ].copy()

    if "Contamination (Y/N)" in variables:

        datos["Contamination (Y/N)"] = (
            datos["Contamination (Y/N)"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "Y": 1,
                "YES": 1,
                "N": 0,
                "NO": 0
            })
        )

    for variable in variables:

        if variable != "Contamination (Y/N)":

            datos[variable] = pd.to_numeric(
                datos[variable],
                errors="coerce"
            )

    datos[VARIABLE_RESPUESTA] = pd.to_numeric(
        datos[VARIABLE_RESPUESTA],
        errors="coerce"
    )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan
    )

    datos = datos.dropna(
        subset=[VARIABLE_RESPUESTA]
    )

    # Eliminar variables vacías o constantes
    variables = [
        v for v in variables
        if datos[v].notna().any()
        and datos[v].nunique(dropna=True) > 1
    ]

    if len(variables) < 2:
        return None, None, None

    X = datos[variables]
    y = datos[VARIABLE_RESPUESTA]

    # Imputación de datos faltantes
    imputador = SimpleImputer(
        strategy="median"
    )

    X_modelo = pd.DataFrame(
        imputador.fit_transform(X),
        columns=variables,
        index=X.index
    )

    return X_modelo, y, variables


# =========================================================
# 6. ENCABEZADO
# =========================================================

st.components.v1.html("""
<div style="
    background: linear-gradient(120deg, #17365D, #244A73);
    padding: 32px;
    border-radius: 16px;
    font-family: Arial, sans-serif;
">
    <div style="color: #79B8E8; font-size: 12px;
                font-weight: bold; letter-spacing: 2px;">
        Biotecnología · Análisis de Datos
    </div>

    <h1 style="color: white; font-size: 30px;">
        Análisis y optimización de procesos fermentativos
    </h1>

    <p style="color: #D8E9F8; font-size: 16px;">
        Exploración de variables operativas y su relación
        con la producción de celulasa.
    </p>
</div>
""", height=190)


# =========================================================
# 7. BARRA LATERAL
# =========================================================

with st.sidebar:

    st.markdown("## 🧪 Análisis de fermentación")

    st.caption(
        "Panel de exploración de datos"
    )

    st.markdown("---")

    st.markdown("### Cargar conjunto de datos")

    archivo = st.file_uploader(
        "Seleccioná un archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Se admiten archivos CSV, XLSX y XLS."
    )

    st.markdown("---")

    st.markdown("### Acerca del dashboard")

    st.caption(
        "Herramienta de análisis exploratorio para estudiar "
        "variables operativas de un proceso fermentativo "
        "y su relación con la actividad celulasa."
    )

    st.markdown("---")

    st.caption(
        "Fuente de datos: Kaggle · Fermentation Optimization"
    )


# =========================================================
# 8. DESCRIPCIÓN DEL PROYECTO
# =========================================================

st.markdown("### Descripción del proyecto")

st.markdown("""
Este dashboard permite explorar y analizar datos de un proceso de
fermentación a escala de biorreactor de 5 litros, considerando variables
como temperatura, pH, oxígeno disuelto, velocidad de agitación, tamaño
del inóculo y tiempo de cultivo.

Mediante herramientas de visualización y análisis exploratorio, se busca
identificar patrones, tendencias y posibles relaciones entre las
condiciones operativas y la actividad celulasa obtenida (FPU/mL).
""")

st.markdown(
    '<div class="source-card"><b>Fuente del conjunto de datos:</b> '
    '<a href="https://www.kaggle.com/datasets/adityanarayankonwar/fermentation-optimization" '
    'target="_blank">Fermentation Optimization · Kaggle</a></div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# 9. CARGA Y PROCESAMIENTO
# =========================================================

if archivo is None:

    st.info(
        "Para comenzar, cargá el archivo de datos desde el panel lateral. "
        "Una vez cargado, se habilitarán las secciones de análisis."
    )

    st.markdown("### ¿Qué podés explorar?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                📋 Calidad y estructura
            </div>
            <div class="info-card-text">
                Revisá registros, variables, tipos de datos y valores faltantes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                📊 Distribuciones
            </div>
            <div class="info-card-text">
                Explorá frecuencia, dispersión y variabilidad.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                🔬 Relaciones
            </div>
            <div class="info-card-text">
                Compará las condiciones del proceso con la actividad celulasa.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

else:

    try:

        extension = Path(archivo.name).suffix.lower()

        if extension in [".xlsx", ".xls"]:

            df_original = leer_archivo(
                archivo,
                hoja=0
            )

        else:

            df_original = leer_archivo(archivo)

        if df_original.empty:

            st.error(
                "El archivo no contiene registros."
            )

            st.stop()

        if len(df_original.columns) > MAX_COLUMNAS:

            st.warning(
                f"El archivo contiene {len(df_original.columns)} columnas. "
                f"Se utilizarán las primeras {MAX_COLUMNAS}."
            )

            df_original = df_original.iloc[
                :, :MAX_COLUMNAS
            ]

        df = limpiar_datos(df_original)

        st.success(
            "Archivo cargado correctamente."
        )

    except Exception as e:

        st.error(
            f"No se pudo procesar el archivo: {e}"
        )

        st.stop()


# =========================================================
# 10. INDICADORES GENERALES
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Registros",
    f"{len(df):,}"
)

col2.metric(
    "Variables",
    len(df.columns)
)

col3.metric(
    "Valores nulos",
    f"{int(df.isna().sum().sum()):,}"
)

if VARIABLE_RESPUESTA not in df.columns:

    st.warning(
        f"No se encontró la variable '{VARIABLE_RESPUESTA}'. "
        "Algunos análisis no estarán disponibles."
    )


# =========================================================
# 11. PESTAÑAS
# =========================================================

tab_inicio, tab_datos, tab_dist, tab_rel, tab_corr, tab_modelo = st.tabs([
    "🏠 Inicio",
    "📋 Datos",
    "📊 Distribuciones",
    "🔬 Relaciones",
    "📉 Correlaciones",
    "🤖 Modelo predictivo"
])


# =========================================================
# PESTAÑA 1: INICIO
# =========================================================

with tab_inicio:

    st.markdown("### Objetivo del análisis")

    st.write("""
    Explorar las variables de operación de un proceso fermentativo
    y su relación con la actividad celulasa, mediante herramientas
    estadísticas y gráficas que faciliten la interpretación de los datos.
    """)

    st.markdown("### Variables de interés")

    variables_info = [

        ("🌡️ Temperatura y pH",
         "Condiciones fisicoquímicas del proceso."),

        ("💨 Oxígeno disuelto",
         "Disponibilidad de oxígeno durante la fermentación."),

        ("⚙️ Agitación",
         "Velocidad máxima de agitación del biorreactor."),

        ("🧫 Tamaño del inóculo",
         "Volumen de inóculo utilizado."),

        ("⏱️ Tiempo de cultivo",
         "Duración total del proceso fermentativo."),

        ("🧪 Actividad celulasa",
         "Producción máxima expresada en FPU/mL.")
    ]

    columnas = st.columns(3)

    for i, (titulo, descripcion) in enumerate(
        variables_info
    ):

        with columnas[i % 3]:

            st.markdown(
                f'<div class="info-card">'
                f'<div class="info-card-title">{titulo}</div>'
                f'<div class="info-card-text">{descripcion}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        if i in [2, 5]:
            st.write("")

    st.markdown("### Fuente de información")

    st.write("""
    El conjunto de datos utilizado corresponde a **Fermentation
    Optimization**, disponible en Kaggle. La información contiene
    registros de corridas de fermentación y variables relacionadas
    con las condiciones de operación y la actividad celulasa.
    """)

    st.markdown(
        "[Consultar conjunto de datos en Kaggle]"
        "(https://www.kaggle.com/datasets/adityanarayankonwar/fermentation-optimization)"
    )


# =========================================================
# PESTAÑA 2: DATOS
# =========================================================

with tab_datos:

    st.markdown("### Exploración del conjunto de datos")

    st.markdown(
        "Revisión de la estructura, calidad y contenido del archivo cargado."
    )

    tabla_eda = crear_tabla_eda(df)

    st.markdown("#### Resumen de variables")

    st.dataframe(
        preparar_dataframe_visual(tabla_eda),
        width="stretch",
        hide_index=True
    )

    st.markdown("#### Datos cargados")

    st.dataframe(
        preparar_dataframe_visual(df),
        width="stretch",
        hide_index=True
    )

    archivo_excel = exportar_excel(
        df,
        tabla_eda
    )

    st.download_button(
        label="⬇️ Descargar Excel con datos y resumen EDA",
        data=archivo_excel,
        file_name="reporte_fermentacion.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# =========================================================
# PESTAÑA 3: DISTRIBUCIONES
# =========================================================

with tab_dist:

    st.markdown("### Distribución de las variables")

    st.markdown(
        "Analizá la frecuencia, variabilidad y distribución "
        "de las variables del proceso."
    )

    # HISTOGRAMA

    st.markdown("#### Histograma")

    variables_hist = [

        variable for variable in VARIABLES_NUMERICAS

        if variable in df.columns

        and pd.to_numeric(
            df[variable],
            errors="coerce"
        ).notna().any()

    ]

    if not variables_hist:

        st.warning(
            "No se encontraron variables numéricas disponibles."
        )

    else:

        variable_hist = st.selectbox(
            "Seleccioná la variable para el histograma",
            variables_hist,
            key="variable_histograma"
        )

        bins = st.slider(
            "Número de intervalos",
            min_value=5,
            max_value=40,
            value=15,
            key="cantidad_bins"
        )

        crear_histograma(
            df,
            variable_hist,
            bins
        )

    st.divider()

    # BOXPLOT

    st.markdown("#### Diagrama de caja (boxplot)")

    st.write(
        "Compará la distribución de la actividad celulasa "
        "entre diferentes condiciones del proceso."
    )

    if VARIABLE_RESPUESTA not in df.columns:

        st.warning(
            f"Se necesita la columna '{VARIABLE_RESPUESTA}' "
            "para construir los boxplots."
        )

    else:

        variables_box_disponibles = [

            variable for variable in VARIABLES_BOX

            if variable in df.columns

        ]

        if not variables_box_disponibles:

            st.warning(
                "No se encontraron variables para comparar."
            )

        else:

            variable_box = st.selectbox(
                "Seleccioná la variable de comparación",
                variables_box_disponibles,
                key="variable_boxplot"
            )

            crear_boxplot(
                df,
                variable_box
            )


# =========================================================
# PESTAÑA 4: RELACIONES
# =========================================================

with tab_rel:

    st.markdown("### Relación entre variables del proceso")

    st.markdown(
        "Explorá cómo se distribuye la actividad celulasa "
        "en función de las condiciones operativas seleccionadas."
    )

    if VARIABLE_RESPUESTA not in df.columns:

        st.error(
            f"No se encontró '{VARIABLE_RESPUESTA}'. "
            "No es posible generar los gráficos de dispersión."
        )

    else:

        variables_disp_disponibles = [

            variable for variable in VARIABLES_DISPERSION

            if variable in df.columns

        ]

        if not variables_disp_disponibles:

            st.warning(
                "No se encontraron variables para generar los gráficos."
            )

        else:

            variable_x = st.selectbox(
                "Seleccioná la variable del eje X",
                options=variables_disp_disponibles,
                key="seleccion_variable_dispersion"
            )

            crear_dispersion(
                df,
                variable_x
            )


# =========================================================
# PESTAÑA 5: CORRELACIONES
# =========================================================

with tab_corr:

    st.markdown("### Matriz de correlación")

    st.markdown(
        "Explorá la asociación lineal entre las variables "
        "numéricas del proceso fermentativo."
    )

    numericas = df.select_dtypes(
        include=np.number
    )

    numericas = numericas.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if numericas.shape[1] < 2:

        st.warning(
            "Se necesitan al menos dos variables numéricas "
            "para calcular la matriz de correlación."
        )

    else:

        correlaciones = numericas.corr(
            method="pearson"
        )

        figura_corr = px.imshow(
            correlaciones,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[
                [0.00, "#2166AC"],
                [0.25, "#67A9CF"],
                [0.50, "#F7F7F7"],
                [0.75, "#F4A582"],
                [1.00, "#B35806"]
            ],
            zmin=-1,
            zmax=1,
            color_continuous_midpoint=0,
            title="Correlación entre variables numéricas"
        )

        figura_corr.update_layout(
            height=650,
            coloraxis_colorbar=dict(
                title="Coeficiente<br>de Pearson",
                tickvals=[-1, -0.5, 0, 0.5, 1]
            )
        )

        aplicar_estilo_figura(
            figura_corr
        )

        st.plotly_chart(
            figura_corr,
            width="stretch",
            key="matriz_correlacion"
        )

        st.markdown("#### ¿Cómo interpretar la matriz?")

        st.markdown("""
        - **Correlación positiva (+1):** cuando una variable aumenta,
          la otra tiende a aumentar.
        - **Correlación negativa (-1):** cuando una variable aumenta,
          la otra tiende a disminuir.
        - **Correlación cercana a cero:** indica una asociación lineal
          débil o no observada entre las variables.
        """)

        st.info(
            "La correlación de Pearson mide asociaciones lineales. "
            "Un valor cercano a cero no descarta una posible relación "
            "no lineal. Además, una correlación no demuestra causalidad."
        )


# =========================================================
# PESTAÑA 6: MODELO PREDICTIVO
# =========================================================

with tab_modelo:

    st.markdown("### Modelo de aprendizaje automático")

    st.markdown("""
    Se utiliza el algoritmo **Random Forest Regressor** para explorar
    la relación entre las variables operativas y la actividad celulasa
    (FPU/mL).
    """)

    # EXPLICACIÓN DEL MODELO

    with st.expander("¿Cómo funciona Random Forest?"):

        st.markdown("""
        Random Forest es un algoritmo de aprendizaje automático
        supervisado que combina múltiples árboles de decisión
        para realizar predicciones.

        En este análisis:

        - **Variables predictoras (X):** temperatura, pH, oxígeno
          disuelto, tamaño del inóculo, agitación, alimentación
          de lactosa, duración del cultivo y volumen del medio.
        - **Variable respuesta (Y):** actividad máxima de celulasa,
          expresada en FPU/mL.

        El algoritmo construye varios árboles de decisión a partir
        de diferentes muestras de los datos. Posteriormente, combina
        las predicciones de esos árboles para obtener una predicción
        final.

        Su utilidad en este análisis consiste en explorar qué variables
        tienen mayor importancia para las predicciones del modelo
        y cómo se relacionan con el rendimiento estimado.
        """)

    X_modelo, y_modelo, variables_modelo = preparar_datos_modelo(df)

    if X_modelo is None:

        st.warning(
            "No hay suficientes datos o variables válidas "
            "para construir el modelo."
        )

    elif len(X_modelo) < 5:

        st.warning(
            "Se necesitan al menos 5 registros para construir "
            "el modelo exploratorio."
        )

    else:

        modelo_rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2
        )

        modelo_rf.fit(
            X_modelo,
            y_modelo
        )

        st.success(
            f"Modelo entrenado con {len(X_modelo)} registros "
            f"y {len(variables_modelo)} variables predictoras."
        )

        st.warning(
            "El modelo se ajustó con todos los registros disponibles. "
            "Los resultados son exploratorios: no representan una "
            "validación independiente de su capacidad predictiva."
        )

        # =================================================
        # IMPORTANCIA DE VARIABLES
        # =================================================

        st.markdown("#### 1. Importancia de las variables")

        st.write("""
        Este gráfico muestra cuáles variables contribuyen más
        a las divisiones realizadas por los árboles de decisión
        del modelo Random Forest.
        """)

        importancia = pd.DataFrame({

            "Variable": variables_modelo,

            "Importancia": modelo_rf.feature_importances_

        }).sort_values(
            "Importancia",
            ascending=True
        )

        fig_importancia = px.bar(
            importancia,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de las variables en Random Forest",
            labels={
                "Importancia": "Importancia relativa",
                "Variable": "Variable del proceso"
            },
            color_discrete_sequence=["#3978A8"]
        )

        aplicar_estilo_figura(
            fig_importancia
        )

        st.plotly_chart(
            fig_importancia,
            width="stretch",
            key="importancia_random_forest"
        )

        st.caption(
            "Una importancia mayor indica que la variable tuvo "
            "una contribución relativa mayor en las divisiones "
            "de los árboles. Esta medida no indica si su relación "
            "con el FPU/ml es positiva o negativa."
        )

        st.info(
            "La importancia de variables de Random Forest no demuestra "
            "que una condición cause cambios en la producción de celulasa."
        )

        # =================================================
        # SHAP SUMMARY PLOT
        # =================================================

        st.divider()

        st.markdown("#### 2. SHAP Summary Plot")

        st.write("""
        SHAP permite interpretar cómo contribuye cada variable
        a las predicciones del modelo. A diferencia de la importancia
        convencional, permite observar tanto la magnitud como
        la dirección de las contribuciones.
        """)

        try:

            explicador = shap.TreeExplainer(
                modelo_rf
            )

            valores_shap = explicador.shap_values(
                X_modelo
            )

            plt.figure(
                figsize=(10, 6)
            )

            shap.summary_plot(
                valores_shap,
                X_modelo,
                show=False,
                cmap="cividis"
            )

            fig_shap = plt.gcf()

            plt.title(
                "Contribución de las variables a las predicciones de FPU/ml"
            )

            plt.tight_layout()

            st.pyplot(
                fig_shap,
                clear_figure=True
            )

            plt.close("all")

            st.markdown("**¿Cómo interpretar este gráfico?**")

            st.markdown("""
            - **Cada punto:** representa una observación del conjunto de datos.
            - **Eje horizontal:** representa el valor SHAP, es decir,
              la contribución de esa variable a la predicción.
            - **Valores SHAP positivos:** contribuyen a aumentar
              la predicción respecto al valor base del modelo.
            - **Valores SHAP negativos:** contribuyen a disminuir
              la predicción respecto al valor base del modelo.
            - **Color:** representa el valor de la variable,
              desde valores bajos hasta valores altos.
            """)

            st.caption(
                "SHAP explica las predicciones del modelo y no establece "
                "relaciones causales entre las variables y la actividad celulasa."
            )

        except Exception as e:

            st.warning(
                f"No fue posible generar el gráfico SHAP: {e}"
            )

        # =================================================
        # PARTIAL DEPENDENCE PLOT
        # =================================================

        st.divider()

        st.markdown("#### 3. Gráfico de dependencia parcial (PDP)")

        st.write("""
        El gráfico de dependencia parcial permite observar cómo
        cambia la predicción promedio del modelo cuando se modifica
        una variable, manteniendo el promedio de las demás variables
        del conjunto de datos.
        """)

        variables_pdp = [

            v for v in variables_modelo

            if v != "Contamination (Y/N)"

        ]

        if variables_pdp:

            variable_pdp = st.selectbox(
                "Seleccioná la variable del proceso",
                variables_pdp,
                key="variable_pdp"
            )

            try:

                fig_pdp, ax = plt.subplots(
                    figsize=(9, 5)
                )

                PartialDependenceDisplay.from_estimator(
                    modelo_rf,
                    X_modelo,
                    [variable_pdp],
                    ax=ax
                )

                ax.set_title(
                    f"Dependencia parcial de {variable_pdp}"
                )

                ax.set_xlabel(
                    variable_pdp
                )

                ax.set_ylabel(
                    "FPU/ml predicho"
                )

                plt.tight_layout()

                st.pyplot(
                    fig_pdp,
                    clear_figure=True
                )

                plt.close(fig_pdp)

                st.markdown("**¿Cómo interpretar este gráfico?**")

                st.markdown("""
                - **Eje X:** representa los valores de la variable
                  seleccionada.
                - **Eje Y:** representa la predicción promedio de
                  FPU/ml estimada por el modelo.
                - **Tendencia ascendente:** indica que, según el modelo,
                  los valores mayores de esa variable se asocian con
                  predicciones promedio mayores.
                - **Tendencia descendente:** indica que, según el modelo,
                  los valores mayores se asocian con predicciones
                  promedio menores.
                """)

                st.caption(
                    "El PDP muestra una tendencia promedio estimada "
                    "por el modelo. No representa un efecto causal y "
                    "puede ser poco confiable en regiones donde existen "
                    "pocas observaciones o combinaciones de variables "
                    "que no aparecen en los datos originales."
                )

            except Exception as e:

                st.warning(
                    f"No fue posible generar el gráfico PDP: {e}"
                )

        else:

            st.warning(
                "No hay variables disponibles para generar el PDP."
            )

    st.divider()

    st.markdown("#### Consideración sobre el modelo")

    st.write("""
    Los gráficos de importancia, SHAP y dependencia parcial permiten
    explorar el comportamiento del modelo y reconocer patrones en los
    datos. Sin embargo, para determinar su capacidad de predecir
    correctamente la actividad celulasa en nuevos experimentos,
    será necesario realizar una evaluación independiente mediante
    técnicas de validación, como validación cruzada.
    """)

