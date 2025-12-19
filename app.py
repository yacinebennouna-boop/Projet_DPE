# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Simulateur DPE - Projet ML",
    page_icon="🏠",
    layout="wide",
)

DATA_PATH = Path("data/df_viz.csv")       # CSV déjà traité pour les dataviz
MODEL_PATH = Path("models/model.joblib")  # pipeline/modèle sérialisé (sklearn/joblib)
# Optionnel: si tu as un scaler/encoder séparé
# PREPROC_PATH = Path("models/preprocess.joblib")

# ----------------------------
# UTILS: chargements en cache
# ----------------------------
@st.cache_data(show_spinner=False)
def load_viz_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    # idéalement: un Pipeline sklearn qui inclut preprocessing + modèle
    return joblib.load(path)

# ----------------------------
# UI: Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    [
        "🏁 Présentation",
        "📊 Dataviz",
        "📈 Résultats d'entraînement",
        "🧮 Simulateur DPE",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Projet ML - Simulation DPE")

# ----------------------------
# PAGE 1: Présentation
# ----------------------------
def page_presentation():
    st.title("🏠 Simulation DPE par Machine Learning")

    st.markdown(
        """
## Contexte
Ici tu présentes le sujet : DPE, enjeux, objectifs.

## Données
- Sources
- Variables (features)
- Target (ex: conso énergie / étiquette)

## Approche ML
- Préprocessing
- Modèles testés
- Métriques
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nb. lignes", "—")
    with col2:
        st.metric("Nb. variables", "—")
    with col3:
        st.metric("Score final", "—")

# ----------------------------
# PAGE 2: Dataviz
# ----------------------------
def page_dataviz():
    st.title("📊 Dataviz")

    # Chargement CSV dataviz
    if not DATA_PATH.exists():
        st.error(f"CSV introuvable : {DATA_PATH}")
        st.stop()

    df = load_viz_data(DATA_PATH)

    st.subheader("Aperçu des données")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Filtres")

    # Exemple de filtres (à adapter à tes colonnes)
    cols = df.columns.tolist()
    col_cat = st.selectbox("Colonne catégorielle (optionnel)", ["(aucune)"] + cols, index=0)
    col_num = st.selectbox("Colonne numérique", cols, index=0)

    df_f = df.copy()

    if col_cat != "(aucune)":
        vals = sorted(df[col_cat].dropna().unique().tolist())
        selected = st.multiselect("Valeurs", vals, default=vals[: min(5, len(vals))])
        if selected:
            df_f = df_f[df_f[col_cat].isin(selected)]

    st.markdown("---")
    st.subheader("Graphiques")

    # 1) Histogramme (Altair)
    try:
        chart_hist = (
            alt.Chart(df_f)
            .mark_bar()
            .encode(
                x=alt.X(f"{col_num}:Q", bin=alt.Bin(maxbins=30)),
                y="count()",
                tooltip=["count()"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_hist, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible de tracer l'histogramme : {e}")

    # 2) Scatter si 2 colonnes numériques
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        xcol = st.selectbox("Scatter X", num_cols, index=0)
        ycol = st.selectbox("Scatter Y", num_cols, index=1)

        try:
            chart_scatter = (
                alt.Chart(df_f)
                .mark_circle(size=40, opacity=0.35)
                .encode(
                    x=alt.X(f"{xcol}:Q"),
                    y=alt.Y(f"{ycol}:Q"),
                    tooltip=[xcol, ycol],
                )
                .properties(height=380)
                .interactive()
            )
            st.altair_chart(chart_scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de tracer le scatter : {e}")

    # 3) Agrégations par catégorie (si une catégorie est choisie)
    if col_cat != "(aucune)" and pd.api.types.is_numeric_dtype(df[col_num]):
        st.markdown("### Moyenne par catégorie")
        try:
            agg = df_f.groupby(col_cat, dropna=False)[col_num].mean().reset_index()
            chart_bar = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X(f"{col_cat}:N", sort="-y"),
                    y=alt.Y(f"{col_num}:Q"),
                    tooltip=[col_cat, col_num],
                )
                .properties(height=360)
            )
            st.altair_chart(chart_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de tracer l'agrégation : {e}")

# ----------------------------
# PAGE 3: Résultats d'entraînement
# ----------------------------
def page_results():
    st.title("📈 Résultats d'entraînement")

    st.markdown(
        """
## Modèles testés
- Baseline
- RandomForest / XGBoost / NN
- Optimisation d'hyperparamètres

## Métriques
- MAE / RMSE / R² (si régression)
- Accuracy / F1 (si classification)

## Analyse d'erreur
- où le modèle se trompe le plus
- biais potentiels
        """
    )

    st.markdown("---")
    st.subheader("Illustrations / Courbes")
    st.info("Ici tu peux ajouter tes figures exportées (PNG) ou des courbes calculées à partir d'un CSV de logs.")

    # Exemple: afficher une image si tu en as
    # st.image("assets/loss_curve.png", caption="Courbe de loss", use_container_width=True)

# ----------------------------
# PAGE 4: Simulateur (Formulaire + Modèle)
# ----------------------------
def page_simulator():
    st.title("🧮 Simulateur DPE")
    st.write("Renseigne les caractéristiques du logement pour obtenir une estimation.")

    if not MODEL_PATH.exists():
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)

    # ---- Définition des valeurs possibles (à adapter à ton dataset) ----
    # Idéalement: tu mets ces listes dans un fichier config (yaml/json) ou tu les derives du training.
    CATS = {
        "type_batiment": ["Maison", "Appartement"],
        "periode_construction": ["< 1948", "1949-1974", "1975-2000", "2001-2012", ">= 2013"],
        "qualite_isolation_murs": ["insuffisante", "moyenne", "bonne", "très bonne"],
        # ...
    }

    # ---- Formulaire ----
    with st.form("dpe_form"):
        st.subheader("Caractéristiques")

        c1, c2, c3 = st.columns(3)

        with c1:
            type_bat = st.selectbox("Type de bâtiment", CATS["type_batiment"])
            periode = st.selectbox("Période de construction", CATS["periode_construction"])

        with c2:
            surface = st.number_input("Surface (m²)", min_value=5.0, max_value=1000.0, value=60.0, step=1.0)
            hauteur = st.number_input("Hauteur sous plafond (m)", min_value=1.8, max_value=4.0, value=2.5, step=0.1)

        with c3:
            iso_murs = st.selectbox("Qualité isolation murs", CATS["qualite_isolation_murs"])
            # Ajoute d'autres champs...

        submitted = st.form_submit_button("Calculer le DPE")

    # ---- Inférence ----
    if submitted:
        # Construire une ligne au format modèle
        # IMPORTANT: les noms de colonnes doivent correspondre à ceux utilisés au training
        X = pd.DataFrame([{
            "type_batiment": type_bat,
            "periode_construction": periode,
            "surface_habitable": surface,
            "hauteur_sous_plafond": hauteur,
            "qualite_isolation_murs": iso_murs,
            # ...
        }])

        try:
            pred = model.predict(X)

            # Si ton modèle renvoie un scalaire
            y = float(np.ravel(pred)[0])

            st.success("Résultat calculé ✅")
            st.metric("Estimation (valeur)", f"{y:,.2f}")

            # Option: transformer en étiquette DPE si tu as un mapping
            # etiquette = to_dpe_label(y)
            # st.metric("Étiquette DPE", etiquette)

            with st.expander("Voir les données envoyées au modèle"):
                st.dataframe(X, use_container_width=True)

        except Exception as e:
            st.error("Erreur lors du calcul. Vérifie la compatibilité features / preprocessing.")
            st.exception(e)

# ----------------------------
# ROUTER
# ----------------------------
if page == "🏁 Présentation":
    page_presentation()
elif page == "📊 Dataviz":
    page_dataviz()
elif page == "📈 Résultats d'entraînement":
    page_results()
elif page == "🧮 Simulateur DPE":
    page_simulator()
