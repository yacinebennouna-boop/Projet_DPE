# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from pathlib import Path
import os
import plotly.express as px
import random

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Simulateur DPE - Projet ML",
    page_icon="🏠",
    layout="wide",
)


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
    # --- Sidebar : L'équipe ---
    with st.sidebar:
        st.markdown("### 👥 L'Équipe")
        st.markdown("""
        * **Yacine Bennouna**
        * **Aymane Karani**
        * **Dylan Nefnaf**
        * **Guillaume Deschamps**
        """)
        st.divider()
        st.info("Projet dans le cadre du cursus Datascientist de Datascientest")

    # --- En-tête Principal ---
    st.title("🏡 Projet DPE : Modélisation & Prédiction")
    
    st.markdown("""
    **Bienvenue sur l'interface de restitution de notre projet.**
    
    Ce projet explore les données du *Diagnostic de Performance Énergétique (DPE)* en France. 
    Il vise à appliquer des modèles de Machine Learning pour prédire l'étiquette énergétique des logements 
    et comprendre les facteurs déterminants de la consommation, à la croisée des enjeux techniques, économiques et scientifiques.
    """)

    st.divider()

    # --- Organisation en Onglets ---
    tab_contexte, tab_objectifs, tab_donnees = st.tabs(["🌍 Contexte & Réforme", "🎯 Objectifs", "💾 Données ADEME"])

    # --- ONGLET 1 : CONTEXTE ---
    with tab_contexte:
        st.header("Contexte Réglementaire et Technique")
        
        st.markdown("""
        Le DPE a subi une **réforme majeure le 1er juillet 2021** pour devenir un outil opposable et plus fiable. 
        Notre projet s'appuie exclusivement sur les données issues de ce nouveau cadre.
        """)

        with st.expander("⚖️ La Réforme DPE 2021 (Ce qui change)", expanded=True):
            st.markdown("""
            * **Méthode de calcul unifiée (3CL) :** Fin de la méthode "sur facture". Le calcul est désormais standardisé pour tous les logements.
            * **Double Seuil :** L'étiquette (A à G) est déterminée par la plus mauvaise note entre la **consommation d'énergie** et les **émissions de gaz à effet de serre (GES)**.
            * **5 Usages :** Prise en compte de l'éclairage et des auxiliaires (en plus du chauffage, de l'eau chaude et du refroidissement).
            """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🛠️ Enjeu Technique")
            st.markdown("""
            La complexité réside dans la reproduction d'une méthode réglementaire stricte par des modèles statistiques.
            Le défi est de gérer des données hétérogènes (matériaux, systèmes) et de prédire une classe définie par des règles physiques.
            """)
        
        with col_b:
            st.markdown("#### 💰 Enjeu Économique")
            st.markdown("""
            Le DPE conditionne la valeur vénale ("Valeur Verte") et locative.
            L'objectif est d'aider à la décision pour prioriser les rénovations et anticiper les interdictions de location (passoires thermiques G+ dès 2023, G en 2025).
            """)

    # --- ONGLET 2 : OBJECTIFS ---
    with tab_objectifs:
        st.header("Objectifs du Projet")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("🤖 **Modélisation ML**")
            st.markdown("""
            * **Classification :** Prédire l'étiquette DPE (7 classes).
            * **Régression :** Estimer la consommation en kWh/m²/an.
            * **Comparaison :** Random Forest vs XGBoost vs Deep Learning.
            """)

        with col2:
            st.warning("📊 **Analyse & Biais**")
            st.markdown("""
            * **Facteurs clés :** Identifier les variables les plus influentes (Feature Importance).
            * **Déséquilibre :** Gérer la sous-représentation des classes extrêmes (A et G).
            * **Simplification :** Tester l'impact de la réduction des variables.
            """)

        with col3:
            st.success("🧠 **Interprétabilité**")
            st.markdown("""
            * **Explicabilité :** Utiliser SHAP pour comprendre les décisions du modèle.
            * **Critique :** Évaluer la capacité du ML à approximer une réglementation.
            * **Outil métier :** Proposer un simulateur interactif.
            """)

    # --- ONGLET 3 : DONNÉES (ENRICHI) ---
    with tab_donnees:
        st.header("Le Jeu de Données ADEME")
        
        # Métriques mises à jour avec les chiffres officiels
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Volume Total", "~13.6 Millions (12M au début du projet)", "DPE (Recensement continu)")
        col_m2.metric("Fréquence", "Hebdomadaire", "Mise à jour")
        col_m3.metric("Périmètre", "France", "Logements Existants")

        st.markdown("---")
        
        st.markdown("### 🔍 Spécificités du Dataset")
        st.warning("""
        **⚠️ Attention aux biais d'interprétation :**
        Selon l'ADEME, cette base n'est **pas représentative de l'ensemble du parc immobilier français**.
        Elle ne contient que les DPE réalisés obligatoirement lors de **ventes, locations ou constructions neuves**. 
        Un redressement statistique (croisement avec données INSEE) serait nécessaire pour une extrapolation nationale parfaite.
        """)

        st.markdown("""
        * **Source :** Base officielle [DPE Logements existants (depuis juillet 2021)](https://data.ademe.fr/datasets/dpe03existant).
        * **Contenu :** Caractéristiques techniques (bâti, isolation, chauffage), consommations énergétiques et émissions GES.
        * **Filtres appliqués pour le projet :**
            * Logements résidentiels uniquement (Maisons & Appartements).
            * Données nettoyées des valeurs aberrantes et doublons.
        """)
        
        st.caption("Données sous Licence Ouverte / Open Licence version 2.0 - Producteur : ADEME")
# ----------------------------
# PAGE 2: Dataviz
# ----------------------------


def display_img(filename, caption=""):
    """Fonction utilitaire pour gérer l'affichage sécurisé des images"""
    path = f"img/{filename}"
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Image manquante : {path}")

def page_dataviz():
    st.title("📊 Visualisation des Données DPE")
    st.markdown("""
    Cette section explore la répartition des classes énergétiques en France et analyse les corrélations 
    avec les caractéristiques physiques et géographiques des logements.
    """)

    # Création d'onglets pour organiser la navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Panorama National", 
        "🗺️ Géographie & Climat", 
        "🏗️ Caractéristiques Bâti", 
        "⏳ Temps & Surface"
    ])

    # --- ONGLET 1 : PANORAMA NATIONAL ---
    with tab1:
        st.header("État des lieux du parc immobilier")
        
        st.markdown("### 1. Répartition DPE & GES")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Étiquette Énergie (DPE)**")
            display_img("repartition_etiquette_DPE_France.png", "Répartition nationale des DPE")
        with col2:
            st.markdown("**Étiquette Climat (GES)**")
            display_img("repartition_etiquette_GES_France.png", "Répartition nationale des GES")
            
        st.info("💡 **Note :** On observe souvent une corrélation entre les étiquettes DPE et GES, bien que le mode de chauffage influence fortement le GES.")

        st.markdown("### 2. Consommation réelle")
        display_img("repartition_conso_France.png", "Distribution de la consommation énergétique (kWh/m²/an)")

    # --- ONGLET 2 : GÉOGRAPHIE ---
    with tab2:
        st.header("Disparités Territoriales")
        
        st.markdown("### 1. La France des passoires vs bâtiments écolos")
        c1, c2 = st.columns(2)
        with c1:
            display_img("part_passoires_thermiques_par_departement.png", "Part des passoires (F & G)")
        with c2:
            display_img("part_batiments_ecolo_par_departements.png", "Part des bâtiments performants (A & B)")

        st.markdown("---")
        
        st.markdown("### 2. Influence de l'environnement")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Par Région")
            display_img("repartition_DPE_regions.png", "DPE par Région administrative")
        with c4:
            st.subheader("Par Zone Climatique")
            display_img("repartition_zone_climatique.png", "Impact du climat local")
            
        st.markdown("#### Focus Altitude")
        display_img("repartition_classe_altitude.png", "Répartition des classes selon l'altitude")

    # --- ONGLET 3 : CARACTÉRISTIQUES BÂTI ---
    with tab3:
        st.header("Impact technique sur la performance")

        st.markdown("### 1. Type de bâtiment & Énergie")
        # Comparaison Maison vs Appartement (DPE & GES)
        c1, c2 = st.columns(2)
        with c1:
            display_img("etiquette_DPE_type_bat.png", "DPE selon le type de logement")
        with c2:
            display_img("etiquette_GES_type_bat.png", "GES selon le type de logement")
            
        st.markdown("#### Source d'énergie principale")
        display_img("repartition_type_energie_n1.png", "Répartition par type d'énergie")

        st.markdown("---")
        st.markdown("### 2. Inertie du bâtiment")
        st.markdown("L'inertie thermique joue un rôle clé dans le confort et la performance.")
        display_img("repartition_classe_inertie_batiment.png", "Classement selon l'inertie")

    # --- ONGLET 4 : TEMPS ET SURFACE ---
    with tab4:
        st.header("Construction et Dimensions")

        st.markdown("### 1. L'impact de l'ancienneté")
        st.markdown("L'évolution des normes de construction au fil du temps :")
        
        c1, c2 = st.columns(2)
        with c1:
            display_img("repartition_etiquette_periode.png", "Étiquettes par période de construction")
        with c2:
            display_img("repartition_periode_etiquette.png", "Périodes de construction par étiquette")

        st.markdown("---")

        st.markdown("### 2. L'impact de la surface")
        st.markdown("Les petites surfaces sont-elles défavorisées par le calcul du DPE ?")
        
        display_img("surface_etiquette_boxplot.png", "Distribution des surfaces par étiquette")

        with st.expander("🔎 Détail du nettoyage des données (Outliers)"):
            st.write("Analyse de la distribution des surfaces avant et après traitement des valeurs aberrantes.")
            col_a, col_b = st.columns(2)
            with col_a:
                display_img("surface_without_outliers.png", "Surface sans outliers")
            with col_b:
                display_img("surface_without_outliers_dist.png", "Distribution nettoyée")               
                
# ----------------------------
# PAGE 3: Résultats d'entraînement
# ----------------------------
def page_results():
    st.title("🤖 Modélisation & Résultats")
    st.markdown("""
    Nous avons testé deux approches pour prédire la performance énergétique :
    1.  **Classification** : Prédire l'étiquette DPE (A à G).
    2.  **Régression** : Prédire la consommation d'énergie primaire ($kWh/m^2/an$).
    
    *Contrainte : Utilisation d'une baseline à 16 colonnes pour gérer la charge mémoire.*
    """)

    tab_classif, tab_reg = st.tabs(["🔠 Approche Classification", "📈 Approche Régression"])

    # --- ONGLET 1 : CLASSIFICATION ---
    with tab_classif:
        st.header("Classification des étiquettes DPE")
        st.markdown("Objectif : Prédire la classe exacte (A, B, C, D, E, F, G).")

        # 1. Comparaison Baseline
        st.subheader("1. Benchmark des modèles (Baseline)")
        data_classif = {
            "Modèle": ["Random Forest", "KNN", "Decision Tree", "Logistic Regression", "Naive Bayes"],
            "Accuracy": [0.577, 0.546, 0.526, 0.526, 0.031],
            "F1-Score": [0.564, 0.539, 0.510, 0.510, 0.024]
        }
        df_classif = pd.DataFrame(data_classif).sort_values(by="Accuracy", ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df_classif.style.highlight_max(axis=0, color="#d1e7dd"), use_container_width=True)
        with col2:
            fig_classif = px.bar(df_classif, x="Accuracy", y="Modèle", orientation='h', 
                                 title="Précision par modèle (Baseline)", color="Accuracy", color_continuous_scale="Viridis")
            st.plotly_chart(fig_classif, use_container_width=True)

        # 2. Focus Meilleur Modèle
        st.subheader("2. Meilleur Modèle : Random Forest Optimisé")
        st.markdown("Après optimisation des hyperparamètres (GridSearch), les gains sont marginaux, suggérant une limite intrinsèque aux données d'entrée.")

        met1, met2, met3 = st.columns(3)
        met1.metric("Accuracy Test", "58.3%", delta="+0.6% vs Baseline")
        met2.metric("F1-Score Weighted", "0.575")
        met3.metric("Meilleur params", "500 arbres, Max Depth 20")

        # Analyse des erreurs
        with st.expander("🔎 Analyse détaillée (Matrice de Confusion & Rapport)"):
            st.markdown("#### Pourquoi plafonne-t-on à 58% ?")
            st.markdown("""
            L'analyse de la matrice de confusion montre que les erreurs sont principalement **"à une classe près"** :
            * Le modèle confond souvent **C et D** (les classes majoritaires).
            * Difficulté sur les extrêmes (A/B et F/G) à cause du déséquilibre de classe.
            """)
            
            st.markdown("#### Rapport de Classification (Optimisé)")
            report_data = {
                "Classe": ["A", "B", "C", "D", "E", "F", "G"],
                "Precision": [0.65, 0.60, 0.72, 0.56, 0.46, 0.39, 0.53],
                "Recall": [0.52, 0.33, 0.72, 0.65, 0.48, 0.18, 0.52],
                "F1-Score": [0.58, 0.43, 0.72, 0.60, 0.47, 0.25, 0.52]
            }
            st.dataframe(pd.DataFrame(report_data).set_index("Classe").style.background_gradient(cmap="Reds", subset=["F1-Score"]))

    # --- ONGLET 2 : REGRESSION ---
    with tab_reg:
        st.header("Estimation de la consommation énergétique")
        st.markdown("Objectif : Prédire une valeur continue (kWh/m²/an).")

        # 1. Benchmark ML Classique
        st.subheader("1. Benchmark Machine Learning")
        data_reg = {
            "Modèle": ["Random Forest", "KNN Regressor", "Lasso/Ridge/Linear", "Decision Tree"],
            "MAE": [44.75, 47.86, 54.79, 59.56],
            "R²": [0.645, 0.576, 0.491, 0.424]
        }
        df_reg = pd.DataFrame(data_reg).sort_values(by="R²", ascending=False)
        
        st.dataframe(df_reg.style.highlight_max(subset=["R²"], color="#d1e7dd").highlight_min(subset=["MAE"], color="#d1e7dd"), use_container_width=True)
        st.caption("Le Random Forest domine largement les modèles linéaires classiques.")

        st.divider()

        # 2. Deep Learning vs Random Forest
        st.subheader("2. Le saut de performance : Deep Learning")
        st.markdown("""
        Nous avons entraîné un réseau de neurones avec plus de colonnes en entrée. 
        C'est l'approche qui donne les **meilleurs résultats globaux**.
        """)

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("MAE (Erreur Moyenne)", "36.6 kWh/m²", delta="-7 kWh vs RF", delta_color="normal")
        col_res2.metric("RMSE", "49.6")
        col_res3.metric("R² (Score)", "0.69", delta="+0.05 vs RF")

        # 3. Image d'analyse Deep Learning
        st.markdown("#### Analyse de l'entraînement (Validation Loss)")
        st.markdown("Comparaison de la convergence selon la taille du batch (Batch Size).")
        
        # Affichage de l'image fournie
        try:
            st.image("img/loss_batch_size.png", caption="Comparaison du Val Loss par Batch Size", use_container_width=True)
            st.info("On remarque qu'un Batch Size plus grand (8192 - courbe verte) converge plus vite et offre une courbe plus stable.")
        except:
            st.warning("⚠️ Image 'img/loss_batch_size.png' introuvable.")

# ----------------------------
# LOGIQUE METIER (Calcul du DPE)
# ----------------------------
def get_classe_dpe(conso, ges):
    """
    Méthode du double seuil (2021) : on prend la pire note entre Conso et GES.
    """
    seuils = {
        "A": [70, 6],
        "B": [110, 11],
        "C": [180, 30],
        "D": [250, 50],
        "E": [330, 70],
        "F": [420, 100],
        "G": [float("inf"), float("inf")],
    }

    def get_letter(val, idx):
        for letter, limits in seuils.items():
            if val < limits[idx]:
                return letter
        return "G"

    letter_c = get_letter(conso, 0)
    letter_g = get_letter(ges, 1)

    order = "ABCDEFG"
    # "pire" = lettre plus loin dans l'alphabet
    return letter_c if order.index(letter_c) > order.index(letter_g) else letter_g


# ----------------------------
# CHARGEMENT MODELE
# ----------------------------
@st.cache_resource
def load_pipeline(model_path: str):
    """
    Charge ton pipeline sklearn (préprocess + modèle) depuis un .joblib
    """
    return joblib.load(model_path)


def predict_conso(pipeline, features: dict) -> float:
    """
    features: dict {col: value} avec EXACTEMENT les colonnes attendues
    Retourne une conso prédite (float).
    """
    X = pd.DataFrame([features])

    # Sécurités simples
    # - évite NaN surprise
    X = X.fillna("Vide")

    y_pred = pipeline.predict(X)

    # y_pred peut être array([val]) ou liste
    conso = float(np.ravel(y_pred)[0])

    # Optionnel : borner et arrondir pour un rendu UI propre
    conso = max(0.0, conso)
    return conso


# ----------------------------
# PAGE STREAMLIT
# ----------------------------
def page_simulator():
    st.title("🏗️ Simulateur de Performance Énergétique")
    st.markdown(
        """
        Remplissez les caractéristiques du logement pour estimer sa consommation et son étiquette DPE.
        *Note : la consommation est prédite par le modèle, les GES sont simulés (random).*
        """
    )

    # Chemin vers ton pipeline joblib
    # -> adapte-le à ton projet
    MODEL_PATH = "models/dpe_pipeline.joblib"

    try:
        pipeline = load_pipeline(MODEL_PATH)
    except Exception as e:
        st.error(
            f"Impossible de charger le modèle depuis `{MODEL_PATH}`.\n\n"
            f"Détail : {e}"
        )
        st.stop()

    # Options EXACTES issues de ton training (celles que tu as listées)
    form_options = {
        "type_batiment": ["appartement", "maison"],
        "periode_construction": [
            "1948-1974",
            "1975-1977",
            "1978-1982",
            "1983-1988",
            "1989-2000",
            "2001-2005",
            "2006-2012",
            "2013-2021",
            "après 2021",
            "avant 1948",
        ],
        "type_installation_chauffage": [
            "Vide",
            "collectif",
            "individuel",
            "mixte (collectif-individuel)",
        ],
        "classe_altitude": ["400-800m", "inférieur à 400m", "supérieur à 800m"],
        "type_energie_principale_chauffage": [
            "Bois – Bûches",
            "Bois – Granulés (pellets) ou briquettes",
            "Bois – Plaquettes d’industrie",
            "Bois – Plaquettes forestières",
            "Butane",
            "Charbon",
            "Fioul domestique",
            "GPL",
            "Gaz naturel",
            "Propane",
            "Réseau de Chauffage urbain",
            "Électricité",
            "Électricité d'origine renouvelable utilisée dans le bâtiment",
        ],
        "type_emetteur_installation_chauffage_n1": [
            "Autres",
            "Convecteur électrique NFC  NF** et NF***",
            "Panneau rayonnant NFC  NF** et NF***",
            "Radiateur bitube avec robinet thermostatique sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)",
            "Radiateur bitube avec robinet thermostatique sur réseau individuel eau chaude haute température(sup ou egal 65°C)",
            "Vide",
            "radiateur électrique NFC  NF** et NF***",
        ],
        "type_energie_generateur_n1_ecs_n1": [
            "Bois – Bûches",
            "Bois – Granulés (pellets) ou briquettes",
            "Bois – Plaquettes d’industrie",
            "Bois – Plaquettes forestières",
            "Butane",
            "Charbon",
            "Fioul domestique",
            "GPL",
            "Gaz naturel",
            "Propane",
            "Réseau de Chauffage urbain",
            "Vide",
            "Électricité",
            "Électricité d'origine renouvelable utilisée dans le bâtiment",
        ],
        "type_energie_n1": [
            "Bois – Bûches",
            "Bois – Granulés (pellets) ou briquettes",
            "Bois – Plaquettes d’industrie",
            "Bois – Plaquettes forestières",
            "Butane",
            "Charbon",
            "Fioul domestique",
            "GPL",
            "Gaz naturel",
            "Propane",
            "Réseau de Chauffage urbain",
            "Électricité",
            "Électricité d'origine renouvelable utilisée dans le bâtiment",
        ],
        "type_energie_principale_ecs": [
            "Bois – Bûches",
            "Bois – Granulés (pellets) ou briquettes",
            "Bois – Plaquettes d’industrie",
            "Bois – Plaquettes forestières",
            "Butane",
            "Charbon",
            "Fioul domestique",
            "GPL",
            "Gaz naturel",
            "Non affecté",
            "Propane",
            "Réseau de Chauffage urbain",
            "Électricité",
            "Électricité d'origine renouvelable utilisée dans le bâtiment",
        ],
        "type_installation_ecs": ["INCONNU", "collectif", "individuel", "mixte (collectif-individuel)"],
        "type_generateur_chauffage_principal": [
            "Autres",
            "Chaudière gaz à condensation 2001-2015",
            "Chaudière gaz à condensation après 2015",
            "Convecteur électrique NFC  NF** et NF***",
            "Panneau rayonnant électrique NFC  NF** et NF***",
            "Radiateur électrique à accumulation",
            "Réseau de chaleur isolé",
            "Vide",
        ],
        "type_generateur_chauffage_principal_ecs": [
            "Autres",
            "Ballon électrique à accumulation vertical Autres ou inconnue",
            "Ballon électrique à accumulation vertical Catégorie B ou 2 étoiles",
            "Chaudière gaz à condensation 2001-2015",
            "Chaudière gaz à condensation après 2015",
            "Vide",
        ],
        "type_energie_n2": [
            "AUCUN",
            "Bois – Bûches",
            "Bois – Granulés (pellets) ou briquettes",
            "Bois – Plaquettes d’industrie",
            "Bois – Plaquettes forestières",
            "Butane",
            "Charbon",
            "Fioul domestique",
            "GPL",
            "Gaz naturel",
            "Propane",
            "Réseau de Chauffage urbain",
            "Électricité",
            "Électricité d'origine renouvelable utilisée dans le bâtiment",
        ],
        "zone_clim_simple": ["H1", "H2", "H3"],
    }

    with st.form("form_simulation"):
        st.subheader("1. Caractéristiques utilisées par le modèle")

        c1, c2, c3 = st.columns(3)

        with c1:
            type_batiment = st.selectbox("Type de bâtiment", form_options["type_batiment"])
            periode_construction = st.selectbox("Période de construction", form_options["periode_construction"])
            classe_altitude = st.selectbox("Classe d'altitude", form_options["classe_altitude"])
            zone_clim_simple = st.selectbox("Zone climatique", form_options["zone_clim_simple"])

        with c2:
            type_installation_chauffage = st.selectbox(
                "Type installation chauffage",
                form_options["type_installation_chauffage"],
            )
            type_energie_principale_chauffage = st.selectbox(
                "Énergie principale chauffage",
                form_options["type_energie_principale_chauffage"],
            )
            type_generateur_chauffage_principal = st.selectbox(
                "Générateur chauffage principal",
                form_options["type_generateur_chauffage_principal"],
            )

        with c3:
            type_emetteur_installation_chauffage_n1 = st.selectbox(
                "Type émetteur chauffage (n1)",
                form_options["type_emetteur_installation_chauffage_n1"],
            )
            type_installation_ecs = st.selectbox(
                "Type installation ECS",
                form_options["type_installation_ecs"],
            )
            type_energie_principale_ecs = st.selectbox(
                "Énergie principale ECS",
                form_options["type_energie_principale_ecs"],
            )

        with st.expander("Paramètres avancés (énergies secondaires / détails ECS)"):
            ac1, ac2 = st.columns(2)
            with ac1:
                type_energie_n1 = st.selectbox("Type énergie n°1", form_options["type_energie_n1"])
                type_energie_generateur_n1_ecs_n1 = st.selectbox(
                    "Type énergie générateur n°1 ECS (n1)",
                    form_options["type_energie_generateur_n1_ecs_n1"],
                )
            with ac2:
                type_energie_n2 = st.selectbox("Type énergie n°2", form_options["type_energie_n2"])
                type_generateur_chauffage_principal_ecs = st.selectbox(
                    "Générateur chauffage principal ECS",
                    form_options["type_generateur_chauffage_principal_ecs"],
                )

        submitted = st.form_submit_button("🚀 Lancer la simulation", use_container_width=True)

    if submitted:
        # ✅ Construction EXACTE des features attendues par le modèle
        features = {
            "type_batiment": type_batiment,
            "periode_construction": periode_construction,
            "type_installation_chauffage": type_installation_chauffage,
            "classe_altitude": classe_altitude,
            "type_energie_principale_chauffage": type_energie_principale_chauffage,
            "type_emetteur_installation_chauffage_n1": type_emetteur_installation_chauffage_n1,
            "type_energie_generateur_n1_ecs_n1": type_energie_generateur_n1_ecs_n1,
            "type_energie_n1": type_energie_n1,
            "type_energie_principale_ecs": type_energie_principale_ecs,
            "type_installation_ecs": type_installation_ecs,
            "type_generateur_chauffage_principal": type_generateur_chauffage_principal,
            "type_generateur_chauffage_principal_ecs": type_generateur_chauffage_principal_ecs,
            "type_energie_n2": type_energie_n2,
            "zone_clim_simple": zone_clim_simple,
        }

        # ✅ conso via modèle
        try:
            conso_pred = predict_conso(pipeline, features)
        except Exception as e:
            st.error(
                "Erreur pendant la prédiction du modèle. "
                "Vérifie que les noms de colonnes et les modalités correspondent à l'entraînement.\n\n"
                f"Détail : {e}"
            )
            st.stop()

        # ✅ GES en random (comme demandé)
        ges_simule = random.randint(2, 68)

        # ✅ Classe DPE (double seuil)
        classe_finale = get_classe_dpe(conso_pred, ges_simule)

        st.divider()
        st.header("Résultats de l'estimation")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.markdown("### Indicateurs")
            st.metric("Consommation (Ep)", f"{conso_pred:.0f} kWh/m²/an")
            st.metric("Émissions (GES)", f"{ges_simule} kgCO₂/m²/an")

            color_map = {
                "A": "#009036",
                "B": "#53af31",
                "C": "#c6d300",
                "D": "#fce600",
                "E": "#fbba00",
                "F": "#eb6105",
                "G": "#d40f14",
            }
            st.markdown(
                f"""
                <div style="text-align: center; background-color: {color_map[classe_finale]};
                            padding: 10px; border-radius: 10px;">
                    <h1 style="color: white; margin:0;">CLASSE {classe_finale}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_res2:
            st.markdown("### Étiquette Officielle")
            base_url = "https://www.outils.immo/outils-immo.php"
            params = (
                f"?type=dpe&modele=2021&valeur={int(round(conso_pred))}"
                f"&lettre={classe_finale}&valeurges={ges_simule}"
            )
            full_url = base_url + params
            st.image(
                full_url,
                caption=f"DPE généré pour {conso_pred:.0f} kWh et {ges_simule} kgCO₂",
                use_container_width=True,
            )

        st.success("Simulation terminée avec succès (Conso via modèle, GES aléatoire).")

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
