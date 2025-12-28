# 🏡 Analyse et Prédiction du DPE des Logements en France

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projetdpe.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> **Projet fil rouge réalisé dans le cadre de la formation Data Scientist chez [DataScientest](https://datascientest.com/).**

Ce projet explore la base de données des **Diagnostics de Performance Énergétique (DPE)** en France. Notre objectif est de comprendre les déterminants de la consommation énergétique résidentielle et de développer des modèles de Machine Learning capables de prédire la classe énergétique et la consommation réelle d'un logement.

👉 **[Accéder à l'application de démonstration](https://projetdpe.streamlit.app/)**

---

## 📑 Table des matières
1. [Contexte et Données](#-contexte-et-données)
2. [Objectifs](#-objectifs)
3. [Méthodologie](#-méthodologie)
4. [Résultats de Modélisation](#-résultats-de-modélisation)
5. [Fonctionnalités de l'Application](#-fonctionnalités-de-lapplication)
6. [Structure du Repository](#-structure-du-repository)
7. [Installation locale](#-installation-locale)
8. [Auteurs](#-auteurs)

---

## 💾 Contexte et Données

Les données sont issues de l'**Observatoire DPE-Audit** de l'ADEME. Elles recensent les diagnostics réalisés sur le territoire français.

* **Source officielle** : [Observatoire DPE - ADEME](https://observatoire-dpe-audit.ademe.fr/donnees-dpe-publiques)
* **Périmètre** : Logements existants (France Métropolitaine).
* **Variables** : Caractéristiques techniques (surface, année de construction, matériaux d'isolation, type de chauffage/ECS), localisation (altitude, zone climatique) et résultats (conso kwh/m², émissions GES, étiquettes).

---

## 🎯 Objectifs

1.  **Analyse Exploratoire (EDA)** : Visualiser la répartition du parc immobilier, identifier les zones de "passoires thermiques" et corréler les caractéristiques physiques aux performances.
2.  **Modélisation** :
    * **Classification** : Prédire l'étiquette DPE (A, B, C, D, E, F, G).
    * **Régression** : Estimer la consommation d'énergie primaire précise ($kWh/m^2/an$).
3.  **Déploiement** : Mettre à disposition un outil de simulation interactif.

---

## ⚙️ Méthodologie

Le projet a suivi le cycle de vie classique d'un projet de Data Science :

1.  **Data Cleaning & Preprocessing** :
    * Filtrage des données aberrantes (surfaces incohérentes, consommations nulles).
    * Imputation des valeurs manquantes.
    * Encodage des variables catégorielles (OneHotEncoding pour les types d'énergie, Ordinal pour les qualités d'isolation).
2.  **Feature Engineering** : Création de variables synthétiques et sélection des features les plus importantes (Feature Importance).
3.  **Modélisation** :
    * *Baseline* : Régression Linéaire, KNN.
    * *Modèles avancés* : Random Forest, XGBoost.
    * *Deep Learning* : Réseaux de neurones denses (DNN) pour la régression.

---

## 📊 Résultats de Modélisation

Nous avons testé deux approches principales. Voici les meilleures performances obtenues sur le jeu de test :

### 1. Classification (Prédiction de l'étiquette)
* **Meilleur Modèle** : Random Forest Classifier (Optimisé).
* **Accuracy** : **58.3%**.
* *Analyse* : Le modèle rencontre des difficultés aux frontières des classes (ex: distinguer un "C bas" d'un "D haut"), mais capture bien la tendance globale.

### 2. Régression (Prédiction de la consommation)
* **Meilleur Modèle** : Réseau de Neurones (Deep Learning).
* **R² (Score)** : **0.69**.
* **MAE (Erreur Absolue Moyenne)** : **36.6 kWh/m²/an**.
* *Conclusion* : L'approche par régression est plus précise. Elle permet de recalculer l'étiquette *a posteriori* en appliquant les seuils officiels du DPE.

---

## 🖥 Fonctionnalités de l'Application

L'application Streamlit est structurée en trois parties :

1.  **Data Visualization** :
    * Cartographie des passoires thermiques par département.
    * Distribution des étiquettes (DPE/GES).
    * Impact de l'année de construction et de la surface.
2.  **Modélisation** :
    * Présentation des métriques de performance.
    * Comparaison des algorithmes (Benchmark).
    * Analyse de l'apprentissage (Loss curves).
3.  **Simulateur** :
    * Formulaire interactif permettant de saisir les caractéristiques d'un bien.
    * Estimation en temps réel de la consommation et de l'étiquette.
    * Génération visuelle de l'étiquette DPE officielle.

---

## 📂 Structure du Repository

```bash
├── img/                # Images pour l'application
├── models/             # Modèles entraînés
├── notebooks/          # Jupyter Notebooks (EDA, Preprocessing, Modeling)
├── app.py              # Application principale Streamlit
├── requirements.txt    # Liste des dépendances
└── README.md           # Documentation du projet
```

## 🚀 Installation locale

Si vous souhaitez faire tourner le projet sur votre machine :

    Cloner le dépôt :
```bash
git clone [https://github.com/VOTRE-USER/VOTRE-REPO.git](https://github.com/VOTRE-USER/VOTRE-REPO.git)
cd VOTRE-REPO
```

Créer un environnement virtuel :
```bash
python -m venv venv
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Lancer l'application :
```bash
    streamlit run app.py
```

## 👥 Auteurs

Projet réalisé par l'équipe DataScientest :

    [Aymane Karani]

    [Dylan Nefnaf]

    [Guillaume Deschamps]

    [Yacine Bennouna]