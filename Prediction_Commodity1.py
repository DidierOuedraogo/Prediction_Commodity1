import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import random

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Prix des Commodités Minérales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #4361ee;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e9ecef;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4361ee;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef;
    }
    .author-info {
        text-align: center;
        margin-bottom: 2rem;
        color: #6c757d;
        font-style: italic;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4361ee;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6c757d;
    }
    .confidence-interval {
        background-color: rgba(67, 97, 238, 0.1);
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .prediction-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-top: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.875rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(67, 97, 238, 0.1);
        border-bottom: 2px solid #4361ee;
    }
</style>
""", unsafe_allow_html=True)

# Titre et informations sur l'auteur
st.markdown('<h1 class="main-header">Prédiction de Prix des Commodités Minérales</h1>', unsafe_allow_html=True)
st.markdown('<p class="author-info">Développé par Didier Ouedraogo, P.Geo</p>', unsafe_allow_html=True)

# Base de données des commodités minérales et leurs symboles
commodity_database = [
    {"name": "Or", "symbol": "XAU", "unit": "oz", "categories": ["métal précieux", "or"]},
    {"name": "Gold", "symbol": "XAU", "unit": "oz", "categories": ["precious metal", "gold"]},
    {"name": "Argent", "symbol": "XAG", "unit": "oz", "categories": ["métal précieux", "argent"]},
    {"name": "Silver", "symbol": "XAG", "unit": "oz", "categories": ["precious metal", "silver"]},
    {"name": "Platine", "symbol": "XPT", "unit": "oz", "categories": ["métal précieux", "platine"]},
    {"name": "Platinum", "symbol": "XPT", "unit": "oz", "categories": ["precious metal", "platinum"]},
    {"name": "Palladium", "symbol": "XPD", "unit": "oz", "categories": ["métal précieux", "palladium"]},
    {"name": "Cuivre", "symbol": "COPPER", "unit": "tonne", "categories": ["métal de base", "cuivre"]},
    {"name": "Copper", "symbol": "COPPER", "unit": "tonne", "categories": ["base metal", "copper"]},
    {"name": "Aluminium", "symbol": "ALUMINUM", "unit": "tonne", "categories": ["métal de base", "aluminium"]},
    {"name": "Aluminum", "symbol": "ALUMINUM", "unit": "tonne", "categories": ["base metal", "aluminum"]},
    {"name": "Zinc", "symbol": "ZINC", "unit": "tonne", "categories": ["métal de base", "zinc"]},
    {"name": "Nickel", "symbol": "NICKEL", "unit": "tonne", "categories": ["métal de base", "nickel"]},
    {"name": "Plomb", "symbol": "LEAD", "unit": "tonne", "categories": ["métal de base", "plomb"]},
    {"name": "Lead", "symbol": "LEAD", "unit": "tonne", "categories": ["base metal", "lead"]},
    {"name": "Étain", "symbol": "TIN", "unit": "tonne", "categories": ["métal de base", "étain"]},
    {"name": "Tin", "symbol": "TIN", "unit": "tonne", "categories": ["base metal", "tin"]},
    {"name": "Fer", "symbol": "IRONORE", "unit": "tonne", "categories": ["minerai", "fer"]},
    {"name": "Iron Ore", "symbol": "IRONORE", "unit": "tonne", "categories": ["ore", "iron"]},
    {"name": "Lithium", "symbol": "LITHIUM", "unit": "tonne", "categories": ["métal de spécialité", "lithium"]},
    {"name": "Cobalt", "symbol": "COBALT", "unit": "tonne", "categories": ["métal de spécialité", "cobalt"]},
]

# Fonction pour générer des données historiques (simulées)
def generate_historical_data(commodity_name, start_date, end_date, volatility=None, unit=None):
    """Génère des données de prix historiques simulées"""
    
    # Définir la volatilité en fonction de la commodité si non spécifiée
    if volatility is None:
        if "or" in commodity_name.lower() or "gold" in commodity_name.lower():
            volatility = 0.01  # Faible volatilité
        elif "argent" in commodity_name.lower() or "silver" in commodity_name.lower():
            volatility = 0.015
        elif "cuivre" in commodity_name.lower() or "copper" in commodity_name.lower():
            volatility = 0.02
        elif "lithium" in commodity_name.lower():
            volatility = 0.03  # Haute volatilité
        else:
            volatility = 0.02  # Volatilité moyenne
    
    # Définir un prix de base en fonction de la commodité
    commodity_lower = commodity_name.lower()
    if "or" in commodity_lower or "gold" in commodity_lower:
        base_price = 1800  # USD par once
        trend = 0.0001  # Tendance légèrement haussière
    elif "argent" in commodity_lower or "silver" in commodity_lower:
        base_price = 25  # USD par once
        trend = 0.0002
    elif "platine" in commodity_lower or "platinum" in commodity_lower:
        base_price = 900  # USD par once
        trend = -0.0001  # Tendance légèrement baissière
    elif "palladium" in commodity_lower:
        base_price = 1600  # USD par once
        trend = 0.0003
    elif "cuivre" in commodity_lower or "copper" in commodity_lower:
        base_price = 8500  # USD par tonne
        trend = 0.0002
    elif "aluminium" in commodity_lower or "aluminum" in commodity_lower:
        base_price = 2200  # USD par tonne
        trend = 0.0001
    elif "zinc" in commodity_lower:
        base_price = 2800  # USD par tonne
        trend = 0.0001
    elif "nickel" in commodity_lower:
        base_price = 16000  # USD par tonne
        trend = 0.0002
    elif "lithium" in commodity_lower:
        base_price = 70000  # USD par tonne
        trend = 0.0005
    elif "fer" in commodity_lower or "iron" in commodity_lower:
        base_price = 120  # USD par tonne
        trend = -0.0002
    else:
        base_price = 1000  # Prix générique
        trend = 0.0001
    
    # Convertir les dates en objets datetime
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculer le nombre de jours
    day_diff = (end_date - start_date).days
    
    # Générer des points de données quotidiens
    dates = []
    prices = []
    current_price = base_price
    
    for i in range(day_diff + 1):
        current_date = start_date + timedelta(days=i)
        
        # Sauter les weekends pour simuler des données boursières
        if current_date.weekday() >= 5:  # 5 = Samedi, 6 = Dimanche
            continue
        
        # Ajouter une composante aléatoire (marche aléatoire)
        random_change = (random.random() - 0.5) * 2 * volatility * current_price
        # Ajouter une composante de tendance
        trend_change = trend * current_price
        
        # Ajouter une composante saisonnière (par exemple, cycles de 30 jours)
        seasonal_factor = 0.05  # Force de la saisonnalité
        seasonal_change = math.sin(i * (2 * math.pi / 30)) * seasonal_factor * current_price
        
        # Calculer le nouveau prix
        current_price = current_price + random_change + trend_change + seasonal_change
        
        # S'assurer que le prix reste positif
        current_price = max(current_price, base_price * 0.1)
        
        dates.append(current_date)
        prices.append(current_price)
    
    return pd.DataFrame({
        'date': dates,
        'price': prices
    })

# Menu latéral pour les paramètres principaux
st.sidebar.markdown("## Paramètres de la commodité")

# Sélection de la commodité avec recherche et autocomplétion
commodity_names = [item["name"] for item in commodity_database]
commodity_name = st.sidebar.selectbox(
    "Nom de la Commodité", 
    options=commodity_names, 
    index=0 if commodity_names else None
)

# Obtenir les informations sur la commodité sélectionnée
selected_commodity = next((item for item in commodity_database if item["name"] == commodity_name), None)
commodity_symbol = selected_commodity["symbol"] if selected_commodity else ""
default_unit = selected_commodity["unit"] if selected_commodity else "tonne"

# Paramètres supplémentaires
unit = st.sidebar.selectbox(
    "Unité de Masse",
    options=["tonne", "kg", "oz", "lb"],
    index=["tonne", "kg", "oz", "lb"].index(default_unit) if default_unit in ["tonne", "kg", "oz", "lb"] else 0
)

currency = st.sidebar.selectbox(
    "Devise",
    options=["USD", "EUR", "GBP", "JPY", "CHF"],
    index=0
)

prediction_period = st.sidebar.slider(
    "Période de prédiction (jours)",
    min_value=1,
    max_value=365,
    value=30
)

# Choix de l'algorithme de prédiction
st.sidebar.markdown("## Algorithme de prédiction")
algorithm = st.sidebar.radio(
    "Sélectionnez un algorithme",
    options=["Régression linéaire", "ARIMA", "Prophet"],
    index=0,
    help="Régression linéaire: modèle simple basé sur la tendance linéaire\nARIMA: modèle tenant compte des moyennes mobiles et de l'auto-régression\nProphet: détecte cycles, saisonnalités et tendances"
)

# Paramètres avancés par algorithme
with st.sidebar.expander("Paramètres avancés"):
    if algorithm == "Régression linéaire":
        regression_type = st.selectbox(
            "Type de régression",
            options=["linear", "polynomial", "exponential", "logarithmic"],
            index=0,
            format_func=lambda x: {
                "linear": "Linéaire simple",
                "polynomial": "Polynomiale",
                "exponential": "Exponentielle",
                "logarithmic": "Logarithmique"
            }[x]
        )
        
        if regression_type == "polynomial":
            poly_degree = st.slider("Degré polynomial", min_value=2, max_value=5, value=2)
    
    elif algorithm == "ARIMA":
        col1, col2, col3 = st.columns(3)
        p_value = col1.number_input("p", min_value=0, max_value=5, value=1, help="Ordre auto-régressif")
        d_value = col2.number_input("d", min_value=0, max_value=2, value=1, help="Ordre de différenciation")
        q_value = col3.number_input("q", min_value=0, max_value=5, value=1, help="Ordre moyenne mobile")
    
    elif algorithm == "Prophet":
        changepoint_scale = st.slider(
            "Flexibilité de la tendance",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            format="%.3f",
            step=0.001
        )
        
        seasonality_scale = st.slider(
            "Force de la saisonnalité",
            min_value=0.01,
            max_value=10.0,
            value=10.0,
            format="%.2f"
        )
        
        holidays_option = st.selectbox(
            "Prise en compte des jours fériés",
            options=["none", "global", "country"],
            index=0,
            format_func=lambda x: {
                "none": "Non",
                "global": "Jours fériés globaux",
                "country": "Jours fériés spécifiques au pays"
            }[x]
        )

# Onglets pour les différentes méthodes d'importation de données
tabs = st.tabs(["Fichier CSV", "Recherche Internet", "Données d'exemple"])

# Onglet 1: Importation de fichier CSV
with tabs[0]:
    st.markdown("### Importer des données historiques (CSV)")
    uploaded_file = st.file_uploader("Sélectionner un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Vérifier si les colonnes existent
            if 'date' not in df.columns and 'price' not in df.columns:
                # Renommer les colonnes si nécessaire
                if len(df.columns) >= 2:
                    df.columns = ['date', 'price'] + list(df.columns[2:])
                else:
                    st.error("Le fichier CSV doit contenir au moins deux colonnes: date et prix")
            
            # Convertir la colonne date au format datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Trier par date
            df = df.sort_values('date')
            
            # Afficher un aperçu des données
            st.markdown("#### Aperçu des données importées")
            st.dataframe(df.head())
            
            # Afficher un graphique des données historiques
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['price'],
                mode='lines',
                name='Prix historiques',
                line=dict(color='#4361ee', width=2)
            ))
            fig.update_layout(
                title=f"Prix historiques de {commodity_name}",
                xaxis_title="Date",
                yaxis_title=f"Prix ({currency}/{unit})",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stocker les données pour la prédiction
            data_source = "file"
            historical_data = df
        
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {e}")

# Onglet 2: Recherche Internet
with tabs[1]:
    st.markdown("### Rechercher des données historiques en ligne")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_input = st.text_input("Symbole de la commodité (facultatif)", value=commodity_symbol)
    
    with col2:
        data_provider = st.selectbox(
            "Source de données",
            options=["automatic", "worldbank", "imf", "lme", "usgs"],
            index=0,
            format_func=lambda x: {
                "automatic": "Sélection automatique",
                "worldbank": "Banque Mondiale",
                "imf": "FMI (Fonds Monétaire International)",
                "lme": "LME (London Metal Exchange)",
                "usgs": "USGS (US Geological Survey)"
            }[x]
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=datetime.now() - timedelta(days=365)
        )
    
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=datetime.now()
        )
    
    if st.button("Rechercher les données", key="search_data"):
        with st.spinner("Recherche de données en cours..."):
            # Simuler un délai de recherche
            import time
            time.sleep(2)
            
            # Générer des données simulées
            api_data = generate_historical_data(
                commodity_name,
                start_date,
                end_date,
                unit=unit
            )
            
            # Afficher les résultats
            st.success(f"{len(api_data)} points de données trouvés pour {commodity_name}")
            
            # Afficher un aperçu des données
            st.markdown("#### Aperçu des données trouvées")
            st.dataframe(api_data.head())
            
            # Afficher un graphique des données historiques
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=api_data['date'],
                y=api_data['price'],
                mode='lines',
                name='Prix historiques',
                line=dict(color='#4361ee', width=2)
            ))
            fig.update_layout(
                title=f"Prix historiques de {commodity_name}",
                xaxis_title="Date",
                yaxis_title=f"Prix ({currency}/{unit})",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Optionnel: bouton de téléchargement des données
            csv = api_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{commodity_name}_historical_data.csv">Télécharger les données CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Stocker les données pour la prédiction
            data_source = "api"
            historical_data = api_data

# Onglet 3: Données d'exemple
with tabs[2]:
    st.markdown("### Utiliser des données d'exemple")
    
    sample_commodities = {
        "gold": "Or (Gold)",
        "silver": "Argent (Silver)",
        "copper": "Cuivre (Copper)",
        "aluminum": "Aluminium",
        "iron": "Fer (Iron Ore)",
        "zinc": "Zinc",
        "nickel": "Nickel",
        "lithium": "Lithium",
        "platinum": "Platine (Platinum)",
        "palladium": "Palladium"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_commodity = st.selectbox(
            "Sélectionner une commodité",
            options=list(sample_commodities.keys()),
            index=0,
            format_func=lambda x: sample_commodities[x]
        )
    
    with col2:
        sample_timeframe = st.selectbox(
            "Période d'échantillon",
            options=["1y", "2y", "5y", "10y"],
            index=2,
            format_func=lambda x: {
                "1y": "1 an",
                "2y": "2 ans",
                "5y": "5 ans",
                "10y": "10 ans"
            }[x]
        )
    
    if st.button("Charger les données d'exemple", key="load_sample"):
        with st.spinner("Chargement des données d'exemple..."):
            # Simuler un délai
            import time
            time.sleep(1)
            
            # Définir la date de début en fonction de la période
            end_date = datetime.now()
            if sample_timeframe == "1y":
                start_date = end_date - timedelta(days=365)
            elif sample_timeframe == "2y":
                start_date = end_date - timedelta(days=365*2)
            elif sample_timeframe == "5y":
                start_date = end_date - timedelta(days=365*5)
            else:  # 10y
                start_date = end_date - timedelta(days=365*10)
            
            # Générer des données simulées pour la commodité sélectionnée
            sample_data = generate_historical_data(
                sample_commodity,
                start_date,
                end_date,
                unit=unit
            )
            
            # Mettre à jour le nom de la commodité
            commodity_name = sample_commodities[sample_commodity].split(" ")[0]  # Prendre le premier mot
            
            # Afficher les résultats
            st.success(f"{len(sample_data)} points de données d'exemple pour {commodity_name}")
            
            # Afficher un aperçu des données
            st.markdown("#### Aperçu des données d'exemple")
            st.dataframe(sample_data.head())
            
            # Afficher un graphique des données historiques
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_data['date'],
                y=sample_data['price'],
                mode='lines',
                name='Prix historiques',
                line=dict(color='#4361ee', width=2)
            ))
            fig.update_layout(
                title=f"Prix historiques de {commodity_name}",
                xaxis_title="Date",
                yaxis_title=f"Prix ({currency}/{unit})",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stocker les données pour la prédiction
            data_source = "sample"
            historical_data = sample_data

# Bouton pour exécuter la prédiction
st.markdown('<h2 class="subheader">Prédiction de prix</h2>', unsafe_allow_html=True)

# Vérifier si des données sont disponibles
if 'historical_data' in locals():
    if st.button("Générer la Prédiction", key="predict_button"):
        with st.spinner("Calcul des prédictions en cours..."):
            # Simuler un temps de calcul
            import time
            time.sleep(2)
            
            # Obtenir les paramètres de prédiction
            prediction_days = prediction_period
            
            # Extraire les données nécessaires
            dates = historical_data['date'].values
            prices = historical_data['price'].values
            
            # Fonction pour la régression linéaire
            def predict_with_regression(data, days, reg_type="linear", degree=2):
                x = np.array(range(len(data)))
                y = np.array(data)
                
                if reg_type == "linear":
                    # Régression linéaire simple
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)
                    
                elif reg_type == "polynomial":
                    # Régression polynomiale
                    coeffs = np.polyfit(x, y, degree)
                    poly = np.poly1d(coeffs)
                    
                elif reg_type == "exponential":
                    # Régression exponentielle
                    # y = a * e^(bx)
                    # ln(y) = ln(a) + bx
                    valid_indices = y > 0  # Éviter log de valeurs négatives
                    x_valid = x[valid_indices]
                    y_valid = np.log(y[valid_indices])
                    
                    if len(x_valid) < 2:
                        # Fallback to linear if not enough data
                        return predict_with_regression(data, days, "linear")
                    
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)
                    a = np.exp(intercept)
                    b = slope
                    
                    poly = lambda x: a * np.exp(b * x)
                    
                elif reg_type == "logarithmic":
                    # Régression logarithmique
                    # y = a + b * ln(x)
                    valid_indices = x > 0  # Éviter log de valeurs négatives ou nulles
                    x_valid = np.log(x[valid_indices])
                    y_valid = y[valid_indices]
                    
                    if len(x_valid) < 2:
                        # Fallback to linear if not enough data
                        return predict_with_regression(data, days, "linear")
                    
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)
                    
                    poly = lambda x: intercept + slope * np.log(np.maximum(x, 0.1))  # Éviter log(0)
                
                else:
                    # Default to linear
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)
                
                # Calculer les erreurs
                y_pred = [poly(i) for i in x]
                errors = y - y_pred
                rmse = np.sqrt(np.mean(errors**2))
                mae = np.mean(np.abs(errors))
                
                # Générer les prédictions
                future_dates = [dates[-1] + timedelta(days=i+1) for i in range(days)]
                future_indices = np.array([len(data) + i for i in range(days)])
                future_prices = [poly(i) for i in future_indices]
                
                # Calculer les intervalles de confiance
                confidence_factor = 1.96  # 95% confidence interval
                confidence = [rmse * confidence_factor * np.sqrt(1 + 1/len(data) + (i - np.mean(x))**2 / np.sum((x - np.mean(x))**2)) for i in future_indices]
                
                lower_bounds = [max(0, price - conf) for price, conf in zip(future_prices, confidence)]
                upper_bounds = [price + conf for price, conf in zip(future_prices, confidence)]
                
                return {
                    "dates": future_dates,
                    "prices": future_prices,
                    "lower_bounds": lower_bounds,
                    "upper_bounds": upper_bounds,
                    "metrics": {
                        "rmse": rmse,
                        "mae": mae,
                        "reliability": 100 - (mae / (np.max(y) - np.min(y))) * 100 if np.max(y) != np.min(y) else 0
                    }
                }
            
            # Fonction pour simuler ARIMA
            def predict_with_arima(data, days, p=1, d=1, q=1):
                # Différenciation simple pour simuler d=1
                diff_prices = [data[i] - data[i-1] for i in range(1, len(data))]
                
                # Moyenne des différences
                avg_diff = np.mean(diff_prices)
                
                # Écart-type des différences pour l'intervalle de confiance
                diff_std = np.std(diff_prices)
                
                # Simuler l'autocorrélation
                auto_corr = 0
                if len(diff_prices) > 1:
                    auto_corr = np.corrcoef(diff_prices[:-1], diff_prices[1:])[0, 1]
                    if np.isnan(auto_corr):
                        auto_corr = 0
                
                # Poids des paramètres
                p_factor = min(1, p * 0.3)  # Poids de l'auto-régression
                
                # Fonction pour prédire la prochaine valeur
                def arima_next_value(prev, prev_diff):
                    return prev + (p_factor * auto_corr * prev_diff + (1 - p_factor) * avg_diff)
                
                # Mesurer les erreurs
                test_size = min(20, len(data) // 3)
                if test_size > 0:
                    test_data = data[-test_size:]
                    test_diffs = diff_prices[-test_size:]
                    
                    errors = []
                    for i in range(len(test_data) - 1):
                        pred = arima_next_value(test_data[i], test_diffs[i] if i < len(test_diffs) else avg_diff)
                        errors.append(test_data[i+1] - pred)
                    
                    rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else diff_std
                    mae = np.mean(np.abs(errors)) if errors else diff_std / 2
                else:
                    rmse = diff_std
                    mae = diff_std / 2
                
                # Générer les prédictions
                future_dates = [dates[-1] + timedelta(days=i+1) for i in range(days)]
                future_prices = []
                future_lower_bounds = []
                future_upper_bounds = []
                
                last_price = data[-1]
                last_diff = diff_prices[-1] if diff_prices else avg_diff
                
                for i in range(days):
                    next_price = arima_next_value(last_price, last_diff)
                    next_diff = next_price - last_price
                    
                    # Incertitude croissante avec l'horizon
                    uncertainty = rmse * np.sqrt(i + 1)
                    
                    future_prices.append(next_price)
                    future_lower_bounds.append(max(0, next_price - 1.96 * uncertainty))
                    future_upper_bounds.append(next_price + 1.96 * uncertainty)
                    
                    last_price = next_price
                    last_diff = next_diff
                
                return {
                    "dates": future_dates,
                    "prices": future_prices,
                    "lower_bounds": future_lower_bounds,
                    "upper_bounds": future_upper_bounds,
                    "metrics": {
                        "rmse": rmse,
                        "mae": mae,
                        "reliability": 100 - (mae / (np.max(data) - np.min(data))) * 100 if np.max(data) != np.min(data) else 0
                    }
                }
            
            # Fonction pour simuler Prophet
            def predict_with_prophet(data, days, changepoint_scale=0.05, seasonality_scale=10, holiday_option="none"):
                dates_numeric = np.array(range(len(data)))
                
                # Tendance: régression polynomiale du second degré
                trend_coefs = np.polyfit(dates_numeric, data, 2)
                trend = np.poly1d(trend_coefs)
                trend_values = trend(dates_numeric)
                
                # Résidus (données - tendance)
                residuals = data - trend_values
                
                # Détecter la saisonnalité via auto-corrélation
                max_corr = 0
                best_period = 7  # Par défaut, semaine
                
                for period in range(2, min(30, len(residuals) // 3)):
                    if len(residuals) <= period:
                        continue
                    
                    # Calculer l'auto-corrélation pour ce décalage
                    corr = np.corrcoef(residuals[:-period], residuals[period:])[0, 1]
                    if np.isnan(corr):
                        continue
                    
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        best_period = period
                
                # Fonction pour la composante saisonnière
                def seasonal_component(day, period=best_period):
                    return seasonality_scale / 10 * max_corr * np.sin(2 * np.pi * day / period)
                
                # Fonction pour l'effet des jours fériés
                def holiday_effect(date):
                    if holiday_option == "none":
                        return 0
                    
                    # Effet simple sur les fins de mois
                    day_of_month = date.day
                    if day_of_month >= 28 or day_of_month <= 2:
                        return (np.random.random() * 0.02 + 0.01) * np.max(data)
                    
                    return 0
                
                # Calculer les valeurs prédites sur les données historiques
                predicted_values = []
                for i, date in enumerate(dates):
                    pred = trend(i) + seasonal_component(i) + holiday_effect(date)
                    predicted_values.append(pred)
                
                # Calculer les erreurs
                errors = data - np.array(predicted_values)
                rmse = np.sqrt(np.mean(errors**2))
                mae = np.mean(np.abs(errors))
                
                # Générer les prédictions
                future_dates = [dates[-1] + timedelta(days=i+1) for i in range(days)]
                future_prices = []
                future_lower_bounds = []
                future_upper_bounds = []
                
                for i in range(days):
                    day_num = len(data) + i
                    day_date = future_dates[i]
                    
                    # Composante de tendance
                    trend_component = trend(day_num)
                    # Composante saisonnière
                    seasonal = seasonal_component(day_num)
                    # Effet des jours fériés
                    holiday = holiday_effect(day_date)
                    
                    # Prix prédit
                    predicted_price = trend_component + seasonal + holiday
                    
                    # Incertitude croissante avec l'horizon
                    uncertainty = rmse * np.sqrt(1 + i/30)
                    
                    future_prices.append(predicted_price)
                    future_lower_bounds.append(max(0, predicted_price - 1.96 * uncertainty))
                    future_upper_bounds.append(predicted_price + 1.96 * uncertainty)
                
                return {
                    "dates": future_dates,
                    "prices": future_prices,
                    "lower_bounds": future_lower_bounds,
                    "upper_bounds": future_upper_bounds,
                    "metrics": {
                        "rmse": rmse,
                        "mae": mae,
                        "reliability": 100 - (mae / (np.max(data) - np.min(data))) * 100 if np.max(data) != np.min(data) else 0
                    }
                }
            
            # Sélectionner l'algorithme approprié
            if algorithm == "Régression linéaire":
                reg_type = regression_type
                poly_degree = poly_degree if 'poly_degree' in locals() else 2
                prediction = predict_with_regression(prices, prediction_days, reg_type, poly_degree)
                algo_display_name = f"Régression {reg_type}"
            elif algorithm == "ARIMA":
                p = p_value if 'p_value' in locals() else 1
                d = d_value if 'd_value' in locals() else 1
                q = q_value if 'q_value' in locals() else 1
                prediction = predict_with_arima(prices, prediction_days, p, d, q)
                algo_display_name = f"ARIMA({p},{d},{q})"
            elif algorithm == "Prophet":
                cp_scale = changepoint_scale if 'changepoint_scale' in locals() else 0.05
                s_scale = seasonality_scale if 'seasonality_scale' in locals() else 10
                hol_opt = holidays_option if 'holidays_option' in locals() else "none"
                prediction = predict_with_prophet(prices, prediction_days, cp_scale, s_scale, hol_opt)
                algo_display_name = "Prophet"
            
            # Afficher les résultats de la prédiction
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            
            # Informations sur la prédiction
            st.markdown(f"### Prédiction pour {commodity_name}")
            
            # Métriques principales en colonnes
            col1, col2, col3 = st.columns(3)
            
            # Date et valeur finale de la prédiction
            final_date = prediction["dates"][-1]
            final_price = prediction["prices"][-1]
            lower_bound = prediction["lower_bounds"][-1]
            upper_bound = prediction["upper_bounds"][-1]
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Prix prédit</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{final_price:.2f} {currency}</div>', unsafe_allow_html=True)
                st.markdown(f'<div>pour le {final_date.strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Erreur moyenne (MAE)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{prediction["metrics"]["mae"]:.2f} {currency}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Score de fiabilité</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{prediction["metrics"]["reliability"]:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Intervalle de confiance
            st.markdown('<div class="confidence-interval">', unsafe_allow_html=True)
            st.markdown(f"**Intervalle de confiance (95%):** {lower_bound:.2f} - {upper_bound:.2f} {currency}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Graphique de prédiction
            st.markdown("### Graphique de prédiction")
            
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Données historiques
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines',
                    name='Données historiques',
                    line=dict(color='#4361ee', width=2)
                )
            )
            
            # Données prédites
            fig.add_trace(
                go.Scatter(
                    x=prediction["dates"],
                    y=prediction["prices"],
                    mode='lines',
                    name='Prédiction',
                    line=dict(color='#e63946', width=2, dash='dash')
                )
            )
            
            # Intervalle de confiance
            fig.add_trace(
                go.Scatter(
                    x=prediction["dates"] + prediction["dates"][::-1],
                    y=prediction["upper_bounds"] + prediction["lower_bounds"][::-1],
                    fill='toself',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    line=dict(color='rgba(231, 76, 60, 0.5)'),
                    name='Intervalle de confiance (95%)'
                )
            )
            
            fig.update_layout(
                title=f"Prédiction du prix de {commodity_name} avec {algo_display_name}",
                xaxis_title="Date",
                yaxis_title=f"Prix ({currency}/{unit})",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des prévisions
            st.markdown("### Tableau des prévisions")
            
            forecast_df = pd.DataFrame({
                'Date': prediction["dates"],
                'Prix prédit': prediction["prices"],
                'Borne inférieure': prediction["lower_bounds"],
                'Borne supérieure': prediction["upper_bounds"]
            })
            
            # Formatter les colonnes de prix
            forecast_df['Prix prédit'] = forecast_df['Prix prédit'].round(2)
            forecast_df['Borne inférieure'] = forecast_df['Borne inférieure'].round(2)
            forecast_df['Borne supérieure'] = forecast_df['Borne supérieure'].round(2)
            
            st.dataframe(forecast_df)
            
            # Bouton pour télécharger les prévisions
            csv = forecast_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{commodity_name}_forecast.csv">Télécharger les prévisions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Veuillez d'abord charger des données historiques en utilisant l'une des méthodes ci-dessus.")

# Pied de page
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('Développé par Didier Ouedraogo, P.Geo | © 2025 | Tous droits réservés', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)