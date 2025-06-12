# Importação das dependencias
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class airQualityMonitor:
    def __init__(self):
        # Padrões da EPA para Índice de Qualidade do Ar (AQI)
        self.aqi_breakpoints = {
            'PM2.5': [
                (0, 12.0, 0, 50),      # Bom
                (12.1, 35.4, 51, 100), # Moderado
                (35.5, 55.4, 101, 150), # Insalubre para grupos sensíveis
                (55.5, 150.4, 151, 200), # Insalubre
                (150.5, 250.4, 201, 300), # Muito insalubre
                (250.5, 500.4, 301, 500)  # Perigoso
            ],
            'PM10': [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ],
            'O3': [
                (0, 0.054, 0, 50),
                (0.055, 0.070, 51, 100),
                (0.071, 0.085, 101, 150),
                (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300),
                (0.201, 0.604, 301, 500)
            ],
            'NO2': [
                (0, 0.053, 0, 50),
                (0.054, 0.100, 51, 100),
                (0.101, 0.360, 101, 150),
                (0.361, 0.649, 151, 200),
                (0.650, 1.249, 201, 300),
                (1.250, 2.049, 301, 500)
            ]
        }
        
        # Categorias de qualidade do ar
        self.aqi_categories = {
            (0, 50): ('Bom', 'green'),
            (51, 100): ('Moderado', 'yellow'),
            (101, 150): ('Insalubre para Grupos Sensíveis', 'orange'),
            (151, 200): ('Insalubre', 'red'),
            (201, 300): ('Muito Insalubre', 'purple'),
            (301, 500): ('Perigoso', 'maroon')
        }
    
    def generate_sample_data(self, days=30):
        """Gera dados sintéticos de qualidade do ar e meteorologia"""
        
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='H')
        
        n_points = len(dates)
        
        # Dados meteorológicos
        temperature = 20 + 10 * np.sin(np.arange(n_points) * 2 * np.pi / 24) + np.random.normal(0, 2, n_points)
        humidity = 60 + 20 * np.sin(np.arange(n_points) * 2 * np.pi / 24 + np.pi) + np.random.normal(0, 5, n_points)
        wind_speed = np.abs(5 + 3 * np.sin(np.arange(n_points) * 2 * np.pi / 168) + np.random.normal(0, 2, n_points))
        wind_direction = (180 + 60 * np.sin(np.arange(n_points) * 2 * np.pi / 168)) % 360
        pressure = 1013 + 10 * np.sin(np.arange(n_points) * 2 * np.pi / 168) + np.random.normal(0, 3, n_points)
        
        # Efeito da inversão térmica (pior qualidade do ar com baixa velocidade do vento)
        inversion_factor = 1 / (wind_speed + 0.5)
        
        # Dados de poluentes (correlacionados com condições meteorológicas)
        pm25 = np.abs(15 + 20 * inversion_factor + np.random.normal(0, 5, n_points))
        pm10 = pm25 * 1.5 + np.random.normal(0, 10, n_points)
        
        # O3 correlacionado com temperatura e radiação solar
        solar_factor = np.maximum(0, np.sin(np.arange(n_points) * 2 * np.pi / 24))
        o3 = 0.03 + 0.05 * (temperature / 30) * solar_factor + np.random.normal(0, 0.01, n_points)
        o3 = np.maximum(0, o3)
        
        # NO2 correlacionado com tráfego (picos nas horas de pico)
        traffic_factor = 1 + 0.5 * (np.sin((np.arange(n_points) % 24 - 8) * np.pi / 6) + 
                                   np.sin((np.arange(n_points) % 24 - 18) * np.pi / 6))
        no2 = 0.02 + 0.03 * traffic_factor * inversion_factor + np.random.normal(0, 0.01, n_points)
        no2 = np.maximum(0, no2)
        
        # Criação do DataFrame
        data = pd.DataFrame({
            'datetime': dates,
            'temperature': temperature,
            'humidity': np.clip(humidity, 0, 100),
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'pressure': pressure,
            'PM2.5': np.maximum(0, pm25),
            'PM10': np.maximum(0, pm10),
            'O3': o3,
            'NO2': no2
        })
        
        return data
    