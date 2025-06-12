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