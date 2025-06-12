# Sistema de Monitoramento de Qualidade do Ar
## Visão Geral

Este projeto implementa um sistema abrangente de monitoramento da qualidade do ar com análise de correlação meteorológica. O sistema calcula o Índice de Qualidade do Ar (AQI) com base nos padrões da EPA, gera dados sintéticos de qualidade do ar, modela a dispersão de poluentes, identifica episódios críticos de poluição, produz relatórios de saúde e cria um painel interativo.
Funcionalidades

    Cálculo de AQI: Computa o AQI para PM2.5, PM10, O3 e NO2 usando os pontos de corte da EPA
    Geração de Dados: Cria dados sintéticos realistas com correlações meteorológicas
    Modelo de Pluma Gaussiana: Simula a dispersão de poluentes com base nas condições atmosféricas
    Detecção de Episódios Críticos: Identifica períodos de má qualidade do ar (AQI > 150)
    Relatórios de Saúde: Gera recomendações de saúde pública com base nas tendências de qualidade do ar
    Painel Interativo: Visualiza séries temporais, correlações e padrões de rosa dos ventos
    Sistema de Alertas: Gera alertas em tempo real para condições de baixa qualidade do ar e meteorológicas

## Requisitos

    Python 3.8+
    Bibliotecas: numpy, pandas, matplotlib, seaborn, plotly

Instale as dependências usando:
bash
pip install numpy pandas matplotlib seaborn plotly
Uso

    Clone o repositório
    Instale as bibliotecas necessárias
    Execute o script principal:

bash
python air_quality_monitor.py

O script irá:

    Gerar 7 dias de dados de exemplo
    Calcular o AQI e identificar episódios críticos
    Produzir um relatório de saúde com recomendações
    Demonstrar o modelo de pluma gaussiana
    Gerar alertas para condições atuais
    Criar um painel interativo em HTML (air_quality_dashboard.html)

## Componentes Principais

    Classe AirQualityMonitor: Núcleo do sistema com métodos para processamento e visualização de dados
    generate_sample_data: Cria dados sintéticos de qualidade do ar e meteorológicos
    calculate_aqi: Calcula o AQI com base nas concentrações de poluentes
    gaussian_plume_model: Modela a dispersão de poluentes
    identify_critical_episodes: Detecta períodos de alta poluição
    generate_health_report: Produz recomendações de saúde pública
    create_dashboard: Gera visualizações interativas com Plotly
    generate_alerts: Cria alertas em tempo real para qualidade do ar

## Saída

    Saída no console com estatísticas, episódios críticos e recomendações de saúde
    Painel interativo em HTML com séries temporais, correlações e gráficos de rosa dos ventos
    Notificações de alerta para condições adversas

## Observações

    O sistema usa dados sintéticos para demonstração, mas pode ser adaptado para dados reais de sensores
    O painel é salvo como air_quality_dashboard.html e pode ser visualizado em qualquer navegador
    Os cálculos de AQI seguem os padrões da EPA para categorização precisa
    O modelo de pluma gaussiana inclui classes de estabilidade para modelagem realista de dispersão