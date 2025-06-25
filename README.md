# Trafico-y-Valenbisi

# Valencia Tráfico Prediction

Una aplicación interactiva en Streamlit que combina datos de tráfico del ayuntamiento y de estaciones Valenbisi para:

- **Visualizar** el estado actual del tráfico y la disponibilidad de bicis en Valencia.  
- **Predecir** la probabilidad de congestión a 15 min usando una cadena de Markov.  
- **Comparar** la cadena de Markov con un modelo de regresión logística.  
- **Calcular** KPIs de movilidad (por ejemplo, % de estaciones vacías).

---

## 📁 Estructura del repositorio

├── app.py               # Aplicación principal de Streamlit
├── markov.py            # Módulo que construye la cadena de Markov y predice congestión
├── hist_traffic.csv     # Histórico de estados de tráfico (timestamp,estado)
├── requirements.txt     # Dependencias Python
├── README.md            # Documentación (este fichero)
└── data/                # (Opcional) Datos adicionales o scripts de ingestión

---

## ⚙️ Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/valencia-trafico-prediction.git
   cd valencia-trafico-prediction

2. Crear y activar un entorno virtual:

  python3 -m venv venv
  source venv/bin/activate    # Linux/macOS
  venv\Scripts\activate       # Windows

3. Instalar dependencias:
  pip install -r requirements.txt
 
