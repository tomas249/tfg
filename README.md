# Sistema de Clasificación de Neuronas - TFG

**UOC - Trabajo Final de Grado en Ciencia de Datos Aplicada**  
**Autor:** Tomas Mirchev

## Demo y Recursos Online

- **Demo interactiva:** [tfg.tomastm.com](https://tfg.tomastm.com)
- **API REST:** [api.tfg.tomastm.com](https://api.tfg.tomastm.com)

## Descripción

Sistema de predicción de propiedades neuronales basado en Graph Neural Networks (GNN) para la clasificación de neuronas del conectoma de *Drosophila melanogaster*. El sistema predice múltiples características neuronales incluyendo superclase funcional, tipo de neurotransmisor, etiquetas de conectividad y tipo celular primario.

## Arquitectura del Sistema

**Stack Tecnológico**
- **Backend:** Flask (Python) + PostgreSQL + RabbitMQ
- **Worker:** Python + PyTorch + PostgreSQL + RabbitMQ
- **Frontend:** React + TailwindCSS
- **Orquestación:** Docker Compose

## Estructura del Proyecto

```
.
├── api/                          # Servicio API REST
│   ├── Dockerfile.api            # Contenedor para la API
│   ├── app.py                    # Aplicación Flask principal
│   └── requirements.txt          # Dependencias Python del API
│
├── worker/                       # Servicio de procesamiento asíncrono
│   ├── Dockerfile.worker         # Contenedor para el worker
│   ├── worker.py                 # Worker principal con RabbitMQ
│   ├── requirements.txt          # Dependencias Python del worker
│   └── common/                   # Recursos compartidos
│       ├── models.py             # Definición de modelos PyTorch
│       └── neuron_model_bundle.pt  # Modelo GNN entrenado
│
├── frontend/                     # Aplicación web React
│   ├── Dockerfile                # Contenedor para frontend
│   ├── src/                      # Código fuente React
│   │   ├── App.jsx               # Componente principal
│   │   └── constants.js          # Constantes del frontend
│   └── package.json              # Configuración npm
│
├── notebooks/                    # Análisis y desarrollo
│   ├── DataAnalysis.ipynb        # Análisis exploratorio de datos
│   └── neuron_gnn_pipeline.py    # Pipeline de entrenamiento del modelo
│
└── docker-compose.yml            # Orquestación completa del sistema
```

## Despliegue

**Inicio rápido**
```bash
# Clonar el repositorio
git clone https://github.com/tomas249/tfg
cd tfg

# Iniciar todos los servicios
docker-compose up --build

# Acceder a la aplicación
# Frontend: http://localhost:4000
# API: http://localhost:4001
```

## Datos

El sistema utiliza datos del proyecto [FlyWire](https://flywire.ai) disponibles a través de [FlyWire Codex](https://codex.flywire.ai/api/download), procesando información de aproximadamente 140,000 neuronas del conectoma completo de *Drosophila melanogaster*. Los datos incluyen características morfológicas, de conectividad, clasificaciones funcionales y neuroquímicas.

**Fuente de datos:** FlyWire public release version 783 (snapshot octubre 2023) bajo licencia CC BY-NC 4.0. Los datos fueron generados por el consorcio FlyWire con reconstrucciones neuronales validadas por la comunidad científica global y predicciones de neurotransmisores utilizando métodos de inteligencia artificial.

## Contexto

Este proyecto se desarrolla con fines académicos como parte del Trabajo Final de Grado en Ciencia de Datos Aplicada de la UOC.
