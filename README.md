[readme_del_repositorio.md](https://github.com/user-attachments/files/33686126/readme_del_repositorio.md)[Uplo# NutriScan AI: Detección y Clasificación Inteligente de Suplementos y Bebidas

 Integrantes del Equipo
- Magdalena Gómez
- Felipe Ruiz
- Sebastián Fuentes

---

 Descripción del Proyecto
NutriScan AI es una aplicación web interactiva desarrollada con Streamlit y desplegada mediante Google Colab y ngrok. Su objetivo principal es clasificar imágenes de suplementos y bebidas deportivas en tiempo real utilizando aprendizaje profundo (Deep Learning) e integrar recomendaciones contextuales personalizadas mediante **Inteligencia Artificial Generativa (Google Gemini)**.

---

Dataset Utilizado
- Fuente: [Roboflow Universe - Beverage Detection (`tribe-tagging/beverage-detection-stpvk`)](https://universe.roboflow.com/tribe-tagging/beverage-detection-stpvk/dataset/12)
- Volumen Total: 759 imágenes etiquetadas.
- División de Datos: 70% Entrenamiento, 15% Validación y 15% Prueba.
- Clases (10):
  1. `Clear Protein`
  2. `Colostrum`
  3. `Creatine`
  4. `Creatine Gummies`
  5. `Energy Drink`
  6. `Energy Stick`
  7. `Greens`
  8. `Greens Gummies`
  9. `Preworkout`
  10. `Soda`

---

 Selección y Justificación del Modelo
- Modelo Elegido: `EfficientNet-B0` (Preentrenado en ImageNet y afinado mediante Transfer Learning).
- **Justificación:** Se seleccionó por su alta eficiencia computacional y su bajo peso (~16 MB), ideal para despliegues ligeros en la nube sin comprometer la precisión.
- **Métricas de Rendimiento (Test Set):**
  - **Accuracy:** 93.8%
  - **Precision Balanceada:** 93.2%
- **Análisis del Error más Costoso:** Confundir productos con alto contenido de estimulantes (como `Preworkout` o `Energy Drink`) con productos libres de cafeína (como `Colostrum` o `Greens`). Se implementaron alertas de confianza mínima (<70%) para exigir revisión humana en estos casos.

---

## 🚀 Instrucciones de Ejecución Local / Google Colab

1. **Instalación de Dependencias:**
```bash
pip install -r requirements.txt
pip install google-genai pyngrok
```

2. **Configuración de Claves API:**
- Configura la variable de entorno para Gemini: `GOOGLE_API_KEY="tu_api_key"`
- Configura el token de ngrok: `ngrok.set_auth_token("tu_ngrok_token")`

3. **Ejecutar la Aplicación Web:**
```bash
streamlit run app.py
```

---

##  Estructura del Repositorio
```text
NutriScan-AI/
├── datos/                  # Script / referencias del dataset
├── ejemplos/               # Imágenes de prueba (.jpg, .png)
├── modelo/                 # Pesos exportados del modelo (.pt)
├── resultados/             # Gráficas de entrenamiento y matrices de confusión
├── app.py                  # Código fuente de la interfaz Streamlit
├── model_io.py             # Carga e inferencia del modelo PyTorch
├── prepare_dataset.py      # Descarga y preprocesamiento de Roboflow
├── training.py             # Script de entrenamiento de EfficientNet-B0
├── requirements.txt        # Dependencias del proyecto
├── ATTRIBUTION.md          # Licencia y atribución del dataset
└── README.md               # Documentación general del proyecto
```

---

##  Declaración del Uso de IA
En cumplimiento con los lineamientos académicos, declaramos que las herramientas de Inteligencia Artificial Generativa (ChatGPT / Google Gemini) fueron utilizadas como asistencia técnica para:
- Depuración y optimización de la estructura del código en `app.py`.
- Generación dinámica de prompts responsables y seguros para la consulta nutricional del usuario final.
- Redacción y maquetación de la documentación técnica (`README.md`).

---
*Proyecto académico para la evaluación de Inteligencia Artificial y Visión por Computadora.*ading readme_del_repositorio.md…]()
