"""NutriScan: CLIP real + Gemini. Prototipo sin métricas de test validadas."""
import hashlib
import io
import json
import os
from pathlib import Path
from model_io import trained_bundle, predict_resnet

import pandas as pd
import streamlit as st
import torch
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor, pipeline

st.set_page_config(page_title="NutriScan AI", page_icon="💪", layout="wide")
MODEL_ID = "openai/clip-vit-base-patch32"
CLASSES = ['Clear protein', 'Creatina', 'Preentreno']
LABELS = ['a container of clear protein supplement powder', 'a container of creatine supplement powder', 'a container of pre workout sports supplement powder']
CATALOG = {'Clear protein': ['Myprotein Clear Whey Protein Powder', 'Optimum Nutrition Clear Protein Powder'], 'Creatina': ['Optimum Nutrition Micronized Creatine Powder', 'Myprotein Creatine Monohydrate Powder'], 'Preentreno': ['Cellucor C4 Original', 'Optimum Nutrition Gold Standard Pre-Workout']}
INFO = {'Clear protein': {'id': 'protein', 'name': 'Proteína en polvo', 'label': 'a container or bag of whey protein powder supplement', 'description': 'Categoría de proteínas que se preparan como una bebida de aspecto claro. La categoría del dataset no confirma la fuente proteica exacta ni la fórmula.', 'compare': ['Fuente proteica', 'Proteína declarada por porción', 'Alérgenos', 'Ingredientes y versión'], 'caution': 'No equiparar automáticamente clear protein con cualquier proteína whey. Revisa el producto exacto.'}, 'Creatina': {'id': 'creatine', 'name': 'Creatina', 'label': 'a container or bag of creatine monohydrate sports supplement', 'description': 'La creatina participa en el suministro rápido de energía muscular. La evidencia depende del ingrediente y del contexto de ejercicio.', 'compare': ['Forma de creatina', 'Ingrediente único o mezcla', 'Cantidad por porción', 'Aditivos declarados'], 'caution': 'La clase creatine del dataset no confirma por sí sola que sea monohidrato puro. Revisa ingredientes y forma declarada.'}, 'Preentreno': {'id': 'preworkout', 'name': 'Preentreno', 'label': 'a container of pre workout energy supplement powder', 'description': 'Categoría comercial de mezclas para antes del ejercicio. Sus ingredientes y la presencia de estimulantes varían entre productos.', 'compare': ['Cafeína declarada', 'Otros estimulantes', 'Ingredientes y cantidades', 'Advertencias de la etiqueta'], 'caution': 'No asumir que dos preentrenos son equivalentes. La cafeína y las mezclas requieren especial atención.'}}
PRODUCTS = {'Clear protein': [{'name': 'Clear Whey Protein Powder', 'brand': 'Myprotein', 'feature': 'Proteína de suero para bebida clara', 'url': 'https://www.myprotein.com/p/sports-nutrition/clear-whey-protein-powder/12081395/'}, {'name': 'Clear Protein Powder', 'brand': 'Optimum Nutrition', 'feature': 'Proteína de suero para bebida clara, versión Reino Unido', 'url': 'https://www.optimumnutrition.com/en-gb/products/clear-protein-powder'}], 'Creatina': [{'id': 'on-creatine', 'category': 'creatine', 'name': 'Micronized Creatine Powder', 'brand': 'Optimum Nutrition', 'feature': 'Creatina monohidrato en polvo', 'url': 'https://www.optimumnutrition.com/en-us/products/creatine-monohydrate-micronized-powder'}, {'id': 'mp-creatine', 'category': 'creatine', 'name': 'Creatine Monohydrate Powder', 'brand': 'Myprotein', 'feature': 'Creatina monohidrato en polvo', 'url': 'https://www.myprotein.com/p/sports-nutrition/creatine-monohydrate-powder/10530050/'}], 'Preentreno': [{'id': 'c4', 'category': 'preworkout', 'name': 'C4 Original', 'brand': 'Cellucor', 'feature': 'Mezcla preentreno con cafeína', 'url': 'https://cellucor.com/products/c4-original'}, {'id': 'on-pre', 'category': 'preworkout', 'name': 'Gold Standard Pre-Workout', 'brand': 'Optimum Nutrition', 'feature': 'Mezcla comercial preentreno', 'url': 'https://www.optimumnutrition.com/en-us/products/gold-standard-pre-workout-powder'}]}

@st.cache_resource
def deployed_model():
    return trained_bundle(Path(__file__).parent / "modelo")

META, DEPLOYED_MODEL = deployed_model()
if META:
    CLASSES = META["classes"]
    LABELS = META["prompts"]


def secret(name):
    value = os.environ.get(name, "")
    if not value:
        try:
            value = st.secrets.get(name, "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    return value


@st.cache_resource
def load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    return model, processor


@st.cache_data(max_entries=24, show_spinner=False)
def classify(image_bytes):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    image.thumbnail((1600, 1600))
    if DEPLOYED_MODEL is not None:
        return predict_resnet(DEPLOYED_MODEL, image)
    model, processor = load_model()
    inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        scores = model(**inputs).logits_per_image.softmax(dim=-1)[0].tolist()
    return scores


@st.cache_resource
def load_language_model():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=-1)


def generate_local(context, instruction):
    generator = load_language_model()
    result = generator([
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ], max_new_tokens=350, do_sample=False, repetition_penalty=1.12)
    generated = result[0]["generated_text"]
    text = generated[-1]["content"] if isinstance(generated, list) else generated
    if not text.strip():
        raise ValueError("Respuesta vacía")
    return text


st.title("💪 NutriScan AI")
st.caption("Felipe Ruiz · Magdalena Gómez · Sebastián Fuentes | Clear protein, creatina y preentreno")
st.info("Clasificador de tres categorías. Un objeto ajeno también puede recibir una categoría: revisa siempre la etiqueta.")
st.write("Sube una foto frontal de un suplemento y compara su categoría con ayuda de IA.")
st.sidebar.header("Modelo y evaluación")
if META:
    st.sidebar.info("Modelo evaluado: " + META["selected"])
    st.sidebar.metric("Accuracy en test", f"{META['test']['accuracy']:.1%}")
    st.sidebar.metric("Balanced accuracy en test", f"{META['test']['balanced_accuracy']:.1%}")
    st.sidebar.caption(f"Prueba: {META['test_size']} imágenes. Estas métricas no son la confianza de esta foto.")
else:
    st.sidebar.info("Modo prototipo: CLIP sin ajuste específico. Aún no hay modelo exportado.")
    st.sidebar.write("Accuracy, balanced accuracy y matriz de confusión: pendientes de evaluación.")
st.sidebar.warning("Prototipo educativo. No verifica autenticidad, ingredientes ni seguridad de consumo.")
generation_mode = st.sidebar.radio("Motor de recomendaciones", ["IA local (sin clave)", "Gemini API"])
st.sidebar.caption("La IA local descarga un modelo de lenguaje y puede tardar; requiere memoria disponible.")
api_key = secret("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API key", type="password")
model_name = st.sidebar.text_input("Modelo Gemini habilitado en tu cuenta", value=secret("GEMINI_MODEL"), placeholder="ID de modelo vigente en Google AI Studio")
st.sidebar.link_button("Consultar modelos Gemini", "https://ai.google.dev/gemini-api/docs/models")

upload = st.file_uploader("Imagen JPG, PNG o WebP (máximo 10 MB)", type=["jpg", "jpeg", "png", "webp"])
if upload is None:
    st.info("Carga una imagen para comenzar.")
    st.stop()
data = upload.getvalue()
if len(data) > 10 * 1024 * 1024:
    st.error("El archivo supera 10 MB.")
    st.stop()
image_key = hashlib.sha256(data).hexdigest()
if st.session_state.get("image_key") != image_key:
    st.session_state.image_key = image_key
    for key in ("scores", "recommendation", "confirm", "product"):
        st.session_state.pop(key, None)

left, right = st.columns(2)
try:
    picture = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    picture.load()
    left.image(picture, caption="Imagen proporcionada", width="stretch")
except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
    st.error("No se pudo abrir una imagen válida.")
    st.stop()

if right.button("Analizar imagen", type="primary"):
    try:
        with st.spinner("Ejecutando CLIP. La primera carga descarga los pesos del modelo..."):
            st.session_state.scores = classify(data)
        st.session_state.pop("recommendation", None)
    except Exception:
        st.error("No se pudo cargar o ejecutar CLIP. Revisa conexión, dependencias y memoria del servidor.")

scores = st.session_state.get("scores")
if scores is None:
    st.stop()
index = max(range(len(scores)), key=scores.__getitem__)
category, confidence = CLASSES[index], scores[index]
ordered = sorted(scores, reverse=True)
uncertain = confidence < 0.70 or ordered[0] - ordered[1] < 0.15
right.metric("Categoría estimada", category)
right.metric("Confianza relativa", f"{confidence:.1%}")
right.caption("Softmax entre las etiquetas propuestas; no es una probabilidad calibrada de acertar.")
right.bar_chart(pd.DataFrame({"Categoría": CLASSES, "Puntuación": scores}).set_index("Categoría"), horizontal=True)
if uncertain:
    st.warning("Confianza baja o categorías muy próximas. Revisa la etiqueta antes de interpretar el resultado.")
if category == "Fuera de alcance":
    st.warning("Sube una imagen nítida de un solo envase. Esta etiqueta tampoco constituye un detector de rechazo validado.")
    st.stop()

st.subheader("Información del suplemento")
st.write(INFO[category]["description"])
st.info(INFO[category]["caution"])
st.write("**Características que debes comparar:** " + "; ".join(INFO[category]["compare"]))
st.subheader("Otros suplementos con características similares")
st.caption("La categoría no identifica marca, fórmula ni sabor. Una alternativa de la misma categoría no es necesariamente equivalente.")
confirmed = st.checkbox("He comprobado que la categoría coincide con la etiqueta", key="confirm")
selected = st.selectbox("Producto identificado manualmente (opcional)", ["No identificado"] + CATALOG[category], key="product")
for product in PRODUCTS[category]:
    full_name = product["brand"] + " " + product["name"]
    if full_name != selected:
        st.markdown(f"**{full_name}** — {product['feature']}")
        st.link_button("Ver ficha oficial de " + product["name"], product["url"])
if st.button("Generar recomendación con IA", disabled=not confirmed):
    st.session_state.pop("recommendation", None)
    if generation_mode == "Gemini API" and (not api_key or not model_name.strip()):
        st.error("Configura la clave y un modelo Gemini vigente antes de generar la recomendación.")
    else:
        context = {"categoria": category, "confianza_relativa": confidence,
                   "incierto": uncertain, "producto_declarado_por_usuario": selected,
                   "informacion": INFO[category]["description"],
                   "caracteristicas_a_comparar": INFO[category]["compare"],
                   "alternativas_catalogadas": [p for p in PRODUCTS[category] if p["brand"] + " " + p["name"] != selected]}
        instruction = (
            "Responde en español, máximo 200 palabras, con tres apartados: Información, Alternativas similares, Qué comparar. Usa sólo el contexto. Explica brevemente la categoría de suplemento. "
            "Compara únicamente los nombres del catálogo del contexto por su categoría común; no inventes "
            "ingredientes, dosis, certificaciones ni equivalencia. Si faltan alternativas dilo claramente. "
            "No confundas alternativas con productos complementarios ni recomiendes combinaciones. "
            "No des pautas de consumo, diagnóstico ni promesas de resultados. No afirmes que la foto identifica la marca. "
            "Si incierto es verdadero o confianza menor a 0.70, empieza pidiendo revisión humana de la etiqueta. "
            "Termina aclarando que la información es educativa y no sustituye asesoramiento profesional."
        )
        try:
            with st.spinner("Generando recomendaciones con " + generation_mode + "..."):
                if generation_mode == "IA local (sin clave)":
                    text = generate_local(context, instruction)
                else:
                    with genai.Client(api_key=api_key) as client:
                        response = client.models.generate_content(
                            model=model_name.strip(), contents=json.dumps(context, ensure_ascii=False),
                            config=types.GenerateContentConfig(system_instruction=instruction, temperature=0.2, max_output_tokens=1000),
                        )
                    text = response.text
                if not text or not text.strip():
                    raise ValueError("Respuesta vacía")
                st.session_state.recommendation = {"text": text, "product": selected, "engine": generation_mode}
        except Exception:
            st.error("No se pudo generar la recomendación. Para IA local revisa la descarga y memoria; para Gemini revisa clave, modelo y cuota. Puedes cambiar el motor en la barra lateral.")
rec = st.session_state.get("recommendation")
if rec and confirmed and rec["product"] == selected:
    st.caption("Generado por: " + rec["engine"])
    st.markdown(rec["text"])
    st.caption("Texto generado por IA: contrasta los detalles con la etiqueta y la fuente oficial.")
st.link_button("Información de referencia: NIH ODS", "https://ods.od.nih.gov/factsheets/ExerciseAndAthleticPerformance-HealthProfessional/")
