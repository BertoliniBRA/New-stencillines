import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Pro - Studio", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico: Studio Pro")
st.write("Crie decalques perfeitos para Tattoo. Escolha entre o Motor de Posterização (Fotografia) ou o Motor de Pontilhismo (Flash/Desenho).")

# --- UPLOAD DE IMAGEM NA PÁGINA PRINCIPAL ---
st.markdown("---")
imagem_subida = st.file_uploader("📂 Faça o upload da sua referência (JPG ou PNG)", type=['jpg', 'jpeg', 'png'])

# --- BARRA LATERAL (MENU DE CONFIGURAÇÕES) ---
st.sidebar.header("🎯 Modo de Trabalho")
tipo_referencia = st.sidebar.radio(
    "Escolha o motor do decalque:", 
    ["1. Fotografia (Posterização / Realismo)", "2. Desenho / Flash (Pontilhismo)"]
)

st.sidebar.header("🎨 Cor da Impressão")
cor_stencil = st.sidebar.radio("Cor do Estêncil:", ["Roxo Hectográfico", "Preto"])

st.sidebar.markdown("---")

# Variáveis e Controles Condicionais
if "Fotografia" in tipo_referencia:
    st.sidebar.header("🛠️ Controles de Posterização")
    st.sidebar.info("Simula a posterização do Photoshop: agrupa a foto em níveis de sombra e desenha as bordas de cada tom.")
    
    n_niveis = st.sidebar.slider(
        "Níveis de Sombra (Posterizar)", 2, 8, 4,
        help="Quantidade de tons em que a foto será dividida. Menos níveis = linhas mais limpas. Mais níveis = mapeamento mais detalhado."
    )
    espessura_foto = st.sidebar.slider("Espessura das Linhas", 1, 3, 1)
    limpeza_foto = st.sidebar.slider("Limpar Ruídos da Foto", 0, 30, 5)

else:
    st.sidebar.header("🛠️ Controles de Desenho / Flash")
    espessura_linha = st.sidebar.slider("Espessura do Traço Principal", 1, 5, 2)
    corte_sombra = st.sidebar.slider("Captura de Sombra", 50, 250, 180)
    densidade_pontos = st.sidebar.slider("Espaçamento dos Pontos (Pontilhismo)", 2, 8, 4)
    limpeza_flash = st.sidebar.slider("Remover Sujeiras", 0, 50, 0)

# --- MOTOR DE PROCESSAMENTO ---
def processar_posterizacao(gray, niveis, espessura, limpeza):
    # 1. Suavização para evitar linhas 'serrilhadas'
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Posterização (Quantização de Tons - Efeito Photoshop)
    # Divide os 256 tons de cinza no número de níveis escolhido
    fator = 255.0 / (niveis - 1)
    posterizado = np.round(blur / fator) * fator
    posterizado = posterizado.astype(np.uint8)
    
    # 3. Extração das Bordas entre os níveis de tom (Sobel/Canny)
    bordas = cv2.Canny(posterizado, 10, 50)
    
    # 4. Ajuste de Espessura
    if espessura > 1:
        kernel = np.ones((espessura, espessura), np.uint8)
        bordas = cv2.dilate(bordas, kernel, iterations=1)
        
    # 5. Filtro de Limpeza
    if limpeza > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bordas, connectivity=8)
        mask = np.zeros_like(bordas)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= limpeza:
                mask[labels == i] = 255
        bordas = mask
        
    return bordas

def processar_flash(gray, esp_linha, corte_sombra, densidade, limpeza):
    h, w = gray.shape
    linhas = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    if esp_linha > 2:
        kernel = np.ones((esp_linha - 1, esp_linha - 1), np.uint8)
        linhas = cv2.dilate(linhas, kernel, iterations=1)
    elif esp_linha == 1:
        kernel = np.ones((2, 2), np.uint8)
        linhas = cv2.erode(linhas, kernel, iterations=1)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mascara_sombra = cv2.threshold(gray_blur, corte_sombra, 255, cv2.THRESH_BINARY_INV)
    mascara_sombra = cv2.bitwise_and(mascara_sombra, cv2.bitwise_not(linhas))
    
    y_indices, x_indices = np.indices((h, w))
    malha_pontos = (((x_indices % densidade == 0) & (y_indices % densidade == 0))).astype(np.uint8) * 255
    sombras_pontilhadas = cv2.bitwise_and(malha_pontos, mascara_sombra)
    
    stencil_combinado = cv2.bitwise_or(linhas, sombras_pontilhadas)
    
    if limpeza > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stencil_combinado, connectivity=8)
        mascara_limpa = np.zeros_like(stencil_combinado)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= limpeza:
                mascara_limpa[labels == i] = 255
        stencil_combinado = mascara_limpa
        
    return stencil_combinado

# --- RENDERIZAÇÃO DA INTERFACE ---
if imagem_subida is not None:
    file_bytes = np.asarray(bytearray(imagem_subida.read()), dtype=np.uint8)
    img_opencv = cv2.imdecode(file_bytes, 1)
    
    if img_opencv is not None:
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        
        # Seleção do Motor
        if "Fotografia" in tipo_referencia:
            final_edges = processar_posterizacao(gray, n_niveis, espessura_foto, limpeza_foto)
        else:
            final_edges = processar_flash(gray, espessura_linha, corte_sombra, densidade_pontos, limpeza_flash)
            
        # Cor de saída
        h, w = gray.shape
        output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        if "Roxo" in cor_stencil:
            output_rgb[final_edges > 0] = [138, 43, 226]
        else:
            output_rgb[final_edges > 0] = [0, 0, 0]
            
        # Exibição Lado a Lado
        st.write("---")
        col1, col2 = st.columns(2)
        
        img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        with col1:
            st.subheader("🖼️ Imagem Original")
            st.image(img_display_orig, use_container_width=True)
            
        with col2:
            st.subheader(f"✨ Estêncil Técnico ({cor_stencil})")
            st.image(output_rgb, use_container_width=True)
            
        # Download
        resultado_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", resultado_bgr)
        io_buf = io.BytesIO(buffer)
        
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="⬇️ Baixar Estêncil Pronto",
            data=io_buf,
            file_name="stencil_studio.jpg",
            mime="image/jpeg"
        )
