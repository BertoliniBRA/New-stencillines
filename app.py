import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Pro - Studio", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico: Motor de Pontilhismo")
st.write("Converte sombras em pontilhismo ajustável e permite controle total sobre a espessura das linhas principais.")

# --- BARRA LATERAL (AJUSTES) ---
st.sidebar.header("🎨 Configurações de Impressão")
cor_stencil = st.sidebar.radio("Cor do Estêncil:", ["Roxo Hectográfico", "Preto"])

st.sidebar.markdown("---")
st.sidebar.header("1️⃣ Linhas Principais")
espessura_linha = st.sidebar.slider(
    "Espessura do Traço", 1, 5, 2,
    help="1 = Afina os traços originais. 3 a 5 = Engrossa os traços."
)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ Sombras (Pontilhismo)")
corte_sombra = st.sidebar.slider(
    "Captura de Sombra", 100, 250, 220,
    help="Define o que o app considera 'sombra'. Valores mais altos pegam até os cinzas mais claros."
)
densidade_pontos = st.sidebar.slider(
    "Espaçamento dos Pontos", 2, 8, 4,
    help="2 = Pontilhismo muito denso/escuro. 8 = Pontilhismo bem espaçado/claro."
)

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ Limpeza de Detalhes")
limpeza = st.sidebar.slider(
    "Remover Sujeiras (Filtro)", 0, 50, 5,
    help="0 = Mantém todos os pontinhos. Aumente para apagar ruídos e sujeiras soltas no papel."
)

# --- MOTOR DE PROCESSAMENTO ---
def gerar_stencil(img, cor, esp_linha, corte_sombra, densidade, limpeza):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. EXTRAIR LINHAS PRINCIPAIS (Limiar Adaptativo para achar contornos fortes)
    linhas = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Ajuste de espessura das linhas
    if esp_linha > 2:
        kernel = np.ones((esp_linha - 1, esp_linha - 1), np.uint8)
        linhas = cv2.dilate(linhas, kernel, iterations=1)
    elif esp_linha == 1:
        kernel = np.ones((2, 2), np.uint8)
        linhas = cv2.erode(linhas, kernel, iterations=1)

    # 2. EXTRAIR SOMBRAS E TRANSFORMAR EM PONTILHISMO
    # Pega tudo que for mais escuro que o 'corte_sombra'
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mascara_sombra = cv2.threshold(gray_blur, corte_sombra, 255, cv2.THRESH_BINARY_INV)
    
    # Subtrai as linhas principais para a sombra não engrossar o contorno
    mascara_sombra = cv2.bitwise_and(mascara_sombra, cv2.bitwise_not(linhas))
    
    # Cria a malha de pontilhismo (Grid)
    y_indices, x_indices = np.indices((h, w))
    # Uma malha intercalada simples baseada na densidade escolhida
    malha_pontos = (((x_indices % densidade == 0) & (y_indices % densidade == 0))).astype(np.uint8) * 255
    
    # Aplica o pontilhismo apenas onde existe sombra
    sombras_pontilhadas = cv2.bitwise_and(malha_pontos, mascara_sombra)
    
    # 3. JUNTAR LINHAS + PONTILHISMO
    stencil_combinado = cv2.bitwise_or(linhas, sombras_pontilhadas)
    
    # 4. LIMPEZA (Remover micro-sujeiras com Base de Área)
    if limpeza > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stencil_combinado, connectivity=8)
        mascara_limpa = np.zeros_like(stencil_combinado)
        for i in range(1, num_labels):
            # Se a área do ponto/traço for maior que o nível de limpeza, mantém.
            if stats[i, cv2.CC_STAT_AREA] >= limpeza:
                mascara_limpa[labels == i] = 255
        stencil_combinado = mascara_limpa

    # --- MAPEAMENTO DE COR ---
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    if "Roxo" in cor:
        output_rgb[stencil_combinado > 0] = [138, 43, 226] 
    else:
        output_rgb[stencil_combinado > 0] = [0, 0, 0] 
        
    return output_rgb

# --- INTERFACE FLUXO DO USUÁRIO ---
st.write("---")
col1, col2 = st.columns([1, 1])

imagem_subida = st.sidebar.file_uploader("📂 Suba seu desenho (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if imagem_subida is not None:
    # Leitura da imagem
    file_bytes = np.asarray(bytearray(imagem_subida.read()), dtype=np.uint8)
    img_opencv = cv2.imdecode(file_bytes, 1)
    
    # Roda o motor
    resultado = gerar_stencil(
        img_opencv, cor_stencil, espessura_linha, 
        corte_sombra, densidade_pontos, limpeza
    )
    
    # Converte a original para RGB para mostrar no Streamlit
    img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
    with col1:
        st.subheader("🖼️ Original")
        st.image(img_display_orig, use_container_width=True)
        
    with col2:
        st.subheader(f"✨ Estêncil ({cor_stencil})")
        st.image(resultado, use_container_width=True)
            
    # Botão de Download
    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", resultado_bgr)
    io_buf = io.BytesIO(buffer)
    
    st.sidebar.markdown("---")
    st.sidebar.success("Estêncil gerado!")
    st.sidebar.download_button(
        label=f"⬇️ Baixar Imagem Pronta",
        data=io_buf,
        file_name=f"stencil_pontilhismo.jpg",
        mime="image/jpeg"
    )
else:
    st.info("👈 Faça o upload de uma imagem na barra lateral para começar a edição.")
