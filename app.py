import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Pro - Clean Flash", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico: Realismo & Clean Flash")
st.write("Modo 1: Fotografia Realista | Modo 2: Desenho/Flash com limpeza extrema de fundo e hierarquia de linhas.")

# --- BARRA LATERAL (AJUSTES PRINCIPAIS) ---
st.sidebar.header("🎯 Tipo de Referência")
tipo_referencia = st.sidebar.radio(
    "O que você está convertendo?", 
    ["1. Fotografia (Realismo / Rostos)", "2. Desenho / Flash (Fundo Limpo)"]
)

st.sidebar.header("🎨 Cor da Linha")
cor_stencil = st.sidebar.radio("Escolha a cor para impressão:", ["Roxo", "Preto"])

# Variáveis globais seguras
sensibilidade_contorno = 40
espessura_silhueta = 2
distancia_sombras = 5
limite_branco = 200
t1, t2, t3 = 45, 100, 175

# --- CONTROLES DINÂMICOS ---
st.sidebar.markdown("---")
if "Fotografia" in tipo_referencia:
    st.sidebar.header("🛠️ Controles de Fotografia")
    sensibilidade_contorno = st.sidebar.slider("Sensibilidade dos Detalhes", 10, 150, 40)
    espessura_silhueta = st.sidebar.slider("Espessura dos Contornos Principais", 1, 5, 2)
    distancia_sombras = st.sidebar.slider("Distância Segura (Tracejado)", 1, 9, 5)
    
    with st.sidebar.expander("⚙️ Ajustes Avançados de Sombra"):
        t1 = st.slider("Zona 1: Áreas Muito Escuras", 10, 80, 45)
        t2 = st.slider("Zona 2: Tons Escuros / Médios", 81, 150, 100)
        t3 = st.slider("Zona 3: Tons Claros (Transições)", 151, 230, 175)
else:
    st.sidebar.header("🛠️ Controles de Desenho Limpo")
    st.sidebar.info("Este modo remove sujeiras do papel e cria um estêncil hierárquico limpo.")
    limite_branco = st.sidebar.slider("Limpeza de Fundo (Mata-Borrao)", 100, 255, 200, help="Diminua se o desenho estiver apagando. Aumente para matar a sujeira do papel.")
    sensibilidade_contorno = st.sidebar.slider("Sensibilidade dos Detalhes (Internos)", 10, 150, 50)
    espessura_silhueta = st.sidebar.slider("Espessura Externa", 1, 5, 2)
    distancia_sombras = st.sidebar.slider("Distância Segura (Tracejado interno)", 1, 9, 3)

# --- MOTOR DE PROCESSAMENTO ---
def gerar_stencil(img, tipo_ref, cor, sens, esp_silhueta, dist, lim_branco, t1, t2, t3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if "Desenho" in tipo_ref:
        # PRE-PROCESSAMENTO: Fundo Branco Absoluto
        # Tudo que for mais claro que o "lim_branco" vira branco puro (255)
        _, gray = cv2.threshold(gray, lim_branco, 255, cv2.THRESH_TRUNC)
        # Normaliza a imagem para forçar o contraste
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # MOTOR PRINCIPAL (Hierarquia + Tracejado)
    blur_major = cv2.GaussianBlur(gray, (5, 5), 0)
    major_edges = cv2.Canny(blur_major, sens * 1.5, sens * 3)
    
    if esp_silhueta > 1:
        kernel_bold = np.ones((esp_silhueta, esp_silhueta), np.uint8)
        major_edges_bold = cv2.dilate(major_edges, kernel_bold, iterations=1)
    else:
        major_edges_bold = major_edges
        
    blur_detail = cv2.GaussianBlur(gray, (3, 3), 0)
    all_edges = cv2.Canny(blur_detail, sens, sens * 2)
    
    kernel_sub = np.ones((esp_silhueta + 2, esp_silhueta + 2), np.uint8)
    major_dilated_for_sub = cv2.dilate(major_edges, kernel_sub, iterations=1)
    detail_edges = cv2.bitwise_and(all_edges, cv2.bitwise_not(major_dilated_for_sub))
    
    combined_solid = cv2.bitwise_or(major_edges_bold, detail_edges)
    
    # Sombras / Tons Internos
    blurred_shadows = cv2.GaussianBlur(gray, (7, 7), 0)
    mask1 = (blurred_shadows < t1).astype(np.uint8) * 255
    mask2 = (blurred_shadows < t2).astype(np.uint8) * 255
    mask3 = (blurred_shadows < t3).astype(np.uint8) * 255
    
    edges1 = cv2.Canny(mask1, 100, 200)
    edges2 = cv2.Canny(mask2, 100, 200)
    edges3 = cv2.Canny(mask3, 100, 200)
    shadow_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    kernel_dist = np.ones((dist, dist), np.uint8)
    combined_dilated = cv2.dilate(combined_solid, kernel_dist, iterations=1)
    shadow_edges_pure = cv2.bitwise_and(shadow_edges, cv2.bitwise_not(combined_dilated))
    
    y_indices, x_indices = np.indices((h, w))
    dash_mask = (((x_indices + y_indices) % 12) < 6).astype(np.uint8) * 255
    dashed_shadows = cv2.bitwise_and(shadow_edges_pure, dash_mask)
    
    final_edges = cv2.bitwise_or(combined_solid, dashed_shadows)
        
    # Mapeamento de Cor
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    if cor == "Roxo":
        output_rgb[final_edges > 0] = [138, 43, 226] # Roxo Hectográfico
    else:
        output_rgb[final_edges > 0] = [0, 0, 0] # Preto
        
    return output_rgb

# --- INTERFACE FLUXO DO USUÁRIO ---
tab1, tab2 = st.tabs(["📷 Câmera", "📂 Galeria"])
imagem_subida = None

with tab1:
    camera_pic = st.camera_input("Capture a imagem de referência")
    if camera_pic:
        imagem_subida = camera_pic

with tab2:
    arquivo_pic = st.file_uploader("Selecione um arquivo (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    if arquivo_pic:
        imagem_subida = arquivo_pic

if imagem_subida is not None:
    file_bytes = np.asarray(bytearray(imagem_subida.read()), dtype=np.uint8)
    img_opencv = cv2.imdecode(file_bytes, 1)
    
    resultado = gerar_stencil(
        img_opencv, tipo_referencia, cor_stencil, 
        sensibilidade_contorno, espessura_silhueta, distancia_sombras, limite_branco,
        t1, t2, t3
    )
    
    img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_display_orig, caption="Original", use_container_width=True)
    with c2:
        st.image(resultado, caption=f"Estêncil {cor_stencil}", use_container_width=True)
            
    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", resultado_bgr)
    io_buf = io.BytesIO(buffer)
    
    st.success("Estêncil gerado com sucesso!")
    st.download_button(
        label=f"⬇️ Baixar Estêncil {cor_stencil}",
        data=io_buf,
        file_name=f"stencil_{cor_stencil.lower()}.jpg",
        mime="image/jpeg"
    )
