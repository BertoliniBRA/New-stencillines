import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Pro - Hierarquia", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico: Hierarquia de Linhas")
st.write("Estêncil com Contornos Externos, Detalhes Finos e Fusão Antiborrão para Minimalismo.")

# --- BARRA LATERAL (AJUSTES PRINCIPAIS) ---
st.sidebar.header("🎨 Estilo do Estêncil")
cor_stencil = st.sidebar.radio("Cor da Linha:", ["Vermelho", "Preto"])

st.sidebar.header("🛠️ Controles de Traço")
sensibilidade_contorno = st.sidebar.slider("Sensibilidade dos Detalhes", 10, 150, 40, help="Controla a quantidade de detalhes finos internos.")
espessura_silhueta = st.sidebar.slider("Espessura dos Contornos Principais", 1, 5, 2, help="Deixa as formas principais mais grossas.")

# NOVO: Controle de Fusão para Linhas Duplas
st.sidebar.markdown("---")
st.sidebar.markdown("**🛡️ Limpeza de Traço:**")
fusao_linhas = st.sidebar.slider("Fusão (Evitar Linha Dupla)", 1, 9, 3, help="Derrete linhas duplas paralelas transformando-as em um traço único. Aumente para desenhos minimalistas.")
distancia_sombras = st.sidebar.slider("Distância Segura (Tracejado)", 1, 9, 5, help="Afasta as sombras das linhas contínuas.")

# --- BARRA LATERAL (AJUSTES AVANÇADOS DE SOMBRA) ---
with st.sidebar.expander("⚙️ Ajustes Avançados de Sombra"):
    st.caption("Ajuste as 4 zonas tonais.")
    t1 = st.slider("Zona 1: Áreas Muito Escuras", 10, 80, 45)
    t2 = st.slider("Zona 2: Tons Escuros / Médios", 81, 150, 100)
    t3 = st.slider("Zona 3: Tons Claros (Transições)", 151, 230, 175)

# --- MOTOR DE PROCESSAMENTO (HIERARQUIA) ---
def gerar_stencil_hierarquia(img, sens_canny, thresh1, thresh2, thresh3, dist, espessura, fusao, cor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CONTORNOS PRINCIPAIS (Formas Maiores)
    blur_major = cv2.GaussianBlur(gray, (5, 5), 0)
    major_edges = cv2.Canny(blur_major, sens_canny * 1.5, sens_canny * 3)
    
    if espessura > 1:
        kernel_bold = np.ones((espessura, espessura), np.uint8)
        major_edges_bold = cv2.dilate(major_edges, kernel_bold, iterations=1)
    else:
        major_edges_bold = major_edges
    
    # 2. DETALHES INTERNOS (Linha Fina)
    blur_detail = cv2.GaussianBlur(gray, (3, 3), 0)
    all_edges = cv2.Canny(blur_detail, sens_canny, sens_canny * 2)
    
    # Subtração para ficar só com detalhes
    kernel_sub = np.ones((espessura + 2, espessura + 2), np.uint8)
    major_dilated_for_sub = cv2.dilate(major_edges, kernel_sub, iterations=1)
    detail_edges = cv2.bitwise_and(all_edges, cv2.bitwise_not(major_dilated_for_sub))
    
    # 3. COMBINAÇÃO DE TRAÇOS E FUSÃO (O SEGREDO DA LINHA DUPLA)
    combined_solid = cv2.bitwise_or(major_edges_bold, detail_edges)
    
    # Se o usuário ativou a fusão, aplicamos o Morphological Close
    if fusao > 1:
        kernel_close = np.ones((fusao, fusao), np.uint8)
        combined_solid = cv2.morphologyEx(combined_solid, cv2.MORPH_CLOSE, kernel_close)
    
    # 4. MAPEAMENTO DE SOMBRAS (Tracejado)
    blurred_shadows = cv2.GaussianBlur(gray, (7, 7), 0)
    mask1 = (blurred_shadows < thresh1).astype(np.uint8) * 255
    mask2 = (blurred_shadows < thresh2).astype(np.uint8) * 255
    mask3 = (blurred_shadows < thresh3).astype(np.uint8) * 255
    
    edges1 = cv2.Canny(mask1, 100, 200)
    edges2 = cv2.Canny(mask2, 100, 200)
    edges3 = cv2.Canny(mask3, 100, 200)
    shadow_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # ZONA DE EXCLUSÃO (Campo de Força)
    kernel_dist = np.ones((dist, dist), np.uint8)
    combined_dilated = cv2.dilate(combined_solid, kernel_dist, iterations=1)
    
    shadow_edges_pure = cv2.bitwise_and(shadow_edges, cv2.bitwise_not(combined_dilated))
    
    # EFEITO TRACEJADO
    h, w = gray.shape
    y_indices, x_indices = np.indices((h, w))
    dash_mask = (((x_indices + y_indices) % 12) < 6).astype(np.uint8) * 255
    dashed_shadows = cv2.bitwise_and(shadow_edges_pure, dash_mask)
    
    # 5. JUNÇÃO FINAL
    final_edges = cv2.bitwise_or(combined_solid, dashed_shadows)
    
    # 6. APLICAÇÃO DA COR
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    if cor == "Vermelho":
        output_rgb[final_edges > 0] = [255, 0, 0]
    else:
        output_rgb[final_edges > 0] = [0, 0, 0]
        
    return output_rgb

# --- INTERFACE ---
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
    
    resultado = gerar_stencil_hierarquia(
        img_opencv, sensibilidade_contorno, t1, t2, t3, 
        distancia_sombras, espessura_silhueta, fusao_linhas, cor_stencil
    )
    
    img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_display_orig, caption="Original", use_container_width=True)
    with c2:
        st.image(resultado, caption=f"Hierarquia {cor_stencil}", use_container_width=True)
            
    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", resultado_bgr)
    io_buf = io.BytesIO(buffer)
    
    st.success("Estêncil com Hierarquia gerado!")
    st.download_button(
        label=f"⬇️ Baixar Estêncil {cor_stencil}",
        data=io_buf,
        file_name=f"stencil_hierarquia_{cor_stencil.lower()}.jpg",
        mime="image/jpeg"
    )
