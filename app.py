import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Técnico Pro", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico Avançado")
st.write("Estêncil com controle de espessura, distanciamento e opção de cor (Vermelho/Preto).")

# --- BARRA LATERAL (AJUSTES PRINCIPAIS) ---
st.sidebar.header("🎨 Estilo do Estêncil")
cor_stencil = st.sidebar.radio("Cor da Linha:", ["Vermelho", "Preto"])

st.sidebar.header("🛠️ Controles Principais")
sensibilidade_contorno = st.sidebar.slider("Sensibilidade do Contorno", 10, 150, 50, help="Menor valor pega mais detalhes internos.")

# NOVO: Controle de Espessura
espessura_linha = st.sidebar.slider("Espessura da Linha Principal", 1, 5, 1, help="1 é o mais fino, 5 é o mais grosso. Afeta apenas o traço contínuo.")

# MANTIDO: Controle de Distância
distancia_sombras = st.sidebar.slider("Distância Segura (Evita Linha Dupla)", 1, 9, 5, help="Afasta o tracejado da linha principal.")

# --- BARRA LATERAL (AJUSTES AVANÇADOS ESCONDIDOS) ---
# Isso limpa a interface. O usuário só abre se precisar!
with st.sidebar.expander("⚙️ Ajustes Avançados de Sombra"):
    st.caption("Ajuste fino de onde começam e terminam as sombras.")
    t1 = st.slider("Zona 1: Áreas Muito Escuras", 10, 80, 45)
    t2 = st.slider("Zona 2: Tons Escuros / Médios", 81, 150, 100)
    t3 = st.slider("Zona 3: Tons Claros (Transições)", 151, 230, 175)

# --- MOTOR DE PROCESSAMENTO ---
def gerar_stencil_tecnico(img, sens_canny, thresh1, thresh2, thresh3, dist, espessura, cor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CONTORNO PRINCIPAL (Linha Sólida)
    blurred_main = cv2.GaussianBlur(gray, (3, 3), 0)
    main_edges = cv2.Canny(blurred_main, sens_canny, sens_canny * 2)
    
    # APLICAR ESPESSURA NA LINHA PRINCIPAL
    if espessura > 1:
        kernel_espessura = np.ones((espessura, espessura), np.uint8)
        main_edges = cv2.dilate(main_edges, kernel_espessura, iterations=1)
    
    # 2. MAPEAMENTO DE SOMBRAS
    blurred_shadows = cv2.GaussianBlur(gray, (7, 7), 0)
    
    mask1 = (blurred_shadows < thresh1).astype(np.uint8) * 255
    mask2 = (blurred_shadows < thresh2).astype(np.uint8) * 255
    mask3 = (blurred_shadows < thresh3).astype(np.uint8) * 255
    
    edges1 = cv2.Canny(mask1, 100, 200)
    edges2 = cv2.Canny(mask2, 100, 200)
    edges3 = cv2.Canny(mask3, 100, 200)
    
    shadow_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # ZONA DE EXCLUSÃO (CAMPO DE FORÇA)
    kernel_dist = np.ones((dist, dist), np.uint8)
    main_edges_dilated = cv2.dilate(main_edges, kernel_dist, iterations=1)
    
    shadow_edges_pure = cv2.bitwise_and(shadow_edges, cv2.bitwise_not(main_edges_dilated))
    
    # EFEITO TRACEJADO
    h, w = gray.shape
    y_indices, x_indices = np.indices((h, w))
    dash_mask = (((x_indices + y_indices) % 12) < 6).astype(np.uint8) * 255
    
    dashed_shadows = cv2.bitwise_and(shadow_edges_pure, dash_mask)
    
    # 3. JUNÇÃO FINAL
    final_edges = cv2.bitwise_or(main_edges, dashed_shadows)
    
    # 4. APLICAÇÃO DA COR ESCOLHIDA
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    
    if cor == "Vermelho":
        output_rgb[final_edges > 0] = [255, 0, 0] # Vermelho
    else:
        output_rgb[final_edges > 0] = [0, 0, 0]   # Preto absoluto
        
    return output_rgb

# --- INTERFACE FLUXO DO USUÁRIO ---
tab1, tab2 = st.tabs(["📷 Câmera do Dispositivo", "📂 Abrir da Galeria"])
imagem_subida = None

with tab1:
    camera_pic = st.camera_input("Capture a imagem de referência")
    if camera_pic:
        imagem_subida = camera_pic

with tab2:
    arquivo_pic = st.file_uploader("Selecione um arquivo de imagem", type=['jpg', 'jpeg', 'png'])
    if arquivo_pic:
        imagem_subida = arquivo_pic

if imagem_subida is not None:
    file_bytes = np.asarray(bytearray(imagem_subida.read()), dtype=np.uint8)
    img_opencv = cv2.imdecode(file_bytes, 1)
    
    # Executa o processamento com as novas opções
    resultado = gerar_stencil_tecnico(
        img_opencv, sensibilidade_contorno, t1, t2, t3, 
        distancia_sombras, espessura_linha, cor_stencil
    )
    
    img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_display_orig, caption="Sua Referência Original", use_container_width=True)
    with c2:
        st.image(resultado, caption=f"Resultado: Mapa Técnico {cor_stencil}", use_container_width=True)
            
    # Conversão correta de cores para o download final
    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", resultado_bgr)
    io_buf = io.BytesIO(buffer)
    
    st.success("Seu estêncil técnico foi gerado com sucesso!")
    st.download_button(
        label=f"⬇️ Baixar Estêncil {cor_stencil}",
        data=io_buf,
        file_name=f"stencil_{cor_stencil.lower()}.jpg",
        mime="image/jpeg"
    )
