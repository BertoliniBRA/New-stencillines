import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Stencil Técnico Pro", page_icon="🖊️", layout="wide")

st.title("🖊️ Stencil Técnico Avançado - Linha Vermelha")
st.write("Estêncil técnico com separação limpa: o tracejado nunca encosta no traço principal.")

# --- BARRA LATERAL (AJUSTES) ---
st.sidebar.header("🛠️ Ajustes do Mapeamento Técnico")
sensibilidade_contorno = st.sidebar.slider("Sensibilidade do Contorno Principal", 10, 150, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("**🛡️ Limpeza de Linhas Duplas:**")
# Este é o novo controle que afasta a sombra da linha principal
distancia_sombras = st.sidebar.slider("Distância Segura (Evita Linha Dupla)", 1, 9, 5, help="Aumente este valor se o tracejado estiver muito colado na linha contínua.")

st.sidebar.markdown("---")
st.sidebar.markdown("**🎚️ Limiares das 4 Zonas de Sombra:**")
t1 = st.sidebar.slider("Zona 1: Áreas Muito Escuras (Preto)", 10, 80, 45)
t2 = st.sidebar.slider("Zona 2: Tons Escuros / Médios", 81, 150, 100)
t3 = st.sidebar.slider("Zona 3: Tons Claros (Transições)", 151, 230, 175)

# --- MOTOR DE PROCESSAMENTO CORRIGIDO ---
def gerar_stencil_tecnico(img, sens_canny, thresh1, thresh2, thresh3, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CONTORNO PRINCIPAL (Linha Sólida Fina e Contínua)
    blurred_main = cv2.GaussianBlur(gray, (3, 3), 0)
    main_edges = cv2.Canny(blurred_main, sens_canny, sens_canny * 2)
    
    # 2. MAPEAMENTO DE SOMBRAS
    blurred_shadows = cv2.GaussianBlur(gray, (7, 7), 0)
    
    mask1 = (blurred_shadows < thresh1).astype(np.uint8) * 255
    mask2 = (blurred_shadows < thresh2).astype(np.uint8) * 255
    mask3 = (blurred_shadows < thresh3).astype(np.uint8) * 255
    
    edges1 = cv2.Canny(mask1, 100, 200)
    edges2 = cv2.Canny(mask2, 100, 200)
    edges3 = cv2.Canny(mask3, 100, 200)
    
    shadow_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # === A CORREÇÃO DO CAMPO DE FORÇA (ZONA DE EXCLUSÃO) ===
    # Criamos um "pincel" com a espessura da distância escolhida no menu
    kernel = np.ones((dist, dist), np.uint8)
    
    # Engordamos a linha principal para criar a zona de bloqueio
    main_edges_dilated = cv2.dilate(main_edges, kernel, iterations=1)
    
    # Agora apagamos qualquer sombra que caia dentro dessa zona gorda
    # Isso garante que a linha tracejada fique distante da linha contínua!
    shadow_edges_pure = cv2.bitwise_and(shadow_edges, cv2.bitwise_not(main_edges_dilated))
    
    # Criar máscara geométrica para o efeito tracejado
    h, w = gray.shape
    y_indices, x_indices = np.indices((h, w))
    dash_mask = (((x_indices + y_indices) % 12) < 6).astype(np.uint8) * 255
    
    # O tracejado é aplicado APENAS nas sombras que sobreviveram ao campo de força
    dashed_shadows = cv2.bitwise_and(shadow_edges_pure, dash_mask)
    
    # 3. JUNÇÃO FINAL COMPATÍVEL
    final_edges = cv2.bitwise_or(main_edges, dashed_shadows)
    
    # Mapeamento de cor: Fundo Branco limpo, Linhas Vermelhas puras
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    output_rgb[final_edges > 0] = [255, 0, 0] 
    
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
    
    resultado = gerar_stencil_tecnico(img_opencv, sensibilidade_contorno, t1, t2, t3, distancia_sombras)
    img_display_orig = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_display_orig, caption="Sua Referência Original", use_container_width=True)
    with c2:
        st.image(resultado, caption="Resultado: Mapa Técnico Limpo", use_container_width=True)
            
    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", resultado_bgr)
    io_buf = io.BytesIO(buffer)
    
    st.success("Seu estêncil técnico foi gerado com sucesso!")
    st.download_button(
        label="⬇️ Baixar Estêncil Limpo",
        data=io_buf,
        file_name="stencil_tecnico_limpo.jpg",
        mime="image/jpeg"
