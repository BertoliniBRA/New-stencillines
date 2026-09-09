import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Stencil Pro - Studio", page_icon="🖊️", layout="wide")
st.title("🖊️ Stencil Técnico: Linhas Limpas")

tipo_referencia = st.sidebar.radio("Referência", ["1. Fotografia", "2. Desenho Limpo (Sem Pontilhismo)"])
cor_stencil = st.sidebar.radio("Cor", ["Roxo", "Preto"])

sensibilidade_contorno = 40
espessura_silhueta = 2
distancia_sombras = 5
limite_branco = 200
filtro_sujeira = 15
t1, t2, t3 = 45, 100, 175

st.sidebar.markdown("---")
if "Fotografia" not in tipo_referencia:
    st.sidebar.info("Este modo remove pontilhismos e foca apenas nos traços contínuos do desenho.")
    limite_branco = st.sidebar.slider("Corte de Linha", 50, 250, 180)
    filtro_sujeira = st.sidebar.slider("Mata-Sujeira (Remover Pontilhismo)", 0, 100, 20, help="Aumente para apagar pontos isolados e manter apenas traços longos.")

def gerar_stencil(img, tipo_ref, cor, sens, esp_silhueta, dist, lim_branco, filtro, t1, t2, t3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if "Fotografia" in tipo_ref:
        # Modo Foto Omitido para brevidade (mantenha o original se quiser)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        final_edges = cv2.Canny(blur, sens, sens*2)
    else:
        # Binarização simples
        _, ink_mask = cv2.threshold(gray, lim_branco, 255, cv2.THRESH_BINARY_INV)
        
        # Filtro de Sujeira (Remove pontilhismos)
        if filtro > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
            mask_limpa = np.zeros_like(ink_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= filtro:
                    mask_limpa[labels == i] = 255
            ink_mask = mask_limpa
            
        final_edges = ink_mask
        
    output_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    cor_rgb = [138, 43, 226] if "Roxo" in cor else [0, 0, 0]
    output_rgb[final_edges > 0] = cor_rgb
        
    return output_rgb

tab1, tab2 = st.tabs(["📂 Galeria", "📷 Câmera"])
imagem_subida = st.file_uploader("Selecione um arquivo", type=['jpg', 'jpeg', 'png'])

if imagem_subida:
    file_bytes = np.asarray(bytearray(imagem_subida.read()), dtype=np.uint8)
    img_opencv = cv2.imdecode(file_bytes, 1)
    
    resultado = gerar_stencil(img_opencv, tipo_referencia, cor_stencil, sensibilidade_contorno, espessura_silhueta, distancia_sombras, limite_branco, filtro_sujeira, t1, t2, t3)
    
    st.image(resultado, caption=f"Estêncil {cor_stencil}", use_container_width=True)
