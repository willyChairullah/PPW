# ==============================
# keyword_extractor_app_fast_v4.py
# ==============================

# ======= IMPORT LIBRARY =======
import pymupdf
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
from functools import lru_cache
import pandas as pd
import matplotlib.patches as mpatches
import spacy

# ======= FUNGSI DASAR =======

def extract_text_from_pdf(pdf_file):
    """Ekstrak teks dari PDF baik dari file path atau file upload Streamlit"""
    import io
    if hasattr(pdf_file, "read"):  # Streamlit upload (BytesIO)
        file_bytes = pdf_file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    else:  # Path string
        doc = pymupdf.open(pdf_file)

    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    """Bersihkan teks: lowercase, hapus simbol non-alphabet, hapus spasi ganda"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@st.cache_resource
def get_nlp_tools():
    """Inisialisasi stopword dan stemmer hanya sekali"""
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stop_words, stemmer


@lru_cache(maxsize=None)
def cached_stem(word):
    """Stemming cepat dengan cache"""
    stop_words, stemmer = get_nlp_tools()
    return stemmer.stem(word)

# Muat model spaCy global (biar tidak re-load tiap kali)
nlp = spacy.load("xx_ent_wiki_sm")

# Tambahkan stopword tambahan
CUSTOM_STOPWORDS = {
    "menjadi", "memiliki", "adalah", "terhadap", "tahun", "dalam", 
    "untuk", "pada", "oleh", "antara", "itu", "sebagai", "yang", "dapat"
}

@st.cache_data(show_spinner=False)
def preprocess_text(text, use_stemming=False):
    """Tokenisasi, filtering stopword, POS, dan frekuensi minimal"""

    # === 1️⃣ Inisialisasi stopword & stemmer ===
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    # === 2️⃣ Jalankan NLP tagging untuk ambil kata benda & adjektiva ===
    doc = nlp(text)
    filtered_words = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"]]

    # === 3️⃣ Filter stopword (Sastrawi + custom) ===
    words = [w for w in filtered_words if w.isalpha() and w not in stop_words and w not in CUSTOM_STOPWORDS]

    # === 4️⃣ Opsional: Stemming ===
    if use_stemming:
        words = [stemmer.stem(w) for w in words]

    # === 5️⃣ Filter kata berdasarkan frekuensi minimal ===
    freq = Counter(words)
    words = [w for w in words if freq[w] >= 2]  # hanya kata yang muncul ≥ 2 kali

    return words


def build_word_graph_fast(words, window_size=2, max_words=5000, min_edge_weight=2):
    """Bangun graph cepat dengan Counter"""
    if len(words) > max_words:
        words = words[:max_words]

    edges = Counter()
    for i in range(len(words) - window_size):
        window = words[i:i + window_size + 1]
        for w1 in window:
            for w2 in window:
                if w1 != w2:
                    edges[(w1, w2)] += 1

    G = nx.Graph()
    for (w1, w2), count in edges.items():
        if count >= min_edge_weight:
            G.add_edge(w1, w2, weight=count)
    return G


def extract_keywords_fast(G, top_n=20, use_centrality=False):
    """Hitung PageRank atau Degree Centrality"""
    if use_centrality:
        scores = nx.degree_centrality(G)
    else:
        try:
            scores = nx.pagerank_numpy(G, alpha=0.85, weight="weight")
        except Exception:
            scores = nx.pagerank(G, alpha=0.85, weight="weight")

    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n], scores
import matplotlib.patches as mpatches

def plot_word_graph(G, scores, top_n):
    """
    Visualisasi word graph dengan:
    - Node berbentuk oval adaptif (menyesuaikan panjang kata)
    - Garis antar node normal (lembut, seragam)
    - Tema lembut dan nyaman dibaca
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ambil node teratas
    top_nodes = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
    subgraph = G.subgraph(top_nodes.keys())

    # Layout posisi node
    pos = nx.spring_layout(subgraph, k=0.5, seed=42)

    # Warna & colormap
    cmap = plt.cm.Spectral_r
    node_values = [scores[n] for n in subgraph.nodes()]
    norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))

    # 🎨 Gambar edge normal (lembut dan tidak terlalu menonjol)
    nx.draw_networkx_edges(
        subgraph, pos,
        width=1.0,
        alpha=0.3,
        edge_color="#000",
        ax=ax
    )

    # 🔵 Gambar node berbentuk oval (bukan lingkaran)
    for node in subgraph.nodes():
        x, y = pos[node]
        val = scores[node]
        color = cmap(norm(val))
        # Ukuran oval berdasarkan panjang kata & skor PageRank
        width = 0.12 + len(node) * 0.012  # panjang horizontal menyesuaikan kata
        height = 0.07 + val * 0.25        # sedikit proporsional dengan skor
        ellipse = mpatches.Ellipse(
            (x, y), width=width, height=height,
            color=color, ec="#555", lw=0.6, alpha=0.95
        )
        ax.add_patch(ellipse)

    # ✨ Label di tengah node
    for node, (x, y) in pos.items():
        ax.text(
            x, y, node,
            fontsize=9,
            color='#FFFACD',  # krem lembut
            ha='center', va='center',
            fontweight='bold'
        )

    # Tambahkan colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Skor PageRank / Frekuensi Kata", fontsize=10)

    # Judul & tema
    ax.set_title(
        f"Word Graph - {top_n} Kata Kunci Teratas\n"
        f"(Node oval menyesuaikan panjang kata, warna menunjukkan kepentingan)",
        fontsize=12, color='#333', pad=20
    )

    # 🎨 Latar belakang lembut
    fig.patch.set_facecolor("#f7f6f3")
    ax.set_facecolor("#f7f6f3")
    ax.axis("off")

    return fig

def split_into_sentences(text):
    """Pisahkan teks menjadi kalimat menggunakan regex sederhana"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# ======= STREAMLIT UI =======
st.title("⚡ Ekstraktor Kata Kunci PDF")

uploaded_file = st.file_uploader("📄 Upload file PDF", type=["pdf"])

# ====== Opsi di bagian atas (bersebelahan) ======
col1, col2 = st.columns(2)
with col1:
    use_stemming = st.checkbox(
        "Gunakan stemming", 
        value=False
    )
with col2:
    use_centrality = st.checkbox(
        "Gunakan Degree Centrality", 
        value=False
    )
# ===============================================

if uploaded_file:
    with st.spinner("📖 Membaca dan memproses file PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        clean = clean_text(raw_text)

    st.info(f"Teks berhasil diekstrak, panjang: {len(clean)} karakter")

    # ====== 🆕 Tampilkan isi teks dari PDF ======
    sentences = split_into_sentences(raw_text)
    st.subheader("📘 Cuplikan Teks dari PDF")
    preview_text = "\n\n".join(sentences[:5])
    st.text_area("Preview teks (5 kalimat pertama):", preview_text, height=200)

    with st.expander("📄 Lihat seluruh teks hasil ekstraksi"):
        st.text_area("Teks lengkap:", raw_text, height=400)
    # ============================================

    # ✅ Proses teks
    with st.spinner("🧠 Memproses teks..."):
        words = preprocess_text(clean, use_stemming=use_stemming)

    st.success(f"✅ Preprocessing selesai: {len(words)} kata ({len(set(words))} unik)")

    # ===== Graph dan Ekstraksi =====
    window_size = st.slider("Pilih window size (co-occurrence)", 2, 8, 3)
    top_n = st.slider("Jumlah kata kunci ditampilkan", 5, 80, 20)

    with st.spinner("🔗 Membuat word graph dan menghitung skor..."):
        G = build_word_graph_fast(words, window_size=window_size)
        keywords, scores = extract_keywords_fast(G, top_n=top_n, use_centrality=use_centrality)

    # ===== Output =====
    st.subheader("🔑 Top Kata Kunci")

    # Buat DataFrame dan tambahkan nomor urut
    df_keywords = pd.DataFrame(keywords, columns=["Kata", "Skor"])
    df_keywords.index = range(1, len(df_keywords) + 1)  # mulai dari 1
    df_keywords.index.name = "No"  # beri nama kolom index

    st.dataframe(df_keywords, width='stretch')


    # ===== Graph sesuai top_n =====
    st.subheader("🌐 Visualisasi Word Graph")
    fig = plot_word_graph(G, scores, top_n=top_n)
    st.pyplot(fig)

# else:
#     st.info("⬆️ Silakan upload file PDF terlebih dahulu untuk memulai analisis.")