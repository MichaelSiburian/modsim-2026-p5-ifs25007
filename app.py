import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.block-container { padding: 1.5rem 2.5rem 2rem; }
.page-title {
    font-size: 2rem; font-weight: 800; color: #0F172A;
    border-left: 6px solid #2563EB; padding-left: 1rem;
    line-height: 1.3; margin-bottom: 0.3rem;
}
.page-sub { color: #64748B; font-size: 0.95rem; padding-left: 1.3rem; margin-bottom: 1.2rem; }
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1E40AF;
    background: linear-gradient(90deg, #EFF6FF, transparent);
    border-left: 4px solid #2563EB;
    padding: 0.5rem 0.8rem; border-radius: 0 6px 6px 0;
    margin: 1.5rem 0 0.8rem;
}
.kpi-box { background: #0F172A; color: white; border-radius: 10px; padding: 1rem 1.2rem; border-top: 3px solid #2563EB; }
.kpi-val { font-size: 1.8rem; font-weight: 800; color: #60A5FA; line-height: 1.1; }
.kpi-lbl { font-size: 0.78rem; color: #94A3B8; margin-top: 2px; }
.answer-box { background: #F0FDF4; border: 1.5px solid #BBF7D0; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.9rem; color: #14532D; margin-top: 0.5rem; }
.risk-box { background: #FFF7ED; border: 1.5px solid #FED7AA; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.9rem; color: #7C2D12; }
.stButton>button { background: #2563EB; color: white; border: none; border-radius: 8px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.95rem; padding: 0.55rem 1.2rem; }
.stButton>button:hover { background: #1D4ED8; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL: TAHAPAN PROYEK + FAKTOR RISIKO
# ============================================================================
class TahapanProyek:
    def __init__(self, nama, params, risiko=None, dependensi=None):
        self.nama           = nama
        self.optimis        = params['optimis']
        self.paling_mungkin = params['paling_mungkin']
        self.pesimis        = params['pesimis']
        self.risiko         = risiko or {}
        self.dependensi     = dependensi or []

    def sampel_durasi(self, n):
        durasi = np.random.triangular(self.optimis, self.paling_mungkin, self.pesimis, n)
        for _, r in self.risiko.items():
            if r['tipe'] == 'diskrit':
                terjadi = np.random.random(n) < r['probabilitas']
                durasi  = np.where(terjadi, durasi * (1 + r['dampak']), durasi)
            elif r['tipe'] == 'kontinu':
                faktor = np.random.normal(r['rata'], r['std'], n)
                durasi = durasi / np.clip(faktor, 0.5, 1.5)
        return durasi


class SimulasiMonteCarlo:
    def __init__(self, konfigurasi, n_sim=20000):
        self.n_sim   = n_sim
        self.tahapan = {
            nama: TahapanProyek(
                nama=nama,
                params=cfg['params'],
                risiko=cfg.get('risiko', {}),
                dependensi=cfg.get('dependensi', [])
            )
            for nama, cfg in konfigurasi.items()
        }
        self.hasil = None

    def jalankan(self):
        df      = pd.DataFrame(index=range(self.n_sim))
        mulai   = pd.DataFrame(index=range(self.n_sim))
        selesai = pd.DataFrame(index=range(self.n_sim))
        for nama, t in self.tahapan.items():
            df[nama] = t.sampel_durasi(self.n_sim)
        for nama, t in self.tahapan.items():
            mulai[nama]   = 0.0 if not t.dependensi else selesai[t.dependensi].max(axis=1)
            selesai[nama] = mulai[nama] + df[nama]
        df['Total'] = selesai.max(axis=1)
        for nama in self.tahapan:
            df[f'_m_{nama}'] = mulai[nama]
            df[f'_s_{nama}'] = selesai[nama]
        self.hasil = df
        return df

    def critical_path(self):
        total = self.hasil['Total']
        out   = {}
        for nama in self.tahapan:
            fin  = self.hasil[f'_s_{nama}']
            prob = float(np.mean((fin + 0.001) >= total))
            corr = float(self.hasil[nama].corr(total))
            out[nama] = {'Prob. Critical': prob, 'Korelasi': corr, 'Rata-rata (bln)': float(self.hasil[nama].mean())}
        return pd.DataFrame(out).T.sort_values('Prob. Critical', ascending=False)

    def kontribusi_risiko(self):
        tv  = self.hasil['Total'].var()
        out = {}
        for nama in self.tahapan:
            cov = self.hasil[nama].cov(self.hasil['Total'])
            out[nama] = {'Kontribusi (%)': float((cov / tv) * 100), 'Std Dev (bln)': float(self.hasil[nama].std())}
        return pd.DataFrame(out).T.sort_values('Kontribusi (%)', ascending=False)

    def simulasi_resource(self, stages_pct: dict):
        """stages_pct = {nama_stage: persen_percepatan}"""
        h2 = self.hasil.copy()
        for st_n, pct in stages_pct.items():
            h2[st_n] = h2[st_n] * (1 - pct)
        m2 = pd.DataFrame(index=range(self.n_sim))
        s2 = pd.DataFrame(index=range(self.n_sim))
        for nama, t in self.tahapan.items():
            m2[nama] = 0.0 if not t.dependensi else s2[t.dependensi].max(axis=1)
            s2[nama] = m2[nama] + h2[nama]
        return s2.max(axis=1).values


# ============================================================================
# KONFIGURASI TAHAPAN — GEDUNG FITE 5 LANTAI (satuan: BULAN)
# ============================================================================
KONFIGURASI = {
    "Persiapan & Mobilisasi": {
        "params": {"optimis": 0.5, "paling_mungkin": 1.0, "pesimis": 2.0},
        "risiko": {
            "Pembebasan lahan": {"tipe": "diskrit", "probabilitas": 0.20, "dampak": 0.50},
        }
    },
    "Desain & Perizinan": {
        "params": {"optimis": 1.5, "paling_mungkin": 2.5, "pesimis": 4.5},
        "risiko": {
            "Perubahan desain lab":   {"tipe": "diskrit", "probabilitas": 0.45, "dampak": 0.35},
            "Keterlambatan IMB/PBG":  {"tipe": "diskrit", "probabilitas": 0.30, "dampak": 0.30},
            "Produktivitas arsitek":  {"tipe": "kontinu", "rata": 1.0,  "std": 0.15},
        },
        "dependensi": ["Persiapan & Mobilisasi"]
    },
    "Pondasi & Struktur Bawah": {
        "params": {"optimis": 2.0, "paling_mungkin": 3.0, "pesimis": 4.5},
        "risiko": {
            "Cuaca buruk":              {"tipe": "diskrit", "probabilitas": 0.35, "dampak": 0.20},
            "Material beton terlambat": {"tipe": "diskrit", "probabilitas": 0.25, "dampak": 0.15},
            "Produktivitas pekerja":    {"tipe": "kontinu", "rata": 1.0,  "std": 0.20},
        },
        "dependensi": ["Desain & Perizinan"]
    },
    "Struktur Lantai 1-2": {
        "params": {"optimis": 1.5, "paling_mungkin": 2.5, "pesimis": 4.0},
        "risiko": {
            "Cuaca buruk":              {"tipe": "diskrit", "probabilitas": 0.35, "dampak": 0.20},
            "Material baja terlambat":  {"tipe": "diskrit", "probabilitas": 0.20, "dampak": 0.20},
            "Produktivitas pekerja":    {"tipe": "kontinu", "rata": 1.0,  "std": 0.18},
        },
        "dependensi": ["Pondasi & Struktur Bawah"]
    },
    "Struktur Lantai 3-5": {
        "params": {"optimis": 2.0, "paling_mungkin": 3.0, "pesimis": 4.5},
        "risiko": {
            "Cuaca buruk":             {"tipe": "diskrit", "probabilitas": 0.35, "dampak": 0.25},
            "Material teknis khusus":  {"tipe": "diskrit", "probabilitas": 0.30, "dampak": 0.20},
            "Produktivitas pekerja":   {"tipe": "kontinu", "rata": 1.0,  "std": 0.20},
        },
        "dependensi": ["Struktur Lantai 1-2"]
    },
    "MEP & Infrastruktur Jaringan": {
        "params": {"optimis": 1.5, "paling_mungkin": 2.5, "pesimis": 4.0},
        "risiko": {
            "Material teknis khusus terlambat": {"tipe": "diskrit", "probabilitas": 0.40, "dampak": 0.30},
            "Koordinasi subkontraktor":          {"tipe": "diskrit", "probabilitas": 0.30, "dampak": 0.20},
            "Produktivitas teknisi":             {"tipe": "kontinu", "rata": 1.0,  "std": 0.22},
        },
        "dependensi": ["Struktur Lantai 3-5"]
    },
    "Finishing & Arsitektur": {
        "params": {"optimis": 1.5, "paling_mungkin": 2.5, "pesimis": 4.0},
        "risiko": {
            "Cuaca buruk":           {"tipe": "diskrit", "probabilitas": 0.25, "dampak": 0.15},
            "Material finishing":    {"tipe": "diskrit", "probabilitas": 0.20, "dampak": 0.20},
            "Produktivitas pekerja": {"tipe": "kontinu", "rata": 1.0,  "std": 0.18},
        },
        "dependensi": ["MEP & Infrastruktur Jaringan"]
    },
    "Pemasangan Peralatan Lab": {
        "params": {"optimis": 1.0, "paling_mungkin": 2.0, "pesimis": 3.5},
        "risiko": {
            "Pengadaan alat khusus terlambat": {"tipe": "diskrit", "probabilitas": 0.50, "dampak": 0.40},
            "Perubahan spesifikasi lab":        {"tipe": "diskrit", "probabilitas": 0.35, "dampak": 0.25},
            "Ketersediaan teknisi ahli":        {"tipe": "kontinu", "rata": 1.0,  "std": 0.25},
        },
        "dependensi": ["Finishing & Arsitektur"]
    },
    "Pengujian & Serah Terima": {
        "params": {"optimis": 0.5, "paling_mungkin": 1.0, "pesimis": 2.0},
        "risiko": {
            "Defect & perbaikan ulang": {"tipe": "diskrit", "probabilitas": 0.25, "dampak": 0.50},
            "Birokrasi SLF":            {"tipe": "diskrit", "probabilitas": 0.35, "dampak": 0.40},
        },
        "dependensi": ["Pemasangan Peralatan Lab"]
    }
}

SKENARIO_RESOURCE = [
    {
        "label":       "Tambah 2 Alat Berat (Pondasi)",
        "stages_pct":  {"Pondasi & Struktur Bawah": 0.25},
        "biaya_juta":  210,
        "deskripsi":   "2 excavator + concrete pump → percepat penggalian & pengecoran fondasi"
    },
    {
        "label":       "Tambah 15 Pekerja Terampil (Struktur Lt.3-5)",
        "stages_pct":  {"Struktur Lantai 3-5": 0.22},
        "biaya_juta":  135,
        "deskripsi":   "15 pekerja terampil → paralel pengerjaan kolom, balok & pelat lantai"
    },
    {
        "label":       "Tambah 3 Insinyur MEP Senior",
        "stages_pct":  {"MEP & Infrastruktur Jaringan": 0.28},
        "biaya_juta":  180,
        "deskripsi":   "3 insinyur MEP senior → percepat instalasi listrik, jaringan, AC lab"
    },
    {
        "label":       "Tambah 2 Insinyur Ahli (Peralatan Lab)",
        "stages_pct":  {"Pemasangan Peralatan Lab": 0.30},
        "biaya_juta":  160,
        "deskripsi":   "2 insinyur ahli IT/EE → konfigurasi lab VR/AR, Game, Elektro, Mobile"
    },
    {
        "label":       "Paket Lengkap (Semua Tahapan Kritis)",
        "stages_pct":  {
            "Pondasi & Struktur Bawah":   0.20,
            "Struktur Lantai 3-5":        0.20,
            "MEP & Infrastruktur Jaringan": 0.20,
            "Pemasangan Peralatan Lab":   0.20,
        },
        "biaya_juta":  620,
        "deskripsi":   "Penambahan resource serentak di semua tahapan kritis"
    },
]

WARNA = px.colors.qualitative.Bold

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")
n_sim = st.sidebar.slider("Jumlah Iterasi", 5000, 50000, 20000, 1000)
seed  = st.sidebar.number_input("Random Seed", 0, 9999, 42)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Skenario Deadline")
dl_a = st.sidebar.number_input("Deadline A (bulan)", 10, 40, 16)
dl_b = st.sidebar.number_input("Deadline B (bulan)", 10, 40, 20)
dl_c = st.sidebar.number_input("Deadline C (bulan)", 10, 40, 24)
st.sidebar.markdown("---")
jalankan = st.sidebar.button("🚀 Jalankan Simulasi", use_container_width=True)

if 'sim' not in st.session_state:
    st.session_state.sim    = None
    st.session_state.hasil  = None
    st.session_state.n_done = 0

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="page-title">🏛️ Simulasi Monte Carlo<br>Estimasi Waktu Pembangunan Gedung FITE</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Gedung Fakultas Informatika & Teknik Elektro · 5 Lantai · Lab Komputer, Elektro, Mobile, VR/AR, Game · Ruang Kelas, Dosen, Toilet, Serbaguna</div>', unsafe_allow_html=True)

# ============================================================================
# JALANKAN SIMULASI
# ============================================================================
if jalankan:
    np.random.seed(int(seed))
    with st.spinner(f"Menjalankan {n_sim:,} iterasi Monte Carlo…"):
        sim   = SimulasiMonteCarlo(KONFIGURASI, n_sim)
        hasil = sim.jalankan()
        st.session_state.sim    = sim
        st.session_state.hasil  = hasil
        st.session_state.n_done = n_sim
    st.success(f"✅ Selesai — {n_sim:,} iterasi · {len(KONFIGURASI)} tahapan")

if st.session_state.sim is None:
    st.info("👈 Atur parameter di sidebar, lalu klik **Jalankan Simulasi**.")
    with st.expander("📋 Lihat daftar tahapan yang dimodelkan"):
        for i, (nama, cfg) in enumerate(KONFIGURASI.items(), 1):
            bp  = cfg['params']
            dep = ", ".join(cfg.get('dependensi', [])) or "—"
            st.markdown(f"`{i:02d}` **{nama}** · O:{bp['optimis']} / ML:{bp['paling_mungkin']} / P:{bp['pesimis']} bln · *Setelah: {dep}*")
    st.stop()

# ============================================================================
# HASIL SIMULASI TERSEDIA
# ============================================================================
sim   = st.session_state.sim
hasil = st.session_state.hasil
total = hasil['Total']

mean_total   = float(total.mean())
median_total = float(np.median(total))
std_total    = float(total.std())
p10v = float(np.percentile(total, 10))
p90v = float(np.percentile(total, 90))
p80v = float(np.percentile(total, 80))
p95v = float(np.percentile(total, 95))
p16  = float(np.mean(total <= dl_a))
p20  = float(np.mean(total <= dl_b))
p24  = float(np.mean(total <= dl_c))
baseline_det = sum(cfg['params']['paling_mungkin'] for cfg in KONFIGURASI.values())

# ── KPI ──
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (f"{mean_total:.1f} bln",        "Rata-rata Durasi"),
    (f"±{std_total:.1f} bln",        "Standar Deviasi"),
    (f"{p10v:.1f}–{p90v:.1f} bln",  "80% Conf. Interval"),
    (f"{p16:.1%}", f"Prob ≤ {dl_a} Bln"),
    (f"{p20:.1%}", f"Prob ≤ {dl_b} Bln"),
    (f"{p24:.1%}", f"Prob ≤ {dl_c} Bln"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4, c5, c6], kpis):
    with col:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ============================================================================
# P1 — TOTAL WAKTU
# ============================================================================
st.markdown('<div class="section-title">❶ Berapa lama total waktu yang dibutuhkan?</div>', unsafe_allow_html=True)

fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=total, nbinsx=70, histnorm='probability density',
    marker_color='#3B82F6', opacity=0.75, name='Distribusi'))
fig1.add_vrect(x0=p10v, x1=p90v, fillcolor='#FCD34D', opacity=0.18, line_width=0, annotation_text="80% CI", annotation_position="top left")
fig1.add_vrect(x0=float(np.percentile(total, 2.5)), x1=float(np.percentile(total, 97.5)), fillcolor='#F97316', opacity=0.08, line_width=0)
fig1.add_vline(x=mean_total,   line_dash='dash', line_color='#EF4444', line_width=2.5, annotation_text=f"Mean {mean_total:.1f}", annotation_position="top right")
fig1.add_vline(x=median_total, line_dash='dot',  line_color='#10B981', line_width=2,   annotation_text=f"Median {median_total:.1f}")
fig1.add_vline(x=p80v, line_dash='longdash', line_color='#7C3AED', line_width=1.8, annotation_text=f"P80={p80v:.1f}", annotation_position="top left")
for dl, col in [(dl_a,'#7C3AED'), (dl_b,'#EA580C'), (dl_c,'#B45309')]:
    fig1.add_vline(x=dl, line_dash='dashdot', line_color=col, line_width=1.5, annotation_text=f"{dl} bln")
fig1.update_layout(title='Distribusi Total Durasi Pembangunan Gedung FITE (Monte Carlo)',
    xaxis_title='Durasi Total (Bulan)', yaxis_title='Densitas Probabilitas',
    height=400, showlegend=False, margin=dict(t=50, b=40))
st.plotly_chart(fig1, use_container_width=True)

# Statistik lengkap
s1c1, s1c2, s1c3 = st.columns(3)
with s1c1:
    st.markdown("**Statistik Deskriptif**")
    st.markdown(f"- Mean: **{mean_total:.2f} bln**")
    st.markdown(f"- Median: **{median_total:.2f} bln**")
    st.markdown(f"- Std Dev: **{std_total:.2f} bln**")
with s1c2:
    st.markdown("**Persentil Utama**")
    for p in [10, 25, 50, 75, 80, 90, 95]:
        st.markdown(f"- P{p}: **{np.percentile(total, p):.2f} bln**")
with s1c3:
    st.markdown("**Confidence Intervals**")
    st.markdown(f"- 80% CI: **{p10v:.2f} – {p90v:.2f} bln**")
    st.markdown(f"- 90% CI: **{np.percentile(total,5):.2f} – {p95v:.2f} bln**")
    st.markdown(f"- 95% CI: **{np.percentile(total,2.5):.2f} – {np.percentile(total,97.5):.2f} bln**")

st.markdown(f"""
<div class="answer-box">
<b>📌 Jawaban P1:</b> Berdasarkan {st.session_state.n_done:,} iterasi Monte Carlo, total durasi
pembangunan Gedung FITE 5 lantai diestimasi rata-rata <b>{mean_total:.1f} bulan</b>
(median {median_total:.1f} bln, std {std_total:.1f} bln).
Rentang paling realistis (80% CI): <b>{p10v:.1f} – {p90v:.1f} bulan</b>.
Jadwal aman yang direkomendasikan (probabilitas 80%): <b>{p80v:.1f} bulan</b>.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# P2 — RISIKO KETERLAMBATAN
# ============================================================================
st.markdown('<div class="section-title">❷ Risiko keterlambatan akibat faktor ketidakpastian</div>', unsafe_allow_html=True)

p2c1, p2c2 = st.columns([3, 2])
with p2c1:
    fig2a = go.Figure()
    for i, nama in enumerate(KONFIGURASI):
        fig2a.add_trace(go.Box(y=hasil[nama], name=nama,
            marker_color=WARNA[i % len(WARNA)], boxmean='sd', boxpoints=False, line_width=1.5))
    fig2a.update_layout(title='Variabilitas Durasi per Tahapan (Dampak Risiko)',
        yaxis_title='Durasi (Bulan)', height=420,
        showlegend=False, margin=dict(t=50, b=60), xaxis_tickangle=-25)
    st.plotly_chart(fig2a, use_container_width=True)

with p2c2:
    kontr = sim.kontribusi_risiko()
    fig2b = go.Figure(go.Bar(
        y=[n.replace(' ', '<br>') for n in kontr.index],
        x=kontr['Kontribusi (%)'], orientation='h',
        marker=dict(color=kontr['Kontribusi (%)'], colorscale='Reds', showscale=False),
        text=[f"{v:.1f}%" for v in kontr['Kontribusi (%)']],
        textposition='auto'
    ))
    fig2b.update_layout(title='Kontribusi Risiko ke Variabilitas Total',
        xaxis_title='Kontribusi (%)', height=420, margin=dict(t=50, b=40))
    st.plotly_chart(fig2b, use_container_width=True)

# Tabel faktor risiko
rows_r = []
for nama, cfg in KONFIGURASI.items():
    for r_nama, r in cfg.get('risiko', {}).items():
        if r['tipe'] == 'diskrit':
            rows_r.append({
                'Tahapan': nama, 'Faktor Risiko': r_nama,
                'Probabilitas': f"{r['probabilitas']:.0%}",
                'Dampak Durasi': f"+{r['dampak']*100:.0f}%",
                'Eksp. Tambahan (bln)': f"{r['probabilitas']*r['dampak']*cfg['params']['paling_mungkin']:.2f}"
            })
with st.expander("📋 Tabel Lengkap Faktor Risiko Konstruksi"):
    st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)

delay = mean_total - baseline_det
st.markdown(f"""
<div class="risk-box">
<b>📌 Jawaban P2:</b><br>
• Durasi tanpa faktor risiko (deterministik): <b>{baseline_det:.1f} bln</b> → dengan risiko rata-rata: <b>{mean_total:.1f} bln</b> (selisih <b>+{delay:.1f} bln</b>)<br>
• <b>Cuaca buruk</b> (prob. 25–35%, dampak +15–25%): mempengaruhi fase struktur, pondasi & finishing<br>
• <b>Keterlambatan material teknis khusus</b> (prob. 40%, dampak +30%): terbesar di MEP & Infrastruktur Jaringan<br>
• <b>Perubahan desain laboratorium</b> (prob. 45%, dampak +35%): risiko terbesar di fase Desain & Perizinan<br>
• <b>Produktivitas pekerja</b>: variabilitas kontinu ±15–25% memengaruhi semua fase konstruksi<br>
• Worst-case (P95): <b>{p95v:.1f} bulan</b> · Probabilitas molor > {p80v:.1f} bln: <b>20%</b>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# P3 — CRITICAL PATH
# ============================================================================
st.markdown('<div class="section-title">❸ Tahapan mana yang paling kritis (Critical Path)?</div>', unsafe_allow_html=True)

cp = sim.critical_path()
p3c1, p3c2 = st.columns([3, 2])
with p3c1:
    cp_asc = cp.sort_values('Prob. Critical', ascending=True)
    bar_c  = ['#DC2626' if v > 0.7 else ('#F97316' if v > 0.4 else '#93C5FD') for v in cp_asc['Prob. Critical']]
    fig3a  = go.Figure(go.Bar(
        y=[n.replace(' ', '<br>') for n in cp_asc.index],
        x=cp_asc['Prob. Critical'], orientation='h',
        marker_color=bar_c,
        text=[f"{v:.1%}" for v in cp_asc['Prob. Critical']],
        textposition='outside'
    ))
    fig3a.add_vline(x=0.7, line_dash='dot', line_color='#DC2626', line_width=2,
        annotation_text='Threshold Kritis (70%)', annotation_position='top')
    fig3a.update_layout(
        title='Probabilitas Setiap Tahapan Berada di Critical Path',
        xaxis_title='Probabilitas Critical', xaxis_range=[0, 1.15],
        height=420, margin=dict(t=50, b=40))
    st.plotly_chart(fig3a, use_container_width=True)

with p3c2:
    fig3b = go.Figure(go.Scatter(
        x=cp['Korelasi'], y=cp['Prob. Critical'],
        mode='markers+text',
        text=[n.split()[0] for n in cp.index],
        textposition='top center',
        marker=dict(
            size=[v * 35 + 8 for v in cp['Prob. Critical']],
            color=cp['Prob. Critical'], colorscale='Reds', showscale=True,
            colorbar=dict(title='Prob.'), line=dict(width=1, color='white')
        )
    ))
    fig3b.update_layout(
        title='Korelasi vs Probabilitas Critical Path',
        xaxis_title='Korelasi dengan Durasi Total',
        yaxis_title='Probabilitas Critical',
        height=420, margin=dict(t=50, b=40))
    st.plotly_chart(fig3b, use_container_width=True)

with st.expander("📋 Tabel Detail Critical Path Analysis"):
    st.dataframe(cp.style.format({'Prob. Critical': '{:.2%}', 'Korelasi': '{:.3f}', 'Rata-rata (bln)': '{:.2f}'}), use_container_width=True)

kritis     = cp[cp['Prob. Critical'] > 0.7].index.tolist()
top1, top2 = cp.index[0], cp.index[1]
st.markdown(f"""
<div class="answer-box">
<b>📌 Jawaban P3:</b> Tahapan yang secara konsisten menentukan durasi total proyek (critical path):<br>
• <b>{top1}</b> — probabilitas critical path: <b>{cp.loc[top1,'Prob. Critical']:.1%}</b>,
  korelasi dengan total durasi: {cp.loc[top1,'Korelasi']:.3f}<br>
• <b>{top2}</b> — probabilitas critical path: <b>{cp.loc[top2,'Prob. Critical']:.1%}</b>,
  korelasi: {cp.loc[top2,'Korelasi']:.3f}<br>
{"• Tahapan dengan prob. critical >70%: <b>" + ", ".join(kritis) + "</b>" if kritis else ""}<br>
Keterlambatan di tahapan ini langsung memanjangkan total proyek secara 1:1.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# P4 — PROBABILITAS DEADLINE
# ============================================================================
st.markdown(f'<div class="section-title">❹ Probabilitas penyelesaian: {dl_a}, {dl_b}, {dl_c} bulan</div>', unsafe_allow_html=True)

p4c1, p4c2 = st.columns([3, 2])
with p4c1:
    x_range  = np.linspace(float(total.min()), float(total.max()), 500)
    cdf_vals = [float(np.mean(total <= x)) for x in x_range]
    fig4a = go.Figure()
    fig4a.add_trace(go.Scatter(x=x_range, y=cdf_vals, mode='lines',
        line=dict(color='#2563EB', width=3), fill='tozeroy',
        fillcolor='rgba(37,99,235,0.1)', name='CDF'))
    for lvl, col in [(0.5,'#EF4444'), (0.8,'#10B981'), (0.95,'#6366F1')]:
        fig4a.add_hline(y=lvl, line_dash='dash', line_color=col, line_width=1.5,
            annotation_text=f"{lvl:.0%}", annotation_position="right")
    for dl, col, p in [(dl_a,'#7C3AED',p16), (dl_b,'#EA580C',p20), (dl_c,'#B45309',p24)]:
        fig4a.add_vline(x=dl, line_dash='dashdot', line_color=col, line_width=2)
        fig4a.add_trace(go.Scatter(x=[dl], y=[p], mode='markers+text',
            marker=dict(size=14, color=col, symbol='diamond'),
            text=[f"{dl} bln: {p:.1%}"], textposition='top center',
            textfont=dict(color=col, size=11), showlegend=False))
    fig4a.update_layout(title='Kurva CDF Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Bulan)', yaxis_title='Probabilitas Selesai',
        yaxis_range=[-0.03, 1.08], height=420,
        margin=dict(t=50, b=40), showlegend=False)
    st.plotly_chart(fig4a, use_container_width=True)

with p4c2:
    fig4b = make_subplots(rows=3, cols=1,
        specs=[[{"type":"indicator"}],[{"type":"indicator"}],[{"type":"indicator"}]])
    for i, (dl, p, col) in enumerate([(dl_a,p16,'#7C3AED'),(dl_b,p20,'#EA580C'),(dl_c,p24,'#B45309')], 1):
        fig4b.add_trace(go.Indicator(
            mode="gauge+number", value=p*100,
            title={'text': f"Deadline {dl} Bulan", 'font': {'size': 13}},
            number={'suffix':'%', 'font':{'size':22}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': col},
                'steps': [{'range':[0,30],'color':'#FEE2E2'},
                           {'range':[30,60],'color':'#FEF3C7'},
                           {'range':[60,100],'color':'#D1FAE5'}],
                'threshold': {'line':{'color':'black','width':2},'thickness':0.75,'value':50}
            }
        ), row=i, col=1)
    fig4b.update_layout(height=420, margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig4b, use_container_width=True)

rows4 = []
for dl in range(12, 48, 2):
    p = float(np.mean(total <= dl))
    pot = max(0, p95v - dl)
    rows4.append({
        'Deadline (bln)': dl, 'Prob. Selesai': f"{p:.1%}",
        'Prob. Terlambat': f"{1-p:.1%}",
        'Potensi Molor P95 (bln)': f"{pot:.1f}",
        'Status': '✅ Aman' if p >= 0.8 else ('⚠️ Berisiko' if p >= 0.4 else '❌ Tidak Realistis')
    })
with st.expander("📊 Tabel Probabilitas Semua Skenario Deadline"):
    st.dataframe(pd.DataFrame(rows4), use_container_width=True, hide_index=True)

status_a = "❌ Sangat tidak realistis" if p16 < 0.1 else ("⚠️ Risiko sangat tinggi" if p16 < 0.4 else "✅ Realistis")
status_b = "❌ Sangat tidak realistis" if p20 < 0.1 else ("⚠️ Risiko tinggi" if p20 < 0.4 else "✅ Realistis")
status_c = "❌ Sangat tidak realistis" if p24 < 0.1 else ("⚠️ Berisiko" if p24 < 0.4 else "✅ Realistis")
st.markdown(f"""
<div class="answer-box">
<b>📌 Jawaban P4:</b><br>
• Deadline <b>{dl_a} bulan</b>: probabilitas selesai <b>{p16:.1%}</b> → {status_a}<br>
• Deadline <b>{dl_b} bulan</b>: probabilitas selesai <b>{p20:.1%}</b> → {status_b}<br>
• Deadline <b>{dl_c} bulan</b>: probabilitas selesai <b>{p24:.1%}</b> → {status_c}<br>
• Deadline dengan probabilitas ≥ 50%: <b>≥ {float(np.percentile(total,50)):.1f} bulan</b><br>
• Deadline dengan probabilitas ≥ 80%: <b>≥ {p80v:.1f} bulan</b>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# P5 — PENGARUH PENAMBAHAN RESOURCE
# ============================================================================
st.markdown('<div class="section-title">❺ Pengaruh penambahan resource terhadap percepatan proyek</div>', unsafe_allow_html=True)

with st.spinner("Menghitung skenario resource…"):
    res_rows  = []
    res_arr   = []
    for scen in SKENARIO_RESOURCE:
        np.random.seed(int(seed) + 7)
        arr     = sim.simulasi_resource(scen['stages_pct'])
        new_mn  = float(np.array(arr).mean())
        reduksi = mean_total - new_mn
        pct     = reduksi / mean_total * 100
        s16 = float(np.mean(arr <= dl_a))
        s20 = float(np.mean(arr <= dl_b))
        s24 = float(np.mean(arr <= dl_c))
        penghematan_rp = reduksi * 500
        roi = ((penghematan_rp - scen['biaya_juta']) / scen['biaya_juta']) * 100
        res_rows.append({
            'Skenario':          scen['label'],
            'Deskripsi':         scen['deskripsi'],
            'Durasi Baru (bln)': round(new_mn, 2),
            'Reduksi (bln)':     round(reduksi, 2),
            'Hemat (%)':         round(pct, 1),
            f'Prob ≤{dl_a} bln': f"{s16:.1%}",
            f'Prob ≤{dl_b} bln': f"{s20:.1%}",
            f'Prob ≤{dl_c} bln': f"{s24:.1%}",
            'Biaya (juta Rp)':   scen['biaya_juta'],
            'Est. ROI':          f"{roi:.0f}%"
        })
        res_arr.append({'label': scen['label'], 'arr': arr,
                        'new_mn': new_mn, 'reduksi': reduksi,
                        's16': s16, 's20': s20, 's24': s24})

p5c1, p5c2 = st.columns(2)
with p5c1:
    fig5a = go.Figure()
    labels5  = [r['label'].split('(')[0].strip() for r in res_arr]
    reduksi5 = [r['reduksi'] for r in res_arr]
    fig5a.add_trace(go.Bar(
        y=labels5, x=reduksi5, orientation='h',
        marker_color=[WARNA[i % len(WARNA)] for i in range(len(labels5))],
        text=[f"{v:.2f} bln" for v in reduksi5], textposition='auto'
    ))
    fig5a.add_vline(x=0, line_color='black', line_width=1)
    fig5a.update_layout(title='Pengurangan Durasi per Skenario Resource',
        xaxis_title='Reduksi Durasi (Bulan)', height=380, margin=dict(t=50, b=40))
    st.plotly_chart(fig5a, use_container_width=True)

with p5c2:
    dl_labels = [f"≤{dl_a} bln", f"≤{dl_b} bln", f"≤{dl_c} bln"]
    fig5b = go.Figure()
    fig5b.add_trace(go.Bar(name='Baseline', x=dl_labels, y=[p16, p20, p24],
        marker_color='#94A3B8', opacity=0.7))
    for i, r in enumerate(res_arr):
        fig5b.add_trace(go.Bar(
            name=r['label'].split('(')[0].strip(),
            x=dl_labels, y=[r['s16'], r['s20'], r['s24']],
            marker_color=WARNA[i % len(WARNA)], opacity=0.85
        ))
    fig5b.update_layout(title='Peningkatan Probabilitas Selesai Tepat Waktu',
        yaxis_title='Probabilitas', yaxis_range=[0, 1.1], barmode='group',
        height=380, margin=dict(t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=8)))
    st.plotly_chart(fig5b, use_container_width=True)

# Overlay distribusi
fig5c = go.Figure()
fig5c.add_trace(go.Histogram(x=total, nbinsx=60, histnorm='probability density',
    marker_color='#94A3B8', opacity=0.5, name='Baseline'))
for i, r in enumerate(res_arr):
    fig5c.add_trace(go.Histogram(x=r['arr'], nbinsx=60, histnorm='probability density',
        marker_color=WARNA[i % len(WARNA)], opacity=0.5,
        name=r['label'].split('(')[0].strip()))
for dl, col in [(dl_a,'#7C3AED'),(dl_b,'#EA580C'),(dl_c,'#B45309')]:
    fig5c.add_vline(x=dl, line_dash='dashdot', line_color=col, line_width=1.5,
        annotation_text=f"{dl} bln")
fig5c.update_layout(title='Pergeseran Distribusi Durasi: Baseline vs Semua Skenario Resource',
    xaxis_title='Durasi Total (Bulan)', yaxis_title='Densitas',
    barmode='overlay', height=400, margin=dict(t=50, b=40))
st.plotly_chart(fig5c, use_container_width=True)

st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)

best_r = max(res_arr, key=lambda x: x['reduksi'])
st.markdown(f"""
<div class="answer-box">
<b>📌 Jawaban P5:</b><br>
• Penambahan resource memberikan percepatan <b>{min(r['reduksi'] for r in res_arr):.1f}–{max(r['reduksi'] for r in res_arr):.1f} bulan</b>.<br>
• Skenario terbaik: <b>{best_r['label']}</b> → reduksi <b>{best_r['reduksi']:.2f} bulan</b>
  (prob ≤{dl_b} bln meningkat dari {p20:.1%} → {best_r['s20']:.1%})<br>
• Urutan prioritas: <b>(1) Alat Berat (Pondasi)</b> → <b>(2) Insinyur MEP Senior</b>
  → <b>(3) Insinyur Ahli Lab</b> → <b>(4) Pekerja Terampil Struktur</b><br>
• Paket Lengkap (kombinasi semua) memberikan percepatan maksimal namun biaya tertinggi (Rp 620 juta).
</div>
""", unsafe_allow_html=True)

# ============================================================================
# RINGKASAN EKSEKUTIF
# ============================================================================
st.markdown('<div class="section-title">📋 Ringkasan Eksekutif</div>', unsafe_allow_html=True)
st.markdown(f"""
| # | Permasalahan | Hasil Simulasi |
|:---:|---|---|
| 1 | Total waktu proyek | **{mean_total:.1f} bulan** (mean) · Range 80%: **{p10v:.1f}–{p90v:.1f} bulan** · Jadwal aman: **{p80v:.1f} bulan** |
| 2 | Risiko keterlambatan | Rata-rata molor **+{delay:.1f} bulan** · Faktor utama: perubahan desain lab (45%), material teknis khusus (40%), cuaca buruk (35%) |
| 3 | Critical path | **{top1}** ({cp.loc[top1,'Prob. Critical']:.1%}) dan **{top2}** ({cp.loc[top2,'Prob. Critical']:.1%}) paling kritis |
| 4 | Prob. deadline | {dl_a} bln: **{p16:.1%}** · {dl_b} bln: **{p20:.1%}** · {dl_c} bln: **{p24:.1%}** · Prob ≥80% butuh: **{p80v:.1f} bln** |
| 5 | Penambahan resource | Percepatan **{min(r['reduksi'] for r in res_arr):.1f}–{max(r['reduksi'] for r in res_arr):.1f} bln** · Prioritas: Alat Berat + Insinyur MEP |
""")

st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#94A3B8; font-size:0.82rem;">
Simulasi Monte Carlo · {st.session_state.n_done:,} Iterasi · {len(KONFIGURASI)} Tahapan · Distribusi Triangular (PERT) · Seed: {seed}<br>
⚠️ Hasil ini adalah estimasi probabilistik — bukan prediksi deterministik.
</div>
""", unsafe_allow_html=True)
