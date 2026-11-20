# STREAMLIT APP: IMPLEMENTASI DATA SCIENCE DALAM SUPPLY CHAIN DISTRIBUTION
import os
import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from itertools import combinations
import io

st.set_page_config(layout="wide", page_title="LPG Nonpso - Perencanaan Distribusi Agen")

# --- KONSTANTA ---
KAPASITAS_TRUK = 560
TABLE_NAME = "distribusi"

# Gunakan lokasi file aplikasi, bukan current working directory.
# Jika database sudah ada di folder kerja terminal, gunakan sebagai fallback.
# Jangan biarkan SQLite diam-diam membuat database baru yang kosong.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_CANDIDATES = [
    os.path.join(BASE_DIR, "lpg_distribusi.db"),
    os.path.join(os.getcwd(), "lpg_distribusi.db"),
]
_existing_db = next((path for path in _DB_CANDIDATES if os.path.isfile(path)), None)
DB_FILE = _existing_db or os.path.join(BASE_DIR, "lpg_distribusi.db")


# ---------------------------- Helper Functions ----------------------------
@st.cache_data(show_spinner="Memuat data dari database...")
def load_data_from_db():
    if not os.path.isfile(DB_FILE):
        raise FileNotFoundError(
            "File database tidak ditemukan. Pastikan lpg_distribusi.db berada "
            f"di folder aplikasi: {BASE_DIR}"
        )

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,),
        )
        if cursor.fetchone() is None:
            raise ValueError(
                f"Tabel '{TABLE_NAME}' tidak ditemukan pada database: {DB_FILE}"
            )

        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total_record = cursor.fetchone()[0]
        if total_record == 0:
            return pd.DataFrame()

        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    if "date" in df.columns:
        parsed_date = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

        if parsed_date.isna().all() and not df["date"].isna().all():
            try:
                parsed_date = pd.to_datetime(
                    df["date"], format="mixed", errors="coerce"
                )
            except (TypeError, ValueError):
                pass

        df["date"] = parsed_date
        invalid_count = int(df["date"].isna().sum())

        if invalid_count == len(df):
            raise ValueError(
                "Data ditemukan di tabel distribusi, tetapi seluruh nilai kolom "
                "'date' gagal dikonversi. Periksa format tanggal pada database."
            )

        if invalid_count:
            df = df.dropna(subset=["date"]).copy()

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (
        sin(dLat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) ** 2
    )
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))


# ---------------------------- Database CRUD Functions ----------------------------
def execute_query(query, params=(), fetch=None):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        if fetch == "one":
            return cursor.fetchone()
        if fetch == "all":
            return cursor.fetchall()


def tambah_agen(nama, lat, lon):
    execute_query(
        "INSERT INTO agen (nama_agen, latitude_agen, longitude_agen) VALUES (?, ?, ?)",
        (nama, lat, lon),
    )


def get_semua_agen():
    return execute_query("SELECT * FROM agen ORDER BY nama_agen", fetch="all")


def update_agen(id, nama, lat, lon):
    execute_query(
        "UPDATE agen SET nama_agen=?, latitude_agen=?, longitude_agen=? WHERE agen_id=?",
        (nama, lat, lon, id),
    )


def hapus_agen(id):
    execute_query("DELETE FROM pangkalan WHERE agen_id=?", (id,))
    execute_query("DELETE FROM agen WHERE agen_id=?", (id,))


def tambah_pangkalan(nama, lat, lon, agen_id):
    execute_query(
        "INSERT INTO pangkalan (nama_pangkalan, latitude_pangkalan, longitude_pangkalan, agen_id) VALUES (?, ?, ?, ?)",
        (nama, lat, lon, agen_id),
    )


def get_semua_pangkalan():
    query = "SELECT p.pangkalan_id, p.nama_pangkalan, p.latitude_pangkalan, p.longitude_pangkalan, a.nama_agen, p.agen_id FROM pangkalan p LEFT JOIN agen a ON p.agen_id = a.agen_id ORDER BY a.nama_agen, p.nama_pangkalan"
    return execute_query(query, fetch="all")


def get_pangkalan_by_agen(agen_id):
    query = "SELECT p.pangkalan_id, p.nama_pangkalan, p.latitude_pangkalan, p.longitude_pangkalan FROM pangkalan p WHERE p.agen_id = ? ORDER BY p.nama_pangkalan"
    return execute_query(query, (agen_id,), fetch="all")


def update_pangkalan(id, nama, lat, lon, agen_id):
    execute_query(
        "UPDATE pangkalan SET nama_pangkalan=?, latitude_pangkalan=?, longitude_pangkalan=?, agen_id=? WHERE pangkalan_id=?",
        (nama, lat, lon, agen_id, id),
    )


def hapus_pangkalan(id):
    execute_query("DELETE FROM pangkalan WHERE pangkalan_id=?", (id,))


# ---------------------------- Main App Logic ----------------------------
st.sidebar.title("DATA SCIENCE DALAM SUPPLY CHAIN DISTRIBUTION DI DUNIA INDUSTRI")
st.sidebar.title("Kontrol Aplikasi")
try:
    df = load_data_from_db()
except Exception as e:
    st.error(f"Gagal memuat database '{DB_FILE}'. Error: {e}")
    st.info(
        "Pastikan file lpg_distribusi.db yang berisi data berada satu folder "
        "dengan data_science.py, lalu jalankan ulang aplikasi."
    )
    st.stop()

with st.sidebar.expander("Diagnostik Database"):
    st.write("File database:")
    st.code(DB_FILE)
    st.write("File ditemukan:", os.path.isfile(DB_FILE))
    st.write("Jumlah record terbaca:", len(df))
    st.write("Kolom:", df.columns.tolist())

# Inisialisasi semua state yang dibutuhkan untuk persistensi
if "editing_truck_no" not in st.session_state:
    st.session_state.editing_truck_no = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "prediction_figure" not in st.session_state:
    st.session_state.prediction_figure = None
if "anomaly_results" not in st.session_state:
    st.session_state.anomaly_results = None
if "anomaly_df_plot" not in st.session_state:
    st.session_state.anomaly_df_plot = None
if "selected_agen_anomaly" not in st.session_state:
    st.session_state.selected_agen_anomaly = None
if "rencana_per_truk" not in st.session_state:
    st.session_state.rencana_per_truk = {}
if "pangkalan_terjadwal_ids" not in st.session_state:
    st.session_state.pangkalan_terjadwal_ids = []
if "current_selection" not in st.session_state:
    st.session_state.current_selection = None

st.sidebar.title("Menu Aplikasi")
mode = st.sidebar.radio(
    "Pilih Menu:",
    [
        "Dashboard & Peta Sebaran",
        "Prediksi Rencana Penyaluran",
        "Analisis Pangkalan Prioritas",
        "Perencanaan Rute",
        "Manajemen Data",
        "Import Data",
    ],
    key="mode",
)

if df.empty and mode not in ["Import Data", "Manajemen Data"]:
    st.error(
        "Tabel 'distribusi' kosong. Harap isi data dari menu 'Import Data' terlebih dahulu."
    )
    st.stop()

# ---------------------------- 1. Dashboard & Peta Sebaran ----------------------------
if mode == "Dashboard & Peta Sebaran":
    st.title("Dashboard & Peta Sebaran")
    list_nama_agen = ["Semua Agen"] + sorted(df["nama_agen"].unique().tolist())
    selected_agen_nama = st.selectbox("Pilih Agen:", list_nama_agen)
    df_display = (
        df
        if selected_agen_nama == "Semua Agen"
        else df[df["nama_agen"] == selected_agen_nama]
    ).copy()
    st.subheader("Data Rencana Penyaluran Harian")
    if not df_display.empty:
        rows_to_show = st.slider(
            "Jumlah baris data:",
            min_value=5,
            max_value=len(df_display),
            value=min(10, len(df_display)),
        )
        st.dataframe(df_display.head(rows_to_show))
    else:
        st.info("Tidak ada data untuk ditampilkan.")
    if not df_display.empty:
        st.subheader("Peta Sebaran Agen dan Pangkalan")

        # --- Ubah typo koordinat menjadi tipe data numerik ---
        df_display["latitude_pangkalan"] = pd.to_numeric(
            df_display["latitude_pangkalan"], errors="coerce"
        )
        df_display["longitude_pangkalan"] = pd.to_numeric(
            df_display["longitude_pangkalan"], errors="coerce"
        )
        df_display.dropna(
            subset=["latitude_pangkalan", "longitude_pangkalan"], inplace=True
        )
        # --- Ubah typo koordinat menjadi tipe data numerik ---

        map_center = [
            df_display["latitude_pangkalan"].mean(),
            df_display["longitude_pangkalan"].mean(),
        ]
        m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
        agen_colors = {
            nama: color
            for nama, color in zip(
                sorted(df["nama_agen"].unique()),
                ["#0068C9", "#FF4B4B", "#2E2E2E", "#FFAA00"],
            )
        }
        for _, row in df_display.drop_duplicates("pangkalan_id").iterrows():
            folium.CircleMarker(
                location=[row["latitude_pangkalan"], row["longitude_pangkalan"]],
                radius=5,
                color=agen_colors.get(row["nama_agen"], "grey"),
                fill=True,
                fill_color=agen_colors.get(row["nama_agen"], "grey"),
                tooltip=f"{row['nama_pangkalan']} ({row['nama_agen']})",
            ).add_to(m)
        for _, row in df_display.drop_duplicates("agen_id").iterrows():
            folium.Marker(
                location=[row["latitude_agen"], row["longitude_agen"]],
                tooltip=row["nama_agen"],
                icon=folium.Icon(color="black", icon="home"),
            ).add_to(m)
        st_folium(m, width=1200, height=500)

# ---------------------------- 2. Prediksi Rencana Penyaluran ----------------------------
elif mode == "Prediksi Rencana Penyaluran":
    st.title("Prediksi Rencana Penyaluran (Forecasting)")
    agen_list_pred = sorted(df["nama_agen"].unique().tolist())
    selected_agen_pred = st.selectbox(
        "Pilih Agen:", options=agen_list_pred, key="pred_agen"
    )
    pangkalan_list = ["Semua Pangkalan"] + sorted(
        df[df["nama_agen"] == selected_agen_pred]["pangkalan_id"].unique().tolist()
    )
    selected_pangkalan = st.selectbox(
        "Pilih pangkalan:", options=pangkalan_list, key="pred_pangkalan"
    )
    days_to_predict = st.number_input(
        "Prediksi berapa hari ke depan?", min_value=1, max_value=30, value=7
    )

    if st.button("Latih & Buat Prediksi"):
        st.session_state.prediction_results, st.session_state.prediction_figure = (
            None,
            None,
        )
        with st.spinner("Memproses prediksi..."):
            all_predictions = []
            pangkalan_to_predict = (
                df[df["nama_agen"] == selected_agen_pred]["pangkalan_id"].unique()
                if selected_pangkalan == "Semua Pangkalan"
                else [selected_pangkalan]
            )

            for p_id in pangkalan_to_predict:
                df_p = df[df["pangkalan_id"] == p_id].copy().sort_values("date")
                if len(df_p) < 5:
                    continue
                df_p["penyaluran_lag1"] = df_p["rencana_penyaluran"].shift(1).fillna(0)
                df_p["penyaluran_lag7"] = df_p["rencana_penyaluran"].shift(7).fillna(0)
                df_p["dow"] = df_p["date"].dt.weekday
                features, X, y = (
                    ["dow", "penyaluran_lag1", "penyaluran_lag7"],
                    df_p[["dow", "penyaluran_lag1", "penyaluran_lag7"]],
                    df_p["rencana_penyaluran"],
                )

                if selected_pangkalan != "Semua Pangkalan":
                    train_size = int(len(df_p) * 0.8)
                    if train_size > 0 and len(df_p) - train_size > 0:
                        X_train, X_test, y_train, y_test = (
                            X[:train_size],
                            X[train_size:],
                            y[:train_size],
                            y[train_size:],
                        )
                        model_for_plot = RandomForestRegressor(
                            n_estimators=100, random_state=42
                        )
                        model_for_plot.fit(X_train, y_train)
                        y_pred = model_for_plot.predict(X_test)
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.set_title(
                            f"Prediksi vs Aktual untuk {df_p['nama_pangkalan'].iloc[0]}"
                        )
                        ax.plot(
                            df_p["date"].iloc[train_size:],
                            y_test.values,
                            label="Penyaluran Aktual",
                            marker="o",
                        )
                        ax.plot(
                            df_p["date"].iloc[train_size:],
                            y_pred,
                            label="Hasil Prediksi",
                            linestyle="--",
                        )
                        ax.legend()
                        ax.grid(True)
                        st.session_state.prediction_figure = fig

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                future_preds, history = [], df_p.copy()
                for _ in range(days_to_predict):
                    last_row = history.iloc[-1]
                    next_date = last_row["date"] + timedelta(days=1)
                    lag1, lag7 = last_row["rencana_penyaluran"], (
                        history.iloc[-7]["rencana_penyaluran"]
                        if len(history) >= 7
                        else lag1
                    )
                    features_for_pred = pd.DataFrame(
                        [
                            {
                                "dow": next_date.weekday(),
                                "penyaluran_lag1": lag1,
                                "penyaluran_lag7": lag7,
                            }
                        ]
                    )
                    prediction = model.predict(features_for_pred)[0]
                    future_preds.append(int(prediction))
                    new_row_data = {
                        "date": next_date,
                        "rencana_penyaluran": prediction,
                        "penyaluran_lag1": lag1,
                        "penyaluran_lag7": lag7,
                        "dow": next_date.weekday(),
                    }
                    history = pd.concat(
                        [history, pd.DataFrame([new_row_data])], ignore_index=True
                    )
                pred_dict = {
                    "Pangkalan ID": p_id,
                    "Nama Pangkalan": df_p["nama_pangkalan"].iloc[0],
                }
                for k, pred in enumerate(future_preds):
                    pred_dict[f"Prediksi Hari +{k+1}"] = pred
                all_predictions.append(pred_dict)
            if all_predictions:
                st.session_state.prediction_results = pd.DataFrame(all_predictions)

    if st.session_state.prediction_results is not None:
        st.subheader("Hasil Prediksi Terakhir")
        st.dataframe(st.session_state.prediction_results, use_container_width=True)
    if st.session_state.prediction_figure is not None:
        st.subheader("Grafik Perbandingan Aktual vs Prediksi")
        st.pyplot(st.session_state.prediction_figure)

# ---------------------------- 3. Analisis Pangkalan Prioritas ----------------------------
elif mode == "Analisis Pangkalan Prioritas":
    st.title("Analisis Pangkalan Prioritas & Anomali Penyaluran")
    st.subheader("Peringkat Pangkalan")
    penyaluran_summary = (
        df.groupby(["nama_agen", "nama_pangkalan"])["rencana_penyaluran"]
        .agg(["sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "Total", "mean": "Rata-Rata"})
    )
    st.dataframe(
        penyaluran_summary.sort_values("Total", ascending=False),
        use_container_width=True,
    )
    st.subheader("Deteksi Anomali")
    agen_list_anomaly = sorted(df["nama_agen"].unique().tolist())
    selected_agen = st.selectbox(
        "Pilih Agen:", options=agen_list_anomaly, key="anomaly_agen"
    )
    if st.button("Deteksi Anomali"):
        with st.spinner("Mendeteksi anomali..."):
            df_a = df[df["nama_agen"] == selected_agen].copy()
            model_iso = IsolationForest(contamination=0.05, random_state=42)
            df_a["anomaly"] = model_iso.fit_predict(df_a[["rencana_penyaluran"]])
            (
                st.session_state.anomaly_results,
                st.session_state.anomaly_df_plot,
                st.session_state.selected_agen_anomaly,
            ) = (df_a[df_a["anomaly"] == -1], df_a, selected_agen)
    if st.session_state.anomaly_results is not None:
        st.subheader("Hasil Deteksi Anomali Terakhir")
        if not st.session_state.anomaly_results.empty:
            st.dataframe(
                st.session_state.anomaly_results[
                    ["date", "nama_pangkalan", "rencana_penyaluran"]
                ],
                use_container_width=True,
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_title(
                f"Distribusi Penyaluran dan Anomali untuk {st.session_state.selected_agen_anomaly}"
            )
            ax.plot(
                st.session_state.anomaly_df_plot["date"],
                st.session_state.anomaly_df_plot["rencana_penyaluran"],
                label="Normal",
            )
            ax.scatter(
                st.session_state.anomaly_results["date"],
                st.session_state.anomaly_results["rencana_penyaluran"],
                color="red",
                label="Anomali",
            )
            ax.legend()
            st.pyplot(fig)
        else:
            st.success(
                f"Tidak ditemukan anomali signifikan untuk agen {st.session_state.selected_agen_anomaly}."
            )

# ---------------------------- 4. Perencanaan Rute ----------------------------
elif mode == "Perencanaan Rute":
    st.title("Perencanaan Rute Manual per Truk")

    # PERBAIKAN 3: Fungsi reset yang lebih aman
    def reset_perencanaan(full_reset=False):
        if full_reset:
            st.session_state.rencana_per_truk = {}
            st.session_state.pangkalan_terjadwal_ids = []
            st.session_state.editing_truck_no = None
            st.session_state.current_selection = None

    st.sidebar.subheader("Opsi Perencanaan")
    if st.sidebar.button("Reset Semua Perencanaan Rute"):
        reset_perencanaan(full_reset=True)
        st.rerun()

    st.subheader("1. Pilih Agen dan Tanggal")
    agen_list_opt = sorted(df["nama_agen"].unique().tolist())
    selected_agen_opt = st.selectbox(
        "Pilih Agen:", options=agen_list_opt, key="agen_rute"
    )
    df_agen = df[df["nama_agen"] == selected_agen_opt]
    if not df_agen.empty:
        latest_date = df_agen["date"].max()
        selected_date = st.date_input(
            "Pilih Tanggal Penyaluran", value=latest_date, key="tanggal_rute"
        )
        current_key = (selected_agen_opt, selected_date)
        if st.session_state.current_selection != current_key:
            reset_perencanaan(full_reset=True)
            st.session_state.current_selection = current_key
        df_today = df_agen[df_agen["date"].dt.date == selected_date].copy()

        if not df_today.empty:
            stok_awal_agen = df_today["stok_agen"].iloc[0]
            LOKASI_AGEN_TERPILIH = (
                df_today[["nama_agen", "latitude_agen", "longitude_agen"]]
                .iloc[0]
                .to_dict()
            )
            df_today["rencana_penyaluran"] = df_today["rencana_penyaluran"].clip(
                upper=200
            )

            # --- MODE EDIT RUTE ---
            if st.session_state.editing_truck_no:
                no_truk_edit = st.session_state.editing_truck_no
                st.header(f"📝 Mengedit Rute Truk #{no_truk_edit}")
                pangkalan_di_truk = st.session_state.rencana_per_truk[no_truk_edit][
                    "pangkalan"
                ]
                pangkalan_tersedia = df_today[
                    ~df_today["pangkalan_id"].isin(
                        st.session_state.pangkalan_terjadwal_ids
                    )
                    | df_today["pangkalan_id"].isin(pangkalan_di_truk["pangkalan_id"])
                ].copy()

                # PERBAIKAN 1: Tampilkan filter strategi saat mengedit
                st.subheader("Strategi Prioritas Pangkalan (opsional)")
                strategi = st.radio(
                    "Urutkan pangkalan tersedia:",
                    ("Jumlah Tabung Terbanyak", "Jarak Terdekat", "Kombinasi"),
                    horizontal=True,
                    key=f"strategi_edit_{no_truk_edit}",
                )

                # PERBAIKAN 1: Terapkan logika sorting dengan .apply
                if not pangkalan_tersedia.empty:
                    if strategi == "Jarak Terdekat":
                        pangkalan_tersedia["jarak"] = pangkalan_tersedia.apply(
                            lambda r: haversine_distance(
                                LOKASI_AGEN_TERPILIH["latitude_agen"],
                                LOKASI_AGEN_TERPILIH["longitude_agen"],
                                r["latitude_pangkalan"],
                                r["longitude_pangkalan"],
                            ),
                            axis=1,
                        )
                        pangkalan_tersedia = pangkalan_tersedia.sort_values("jarak")
                    elif strategi == "Kombinasi":
                        pangkalan_tersedia["jarak"] = pangkalan_tersedia.apply(
                            lambda r: haversine_distance(
                                LOKASI_AGEN_TERPILIH["latitude_agen"],
                                LOKASI_AGEN_TERPILIH["longitude_agen"],
                                r["latitude_pangkalan"],
                                r["longitude_pangkalan"],
                            ),
                            axis=1,
                        )
                        pangkalan_tersedia["skor"] = pangkalan_tersedia[
                            "rencana_penyaluran"
                        ] / (pangkalan_tersedia["jarak"] + 0.1)
                        pangkalan_tersedia = pangkalan_tersedia.sort_values(
                            "skor", ascending=False
                        )
                    else:
                        pangkalan_tersedia = pangkalan_tersedia.sort_values(
                            "rencana_penyaluran", ascending=False
                        )

                default_selection_display = (
                    pangkalan_di_truk["nama_pangkalan"]
                    + " ("
                    + pangkalan_di_truk["rencana_penyaluran"].astype(str)
                    + " tbg)"
                ).tolist()
                pangkalan_tersedia["display"] = (
                    pangkalan_tersedia["nama_pangkalan"]
                    + " ("
                    + pangkalan_tersedia["rencana_penyaluran"].astype(str)
                    + " tbg)"
                )

                with st.form(f"form_edit_truk_{no_truk_edit}"):
                    pilihan_baru = st.multiselect(
                        "Edit pilihan pangkalan:",
                        options=pangkalan_tersedia["display"].tolist(),
                        default=default_selection_display,
                    )
                    df_pilihan_baru = pangkalan_tersedia[
                        pangkalan_tersedia["display"].isin(pilihan_baru)
                    ]
                    muatan_baru = df_pilihan_baru["rencana_penyaluran"].sum()
                    if muatan_baru > KAPASITAS_TRUK:
                        st.error(f"Kelebihan Muatan! {muatan_baru}/{KAPASITAS_TRUK}.")
                    else:
                        st.success(f"Muatan Baru: {muatan_baru}/{KAPASITAS_TRUK}.")
                    col1, col2 = st.columns(2)
                    if col1.form_submit_button("Simpan Perubahan"):
                        if muatan_baru <= KAPASITAS_TRUK:
                            st.session_state.rencana_per_truk[no_truk_edit] = {
                                "pangkalan": df_pilihan_baru,
                                "total_muatan": muatan_baru,
                            }
                            all_scheduled_ids = [
                                p_id
                                for truk in st.session_state.rencana_per_truk.values()
                                for p_id in truk["pangkalan"]["pangkalan_id"].tolist()
                            ]
                            st.session_state.pangkalan_terjadwal_ids = list(
                                set(all_scheduled_ids)
                            )
                            st.session_state.editing_truck_no = None
                            st.rerun()
                    if col2.form_submit_button("Batal"):
                        st.session_state.editing_truck_no = None
                        st.rerun()

            # --- MODE PERENCANAAN BARU ---
            else:
                total_penyaluran_terjadwal = sum(
                    detail["total_muatan"]
                    for detail in st.session_state.rencana_per_truk.values()
                )
                sisa_stok = stok_awal_agen - total_penyaluran_terjadwal
                col1, col2 = st.columns(2)
                col1.metric(label="Stok Awal", value=f"{stok_awal_agen} tbg")
                col2.metric(label="Sisa Stok", value=f"{sisa_stok} tbg")
                st.markdown("---")
                pangkalan_tersedia = df_today[
                    ~df_today["pangkalan_id"].isin(
                        st.session_state.pangkalan_terjadwal_ids
                    )
                ]
                st.subheader("2. Pilih Strategi Prioritas Pangkalan")
                strategi = st.radio(
                    "Urutkan daftar pangkalan:",
                    ("Jumlah Tabung Terbanyak", "Jarak Terdekat", "Kombinasi"),
                    horizontal=True,
                    key="strategi_sorting",
                )
                if not pangkalan_tersedia.empty:
                    pangkalan_tersedia = (
                        pangkalan_tersedia.copy()
                    )  # Hindari SettingWithCopyWarning
                    if strategi == "Jarak Terdekat":
                        pangkalan_tersedia["jarak"] = pangkalan_tersedia.apply(
                            lambda r: haversine_distance(
                                LOKASI_AGEN_TERPILIH["latitude_agen"],
                                LOKASI_AGEN_TERPILIH["longitude_agen"],
                                r["latitude_pangkalan"],
                                r["longitude_pangkalan"],
                            ),
                            axis=1,
                        )
                        pangkalan_tersedia = pangkalan_tersedia.sort_values("jarak")
                    elif strategi == "Kombinasi":
                        pangkalan_tersedia["jarak"] = pangkalan_tersedia.apply(
                            lambda r: haversine_distance(
                                LOKASI_AGEN_TERPILIH["latitude_agen"],
                                LOKASI_AGEN_TERPILIH["longitude_agen"],
                                r["latitude_pangkalan"],
                                r["longitude_pangkalan"],
                            ),
                            axis=1,
                        )
                        pangkalan_tersedia["skor"] = pangkalan_tersedia[
                            "rencana_penyaluran"
                        ] / (pangkalan_tersedia["jarak"] + 0.1)
                        pangkalan_tersedia = pangkalan_tersedia.sort_values(
                            "skor", ascending=False
                        )
                    else:
                        pangkalan_tersedia = pangkalan_tersedia.sort_values(
                            "rencana_penyaluran", ascending=False
                        )

                jumlah_truk_tersedia = math.ceil(stok_awal_agen / KAPASITAS_TRUK)
                truk_selanjutnya = len(st.session_state.rencana_per_truk) + 1
                if (
                    sisa_stok > 0
                    and truk_selanjutnya <= jumlah_truk_tersedia
                    and not pangkalan_tersedia.empty
                ):
                    st.header(f"🚚 Perencanaan untuk Truk #{truk_selanjutnya}")
                    options = pangkalan_tersedia
                    options["display"] = (
                        options["nama_pangkalan"]
                        + " ("
                        + options["rencana_penyaluran"].astype(str)
                        + " tbg)"
                    )
                    with st.form(f"form_truk_{truk_selanjutnya}"):
                        pilihan_pangkalan = st.multiselect(
                            "Pilih pangkalan:", options=options["display"].tolist()
                        )
                        df_pilihan = options[options["display"].isin(pilihan_pangkalan)]
                        total_muatan = df_pilihan["rencana_penyaluran"].sum()
                        if total_muatan > KAPASITAS_TRUK:
                            st.error(
                                f"Kelebihan Muatan! {total_muatan}/{KAPASITAS_TRUK}."
                            )
                        elif total_muatan > sisa_stok:
                            st.error(f"Stok Tidak Cukup! Sisa: {sisa_stok}.")
                        else:
                            st.success(
                                f"Muatan Saat Ini: {total_muatan}/{KAPASITAS_TRUK}."
                            )
                        if st.form_submit_button(
                            f"Konfirmasi Truk #{truk_selanjutnya}"
                        ):
                            if (
                                total_muatan > KAPASITAS_TRUK
                                or total_muatan > sisa_stok
                                or not pilihan_pangkalan
                            ):
                                st.warning("Gagal.")
                            else:
                                st.session_state.rencana_per_truk[truk_selanjutnya] = {
                                    "pangkalan": df_pilihan,
                                    "total_muatan": total_muatan,
                                }
                                st.session_state.pangkalan_terjadwal_ids.extend(
                                    df_pilihan["pangkalan_id"].tolist()
                                )
                                st.rerun()
            st.markdown("---")
            st.header("Ringkasan Rencana Pengiriman")
            if st.session_state.rencana_per_truk:
                for no_truk, detail in st.session_state.rencana_per_truk.items():
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.subheader(
                                f"Rute Truk #{no_truk} (Muatan: {detail['total_muatan']} tbg)"
                            )
                            for _, p in detail["pangkalan"].iterrows():
                                st.markdown(
                                    f"- {p['nama_pangkalan']} ({p['rencana_penyaluran']} tbg)"
                                )
                        with col2:
                            if st.button("Edit", key=f"edit_{no_truk}"):
                                st.session_state.editing_truck_no = no_truk
                                st.rerun()
                st.subheader("Peta Visualisasi Rute")
                m = folium.Map(
                    location=[
                        LOKASI_AGEN_TERPILIH["latitude_agen"],
                        LOKASI_AGEN_TERPILIH["longitude_agen"],
                    ],
                    zoom_start=12,
                    tiles="OpenStreetMap",
                )
                folium.Marker(
                    location=[
                        LOKASI_AGEN_TERPILIH["latitude_agen"],
                        LOKASI_AGEN_TERPILIH["longitude_agen"],
                    ],
                    tooltip=LOKASI_AGEN_TERPILIH["nama_agen"],
                    icon=folium.Icon(color="black", icon="home"),
                ).add_to(m)
                colors = ["#0068C9", "#FF4B4B", "#2E2E2E", "#FFAA00"]
                for i, (no_truk, detail) in enumerate(
                    st.session_state.rencana_per_truk.items()
                ):
                    color = colors[i % len(colors)]
                    route_points = [
                        [
                            LOKASI_AGEN_TERPILIH["latitude_agen"],
                            LOKASI_AGEN_TERPILIH["longitude_agen"],
                        ]
                    ]
                    stop_number = 1
                    for _, pangkalan in detail["pangkalan"].iterrows():
                        p_lat, p_lon = (
                            pangkalan["latitude_pangkalan"],
                            pangkalan["longitude_pangkalan"],
                        )
                        route_points.append([p_lat, p_lon])
                        icon_html = f"""<div style="font-family: Arial; font-size: 12px; font-weight: bold; color: white; background-color: {color}; border: 2px solid white; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 21px; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">{stop_number}</div>"""
                        folium.Marker(
                            location=[p_lat, p_lon],
                            tooltip=f"{stop_number}. {pangkalan['nama_pangkalan']}",
                            icon=DivIcon(
                                icon_size=(30, 30), icon_anchor=(15, 15), html=icon_html
                            ),
                        ).add_to(m)
                        stop_number += 1
                    if len(route_points) > 1:
                        folium.PolyLine(
                            route_points,
                            color=color,
                            weight=2,
                            opacity=0.8,
                            dash_array="5, 5",
                        ).add_to(m)
                st_folium(m, width=1200, height=600)

# ---------------------------- 5. Manajemen Data (CRUD) ----------------------------
elif mode == "Manajemen Data":
    st.title("Manajemen Data Agen dan Pangkalan")
    try:
        tab1, tab2 = st.tabs(["Manajemen Agen", "Manajemen Pangkalan"])
        with tab1:
            st.header("Tambah Agen")
            with st.form("form_tambah_agen", clear_on_submit=True):
                nama_agen_baru = st.text_input("Nama Agen")
                lat_agen_baru = st.number_input("Latitude", format="%.6f")
                lon_agen_baru = st.number_input("Longitude", format="%.6f")
                if st.form_submit_button("Simpan"):
                    if nama_agen_baru:
                        tambah_agen(nama_agen_baru, lat_agen_baru, lon_agen_baru)
                        st.success(f"Agen '{nama_agen_baru}' ditambahkan!")
                        st.rerun()
            st.header("Edit atau Hapus Agen")
            list_agen = get_semua_agen()
            if list_agen:
                nama_agen_options = {agen[1]: agen[0] for agen in list_agen}
                nama_agen_pilihan = st.selectbox(
                    "Pilih Agen",
                    options=nama_agen_options.keys(),
                    key="agen_edit_select",
                )
                id_agen_pilihan = nama_agen_options[nama_agen_pilihan]
                data_agen_pilihan = [
                    agen for agen in list_agen if agen[0] == id_agen_pilihan
                ][0]
                with st.form("form_edit_agen"):
                    nama_agen_edit = st.text_input(
                        "Nama Agen", value=data_agen_pilihan[1]
                    )
                    lat_agen_edit = st.number_input(
                        "Latitude", value=data_agen_pilihan[2], format="%.6f"
                    )
                    lon_agen_edit = st.number_input(
                        "Longitude", value=data_agen_pilihan[3], format="%.6f"
                    )
                    col1, col2 = st.columns(2)
                    if col1.form_submit_button("Update"):
                        update_agen(
                            id_agen_pilihan,
                            nama_agen_edit,
                            lat_agen_edit,
                            lon_agen_edit,
                        )
                        st.success(f"Data diperbarui!")
                        st.rerun()
                    if col2.form_submit_button("Hapus"):
                        hapus_agen(id_agen_pilihan)
                        st.warning(f"Agen '{nama_agen_pilihan}' dihapus!")
                        st.rerun()
        with tab2:
            st.header("Tambah Pangkalan")
            list_agen = get_semua_agen()
            if list_agen:
                agen_options_dict = {agen[1]: agen[0] for agen in list_agen}
                with st.form("form_tambah_pangkalan", clear_on_submit=True):
                    nama_pangkalan_baru = st.text_input("Nama Pangkalan")
                    lat_pangkalan_baru = st.number_input(
                        "Latitude", format="%.6f", key="lat_p_new"
                    )
                    lon_pangkalan_baru = st.number_input(
                        "Longitude", format="%.6f", key="lon_p_new"
                    )
                    nama_agen_induk = st.selectbox(
                        "Pilih Agen Induk",
                        options=agen_options_dict.keys(),
                        key="agen_p_new",
                    )
                    if st.form_submit_button("Simpan"):
                        if nama_pangkalan_baru:
                            id_agen_induk = agen_options_dict[nama_agen_induk]
                            tambah_pangkalan(
                                nama_pangkalan_baru,
                                lat_pangkalan_baru,
                                lon_pangkalan_baru,
                                id_agen_induk,
                            )
                            st.success(f"Pangkalan ditambahkan!")
                            st.rerun()
                st.header("Edit atau Hapus Pangkalan")
                # PERBAIKAN 4: Filter pangkalan berdasarkan agen
                selected_agen_name = st.selectbox(
                    "1. Filter berdasarkan Agen",
                    options=agen_options_dict.keys(),
                    key="filter_agen_pangkalan",
                )
                id_agen_filter = agen_options_dict[selected_agen_name]
                list_pangkalan_filtered = get_pangkalan_by_agen(id_agen_filter)
                if list_pangkalan_filtered:
                    pangkalan_options = {p[1]: p[0] for p in list_pangkalan_filtered}
                    pangkalan_name_selected = st.selectbox(
                        "2. Pilih Pangkalan untuk Diedit/Dihapus",
                        options=pangkalan_options.keys(),
                    )
                    p_id_selected = pangkalan_options[pangkalan_name_selected]
                    data_p_selected = [
                        p for p in get_semua_pangkalan() if p[0] == p_id_selected
                    ][0]
                    with st.form("form_edit_pangkalan"):
                        nama_pangkalan_edit = st.text_input(
                            "Nama Pangkalan", value=data_p_selected[1]
                        )
                        lat_pangkalan_edit = st.number_input(
                            "Latitude", value=data_p_selected[2], format="%.6f"
                        )
                        lon_pangkalan_edit = st.number_input(
                            "Longitude", value=data_p_selected[3], format="%.6f"
                        )
                        agen_keys = list(agen_options_dict.keys())
                        current_agen_name = data_p_selected[4]
                        default_index = (
                            agen_keys.index(current_agen_name)
                            if current_agen_name in agen_keys
                            else 0
                        )
                        nama_agen_induk_edit = st.selectbox(
                            "Pilih Agen Induk", options=agen_keys, index=default_index
                        )
                        col1, col2 = st.columns(2)
                        if col1.form_submit_button("Update"):
                            id_agen_baru = agen_options_dict[nama_agen_induk_edit]
                            update_pangkalan(
                                p_id_selected,
                                nama_pangkalan_edit,
                                lat_pangkalan_edit,
                                lon_pangkalan_edit,
                                id_agen_baru,
                            )
                            st.success(f"Data diperbarui!")
                            st.rerun()
                        if col2.form_submit_button("Hapus"):
                            hapus_pangkalan(p_id_selected)
                            st.warning(f"Pangkalan dihapus!")
                            st.rerun()
                else:
                    st.info(
                        f"Tidak ada pangkalan yang terdaftar di bawah agen {selected_agen_name}."
                    )
    except Exception as e:
        st.error(
            f"Gagal memuat manajemen data. Pastikan tabel 'agen' dan 'pangkalan' sudah ada. Error: {e}"
        )

# ---------------------------- 6. Import Data ----------------------------
elif mode == "Import Data":
    st.title("Import Data Perencanaan Penyaluran")
    st.info(
        "Unggah file Excel (.xlsx) untuk menambahkan data perencanaan dan secara otomatis menyinkronkan data master Agen & Pangkalan."
    )

    @st.cache_data
    def create_template():
        template_df = pd.DataFrame(
            {
                "date": ["2025-09-24"],
                "nama_agen": ["AGEN BARU DARI EXCEL"],
                "agen_id": [99],
                "nama_pangkalan": ["PANGKALAN BARU DARI EXCEL"],
                "pangkalan_id": [999],
                "latitude_agen": [-3.60],
                "longitude_agen": [98.60],
                "latitude_pangkalan": [-3.61],
                "longitude_pangkalan": [98.61],
                "rencana_penyaluran": [150],
                "stok_agen": [2000],
            }
        )
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="Sheet1")
        return buffer.getvalue()

    st.download_button(
        label="📥 Unduh Template Excel",
        data=create_template(),
        file_name="template_perencanaan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    uploaded_file = st.file_uploader("Pilih file untuk diunggah", type=["xlsx"])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            st.subheader("Pratinjau Data")
            st.dataframe(df_upload.head())
            required_columns = {
                "date",
                "nama_agen",
                "agen_id",
                "nama_pangkalan",
                "pangkalan_id",
                "rencana_penyaluran",
                "stok_agen",
                "latitude_agen",
                "longitude_agen",
                "latitude_pangkalan",
                "longitude_pangkalan",
            }
            if not required_columns.issubset(df_upload.columns):
                st.error(
                    f"File tidak valid. Pastikan ada semua kolom ini: {', '.join(required_columns)}"
                )
            else:
                if st.button("Mulai Proses Import"):
                    with st.spinner("Menyimpan data..."):
                        with sqlite3.connect(DB_FILE) as conn:
                            df_agen_unique = df_upload[
                                [
                                    "agen_id",
                                    "nama_agen",
                                    "latitude_agen",
                                    "longitude_agen",
                                ]
                            ].drop_duplicates(subset=["agen_id"])
                            for _, r in df_agen_unique.iterrows():
                                conn.cursor().execute(
                                    "INSERT OR IGNORE INTO agen (agen_id, nama_agen, latitude_agen, longitude_agen) VALUES (?, ?, ?, ?)",
                                    (
                                        r["agen_id"],
                                        r["nama_agen"],
                                        r["latitude_agen"],
                                        r["longitude_agen"],
                                    ),
                                )
                            df_pangkalan_unique = df_upload[
                                [
                                    "pangkalan_id",
                                    "nama_pangkalan",
                                    "latitude_pangkalan",
                                    "longitude_pangkalan",
                                    "agen_id",
                                ]
                            ].drop_duplicates(subset=["pangkalan_id"])
                            for _, r in df_pangkalan_unique.iterrows():
                                conn.cursor().execute(
                                    "INSERT OR IGNORE INTO pangkalan (pangkalan_id, nama_pangkalan, latitude_pangkalan, longitude_pangkalan, agen_id) VALUES (?, ?, ?, ?, ?)",
                                    (
                                        r["pangkalan_id"],
                                        r["nama_pangkalan"],
                                        r["latitude_pangkalan"],
                                        r["longitude_pangkalan"],
                                        r["agen_id"],
                                    ),
                                )
                            df_upload.to_sql(
                                TABLE_NAME, conn, if_exists="append", index=False
                            )
                    st.success(
                        f"{len(df_upload)} baris data berhasil ditambahkan dan disinkronkan!"
                    )
                    st.cache_data.clear()
                    st.rerun()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
