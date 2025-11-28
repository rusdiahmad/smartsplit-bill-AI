import streamlit as st
from PIL import Image
import pandas as pd
import json
import io
import torch
import numpy as np

# Import Donut model and processor from Hugging Face
try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError:
    st.error("Pastikan library 'transformers' sudah terinstal. Cek file requirements.txt.")

# --- 1. SETUP MODEL AI ---
# Inisialisasi variabel global untuk mencegah NameError
PROCESSOR = None
MODEL = None
DEVICE = "cpu"

@st.cache_resource
def load_ai_model():
    """Memuat model AI Donut untuk ekstraksi nota."""
    try:
        model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Pindahkan model ke GPU jika tersedia, jika tidak, gunakan CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return processor, model, device
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model AI: {e}")
        return None, None, "cpu"

# Panggil fungsi pemuatan model
PROCESSOR, MODEL, DEVICE = load_ai_model()

if MODEL is not None:
    st.sidebar.success("Model AI siap digunakan!")
else:
    st.sidebar.error("Model AI gagal dimuat. Cek log dan dependensi.")


# --- 2. FUNGSI UTILITY DATA CLEANSING ---

def clean_price_string(price_str):
    """Membersihkan string harga dari karakter non-numerik dan mengkonversi ke float."""
    if not isinstance(price_str, str):
        return 0.0
    # Hapus koma/titik sebagai pemisah ribuan, lalu konversi
    cleaned = price_str.replace(',', '').replace('.', '')
    # Coba konversi ke float
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

# --- 3. FUNGSI EKSTRAKSI DATA DARI GAMBAR ---

def extract_data_from_image(image_file):
    """Menjalankan inference model AI (Donut) pada gambar nota dan mengembalikan data terstruktur."""
    if MODEL is None:
        st.error("Model AI tidak tersedia.")
        return None, None, None, None, None

    try:
        image = Image.open(image_file).convert("RGB")
        
        # --- LOGIKA INFERENCE DONUT ---
        prompt = "<s>"
        pixel_values = PROCESSOR(image, return_tensors="pt").pixel_values
        decoder_input_ids = PROCESSOR.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

        outputs = MODEL.generate(
            pixel_values.to(DEVICE),
            decoder_input_ids=decoder_input_ids.to(DEVICE),
            max_length=MODEL.decoder.config.max_length,
            early_stopping=True,
            num_beams=1,
            return_dict_in_generate=True,
        )
        
        # Ambil sequences dari output objek (Solusi error 'Tensor' object has no attribute 'sequences')
        output_ids = outputs.sequences
        output = PROCESSOR.batch_decode(output_ids[:, decoder_input_ids.shape[1]:])[0]
        extracted_json = PROCESSOR.token2json(output)
        
        st.info("Ekstraksi Data Selesai.")
        
        # --- Pemrosesan dan Konversi ke DataFrame ---
        items_list = []
        if 'menu' in extracted_json:
            for item in extracted_json['menu']:
                # Bersihkan data harga
                price_per_item = clean_price_string(item.get('price', '0'))
                total_item_price = clean_price_string(item.get('cnt', '0'))
                qty = int(item.get('qty', 1))

                items_list.append({
                    'Item': item.get('nm', 'Item Tanpa Nama'),
                    'Qty': qty,
                    'Price_per_item': price_per_item,
                    'Total_Item_Price': total_item_price
                })
        
        df_items = pd.DataFrame(items_list)
        
        # Ekstrak dan bersihkan biaya tambahan dan total
        subtotal = clean_price_string(extracted_json.get('sub_total', '0'))
        total_bill = clean_price_string(extracted_json.get('total', '0'))
        
        # Hitung biaya tambahan (Add-ons)
        # Sesuai contoh nota yang diupload, subtotal dan total bisa berbeda karena diskon/biaya lain.
        # Total Addons = Total Bill - Subtotal Item (semua item yang ada di list)
        total_item_cost_from_df = df_items['Total_Item_Price'].sum()
        total_addons = total_bill - total_item_cost_from_df
        
        # Set total_addons minimal 0
        total_addons = max(0, total_addons)
        
        return df_items, total_item_cost_from_df, total_bill, total_addons, extracted_json

    except Exception as e:
        st.error(f"Error saat menjalankan AI inference: {e}")
        return pd.DataFrame(), 0, 0, 0, {}


# --- 4. LOGIKA PEMBAGIAN BILL (SPLIT LOGIC) ---

def calculate_final_split(df_items_assigned, total_bill, total_addons, participants):
    """
    Menghitung total yang harus dibayar per orang.
    Logika: (Total Harga Item per Orang) + (Porsi Biaya Tambahan dibagi rata)
    """
    if not participants or total_bill <= 0:
        return {}

    # 1. Hitung Total Biaya Item per Orang
    person_item_totals = df_items_assigned.groupby('Assigned_To')['Total_Item_Price'].sum()

    # 2. Hitung Porsi Biaya Tambahan per Orang (Dibagi Rata)
    # Total biaya tambahan dibagi rata kepada semua partisipan yang dipilih.
    num_participants = len(participants)
    addon_per_person = total_addons / num_participants if num_participants > 0 else 0

    # 3. Hitung Total Bayar Akhir
    final_bill = {}
    total_calculated_check = 0
    
    for name in participants:
        # Gunakan get(name, 0) agar tidak error jika nama tidak ada di group (misal, tidak ada item yang ditugaskan)
        item_cost = person_item_totals.get(name, 0)
        
        # Total bayar = Biaya Item + Biaya Tambahan (porsi rata)
        final_cost = item_cost + addon_per_person
        final_bill[name] = final_cost
        total_calculated_check += final_cost

    # Verifikasi total (untuk debugging/audit)
    if abs(total_calculated_check - total_bill) > 0.01:
        st.sidebar.warning(f"Total Verifikasi: {total_calculated_check:.2f} TIDAK SAMA dengan Bill: {total_bill:.2f}")

    return final_bill


# --- 5. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="SmartSplit Bill AI", layout="wide")
st.title("💰 SmartSplit Bill AI Prototype")
st.markdown("Aplikasi untuk mengekstrak data nota dan membagi tagihan secara otomatis.")
st.markdown("---")

# Inisialisasi session state
if 'items_df' not in st.session_state:
    st.session_state['items_df'] = pd.DataFrame()

# --- Bagian A: Upload Gambar dan Ekstraksi AI ---
st.header("1. Upload Nota Pembelian")
uploaded_file = st.file_uploader("Pilih gambar nota (.jpg atau .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col_img, col_metrics = st.columns([1, 1])
    
    with col_img:
        st.image(uploaded_file, caption="Nota Diunggah", use_column_width=True)
    
    df_items, subtotal, total_bill, total_addons, extracted_json = extract_data_from_image(uploaded_file)
    
    if total_bill > 0:
        st.session_state['items_df'] = df_items
        st.session_state['total_bill'] = total_bill
        st.session_state['total_addons'] = total_addons
        
        with col_metrics:
            st.subheader("Ringkasan Bill")
            st.metric(label="Subtotal Item (Dari AI)", value=f"Rp{subtotal:,.0f}")
            st.metric(label="Total Biaya Tambahan (Diestimasi)", value=f"Rp{total_addons:,.0f}")
            st.metric(label="TOTAL AKHIR BILL", value=f"Rp{total_bill:,.0f}")
        
        st.markdown("---")
        
        # --- Bagian B: Input Partisipan ---
        st.header("2. Input Partisipan")
        
        participants_input = st.text_input(
            "Masukkan nama-nama partisipan (dipisahkan koma, contoh: Andi, Budi, Clara)",
            value="Partisipan A, Partisipan B",
            key="participants_input"
        )
        participants = [name.strip() for name in participants_input.split(',') if name.strip()]
        
        if participants:
            st.session_state['participants'] = participants
            st.success(f"Partisipan: {', '.join(participants)}")
            st.markdown("---")
            
            # --- Bagian C: Penugasan Item ---
            st.header("3. Tugaskan Item ke Partisipan")
            
            # Persiapan DataFrame untuk penugasan
            df_assignment = st.session_state['items_df'].copy()
            assigned_data = []
            
            # Buat kolom untuk input item
            col1, col2, col3 = st.columns([0.5, 2, 1])
            col1.write("**Qty**")
            col2.write("**Item**")
            col3.write("**Ditugaskan Kepada**")

            for index, row in df_assignment.iterrows():
                with col1:
                    st.write(f"**{row['Qty']}**")
                with col2:
                    st.write(f"{row['Item']} - *Rp{row['Total_Item_Price']:,.0f}*")
                with col3:
                    assigned_person = st.selectbox(
                        "Pilih Orang",
                        options=participants,
                        key=f"assign_{index}",
                        label_visibility="collapsed"
                    )
                
                assigned_data.append({
                    'Item': row['Item'],
                    'Total_Item_Price': row['Total_Item_Price'],
                    'Assigned_To': assigned_person
                })

            df_items_assigned = pd.DataFrame(assigned_data)
            st.session_state['df_items_assigned'] = df_items_assigned
            st.markdown("---")
            
            # --- Bagian D: Laporan Akhir ---
            st.header("4. Total Bayar per Orang")
            
            final_split = calculate_final_split(
                df_items_assigned, 
                st.session_state['total_bill'], 
                st.session_state['total_addons'], 
                participants
            )
            
            if final_split:
                final_df = pd.DataFrame(list(final_split.items()), columns=['Partisipan', 'Total Bayar'])
                
                st.subheader("Laporan Akhir Pembayaran")
                # Tampilkan hasil dalam tabel yang diformat dengan baik
                st.table(final_df.style.format({'Total Bayar': 'Rp{:,.0f}'}))
                
                st.markdown(
                    f"""
                    ### Total Verifikasi: **Rp{final_df['Total Bayar'].sum():,.0f}**
                    *(Total ini sama dengan Total Bill Akhir: Rp{total_bill:,.0f})*
                    """
                )
            
        else:
            st.warning("Masukkan minimal satu nama partisipan di Langkah 2.")
    else:
        st.warning("Silakan unggah nota atau periksa apakah ekstraksi AI berhasil mendapatkan Total Bill.")
