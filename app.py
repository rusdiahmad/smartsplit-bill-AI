## app.py

import streamlit as st
from PIL import Image
import pandas as pd
import json
# Import libraries untuk model AI yang Anda pilih (contoh: Donut/LayoutLM)
from transformers import DonutProcessor, VisionEncoderDecoderModel 
import torch
import io
import time # Untuk mengukur latency (opsional di app.py, wajib di research_notebook)

# --- 1. SETUP MODEL AI ---

# Definisikan variabel global sebelum blok try (Solusi NameError)
PROCESSOR = None
MODEL = None
DEVICE = "cpu"

try:
    @st.cache_resource
    def load_ai_model():
        # ... (Logika pemuatan model Anda, contoh Donut) ...
        model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return processor, model, device

    PROCESSOR, MODEL, DEVICE = load_ai_model()
    st.sidebar.success("Model AI siap digunakan!")
except Exception as e:
    # Jika gagal, MODEL dan PROCESSOR sudah diset None di awal.
    st.sidebar.error(f"Gagal memuat model AI: {e}. Pastikan dependensi diinstal.")
    
# --- 2. FUNGSI EKSTRAKSI DATA DARI GAMBAR ---
def extract_data_from_image(image_file):
    """
    Fungsi ini menjalankan inference model AI pada gambar nota.
    Output harus berupa DataFrame atau JSON yang terstruktur.
    """
    if MODEL is None:
        st.error("Model AI belum dimuat.")
        return None

    try:
        # Konversi file upload Streamlit ke objek Image
        image = Image.open(image_file).convert("RGB")
        
        # --- LOGIKA INFERENCE MODEL Pilihan Anda ---
        # Contoh Donut (Ganti dengan logika model Anda)
        prompt = "<s>"
        pixel_values = PROCESSOR(image, return_tensors="pt").pixel_values
        decoder_input_ids = PROCESSOR.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

        output_ids = MODEL.generate(
            pixel_values.to(DEVICE),
            decoder_input_ids=decoder_input_ids.to(DEVICE),
            max_length=MODEL.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            eos_token_id=PROCESSOR.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[PROCESSOR.tokenizer.unk_token_id]],
        ).sequences
        
        output = PROCESSOR.batch_decode(output_ids[:, decoder_input_ids.shape[1]:])[0]
        extracted_json = PROCESSOR.token2json(output)
        
        st.info("Ekstraksi Data Selesai.")
        
        # --- Konversi hasil JSON menjadi DataFrame (Penting untuk UI) ---
        # Anda perlu menyesuaikan logika ini berdasarkan format output model Anda (misal: CORD/Donut)
        
        # Asumsi struktur data item dari JSON Donut (Anda harus menyesuaikan)
        items_list = []
        if 'menu' in extracted_json:
            for item in extracted_json['menu']:
                # Hati-hati dengan data yang hilang atau tidak terstruktur
                items_list.append({
                    'Item': item.get('nm', 'N/A'),
                    'Qty': int(item.get('qty', 1)), 
                    'Price_per_item': float(item.get('price', '0').replace(',', '').replace('.', '')), # Perlu cleansing data
                    'Total_Item_Price': float(item.get('cnt', '0').replace(',', '').replace('.', '')) # Perlu cleansing data
                })
        
        df_items = pd.DataFrame(items_list)
        
        # Ekstrak biaya tambahan dan total
        subtotal = float(extracted_json.get('sub_total', '0').replace(',', '').replace('.', ''))
        total_bill = float(extracted_json.get('total', '0').replace(',', '').replace('.', ''))
        
        # Hitung biaya tambahan: Total Biaya Tambahan = Total Bill - Subtotal
        total_addons = total_bill - subtotal
        
        return df_items, subtotal, total_bill, total_addons, extracted_json # Kembalikan juga JSON mentah untuk debugging

    except Exception as e:
        st.error(f"Error saat menjalankan AI inference: {e}")
        return None, None, None, None, None


# --- 3. LOGIKA PEMBAGIAN BILL (SPLIT LOGIC) ---

def calculate_final_split(df_items_assigned, total_bill, total_addons, participants):
    """
    Menghitung total yang harus dibayar per orang.
    Logika: (Total Harga Item per Orang) + (Porsi Biaya Tambahan)
    """
    
    # 1. Hitung Total Biaya Item per Orang
    person_item_totals = df_items_assigned.groupby('Assigned_To')['Total_Item_Price'].sum()

    # 2. Hitung Porsi Biaya Tambahan per Orang (dibagi rata)
    if not participants:
        return {} # Handle case jika tidak ada partisipan
        
    num_participants = len(participants)
    
    # Pembagian proporsional: Biaya Tambahan = Total Addons * (Item Cost Part/Total Item Cost)
    # ATAU Pembagian Rata (lebih sederhana dan umum):
    addon_per_person = total_addons / num_participants

    # 3. Hitung Total Bayar Akhir
    final_bill = {}
    total_calculated_check = 0
    for name in participants:
        item_cost = person_item_totals.get(name, 0)
        # Total bayar = Biaya Item + Biaya Tambahan (porsi rata)
        final_cost = item_cost + addon_per_person
        final_bill[name] = final_cost
        total_calculated_check += final_cost

    # Verifikasi total
    # Pastikan jumlah harga semua orang sama dengan total harga bill (Requirement G)
    st.sidebar.caption(f"Verifikasi Total: {total_calculated_check:.2f} (Calc) vs {total_bill:.2f} (Bill)")
    
    return final_bill


# --- 4. STREAMLIT APP LAYOUT ---
st.title("💰 SmartSplit Bill AI Prototype")
st.markdown("---")

# --- Bagian A: Upload Gambar dan Ekstraksi AI ---
st.header("1. Upload Nota Pembelian")
uploaded_file = st.file_uploader("Pilih gambar nota (.jpg atau .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Nota Diunggah", use_column_width=True)
    
    df_items, subtotal, total_bill, total_addons, extracted_json = extract_data_from_image(uploaded_file)
    
    if df_items is not None and not df_items.empty:
        st.header("2. Hasil Ekstraksi Item AI")
        
        # Pastikan kolom 'Total_Item_Price' ada sebelum ditampilkan
        st.dataframe(df_items[['Item', 'Qty', 'Total_Item_Price']].style.format({
            'Total_Item_Price': 'Rp{:,.0f}'
        }))
        
        st.metric(label="Subtotal Item", value=f"Rp{subtotal:,.0f}")
        st.metric(label="Total Biaya Tambahan (Pajak, Service, Dll.)", value=f"Rp{total_addons:,.0f}")
        st.metric(label="TOTAL AKHIR BILL", value=f"Rp{total_bill:,.0f}")

        # Simpan data ke state Streamlit untuk langkah selanjutnya
        st.session_state['items_df'] = df_items
        st.session_state['total_bill'] = total_bill
        st.session_state['total_addons'] = total_addons
        
        st.markdown("---")
        
        # --- Bagian B: Input Partisipan ---
        st.header("3. Input Partisipan")
        
        # Input nama partisipan (dipisahkan koma)
        participants_input = st.text_input(
            "Masukkan nama-nama partisipan (dipisahkan koma, contoh: Andi, Budi, Clara)",
            value="Andi, Budi"
        )
        participants = [name.strip() for name in participants_input.split(',') if name.strip()]
        
        if participants:
            st.session_state['participants'] = participants
            st.success(f"Partisipan: {', '.join(participants)}")
            st.markdown("---")
            
            # --- Bagian C: Penugasan Item ---
            st.header("4. Tugaskan Item ke Partisipan")
            
            # Persiapan DataFrame untuk penugasan
            # Tambahkan kolom 'Assigned_To'
            df_assignment = st.session_state['items_df'].copy()
            df_assignment['Assigned_To'] = participants[0] # Default ke partisipan pertama
            
            assigned_data = []

            for index, row in df_assignment.iterrows():
                # Streamlit membutuhkan key unik untuk setiap widget selectbox
                assigned_person = st.selectbox(
                    f"**{row['Item']} (Rp{row['Total_Item_Price']:,.0f})**",
                    options=participants,
                    key=f"assign_{index}"
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
            st.header("5. Total Bayar per Orang")
            
            final_split = calculate_final_split(
                df_items_assigned, 
                st.session_state['total_bill'], 
                st.session_state['total_addons'], 
                participants
            )
            
            if final_split:
                final_df = pd.DataFrame(list(final_split.items()), columns=['Partisipan', 'Total Bayar'])
                
                # Tampilkan detail total item yang dibayar
                st.subheader("Rincian Pembayaran")
                st.table(final_df.style.format({'Total Bayar': 'Rp{:,.0f}'}))
                
                st.markdown(
                    f"""
                    ---
                    **TOTAL BILL KESELURUHAN (Verifikasi):** **Rp{final_df['Total Bayar'].sum():,.0f}**
                    *(Total ini mencakup item cost dan porsi biaya tambahan yang dibagi rata.)*
                    """
                )
            
        else:
            st.warning("Masukkan minimal satu nama partisipan untuk melanjutkan.")
    else:
        st.warning("Gagal mendapatkan data item dari nota. Coba lagi atau periksa model AI.")
