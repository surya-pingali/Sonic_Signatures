import streamlit as st
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import audio_engine

st.set_page_config(page_title="Sonic Signatures", page_icon="🎵", layout="wide")
st.title("🎵 Sonic Signatures: Audio Identifier")

@st.cache_resource
def load_database():
    db_path = "song_database.pkl"
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    return None

song_database = load_database()

if song_database is None:
    st.error("Database not found!")
    st.stop()

tab1, tab2 = st.tabs(["Single-Clip Analysis", "Batch Processing Mode"])

# SINGLE CLIP MODE
with tab1:
    st.header("Single-Clip Analysis")
    single_file = st.file_uploader("Upload an audio clip", type=['wav', 'mp3'], key="single")
    
    if single_file is not None:
        # Save uploaded file to disk temporarily
        temp_path = os.path.join("temp_query" + os.path.splitext(single_file.name)[1])
        with open(temp_path, "wb") as f:
            f.write(single_file.getbuffer())
        
        st.info("Processing audio... This may take a moment.")
        
        # Execute the matching engine
        match, conf, spec_data, peak_data, offsets = audio_engine.process_and_match(temp_path, song_database)
        
        # Display the Output
        if match:
            st.success(f"**Identified Song:** {match} (Confidence: {conf:.2%})")
            
            # Visualizations
            st.subheader("Intermediate Processing Steps")
            
            col1, col2 = st.columns(2)
            f, t, spec = spec_data
            p_t, p_f = peak_data
            
            with col1:
                # Plot 1: Spectrogram & Constellation
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.pcolormesh(t, f, spec, shading='gouraud', cmap='magma', alpha=0.7)
                ax1.scatter(p_t, p_f, color='cyan', s=5, label='Extracted Peaks')
                ax1.set_ylim(0, 5000)
                ax1.set_title("Spectrogram & Feature Constellation")
                ax1.set_ylabel("Frequency (Hz)")
                ax1.set_xlabel("Time (s)")
                st.pyplot(fig1)

            with col2:
                # Plot 2: Offset Histogram
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                if offsets:
                    ax2.hist(offsets, bins=100, color='green', alpha=0.7)
                    ax2.set_title("Time-Offset Alignment Histogram")
                    ax2.set_xlabel("Time Offset (s)")
                    ax2.set_ylabel("Hash Collisions")
                else:
                    ax2.text(0.5, 0.5, 'No matches found', ha='center')
                st.pyplot(fig2)
                
        else:
            st.warning("No match found in the database.")
            
        # Clean up the temporary file
        os.remove(temp_path)

# BATCH PROCESSING MODE
with tab2:
    st.header("Batch Processing Mode")
    st.write("Upload multiple audio files to generate an automated evaluation!")
    
    # Accept multiple files
    batch_files = st.file_uploader("Choose multiple audio files", type=['wav', 'mp3'], accept_multiple_files=True, key="batch")
    
    if batch_files:
        st.info(f"{len(batch_files)} files loaded. Click below to begin extraction.")
        
        if st.button("Process Batch"):
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, file in enumerate(batch_files):
                status_text.text(f"Processing {i+1} of {len(batch_files)}: {file.name}")
                
                # Save temporarily
                temp_path = os.path.join(f"temp_batch_{i}" + os.path.splitext(file.name)[1])
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Execute matching engine
                match, _, _, _, _ = audio_engine.process_and_match(temp_path, song_database)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # If no match is found, assign a default string to prevent null errors in evaluation scripts
                prediction = match if match else "Unrecognized"
                results.append({"filename": file.name, "prediction": prediction})
                
                # Update UI
                progress_bar.progress((i + 1) / len(batch_files))
                
            status_text.text("Batch processing complete!")
            st.success("Results compiled successfully.")
            
            # Generate and serve the CSV
            df = pd.DataFrame(results)
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )