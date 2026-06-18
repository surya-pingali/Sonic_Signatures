# 🎵 Sonic Signatures: Frequency-Domain Audio Identification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sonicsignatures-suryapingali.streamlit.app/)

**Sonic Signatures** is a robust, frequency-domain audio identification system developed as a course project for EE200. It utilizes Short-Time Fourier Transforms (STFT) and combinatorial paired hashing to accurately identify noisy audio clips against a pre-indexed reference database.

### 🚀 Live Application
Access the deployed interactive web application here: **[Sonic Signatures App](https://sonicsignatures-suryapingali.streamlit.app/)**

---

## 🛠️ Features

* **Single-Clip Analysis:** Upload a `.wav` or `.mp3` file to identify the track. The UI dynamically generates visual feedback, including:
  * Spectrogram and Feature Constellation plots.
  * Time-Offset Alignment Histograms demonstrating hash collisions.
* **Batch Processing Mode:** Upload multiple audio files simultaneously to generate an automated, downloadable `results.csv` evaluation.
* **Adversarial Robustness:** Engineered to isolate signals against ambient interference (Additive White Gaussian Noise) through highly specific 3-tuple hashing $(f_1, f_2, \Delta t)$.

## 💻 Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Signal Processing:** [Librosa](https://librosa.org/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/)
* **Data Visualization:** [Matplotlib](https://matplotlib.org/)

---

## 📂 Repository Structure

```text
EE200_Audio_Identifier/
│
├── app.py                     # Main Streamlit frontend application
├── audio_engine.py            # Backend signal processing and hashing logic
├── song_database.pkl          # Pre-computed feature index (50 tracks)
├── requirements.txt           # Environment dependencies
├── SonicSignatures_Report.pdf # Comprehensive Q3A project report and methodology
└── Q3A_Sonic_Signatures.ipynb # Original Jupyter Notebook with data generation scripts
