import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import librosa
from collections import Counter

def load_audio(file_path, target_sr=22050):
    audio_array, sample_rate = librosa.load(file_path, sr=target_sr)
    return audio_array, sample_rate

def generate_spectrogram(audio_array, sample_rate, window_length=1024, overlap=512):
    frequencies, times, Sxx = signal.spectrogram(
        audio_array, fs=sample_rate, window='hann', 
        nperseg=window_length, noverlap=overlap
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return frequencies, times, Sxx_db

def extract_peaks(spectrogram_db, freqs, times, neighborhood_size=20, amplitude_threshold=-40):
    local_max = ndimage.maximum_filter(spectrogram_db, size=neighborhood_size)
    peak_mask = (spectrogram_db == local_max)
    threshold_mask = (spectrogram_db > amplitude_threshold)
    valid_peaks = peak_mask & threshold_mask
    
    freq_indices, time_indices = np.where(valid_peaks)
    return times[time_indices], freqs[freq_indices]

def generate_hashes(peak_times, peak_freqs, fan_value=15):
    hashes = []
    num_peaks = len(peak_times)
    for i in range(num_peaks):
        for j in range(1, fan_value):
            if (i + j) < num_peaks:
                freq1 = peak_freqs[i]
                freq2 = peak_freqs[i + j]
                t1 = peak_times[i]
                t2 = peak_times[i + j]
                time_delta = round(t2 - t1, 3)
                hashes.append(((freq1, freq2, time_delta), t1))
    return hashes

def process_and_match(audio_path, song_database):
    # Runs the full pipeline and returns data needed for visualization
    audio, sr = load_audio(audio_path)
    f, t, spec = generate_spectrogram(audio, sr)
    p_t, p_f = extract_peaks(spec, f, t)
    hashes = generate_hashes(p_t, p_f)
    
    if len(hashes) == 0:
        return None, 0.0, (f, t, spec), (p_t, p_f), []

    matches_per_song = {}
    for q_hash, q_time in hashes:
        if q_hash in song_database:
            for db_song, db_time in song_database[q_hash]:
                offset = round(db_time - q_time, 3)
                if db_song not in matches_per_song:
                    matches_per_song[db_song] = []
                matches_per_song[db_song].append(offset)
    
    best_match = None
    max_count = 0
    best_offsets = []
    
    for song, offsets in matches_per_song.items():
        if offsets:
            most_common, count = Counter(offsets).most_common(1)[0]
            if count > max_count:
                max_count = count
                best_match = song
                best_offsets = offsets # Save offsets of the winning song for the histogram
                
    confidence = max_count / len(hashes)
    return best_match, confidence, (f, t, spec), (p_t, p_f), best_offsets