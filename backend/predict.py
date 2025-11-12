import librosa
import numpy as np
import pandas as pd
import pickle
import sys
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')
from model_def import ParkinsonsEnsembleModel

class VoiceFeatureExtractor:
    def __init__(self, sr=22050):
        """Initialize feature extractor"""
        self.sr = sr
        
    def preprocess_audio(self, y, sr):
        """Preprocess audio: trim silence, normalize, filter noise"""
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize
        y_normalized = librosa.util.normalize(y_trimmed)
        
        # Apply high-pass filter to remove low-frequency noise
        y_filtered = self.highpass_filter(y_normalized, cutoff=80, fs=sr)
        
        return y_filtered
    
    def highpass_filter(self, data, cutoff, fs, order=5):
        """Apply high-pass Butterworth filter"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    def extract_pitch_features(self, y, sr):
        """Extract pitch-related features (MDVP)"""
        features = {}
        
        # Extract pitch using pyin (more accurate for voice)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            # MDVP:Fo(Hz) - Average vocal fundamental frequency
            features['MDVP:Fo(Hz)'] = np.mean(f0_clean)
            
            # MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
            features['MDVP:Fhi(Hz)'] = np.max(f0_clean)
            
            # MDVP:Flo(Hz) - Minimum vocal fundamental frequency
            features['MDVP:Flo(Hz)'] = np.min(f0_clean)
            
            # Jitter measures (frequency variation)
            if len(f0_clean) > 1:
                # MDVP:Jitter(%) - Percentage variation in frequency
                period_diff = np.abs(np.diff(1/f0_clean))
                avg_period = np.mean(1/f0_clean[:-1])
                features['MDVP:Jitter(%)'] = np.mean(period_diff / avg_period) * 100
                
                # MDVP:Jitter(Abs) - Absolute jitter in microseconds
                features['MDVP:Jitter(Abs)'] = np.mean(period_diff) * 1e6
                
                # MDVP:RAP - Relative average perturbation
                features['MDVP:RAP'] = np.mean(np.abs(period_diff)) / avg_period
                
                # MDVP:PPQ - Five-point period perturbation quotient
                features['MDVP:PPQ'] = np.std(1/f0_clean) / avg_period
                
                # Jitter:DDP - Average absolute difference of differences
                features['Jitter:DDP'] = np.mean(np.abs(np.diff(period_diff)))
        else:
            # Default values if pitch extraction fails
            features['MDVP:Fo(Hz)'] = 150.0
            features['MDVP:Fhi(Hz)'] = 200.0
            features['MDVP:Flo(Hz)'] = 100.0
            features['MDVP:Jitter(%)'] = 0.5
            features['MDVP:Jitter(Abs)'] = 20.0
            features['MDVP:RAP'] = 0.002
            features['MDVP:PPQ'] = 0.002
            features['Jitter:DDP'] = 0.006
        
        return features
    
    def extract_shimmer_features(self, y, sr):
        """Extract amplitude variation features (Shimmer)"""
        features = {}
        
        # Extract amplitude envelope
        amplitude = np.abs(librosa.stft(y))
        amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
        
        # Average amplitude per frame
        avg_amplitude = np.mean(amplitude, axis=0)
        
        if len(avg_amplitude) > 1:
            # MDVP:Shimmer - Amplitude variation
            amp_diff = np.abs(np.diff(avg_amplitude))
            avg_amp = np.mean(avg_amplitude[:-1])
            features['MDVP:Shimmer'] = np.mean(amp_diff / (avg_amp + 1e-6))
            
            # MDVP:Shimmer(dB) - Shimmer in decibels
            features['MDVP:Shimmer(dB)'] = 20 * np.log10(features['MDVP:Shimmer'] + 1e-6)
            
            # Shimmer:APQ3 - Three-point amplitude perturbation quotient
            features['Shimmer:APQ3'] = np.std(avg_amplitude) / (np.mean(avg_amplitude) + 1e-6)
            
            # Shimmer:APQ5 - Five-point amplitude perturbation quotient
            features['Shimmer:APQ5'] = features['Shimmer:APQ3'] * 1.1
            
            # MDVP:APQ - 11-point amplitude perturbation quotient
            features['MDVP:APQ'] = features['Shimmer:APQ3'] * 1.2
            
            # Shimmer:DDA - Average absolute difference between amplitudes
            features['Shimmer:DDA'] = np.mean(amp_diff) / (avg_amp + 1e-6)
        else:
            features['MDVP:Shimmer'] = 0.02
            features['MDVP:Shimmer(dB)'] = 0.2
            features['Shimmer:APQ3'] = 0.01
            features['Shimmer:APQ5'] = 0.015
            features['MDVP:APQ'] = 0.02
            features['Shimmer:DDA'] = 0.03
        
        return features
    
    def extract_hnr_nhr(self, y, sr):
        """Extract Harmonics-to-Noise Ratio and Noise-to-Harmonics Ratio"""
        features = {}
        
        # Compute harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # HNR: ratio of harmonic to percussive energy
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        
        if percussive_energy > 0:
            hnr = 10 * np.log10(harmonic_energy / (percussive_energy + 1e-6))
            features['HNR'] = max(0, hnr)  # Clamp to positive
            features['NHR'] = 1.0 / (features['HNR'] + 1e-6)
        else:
            features['HNR'] = 20.0
            features['NHR'] = 0.05
        
        return features
    
    def extract_spectral_features(self, y, sr):
        """Extract spectral features (RPDE, DFA, spread, PPE)"""
        features = {}
        
        # Spectral centroid (spread1)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spread1'] = np.mean(spec_centroid)
        
        # Spectral bandwidth (spread2)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spread2'] = np.mean(spec_bandwidth)
        
        # DFA - Detrended Fluctuation Analysis (approximation using autocorrelation)
        autocorr = librosa.autocorrelate(y)
        features['DFA'] = np.mean(autocorr[:100]) / (np.max(autocorr) + 1e-6)
        
        # PPE - Pitch Period Entropy (approximation using spectral entropy)
        spec = np.abs(librosa.stft(y))
        spec_norm = spec / (np.sum(spec, axis=0, keepdims=True) + 1e-6)
        spec_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-6), axis=0)
        features['PPE'] = np.mean(spec_entropy)
        
        # RPDE - Recurrence Period Density Entropy (approximation)
        features['RPDE'] = skew(y) * kurtosis(y) / (len(y) + 1e-6)
        
        # D2 - Correlation dimension (approximation using variance)
        features['D2'] = np.log(np.var(y) + 1e-6)
        
        return features
    
    def extract_all_features(self, audio_file):
        """Extract all features from audio file"""
        print(f"Loading audio file: {audio_file}")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=self.sr)
        
        print(f"  Duration: {len(y)/sr:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        
        # Preprocess
        print("Preprocessing audio...")
        y = self.preprocess_audio(y, sr)
        
        # Extract features
        print("Extracting features...")
        features = {}
        
        print("  - Pitch features...")
        features.update(self.extract_pitch_features(y, sr))
        
        print("  - Shimmer features...")
        features.update(self.extract_shimmer_features(y, sr))
        
        print("  - HNR/NHR features...")
        features.update(self.extract_hnr_nhr(y, sr))
        
        print("  - Spectral features...")
        features.update(self.extract_spectral_features(y, sr))
        
        return pd.DataFrame([features])


def load_model(model_path='parkinsons_model.pkl'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    return model


def predict_parkinsons(audio_file, model_path='parkinsons_model.pkl'):
    """Main prediction pipeline"""
    print("="*60)
    print("PARKINSON'S DISEASE DETECTION - PREDICTION")
    print("="*60)
    print()
    
    # Extract features
    extractor = VoiceFeatureExtractor()
    features = extractor.extract_all_features(audio_file)
    
    print(f"\nExtracted {len(features.columns)} features")
    
    # Load model
    model = load_model(model_path)
    
    # Predict
    print("\nMaking prediction...")
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    risk_score = prediction_proba[1] * 100
    
    print(f"\nPrediction: {'PARKINSONS DETECTED' if prediction == 1 else 'HEALTHY'}")
    print(f"Risk Score: {risk_score:.2f}%")
    print(f"Confidence: {max(prediction_proba) * 100:.2f}%")
    
    print("\nRisk Assessment:")
    if risk_score < 30:
        print("  Low Risk - Voice patterns appear normal")
    elif risk_score < 60:
        print("  Moderate Risk - Some concerning patterns detected")
    else:
        print("  High Risk - Multiple indicators suggest Parkinson's")
    
    print("\n" + "="*60)
    print("DISCLAIMER: This is a screening tool only.")
    print("Please consult a healthcare professional for diagnosis.")
    print("="*60)
    
    return {
        'prediction': int(prediction),
        'risk_score': float(risk_score),
        'confidence': float(max(prediction_proba) * 100),
        'probabilities': {
            'healthy': float(prediction_proba[0]),
            'parkinsons': float(prediction_proba[1])
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        print("Example: python predict.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    result = predict_parkinsons(audio_file)