import soundfile as sf
import librosa

input_file = input("Enter path to input audio file (.m4a): ")

try:
    y, sr = sf.read(input_file)
    print("Audio data shape:", y.shape)
except Exception as e:
    print("Error loading audio file:", str(e))

# Compute spectrogram
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to dB scale
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()

# Prompt user for output spectrogram image file path
output_file = input("Enter path to output spectrogram image file (.png): ")

# Save spectrogram image
plt.savefig(output_file)
