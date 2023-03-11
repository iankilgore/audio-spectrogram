import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# Load the audio file
audio_file_path = input("Enter path to input audio file (.m4a): ")
audio = AudioSegment.from_file(audio_file_path)

# Convert the audio data to a numpy array
data = np.array(audio.get_array_of_samples())
data = data / np.max(np.abs(data))  # normalize the data
fs = audio.frame_rate

# Convert the audio data to a spectrogram
spectrogram, frequencies, times, _ = plt.specgram(data, NFFT=4096, Fs=fs, noverlap=4096 // 2)

# Plot the spectrogram
plt.axis('off')

# Prompt the user for an output file name and extension
output_file = input("Enter the output file name and extension (e.g. spectrogram.png): ")

# Save the spectrogram as an image file
plt.savefig('spectrogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
# Show the spectrogram
plt.show()
