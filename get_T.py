import numpy as np

# t = np.load('time_8_5.npy')
# alpha = np.load('y2_8_5.npy')

# al_vs_t = [t, alpha]


# import numpy as np
# import matplotlib.pyplot as plt
# # Load the .npy file
# data = al_vs_t  # Replace with actual filename
# # Extract time and alpha
# time = data[0]
# alpha = data[1]
# # Check if time steps are uniform (important for FFT)
# dt = np.diff(time)
# if not np.allclose(dt, dt[0]):
#     raise ValueError("Time steps are not uniform. Interpolate before FFT.")
# # Compute sampling frequency
# fs = 1 / dt[0]
# # Perform FFT
# alpha_fft = np.fft.fft(alpha)
# freq = np.fft.fftfreq(len(alpha), d=dt[0])
# # Get magnitude (power spectrum)
# magnitude = np.abs(alpha_fft)
# # Plot only the positive frequencies
# positive_freq = freq > 0
# positive_freq = freq > 0
# freq_pos = freq[positive_freq]
# magnitude_pos = magnitude[positive_freq]
# # Find the frequency with maximum magnitude (excluding DC at 0 Hz)
# dominant_idx = np.argmax(magnitude_pos)
# dominant_freq = freq_pos[dominant_idx]
# print(f"Dominant frequency of oscillation: {dominant_freq:.4f} Hz")
# plt.figure(figsize=(8, 4))
# plt.plot(freq[positive_freq], magnitude[positive_freq])
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
# plt.title("FFT of Alpha")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

def get_T(data):
    time = data[0]
    alpha = data[1]
    # Check if time steps are uniform (important for FFT)
    dt = np.diff(time)
    if not np.allclose(dt, dt[0]):
        raise ValueError("Time steps are not uniform. Interpolate before FFT.")
    # Compute sampling frequency
    fs = 1 / dt[0]
    # Perform FFT
    alpha_fft = np.fft.fft(alpha)
    freq = np.fft.fftfreq(len(alpha), d=dt[0])
    # Get magnitude (power spectrum)
    magnitude = np.abs(alpha_fft)
    # Plot only the positive frequencies
    positive_freq = freq > 0
    positive_freq = freq > 0
    freq_pos = freq[positive_freq]
    magnitude_pos = magnitude[positive_freq]
    # Find the frequency with maximum magnitude (excluding DC at 0 Hz)
    dominant_idx = np.argmax(magnitude_pos)
    dominant_freq = freq_pos[dominant_idx]
    # print(f"Dominant frequency of oscillation: {dominant_freq:.4f} Hz")
    return 1 / dominant_freq




