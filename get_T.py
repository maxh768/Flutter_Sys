import numpy as np

def get_T(data):
    data = data[1:-1, :]
    # Use only the second half of the data
    half_length = len(data) // 2
    data = data[half_length:]
    time = data[:,0]
    alpha = data[:,1]
    # Check if time steps are uniform (important for FFT)
    dt = np.diff(time)
    if not np.allclose(dt, dt[0], atol=1e-6):
        not_close = np.where(np.abs(dt - dt[0]) > 1e-6)[0]
        print("Indices with non-uniform time steps:", not_close)
        print("Values:", dt[not_close])
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




