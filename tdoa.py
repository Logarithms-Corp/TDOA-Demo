import matplotlib.pyplot as plt
import numpy as np
import msgpack
import json
import sys

from scipy.signal import correlate, spectrogram, butter, sosfilt, firwin, lfilter
from scipy.optimize import curve_fit


# TODO docs
filter_settings = 0



# Plot function for testing

def plot_correlation_spectrogram(correlations, fs=2.4e6, title="Correlation Spectrogram"):
	"""
	Plot a spectrogram of correlation arrays, with each row showing correlation magnitude.

	Args:
		correlations (list): List of correlation arrays from compute_tar_offset_time_bins.
		fs (float): Sampling frequency in Hz (default 2.4e6).
		title (str): Plot title.
	"""

	K = len(correlations)
	if K == 0:
		print("No correlations to plot.")
		return

	Nx_seg = len(correlations[0])
	corr_matrix = np.zeros((K, Nx_seg), dtype=float)

	for k, corr in enumerate(correlations):
		corr_abs = (corr)
		if len(corr_abs) == Nx_seg:
			corr_matrix[k, :] = corr_abs
		else:
			corr_matrix[k, :len(corr_abs)] = corr_abs

	max_per_row = corr_matrix.max(axis=1, keepdims=True)
	max_per_row[max_per_row == 0] = 1
	corr_matrix = corr_matrix / max_per_row

	lags = np.arange(-Nx_seg // 2, Nx_seg // 2 + 1)
	time_per_lag = lags / fs

	bin_indices = np.arange(K)

	plt.figure(figsize=(10, 6))
	#plt.style.use('dark_background')
	plt.imshow(
		corr_matrix,
		aspect='auto',
		origin='lower',
		extent=[time_per_lag[0], time_per_lag[-1], 0, K - 1],
		cmap='viridis'
	)

	plt.colorbar(label='Normalized Correlation Magnitude')
	plt.xlabel('Lag (seconds)')
	plt.ylabel('Time Bin Index')
	plt.title(title)
	plt.tight_layout()
	plt.savefig("output2.png")
	plt.close()



# Interp functions

def parabolic_interp(corr, num_points=5):
	"""
	Perform parabolic interpolation on the correlation peak using num_points.

	Args:
		corr (np.ndarray): Cross-correlation array (complex or real).
		num_points (int): Number of points to fit (odd, e.g., 5 or 7).

	Returns:
		float: Sub-sample offset of the peak.
	"""


	if num_points % 2 == 0 or num_points < 3:
		raise ValueError("num_points must be odd and >= 3")

	peak_idx = np.argmax(np.abs(corr))
	half_window = num_points // 2
	indices = np.arange(peak_idx - half_window, peak_idx + half_window + 1)

	if indices[0] < 0 or indices[-1] >= len(corr):
		raise ValueError("Not enough samples around peak for given num_points")

	y = np.abs(corr[indices])
	x = np.arange(-half_window, half_window + 1)  # Centered at peak

	coeffs = np.polyfit(x, y, 2)
	a, b, _ = coeffs

	if a == 0:
		return float(peak_idx)

	subsample_offset = peak_idx - b / (2 * a)

	return subsample_offset


def gaussian_interp(corr, num_points=9):
	"""
	Perform Gaussian interpolation on the correlation peak using num_points.

	Args:
		corr (np.ndarray): Cross-correlation array (complex or real).
		num_points (int): Number of points to fit (odd, e.g., 5 or 7).

	Returns:
		float: Sub-sample offset of the peak.
	"""

	if num_points % 2 == 0 or num_points < 3:
		raise ValueError("num_points must be odd and >= 3")

	peak_idx = np.argmax(np.abs(corr))
	half_window = num_points // 2
	indices = np.arange(peak_idx - half_window, peak_idx + half_window + 1)

	if indices[0] < 0 or indices[-1] >= len(corr):
		raise ValueError("Not enough samples around peak for given num_points")

	y = np.abs(corr[indices])
	x = np.arange(-half_window, half_window + 1, dtype=float)  # Centered at peak

	def gaussian(x, A, mu, sigma):
		return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

	p0 = [max(y), 0, 1]

	try:
		popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=(0, [np.inf, np.inf, np.inf]))
		A, mu, sigma = popt

		subsample_offset = peak_idx + mu
	except RuntimeError:
		# Fallback to peak index if fit fails
		subsample_offset = float(peak_idx)

	return subsample_offset



# DSP Helper Functions

def format_samps(samp):
	samp_iq = np.frombuffer(samp, dtype=np.uint8)

	samp_out_I = samp_iq[0::2]
	samp_out_Q = samp_iq[1::2]

	samp_out_I = (samp_out_I.astype(np.float32) - 128) / 128.0
	samp_out_Q = (samp_out_Q.astype(np.float32) - 128) / 128.0

	samp_out = samp_out_I + 1j * samp_out_Q

	return samp_out


def lowpass(samp, fs=2.4e6, cutoff=6.5e3, taps=351):
	if cutoff == 100e3:
		taps = 31
	if cutoff == 6e3:
		cutoff = 6.5e3

	h = firwin(taps, cutoff, fs=fs)

	filtered = lfilter(h, 1.0, samp)

	delay = (taps - 1) // 2
	filtered = filtered[delay:]

	return filtered


def phase_diff(samp):
	phase = np.angle(samp)
	phase = np.unwrap(phase)
	phase_diff = np.diff(phase)

	return phase_diff


def largest_power_of_2(n):
	new_length = 2 ** int(np.floor(np.log2(n))) + 1
	return new_length



# TDOA Processing

def process_target_bins(tar_1, tar_2, ref_offset, bin_size=32768//12, use_fft=False):

	int_shift = int(np.floor(ref_offset))
	frac_corr = ref_offset - int_shift


	if int_shift < 0:
		tar_2_shifted = tar_2[abs(int_shift):]
		tar_1_shifted = tar_1[:len(tar_2_shifted)]

	elif int_shift > 0:
		tar_1_shifted = tar_1[int_shift:]
		tar_2_shifted = tar_2[:len(tar_1_shifted)]

	else:
		tar_1_shifted = tar_1
		tar_2_shifted = tar_2


	num_bins = (len(tar_1_shifted)) // bin_size

	if num_bins < 1:
		raise ValueError("Insufficient data length after shifting to form one bin.")


	def process_bin(bin_ch0, bin_ch1):

		bin_ch0 = phase_diff(bin_ch0)
		bin_ch1 = phase_diff(bin_ch1)

		if use_fft:
			K = 1
			N = K * (2 * len(bin_ch0)) - 1

			fft0 = np.fft.fft(bin_ch0, n=N)
			fft1 = np.fft.fft(bin_ch1, n=N)

			corr = np.fft.fftshift(np.fft.ifft(fft0 * np.conj(fft1)))
			interp_peak = gaussian_interp(corr)
			# interp_peak = np.argmax(np.abs(corr))

			offset = interp_peak - (len(corr) // 2)
			scaled_offset = offset / K
		else:
			corr = correlate(bin_ch0, bin_ch1)
		
			interp_peak = gaussian_interp(corr)
			# interp_peak = np.argmax(np.abs(corr))

			offset = interp_peak - (len(corr) // 2)
			scaled_offset = offset

		return scaled_offset


	tar_offsets = []
	correlation_graph = []

	
	for b in range(num_bins):
		start_idx = b * bin_size
		end_idx = start_idx + bin_size

		bin_0 = tar_1_shifted[start_idx:end_idx]
		bin_1 = tar_2_shifted[start_idx:end_idx]

		offset = process_bin(bin_0, bin_1)

		# Adjust D to equal the maximum possible offset between nodes to discard bad correlations
		D = 400

		if abs(offset) < D:
			# TODO need to handle poor correlations
			tar_offsets.append(offset + frac_corr)

		#correlation_graph.append(corr)

	#plot_correlation_spectrogram(correlation_graph)

	#vals = sum(tar_offsets) / len(tar_offsets)
	#return [vals]

	return tar_offsets



# Main compute function

def compute_samps(samps, N):

	node_ids = list(samps.keys())

	node_0 = node_ids[0]
	node_1 = node_ids[1]
	node_2 = node_ids[2]

	# split signals to reference and target
	# configured for target first, then reference
	sig1_b = samps[node_0][:N-200000]
	sig2_b = samps[node_1][:N-200000]
	sig3_b = samps[node_2][:N-200000]

	sig1_a = samps[node_0][-(N-200000):]
	sig2_a = samps[node_1][-(N-200000):]
	sig3_a = samps[node_2][-(N-200000):]

	ref_signals = [sig1_a, sig2_a, sig3_a]
	tar_signals = [sig1_b, sig2_b, sig3_b]

	# subtract mean for DC offset
	ref_signals = [s - np.mean(s) for s in ref_signals]
	tar_signals = [s - np.mean(s) for s in tar_signals]

	root2_length = largest_power_of_2(len(ref_signals[0]))

	# phase difference
	ref_signals = [phase_diff(s[:root2_length]) for s in ref_signals]

	# zero pad offset for FFT, assumes all samps same length
	Nx = len(ref_signals[0]) + len(ref_signals[0]) - 1
	ref_signals = [np.fft.fft(s, n=Nx) for s in ref_signals]

	# lowpass for target signals
	# TODO better implementation, this is just yoinked from AEDA
	if int(filter_settings) != 0:
		tar_signals = [lowpass(samp=s, cutoff=(filter_settings * 1e3)) for s in tar_signals]

	filtered = tar_signals


	# Reference signal correlation
	# Gaussian interp seems to work fine here, but need to test more.

	# You can switch interp methods if you want following the example here
	#max_index_ref_2 = np.argmax(np.abs(correlation_ref_2))
	#max_index_ref_2 = parabolic_interp(correlation_ref_2)

	correlation_ref_2 = np.fft.fftshift(np.fft.ifft(ref_signals[0] * np.conj(ref_signals[1])))
	max_index_ref_2 = gaussian_interp(correlation_ref_2)
	offset_index_ref_2 = max_index_ref_2 - int(len(correlation_ref_2) / 2)

	correlation_ref_3 = np.fft.fftshift(np.fft.ifft(ref_signals[0] * np.conj(ref_signals[2])))
	max_index_ref_3 = gaussian_interp(correlation_ref_3)
	offset_index_ref_3 = max_index_ref_3 - int(len(correlation_ref_3) / 2)

	correlation_ref_4 = np.fft.fftshift(np.fft.ifft(ref_signals[1] * np.conj(ref_signals[2])))
	max_index_ref_4 = gaussian_interp(correlation_ref_4)
	offset_index_ref_4 = max_index_ref_4 - int(len(correlation_ref_4) / 2)


	tar_1_2_offsets = process_target_bins(tar_signals[0], tar_signals[1], offset_index_ref_2)
	tar_1_3_offsets = process_target_bins(tar_signals[0], tar_signals[2], offset_index_ref_3)
	tar_2_3_offsets = process_target_bins(tar_signals[1], tar_signals[2], offset_index_ref_4)


	offsets = [
		{
			'reference': node_0,
			'target': node_1,
			'offset': tar_1_2_offsets,
			'len': len(samps[node_0])
		},
		{
			'reference': node_0,
			'target': node_2,
			'offset': tar_1_3_offsets,
			'len': len(samps[node_1])
		},
		{
			'reference': node_1,
			'target': node_2,
			'offset': tar_2_3_offsets,
			'len': len(samps[node_2])
		}
	]

	print(offsets)



# TODO change input format and add helper/demo script

def main():
	tdoa_data = sys.stdin.buffer.read()

	samps = {}

	N = 0

	for node in tdoa_data:
		samp_IQ = format_samps(tdoa_data[node]['data'])

		N = int(len(samp_IQ) / 2)

		samps[node] = samp_IQ


	compute_samps(samps, N)



if __name__ == '__main__':
	main()
