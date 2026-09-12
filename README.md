# Experiment 7: Pulse Shaping and the Nyquist Criterion
# Software-Based Digital Communication Laboratory
#
# Requirements covered:
# 1. Compare rectangular, sinc, raised-cosine (RC)
#    and root-raised-cosine (RRC) pulses.
# 2. Test roll-off factors alpha = 0, 0.25, 0.5, 1.
# 3. Plot time-domain and frequency-domain responses.
# 4. Verify Nyquist zero crossings at symbol intervals.
# 5. Cascade transmitter and receiver RRC filters.
#
# Libraries:
# NumPy, Matplotlib

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PARAMETERS
# ============================================================

symbol_rate = 1.0       # symbols/second
Ts = 1.0 / symbol_rate  # symbol period

samples_per_symbol = 100
span_symbols = 8

alpha_values = [0.0, 0.25, 0.5, 1.0]

rng = np.random.default_rng(7)


# ============================================================
# 2. TIME AXIS
# ============================================================

def make_time_axis(span, sps):
    """
    Create a symmetric time axis covering +/- span/2 symbols.
    """
    return np.arange(
        -span / 2,
        span / 2 + 1 / sps,
        1 / sps
    )


t = make_time_axis(span_symbols, samples_per_symbol)


# ============================================================
# 3. PULSE DEFINITIONS
# ============================================================

def rectangular_pulse(t, Ts=1.0):
    """
    Rectangular pulse:
        p(t) = 1, |t| <= T/2
             = 0, otherwise
    """
    return (np.abs(t) <= Ts / 2).astype(float)


def sinc_pulse(t, Ts=1.0):
    """
    Ideal sinc pulse:
        p(t) = sinc(t/T)
    """
    return np.sinc(t / Ts)


def raised_cosine_pulse(t, Ts=1.0, alpha=0.25):
    """
    Raised-cosine impulse response.

    p(t) =
        sinc(t/T) * cos(pi*alpha*t/T)
        --------------------------------
        1 - (2*alpha*t/T)^2

    Removable singularities are handled explicitly.
    """
    t = np.asarray(t, dtype=float)

    x = t / Ts

    # Start with the normal expression
    numerator = np.sinc(x) * np.cos(np.pi * alpha * x)
    denominator = 1.0 - (2.0 * alpha * x) ** 2

    h = np.empty_like(x)

    # Normal points
    normal = np.abs(denominator) > 1e-12
    h[normal] = numerator[normal] / denominator[normal]

    # t = 0
    zero = np.isclose(x, 0.0, atol=1e-12)

    # For 0 < alpha <= 1:
    # t = +/- T/(2 alpha)
    singular = np.isclose(
        np.abs(x),
        1.0 / (2.0 * alpha) if alpha > 0 else np.inf,
        atol=1e-10
    )

    h[zero] = 1.0

    if alpha > 0:
        h[singular] = (
            alpha / 2.0
        ) * np.sin(np.pi / (2.0 * alpha))

    return h


def root_raised_cosine_pulse(t, Ts=1.0, alpha=0.25):
    """
    Root-raised-cosine impulse response.

    The removable singularities at t = 0 and
    t = +/- T/(4 alpha) are handled explicitly.
    """
    t = np.asarray(t, dtype=float)
    x = t / Ts

    h = np.zeros_like(x)

    if alpha == 0:
        return sinc_pulse(t, Ts)

    denominator = (
        np.pi * x *
        (1.0 - (4.0 * alpha * x) ** 2)
    )

    numerator = (
        np.sin(np.pi * x * (1.0 - alpha))
        + 4.0 * alpha * x *
        np.cos(np.pi * x * (1.0 + alpha))
    )

    normal = (
        np.abs(x) > 1e-12
    ) & (
        np.abs(1.0 - (4.0 * alpha * x) ** 2) > 1e-12
    )

    h[normal] = numerator[normal] / denominator[normal]

    # At t = 0:
    h[np.isclose(x, 0.0, atol=1e-12)] = (
        1.0 - alpha + 4.0 * alpha / np.pi
    )

    # At t = +/- T/(4 alpha)
    special_x = 1.0 / (4.0 * alpha)

    special = np.isclose(
        np.abs(x),
        special_x,
        atol=1e-10
    )

    special_value = (
        alpha / np.sqrt(2.0)
    ) * (
        (1.0 + 2.0 / np.pi)
        * np.sin(np.pi / (4.0 * alpha))
        +
        (1.0 - 2.0 / np.pi)
        * np.cos(np.pi / (4.0 * alpha))
    )

    h[special] = special_value

    return h


# ============================================================
# 4. FREQUENCY RESPONSE
# ============================================================

def frequency_response(h, dt):
    """
    Calculate centered FFT frequency response.
    """
    nfft = 16384

    H = np.fft.fftshift(
        np.fft.fft(h, n=nfft)
    )

    f = np.fft.fftshift(
        np.fft.fftfreq(nfft, d=dt)
    )

    magnitude = np.abs(H)

    # Normalize
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)

    magnitude_db = 20 * np.log10(
        np.maximum(magnitude, 1e-12)
    )

    return f, magnitude_db


# ============================================================
# 5. PLOT BASIC PULSE COMPARISON
# ============================================================

dt = 1.0 / samples_per_symbol

rect = rectangular_pulse(t, Ts)
sinc = sinc_pulse(t, Ts)
rc = raised_cosine_pulse(t, Ts, alpha=0.5)
rrc = root_raised_cosine_pulse(t, Ts, alpha=0.5)


plt.figure(figsize=(10, 6))

plt.plot(t, rect, label="Rectangular")
plt.plot(t, sinc, label="Sinc")
plt.plot(t, rc, label="Raised-Cosine (α=0.5)")
plt.plot(t, rrc, label="Root-Raised-Cosine (α=0.5)")

plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("Comparison of Pulse Shapes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 6. RAISED-COSINE PULSE FOR DIFFERENT ROLL-OFF FACTORS
# ============================================================

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    h = raised_cosine_pulse(
        t,
        Ts,
        alpha=alpha
    )

    plt.plot(
        t,
        h,
        label=f"α = {alpha}"
    )

plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("Raised-Cosine Pulses for Different Roll-Off Factors")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. RRC PULSE FOR DIFFERENT ROLL-OFF FACTORS
# ============================================================

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    h = root_raised_cosine_pulse(
        t,
        Ts,
        alpha=alpha
    )

    plt.plot(
        t,
        h,
        label=f"α = {alpha}"
    )

plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("Root-Raised-Cosine Pulses for Different Roll-Off Factors")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. FREQUENCY RESPONSE OF RAISED-COSINE PULSES
# ============================================================

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    h = raised_cosine_pulse(
        t,
        Ts,
        alpha=alpha
    )

    f, H_db = frequency_response(h, dt)

    plt.plot(
        f,
        H_db,
        label=f"α = {alpha}"
    )

plt.xlim(-1.5, 1.5)
plt.ylim(-80, 5)

plt.xlabel("Frequency")
plt.ylabel("Magnitude (dB)")
plt.title("Raised-Cosine Frequency Responses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. FREQUENCY RESPONSE OF RRC PULSES
# ============================================================

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    h = root_raised_cosine_pulse(
        t,
        Ts,
        alpha=alpha
    )

    f, H_db = frequency_response(h, dt)

    plt.plot(
        f,
        H_db,
        label=f"α = {alpha}"
    )

plt.xlim(-1.5, 1.5)
plt.ylim(-80, 5)

plt.xlabel("Frequency")
plt.ylabel("Magnitude (dB)")
plt.title("Root-Raised-Cosine Frequency Responses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. NYQUIST ZERO-CROSSING VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NYQUIST ZERO-CROSSING VALIDATION")
print("=" * 60)

for alpha in alpha_values:

    h = raised_cosine_pulse(
        t,
        Ts,
        alpha=alpha
    )

    print(f"\nRoll-off factor α = {alpha}")
    print("-" * 40)

    for k in range(-4, 5):
        target_time = k * Ts

        index = np.argmin(
            np.abs(t - target_time)
        )

        value = h[index]

        print(
            f"k = {k:+d}, "
            f"t/T = {k:+d}, "
            f"p(kT) = {value:+.6f}"
        )


# ============================================================
# 11. OVERALL RC RESPONSE FROM RRC CASCADE
# ============================================================

alpha = 0.5

rrc_tx = root_raised_cosine_pulse(
    t,
    Ts,
    alpha
)

rrc_rx = root_raised_cosine_pulse(
    t,
    Ts,
    alpha
)

# Convolution approximates cascade
overall_rc = np.convolve(
    rrc_tx,
    rrc_rx,
    mode="full"
) * dt

# Time axis for convolution result
t_rc = np.arange(
    -(len(overall_rc) // 2),
    len(overall_rc) // 2 + (len(overall_rc) % 2),
    1
) * dt

# Make the lengths match exactly
if len(t_rc) > len(overall_rc):
    t_rc = t_rc[:len(overall_rc)]

if len(t_rc) < len(overall_rc):
    overall_rc = overall_rc[:len(t_rc)]


# Normalize
overall_rc = overall_rc / np.max(
    np.abs(overall_rc)
)


# ============================================================
# 12. PLOT RRC AND CASCADED RESPONSE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    t,
    rrc_tx,
    label="Transmit RRC"
)

plt.plot(
    t_rc,
    overall_rc,
    label="RRC × RRC (Overall RC)"
)

plt.xlim(-4, 4)
plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("RRC Pulse and Cascaded Overall Response")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 13. NYQUIST VALIDATION OF CASCADED RESPONSE
# ============================================================

print("\n" + "=" * 60)
print("CASCADED RRC NYQUIST VALIDATION")
print("=" * 60)

print(f"\nRoll-off factor α = {alpha}")

for k in range(-4, 5):

    target_time = k * Ts

    index = np.argmin(
        np.abs(t_rc - target_time)
    )

    value = overall_rc[index]

    print(
        f"k = {k:+d}, "
        f"t/T = {k:+d}, "
        f"overall_response(kT) = {value:+.6f}"
    )


# ============================================================
# 14. SYMBOL STREAM + RRC PULSE SHAPING
# ============================================================

num_symbols = 16

# BPSK symbols
bits = rng.integers(
    0,
    2,
    num_symbols
)

symbols = 2 * bits - 1

# Upsample symbols
upsampled = np.zeros(
    num_symbols * samples_per_symbol
)

upsampled[
    ::samples_per_symbol
] = symbols

# RRC filter with a longer practical span
filter_span = 10

filter_t = np.arange(
    -filter_span / 2,
    filter_span / 2 + 1 / samples_per_symbol,
    1 / samples_per_symbol
)

rrc = root_raised_cosine_pulse(
    filter_t,
    Ts,
    alpha=0.5
)

# Normalize filter energy
rrc = rrc / np.sqrt(
    np.sum(rrc ** 2)
)

# Transmitter pulse shaping
tx_signal = np.convolve(
    upsampled,
    rrc,
    mode="full"
)


# ============================================================
# 15. RECEIVER MATCHED FILTER
# ============================================================

rx_signal = np.convolve(
    tx_signal,
    rrc,
    mode="full"
)

# Overall filter delay
delay = len(rrc) - 1

# Sample output after removing cascade delay
sample_indices = (
    delay
    + np.arange(num_symbols)
    * samples_per_symbol
)

sample_indices = sample_indices[
    sample_indices < len(rx_signal)
]

detected_symbols = rx_signal[
    sample_indices.astype(int)
]


# ============================================================
# 16. DISPLAY TRANSMITTED SIGNAL
# ============================================================

time_tx = np.arange(
    len(tx_signal)
) / samples_per_symbol

plt.figure(figsize=(12, 5))

plt.plot(
    time_tx,
    tx_signal,
    label="Pulse-Shaped Signal"
)

plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("Pulse-Shaped Symbol Stream Using RRC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17. DISPLAY MATCHED-FILTER OUTPUT
# ============================================================

time_rx = np.arange(
    len(rx_signal)
) / samples_per_symbol

plt.figure(figsize=(12, 5))

plt.plot(
    time_rx,
    rx_signal,
    label="Matched-Filter Output"
)

plt.scatter(
    sample_indices / samples_per_symbol,
    detected_symbols,
    label="Sampling Instants"
)

plt.xlabel("Time / Symbol Period")
plt.ylabel("Amplitude")
plt.title("Matched Filter Output and Symbol Sampling")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18. DISPLAY TRANSMITTED AND DETECTED SYMBOLS
# ============================================================

print("\n" + "=" * 60)
print("SYMBOL RECOVERY")
print("=" * 60)

print("\nOriginal symbols:")
print(symbols)

print("\nDetected samples:")
print(
    np.round(
        detected_symbols,
        4
    )
)

detected_bits = (
    detected_symbols >= 0
).astype(int)

print("\nDetected bits:")
print(detected_bits)

print("\nOriginal bits:")
print(bits)


# ============================================================
# 19. SIMPLE ERROR COUNT
# ============================================================

num_valid = min(
    len(bits),
    len(detected_bits)
)

bit_errors = np.sum(
    bits[:num_valid]
    != detected_bits[:num_valid]
)

print(
    f"\nBit errors = {bit_errors}"
)

print(
    f"Valid bits = {num_valid}"
)


# ============================================================
# 20. BANDWIDTH VS ROLL-OFF
# ============================================================

# For a raised-cosine filter:
# theoretical one-sided baseband bandwidth:
#
# B = (1 + alpha) / (2T)

bandwidths = []

for alpha in alpha_values:
    B = (
        (1 + alpha)
        / (2 * Ts)
    )

    bandwidths.append(B)


plt.figure(figsize=(8, 5))

plt.plot(
    alpha_values,
    bandwidths,
    marker="o"
)

plt.xlabel("Roll-Off Factor α")
plt.ylabel("One-Sided Bandwidth")
plt.title("Bandwidth vs Roll-Off Factor")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 21. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 7 SUMMARY")
print("=" * 60)

print("""
1. Rectangular pulses have sharp transitions and high sidelobes.
2. The sinc pulse satisfies the ideal Nyquist zero-crossing condition.
3. Raised-cosine pulses control bandwidth using roll-off factor α.
4. Increasing α increases bandwidth but reduces time-domain ringing.
5. Root-raised-cosine filtering splits the raised-cosine response
   between transmitter and receiver.
6. Cascading identical RRC filters gives an overall raised-cosine
   response.
7. The overall response shows zero crossings at integer symbol
   intervals, satisfying the Nyquist criterion.
8. The receiver samples only after compensating the filter delay.
""")
