"""
This is the import of the data augmentation module
"""

from .compose import CircularAug, DynamicSingleAug, RandomAug, SequentialAug, StaticSingleAug
from .functional import (
    add_band_noise,
    add_eeg_artifact,
    add_gaussian_noise,
    add_noise_SNR,
    change_ref,
    channel_dropout,
    crop_and_resize,
    filter_bandpass,
    filter_bandstop,
    filter_highpass,
    filter_lowpass,
    flip_horizontal,
    flip_vertical,
    get_channel_map_and_networks,
    get_eeg_channel_network_names,
    get_filter_coeff,
    identity,
    masking,
    moving_avg,
    permutation_signal,
    permute_channels,
    phase_swap,
    random_FT_phase,
    random_slope_scale,
    scaling,
    shift_frequency,
    shift_horizontal,
    shift_vertical,
    warp_signal,
)
