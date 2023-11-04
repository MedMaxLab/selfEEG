"""
This is the import of the data augmentation module
"""

from .functional import (identity, shift_vertical, shift_horizontal, shift_frequency, 
                         flip_vertical, flip_horizontal, scaling, random_slope_scale, 
                         random_FT_phase, add_gaussian_noise, add_noise_SNR, add_band_noise, 
                         add_eeg_artifact, get_filter_coeff, moving_avg, filter_lowpass, 
                         filter_highpass, filter_bandpass, filter_bandstop,
                         get_eeg_channel_network_names, get_channel_map_and_networks, 
                         permute_channels, permutation_signal, warp_signal, crop_and_resize, 
                         change_ref, masking, channel_dropout)

from .compose import StaticSingleAug, DynamicSingleAug, SequentialAug, RandomAug