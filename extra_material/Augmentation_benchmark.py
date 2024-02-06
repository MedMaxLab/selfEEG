import os
import glob
import time
import timeit
import math
import random
import numpy as np
import pandas as pd
import torch
try:
    from selfeeg import augmentation as aug
except:
    import sys
    sys.path.append(os.getcwd().split('/extra_material')[0])
    from selfeeg import augmentation as aug
import argparse

help_d = """
run a benchmark of the available data augmentations. Augmentations will be run with a tensor or
array with dimension (64 x 61 x 512). The following configuration will be tested, leaving NaN
values if not possible (e.g., functions with no batch_equal arg or not available GPU device):

1. Numpy Array with no batch equal
2. Numpy Array with batch equal
3. Torch Tensor with no batch equal
4. Torch Tensor with batch equal
5. Torch Tensor on GPU with no batch equal
6. Torch Tensor on GPU with batch equal

Each augmentation will be run 10 times, while the number of repetitions of the
timeit function (so the total is 10*repetition) can be parsed.
For example:

$ python Augmentation_benchmark -r 10

"""
parser = argparse.ArgumentParser(description=help_d)
parser.add_argument("-r","--repetition", metavar='r', type=int, nargs='?', const=1,
                    help="""an integer for the number of times
                    an augmentation is called 10 times""")
args = parser.parse_args()
n = args.repetition

print('start benchmark with ' , str(n), ' repetition of 10 calls')
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
if device.type == 'cpu':
    device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


sup_torch= """
import sys
import os
try:
    from selfeeg import augmentation as aug
except:
    import sys
    sys.path.append(os.getcwd().split('/extra_material')[0])
    from selfeeg import augmentation as aug
import torch
import random
import numpy as np
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
x = torch.randn(64,61,512)
xaug = torch.clone(x)
"""

sup_torch_gpu= """
import sys
import os
try:
    from selfeeg import augmentation as aug
except:
    import sys
    sys.path.append(os.getcwd().split('/extra_material')[0])
    from selfeeg import augmentation as aug
import torch
import random
import numpy as np
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
if device.type == 'cpu':
    device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
x = torch.randn(64,61,512).to(device=device)
xaug = torch.clone(x)
"""

sup_np= """
import sys
import os
try:
    from selfeeg import augmentation as aug
except:
    import sys
    sys.path.append(os.getcwd().split('/extra_material')[0])
    from selfeeg import augmentation as aug
import torch
import random
import numpy as np
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
x = np.random.randn(64,61,512)
xaug = np.copy(x)
"""
aug_list = ['add_band_noise', 'add_eeg_artifact', 'add_gaussian_noise', 'add_noise_SNR',
            'change_ref', 'channel_dropout', 'crop_and_resize', 'filter_bandpass',
            'filter_bandstop','filter_highpass','filter_lowpass', 'flip_horizontal',
            'flip_vertical', 'masking', 'moving_avg', 'permutation_signal',
            'permute_channels_network', 'permute_channels', 'random_FT_phase',
            'random_slope_scale', 'scaling','shift_frequency', 'shift_horizontal',
            'shift_vertical','warp_signal' ]
bench_dict = {i: [None]*6 for i in aug_list }





print('evaluating add_band_noise')
s ="""
for i in range(10):
    xaug = aug.add_band_noise(x, ['theta',(10,20),50], 128)
"""
bench_dict['add_band_noise'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['add_band_noise'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['add_band_noise'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating add_eeg_artifact')
s_false="""
for i in range(10):
    xaug = aug.add_eeg_artifact(x, 128 , batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.add_eeg_artifact(x, 128 , batch_equal=True)
"""
bench_dict['add_eeg_artifact'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['add_eeg_artifact'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['add_eeg_artifact'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['add_eeg_artifact'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['add_eeg_artifact'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['add_eeg_artifact'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)




print('evaluating add_gaussian_noise')
s ="""
for i in range(10):
    xaug = aug.add_gaussian_noise(x)
"""
bench_dict['add_gaussian_noise'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['add_gaussian_noise'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['add_gaussian_noise'][5] = timeit.timeit(s, sup_torch_gpu, number=n)
print(bench_dict['add_gaussian_noise'])





s="""
for i in range(10):
    xaug = aug.add_noise_SNR(x)
"""
bench_dict['add_noise_SNR'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['add_noise_SNR'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['add_noise_SNR'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating change_ref')
s="""
for i in range(10):
    xaug = aug.change_ref(x)
"""
bench_dict['change_ref'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['change_ref'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['change_ref'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating channel_dropout')
s_false="""
for i in range(10):
    xaug = aug.channel_dropout(x, 8 , batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.channel_dropout(x, 8 , batch_equal=True)
"""
bench_dict['channel_dropout'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['channel_dropout'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['channel_dropout'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['channel_dropout'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['channel_dropout'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['channel_dropout'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating crop_and_resize')
s_false="""
for i in range(10):
    xaug = aug.crop_and_resize(x, 10 , 2, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.crop_and_resize(x, 10, 2, batch_equal=True)
"""
bench_dict['crop_and_resize'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['crop_and_resize'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['crop_and_resize'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['crop_and_resize'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['crop_and_resize'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['crop_and_resize'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating filter_bandpass')
s="""
for i in range(10):
    xaug = aug.filter_bandpass(x, 128)
"""
bench_dict['filter_bandpass'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['filter_bandpass'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['filter_bandpass'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating filter_bandstop')
s="""
for i in range(10):
    xaug = aug.filter_bandstop(x, 128)
"""
bench_dict['filter_bandstop'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['filter_bandstop'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['filter_bandstop'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating filter_highpass')
s="""
for i in range(10):
    xaug = aug.filter_highpass(x, 128)
"""
bench_dict['filter_highpass'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['filter_highpass'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['filter_highpass'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating filter_lowpass')
s="""
for i in range(10):
    xaug = aug.filter_lowpass(x, 128)
"""
bench_dict['filter_lowpass'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['filter_lowpass'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['filter_lowpass'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating flip_horizontal')
s="""
for i in range(10):
    xaug = aug.flip_horizontal(x)
"""
bench_dict['flip_horizontal'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['flip_horizontal'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['flip_horizontal'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating flip_vertical')
s="""
for i in range(10):
    xaug = aug.flip_vertical(x)
"""
bench_dict['flip_vertical'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['flip_vertical'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['flip_vertical'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating masking')
s_false="""
for i in range(10):
    xaug = aug.masking(x, 4 , 0.4, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.masking(x, 4, 0.4 , batch_equal=True)
"""
bench_dict['masking'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['masking'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['masking'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['masking'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['masking'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['masking'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating moving_avg')
s="""
for i in range(10):
    xaug = aug.moving_avg(x)
"""
bench_dict['moving_avg'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['moving_avg'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['moving_avg'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating permutation_signal')
s_false="""
for i in range(10):
    xaug = aug.permutation_signal(x, 15 , 5, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.permutation_signal(x, 15, 5 , batch_equal=True)
"""
bench_dict['permutation_signal'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['permutation_signal'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['permutation_signal'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['permutation_signal'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['permutation_signal'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['permutation_signal'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating permute_channels_network')
s_false="""
for i in range(10):
    xaug = aug.permute_channels(x, 35 , 'network', batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.permute_channels(x, 35 , 'network' , batch_equal=True)
"""
bench_dict['permute_channels_network'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['permute_channels_network'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['permute_channels_network'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['permute_channels_network'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['permute_channels_network'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['permute_channels_network'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating permute_channels')
s_false="""
for i in range(10):
    xaug = aug.permute_channels(x, 35, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.permute_channels(x, 35, batch_equal=True)
"""
bench_dict['permute_channels'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['permute_channels'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['permute_channels'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['permute_channels'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['permute_channels'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['permute_channels'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating random_FT_phase')
s_false="""
for i in range(10):
    xaug = aug.random_FT_phase(x, 0.2, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.random_FT_phase(x, 0.2, batch_equal=True)
"""
bench_dict['random_FT_phase'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['random_FT_phase'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['random_FT_phase'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['random_FT_phase'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type == 'cuda':
    bench_dict['random_FT_phase'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['random_FT_phase'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating random_slope_scale')
s_false="""
for i in range(10):
    xaug = aug.random_slope_scale(x, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.random_slope_scale(x, batch_equal=True)
"""
bench_dict['random_slope_scale'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['random_slope_scale'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['random_slope_scale'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['random_slope_scale'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['random_slope_scale'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['random_slope_scale'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating scaling')
s_false="""
for i in range(10):
    xaug = aug.scaling(x, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.scaling(x, batch_equal=True)
"""
bench_dict['scaling'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['scaling'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['scaling'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['scaling'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['scaling'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['scaling'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating shift_frequency')
s_false="""
for i in range(10):
    xaug = aug.shift_frequency(x, 4,128, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.shift_frequency(x, 4, 128, batch_equal=True)
"""
bench_dict['shift_frequency'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['shift_frequency'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['shift_frequency'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['shift_frequency'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type == 'cuda':
    bench_dict['shift_frequency'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['shift_frequency'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating shift_horizontal')
s_false="""
for i in range(10):
    xaug = aug.shift_horizontal(x, 0.2, 128, random_shift=True, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.shift_horizontal(x, 0.2 , 128, batch_equal=True)
"""
bench_dict['shift_horizontal'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['shift_horizontal'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['shift_horizontal'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['shift_horizontal'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['shift_horizontal'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['shift_horizontal'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





print('evaluating shift_vertical')
s="""
for i in range(10):
    xaug = aug.shift_vertical(x, 2)
"""
bench_dict['shift_vertical'][1] = timeit.timeit(s, sup_np, number=n)
bench_dict['shift_vertical'][3] = timeit.timeit(s, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['shift_vertical'][5] = timeit.timeit(s, sup_torch_gpu, number=n)





print('evaluating warp_signal')
s_false="""
for i in range(10):
    xaug = aug.warp_signal(x, 16, batch_equal=False)
"""
s_true="""
for i in range(10):
    xaug = aug.warp_signal(x, 16, batch_equal=True)
"""
bench_dict['warp_signal'][0] = timeit.timeit(s_false, sup_np, number=n)
bench_dict['warp_signal'][1] = timeit.timeit(s_true, sup_np, number=n)
bench_dict['warp_signal'][2] = timeit.timeit(s_false, sup_torch, number=n)
bench_dict['warp_signal'][3] = timeit.timeit(s_true, sup_torch, number=n)
if device.type != 'cpu':
    bench_dict['warp_signal'][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
    bench_dict['warp_signal'][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)





Bench_Table = pd.DataFrame.from_dict(bench_dict,
                                     orient='index',
                                     columns=['Numpy Array no BE','Numpy Array BE',
                                              'Torch Tensor no BE','Torch Tensor BE',
                                              'Torch Tensor GPU no BE','Torch Tensor GPU BE']
                                    )
Bench_Table.to_csv('bench_table.csv')
