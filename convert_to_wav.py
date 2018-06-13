import tempfile
import os
import pydub
from pydub.utils import make_chunks
from peakutils.peak import indexes
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np


modulation_dict = {
    'OFDM': ['LTEsample', 'Digital_Radio_Mondiale_USB', 'GW',
             'HD_Radio_AM', 'Hd_radio_audio_sample', ],
    'PSK': ['LTEsample', 'Autocab', 'London_Towncars_Inc_854787kHz_AF',
            'GW', 'Inmarsat_AERO', 'Iridium', 'PACTOR-II', 'PIII_Complete_1',
            'PACTORIVaudio1', 'Signal2', 'RDS', 'Bbc_teleswitch'],
    'QAM': ['LTEsample', 'Digital_Radio_Mondiale_USB', 'PACTORIVaudio1', ],
    'FSK': ['ASCII', 'CCIR_493-4_', 'FLEX', 'GW', 'Golay1', 'Nmt-450',
            'PACTOR-200', 'POCSAG_Sound'],
    'AM': ['AM_Modulatio_2015-06-27T16-57-35Z_198', 'Bbc_teleswitch'],
    'QPSK': ['CDMA_WFM_AF', ],
    'GFSK': ['Dect', ],
    '4FSK': ['Ermes-3125bps', 'FLEX', '152', ],
    'FM': ['WFM', ],
    'GMSK': ['GSMNonHopping_Sound', 'Mobitex'],
    'MFSK': ['Inmarsat-D-audio', 'Inmarsat_mfsk', ],
    'FFSK': ['Mobitex', 'UnknownPagerLike_Sound', 'ReFLEXaudio1',
             'Fouine91_MPT1327-Like', ],
    'BPSK': ['SIGFOX_BPSK',],
}


folder_path = 'wav_file'


def convert_mp3_wav(file_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    files = [f for f in os.listdir(file_path) if f.endswith('.mp3')]
    for f in files:
        mp3 = pydub.AudioSegment.from_mp3(os.path.join(file_path, f))
        mp3.export(os.path.join(folder_path, str(f).split('.')[0]+'.wav'), format='wav')


def convert_ogg_wav(file_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    files = [f for f in os.listdir(file_path) if f.endswith('.ogg')]
    for f in files:
        ogg = pydub.AudioSegment.from_ogg(os.path.join(file_path, f))
        ogg.export(os.path.join(folder_path, str(f).split('.')[0]+'.wav'), format='wav')


def split_wav_ms(file_path, chunks_lengths_ms):
    files = [f for f in os.listdir(file_path) if f.endswith('.wav')]
    save_folder = 'data_sets'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for f in files:
        file_name = str(f).split('.')[0]
        if not os.path.exists(os.path.join(save_folder, file_name)):
            os.makedirs(os.path.join(save_folder, file_name))
        wav_file = pydub.AudioSegment.from_wav(os.path.join(file_path, f))
        chunks = make_chunks(wav_file, chunks_lengths_ms)
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join(save_folder, file_name, 'chunk{0}.wav'.format(str(i))), format='wav')


def split_wav_peaks(file_path):
    files = [f for f in os.listdir(file_path) if f.endswith('.wav')]
    save_folder = 'peak_modulation'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for f in files:
        file_name = str(f).split('.')[0]
        if not os.path.exists(os.path.join(save_folder, file_name)):
            os.makedirs(os.path.join(save_folder, file_name))
        wav_file = pydub.AudioSegment.from_wav(os.path.join(file_path, f))
        sr, data = wavfile.read(os.path.join(file_path, f))
        mean = np.mean(data, axis=0)
        index = 0
        for i in range(500, len(wav_file)-1500, 1000):
            arg_max = np.argmax(data[i:i+1000], axis=0)
            max_value = np.max(data[arg_max])
            if max_value > np.max(mean):
                j = i + np.max(arg_max)
                print(j)
                wav_file[j-500:j+500].export(os.path.join(save_folder, file_name,
                                                                      'chunk{0}.wav'.format(index)), format='wav')
                index += 1


def move_wave(file_path):
    files = [f for f in os.listdir(file_path) if f.endswith('.wav')]
    for f in files:
        wav = pydub.AudioSegment.from_wav(os.path.join(file_path, f))
        wav.export(os.path.join(folder_path, str(f)), format='wav')


def classify_modulation(file_path, chunks_lengths_ms):
    files = [f for f in os.listdir(file_path) if f.endswith('.wav')]
    save_folder = 'modulation'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for modulation in modulation_dict.keys():
        if not os.path.exists(os.path.join(save_folder, modulation)):
            os.makedirs(os.path.join(save_folder, modulation))
        for f in modulation_dict[modulation]:
            file_name = str(f).split('.')[0]
            wav_file = pydub.AudioSegment.from_wav(os.path.join(file_path, f+'.wav'))
            chunks = make_chunks(wav_file, chunks_lengths_ms)
            for i, chunk in enumerate(chunks):
                chunk.export(os.path.join(save_folder, modulation, file_name+'chunk{0}.wav'.format(str(i))), format='wav')


# convert_mp3_wav('audio')
# convert_ogg_wav('audio')
# move_wave('audio')
# classify_modulation('wav_file', 1000)
# split_wav_ms(folder_path, 1000)
split_wav_peaks('wav_file')


