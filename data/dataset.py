import librosa
import librosa.display
import numpy as np
import os
import scipy.signal
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from data.filterbank import FilterBankFeatureTransform
from data.augment import spec_augment

# def load_data_list(data_list, mode="train"):
#     size = len(data_list)
#     dataset = {}
#     clean = "dataset" + "/" + "wavs_" + mode
#     mixed = "dataset" + "/" + "noisy_wavs_" + mode
#
#     dataset['clean'] = []
#     dataset['mixed'] = []
#     dataset['name'] = []
#
#     print("loading dataset")
#     for idx in range(3):
#         wav_name = data_list[idx]["wav"]
#         dataset['name'].append(wav_name)
#         dataset['clean'].append(clean + "/" + wav_name)
#         dataset['mixed'].append(mixed + "/" + wav_name)
#
#         # todo 잘되는지 확인
#     return dataset
#
#
# def load_data(dataset):
#     dataset['clean_audio'] = [None] * len(dataset['clean'])
#     dataset['mixed_audio'] = [None] * len(dataset['mixed'])
#
#     for id in tqdm(range(len(dataset['clean']))):
#
#         if dataset['mixed_audio'][id] is None:
#             cleanData, sr = librosa.load(dataset['clean'][id], sr=None)
#             mixedData, sr = librosa.load(dataset['mixed'][id], sr=None)
#
#             # input mixed임
#             dataset['clean_audio'][id] = np.float32(cleanData)
#             dataset['mixed_audio'][id] = np.float32(mixedData)
#
#     return dataset
#
#
# class AudioDataset(Dataset):
#
#     def __init__(self, data_list, mode):
#         dataset = load_data_list(data_list, mode)
#         self.dataset = load_data(dataset)
#
#         self.file_names = dataset['name']
#
#     def __getitem__(self, idx):
#         mixed = torch.from_numpy(self.dataset['mixed_audio'][idx]).type(torch.FloatTensor)
#         clean = torch.from_numpy(self.dataset['clean_audio'][idx]).type(torch.FloatTensor)
#
#         return mixed, clean
#
#     def __len__(self):
#         return len(self.file_names)
#
#     def zero_pad_concat(self, inputs):
#         max_t = max(inp.shape[0] for inp in inputs)
#         shape = (len(inputs), max_t)
#         input_mat = np.zeros(shape, dtype=np.float32)
#         for e, inp in enumerate(inputs):
#             input_mat[e, :inp.shape[0]] = inp
#         return input_mat
#
#     def collate(self, inputs):
#         mixeds, cleans = zip(*inputs)
#         seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])
#
#         x = torch.FloatTensor(self.zero_pad_concat(mixeds))
#         y = torch.FloatTensor(self.zero_pad_concat(cleans))
#
#         batch = [x, y, seq_lens, self.file_names]
#         return batch

import librosa
import librosa.display
import numpy as np
import os
import scipy.signal
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from data.filterbank import FilterBankFeatureTransform
from data.augment import spec_augment


def load_audio(audio_path, sample_rate):
    assert audio_path.endswith('wav'), "only wav files"
    signal, sr = librosa.load(audio_path, sr=sample_rate)
    # print("AAAA", signal.shape, sr) # 16,000 / signal 은 다양한 np 값
    return signal


class MelFilterBankDataset(Dataset):

    def __init__(self, data_list, mode='train'):
        """
        Dataset 은 wav_name, transcripts, speaker_id 가 dictionary 로 담겨져있는 list으로부터 data 를 load
        """

        super(MelFilterBankDataset, self).__init__()
        self.dataset_path = 'dataset/wavs_' + mode
        self.noisy_dataset_path = 'dataset/noisy_wavs_' + mode

        self.data_list = data_list  # [{"wav": , "text": , "speaker_id": "}]
        self.size = len(self.data_list)  # 59662
        self.mode = mode

    """
    EMA DATA 따로 불러오기 DATALOADER도 고치기
    """

    def __getitem__(self, index):
        wav_name = self.data_list[index]['wav']
        # print("wav: " , wav_name) # 41_0607_213_1_08139_05.wav
        audio_path = os.path.join(self.dataset_path, wav_name)
        # print("audio_path: ", audio_path): data/wavs_train/41_0607_213_1_08139_05.wav
        noisy_audio_path = os.path.join(self.noisy_dataset_path, wav_name)
        # print("1",audio_path)
        # print("2",noisy_audio_path)
        transcript = self.data_list[index]['text']


        clean = load_audio(audio_path, sample_rate=16000)
        mixed = load_audio(noisy_audio_path, sample_rate=16000)
        transcript = self.parse_transcript(transcript)

        clean = torch.FloatTensor(clean)
        mixed = torch.FloatTensor(mixed)
        # spect = self.parse_audio(audio_path)
        # print("spect: ", spect)
        # noisy_spect = self.parse_audio(noisy_audio_path)
        # print("spect2: ", noisy_spect)

        return clean, transcript, mixed, wav_name

    def __len__(self):
        return self.size  # 59662

    def parse_transcript(self, transcript):
        # print(list(transcript))
        # ['아', '기', '랑', ' ', '같', '이', ' ', '갈', '건', '데', '요', ',', ' ', '아', '기', '가', ' ', '먹', '을', ' ', '수', ' ', '있', '는', '것', '도', ' ', '있', '나', '요', '?']
        # ['매', '장', ' ', '전', '용', ' ', '주', '차', '장', '이', ' ', '있', '나', '요', '?']
        # ['카', '드', ' ', '할', '인', '은', ' ', '신', '용', '카', '드', '만', ' ', '되', '나', '요', '?']
        # ['미', '리', ' ', '예', '약', '하', '려', '고', ' ', '하', '는', '데', '요', '.']

        # transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        # filter(조건, 순횐 가능한 데이터): char2index 의 key 에 없는 것(None) 다 삭제 해버림
        # print("transcript: ", transcript):[49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126]

        # transcript = [self.sos_id] + transcript + [self.eos_id]
        # [2001, 49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126, 2002]

        return transcript