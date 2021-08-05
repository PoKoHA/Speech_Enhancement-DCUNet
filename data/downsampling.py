"""
파일의 sampling rate를 조절하는 코드
"""

import librosa
import soundfile as sf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--origin-sr", type=int, default=44100)
parser.add_argument("--resampling-sr", type=int, default=16000)

def main():
    args = parser.parse_args()

    file_list = os.listdir(args.file_path)

    for idx in range(len(file_list)):
        file_name = file_list[idx]
        folder = os.path.join(args.file_path, file_name)
        audio, sr = librosa.load(folder, sr=args.origin_sr)
        resampling = librosa.resample(audio, sr, args.resampling_sr)

        output_folder = os.path.join(args.output_path, file_name)
        sf.write(output_folder, resampling, args.resampling_sr, format='WAV', subtype='PCM_16')

if __name__ == "__main__":
    main()




