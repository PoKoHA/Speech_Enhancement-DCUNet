import librosa
import torchaudio
import soundfile as sf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--origin-sr", type=int, default=44100)
parser.add_argument("--resampling-sr", type=int, default=16000)

def main():
    args = parser.parse_args()

    file_list = os.listdir(args.path)

    for idx in range(len(file_list)):

        file_name = file_list[idx]
        # print(file_name)
        if not file_name.endswith('wav'):
            continue
        folder = os.path.join(args.path, file_name)
        audio, sr = torchaudio.load(folder)
        transform = torchaudio.transforms.Resample(orig_freq=args.origin_sr, new_freq=args.resampling_sr)

        resampling = transform(audio)
        output_folder = os.path.join(args.output, file_name)
        # sf.write(output_folder, resampling, args.resampling_sr, format='WAV', subtype='PCM_16')
        torchaudio.save(output_folder, resampling, args.resampling_sr,
                        bits_per_sample=16) # bit 수 조절
        print(idx)

if __name__ == "__main__":
    main()




