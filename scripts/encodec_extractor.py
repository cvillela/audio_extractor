from encodec import EncodecModel
from encodec.utils import convert_audio
from pydub import AudioSegment
import torchaudio
import torch
import os
from tqdm import tqdm
import numpy as np

def load_preprocess_wav(file_path, model):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(file_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    return wav

def get_encodec_model(model_sr, target_bw):
    # Instantiate a pretrained EnCodec model
    if model_sr == 24:
        model = EncodecModel.encodec_model_24khz()
    elif model_sr == 48:
        model = EncodecModel.encodec_model_48khz()
    else:
        raise ValueError("Select valid Encodec Model [24kHz or 48kHz]")

    model.set_target_bandwidth(target_bw)
    return model


def encode_wav(wav, model, mean_codes=True):
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    if mean_codes:
        codes = torch.mean(codes.float(), dim=1)
    return codes


def segment_audio(file_path, segment_length_s=30, truncate=True):

    # Get file name without extension for caption
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    audio = AudioSegment.from_file(file_path)
    segment_length_ms = segment_length_s * 1000
    num_segments = (len(audio) // segment_length_ms) + 1
    len_segs = 0
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = (i + 1) * segment_length_ms

        # truncated_sample -> last segment:
        if i == num_segments - 1:
            if truncate:
                break
            else:
                end_time = len(audio)

        segment = audio[start_time:end_time]

        chance = np.random.randint(low=1, high=11)
        if chance <=2:
            output_dir = eval_dir
        else:
            output_dir = train_dir

        seg_path = os.path.join(output_dir, str(uuid.uuid4())+'.wav')
        
        # Save the segment
        segment.export(seg_path, format='wav')
        
        len_segs+=len(segment)

model = get_encodec_model(24, 6)
model.cuda()


ds_codes = []
directory = "/home/cvillela/dataland/audio_extractor/data/dvorak_samples"
for root, dirs, files in os.walk(directory):
        for filename in tqdm(files):
            if '.wav' in filename:
                file_path = os.path.join(root, filename)
                wav = load_preprocess_wav(file_path=file_path, model=model)
                wav=wav.cuda()
                codes = encode_wav(wav, model)
                ds_codes.append(codes)


