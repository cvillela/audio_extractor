import numpy as np
import pandas as pd
from time import time
import torch
from torch.utils.data import DataLoader
from datasets import Audio, load_dataset, Features, Value
from tqdm import tqdm
import gc
from transformers import AutoFeatureExtractor, Wav2Vec2Model


def get_ds(csv_path):
    features = Features({'rec_path': Value(dtype='string', id=None)})
    xeno_ds = load_dataset(
        "csv", data_files=csv_path,
        streaming=True,
        split="train",
        features=features,
        )

    xeno_ds=xeno_ds.rename_column("rec_path", "audio")
    xeno_ds_audio = xeno_ds.cast_column("audio", Audio(sampling_rate=16000))
    return xeno_ds_audio


def preprocess_function(examples):
    target_sr = wav2vec_extractor.sampling_rate
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = wav2vec_extractor(
        audio_arrays,
        sampling_rate=target_sr,
        max_length=160000,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def get_features(audio_ds, batch_size):

    xeno_features = audio_ds.map(
        preprocess_function,
        remove_columns=["audio"],
        batched=True,
        batch_size=batch_size
    )
    return xeno_features


def model_inference(feature_ds, batched, batch_size, dl_len, device):
    
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    
    if batched==True:
        dl = DataLoader(
            feature_ds,
            batch_size=batch_size,
            )
    else:
        dl=DataLoader(feature_ds)

    results = []
    for i, examples in tqdm(enumerate(dl), total = dl_len):

        batch = {k: v.to(device) for k, v in examples.items()}
        features = batch['input_values']

        with torch.no_grad():
            o = model(features)
            results.append(o.extract_features.mean(dim=1))

        if i % 64 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return torch.vstack(results)


if __name__ == '__main__':

    start = time()
    BATCH_SIZE = 128

    df_path = pd.read_csv('./data/audio_paths_5.csv')
    model_name = "facebook/wav2vec2-base"
    wav2vec_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    xeno_ds = get_ds(csv_path='./data/audio_paths_5.csv')
    xeno_features = get_features(xeno_ds, BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f'DEVICE IS {device}')
    
    results = model_inference(
        xeno_features,
        batched=False,
        batch_size=BATCH_SIZE,
        dl_len=df_path.shape[0],
        device=device
        )
    
    print(f"results in shape {results.shape}!")
    torch.save(results, './data/xeno_amazon_embeddings_5s.pt')
    
    end = time()
    print(f"Process Took {end-start} seconds!!")