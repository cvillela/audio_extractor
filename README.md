# Audio Extractor
Repo configured with useful functions for preprocessing, segmenting and creating audio datasets, as well as extracting embeddings.

## Installation
 - Create a conda environment and activate it:
```
   conda create --name audioext -c conda-forge python=3.11
   conda activate audioext
```
- Be sure to have [torch>=2.0 and torchaudio>=2.0 installed](https://pytorch.org/get-started/locally/)
- Install the requirements
 ```
    pip install -r ./requirements.txt
 ```
- For jukebox embeddings - Install [Jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)  from the github repo:
 ```
    pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
 ```
- Install ffmpeg
 ```
   conda install -c conda-forge ffmpeg
 ```

## Jukebox Extractor
Extract [OpenAI's Jukebox](https://openai.com/research/jukebox) embeddings from a series of audio files contained in a directory.
From the project directory, run for help on the parameters: 
```
python -m audioext.pipelines.jukebox_extractor -h
```
Note that running this script for the first time will automatically download the model weigths to the machine. Refer to [Jukemirlib](https://github.com/rodrigo-castellon/jukemirlib) for more information.

## Musicgen Dataset Generator
Create [MusiGen](https://github.com/facebookresearch/audiocraft) ready samples and metadata from a series of audio files contained in a directory, and send them to train, val and test splits. Prompting is still unconditional (use 8H38fNdtri as a prompt to all models).
From the project directory, run for help on the parameters: 
```
python -m audioext.pipelines.musicgen_dataset -h
```

## Reduce Noise + Remove Silence
Reduce noise and optionally remove silence in between sounds of files in a given directory. 
From the project directory, run for help on the parameters: 
```
python -m audioext.pipelines.denoise -h
```

## Next Steps
### TO-DO
- [x] Reformat code to have a decoupled audio_segmenter utility
- [x] Reformat code to have a decoupled audio_processer utility
- [x] Jukebox extraction notebook to callable script
- [x] Reformat pipelines for Musicgen Dataset Generation.
- [X] Add Denoise + Remove Silence.
- [ ] Reformat pipeline for Music Emotion Recognition.
- [ ] Create Audio Metadata enriching pipelines.   
- [ ] Add bandwith extension and audio super-resolution?

### Known bugs
 - Handle single audio file with varying sample_rate
 - Debug multi-channel normalization

