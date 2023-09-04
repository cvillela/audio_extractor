# ras_audio_extractor
Repo configured with useful functions for preprocessing, segmenting and creating audio datasets, as well as extracting embeddings.

## Installation
 - Create a conda environment and activate it:
```
   conda create --name ras_audio
   conda activate ras_audio
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
## Current Features:
 - Efficiently segments audio files in given directory with length, arbitrary overlap between samples + padding/cropping
 - Creates splits for train/test/val
 - Jukebox Embedding Extractor
   - (1 -1722) datapoints with (4800) dimensions for anything between (0-24) seconds of audio
 - Wav-2-Vec embedding extraction
 - MusicGen specific metadata creation

### TO-DO
- [ ] Reformat code to have a single Audio Segmenter class + call variations from command line
- [ ] Jukebox extraction notebook to script
- [ ] Create template for Metadata Generation for different models, that can be called upon the Audio Segmenter
- [ ] Add audio Denoise + Silence
- [ ] Add Music Emotion Recognition Pipeline
- [ ] Bandwith Extension / Audio-Super resolution?

### BUGS
 - Handle audio with varying sample_rates -> load with librosa setting the sample_rate prior

