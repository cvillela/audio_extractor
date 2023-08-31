# ras_audio_extractor
Repo configured with useful functions for preprocessing, segmenting and creating audio datasets, as well as extracting embeddings.

### Current Features:
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
