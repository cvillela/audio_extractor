import os
from pydub import AudioSegment

def list_wavs_from_dir(path):
    file_paths = []
    for dirpath, _, filenames in os.walk(path):
            for file_name in filenames:
                if file_name.endswith('.wav'):
                    file_paths.append(os.path.join(dirpath,file_name))
    return file_paths

def get_len_wavs(file_paths):
    dur=0
    for f in file_paths:
        audio = AudioSegment.from_file(f)
        dur+=len(audio)
        
    return dur/(1000*60*60) # in hours    

