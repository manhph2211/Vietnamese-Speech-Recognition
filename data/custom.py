import glob
import os
from tqdm import tqdm
import csv
import json


def get_text(path_to_script_file):
    with open(path_to_script_file,'rb') as f:
        text = f.read().decode("utf8").strip().upper()
    return text


def write_json(data, file_path):
    with open(file_path, 'a', encoding='utf8') as f:
    #     json.dump(data, f, ensure_ascii=False)
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


## convert to ljspeech format
def make_ljspeech(root='data', name='vlsp2020_train_set_02', save='LJSpeech-1.1'):
    if not os.path.isdir(os.path.join(root, save, 'wavs')):
        os.mkdir(os.path.join(root, save, 'wavs'))
    all_audio_file_paths = glob.glob(os.path.join(root, name,"*.wav"))
    all_script_file_paths = glob.glob(os.path.join(root, name,"*.txt"))
    for audio_file in tqdm(all_audio_file_paths):
        file_name = audio_file.split('/')[-1]
        os.rename(audio_file, os.path.join(root,save,"wavs",file_name))
    with open(os.path.join(root,save,'metadata.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter="|")
        for text_file in tqdm(all_script_file_paths):
            file_name = text_file.split("/")[-1].split(".")[0]
            text = get_text(text_file)
            row = [file_name,text,text]
            writer.writerow(row)
            

## comvert to timit format (this one is for finetuning wav2vec)
def make_timit(root='data', name='vlsp2020_train_set_02', save='custom_data', ratio = 0.9):
    if not os.path.isdir(os.path.join(root, save)):
        os.mkdir(os.path.join(root, save))
    all_script_file_paths = glob.glob(os.path.join(root, name, "*.txt"))
    train_data = []
    test_data = []
    for i,text_file in tqdm(enumerate(all_script_file_paths)):
        text = get_text(text_file)
        wav_file = os.path.join(root,'LJSpeech-1.1','wavs',text_file.split("/")[-1].replace("txt","wav"))
        sample = {"file":wav_file, "text":text}
        if i < int(ratio*len(all_script_file_paths)):
            train_data.append(sample)
        else:
            test_data.append(sample)    
        write_json(sample, os.path.join(root,save,'train.json'))
        write_json(sample, os.path.join(root,save,'test.json'))
      

if __name__ == "__main__":
    # make_ljspeech()

    make_timit()