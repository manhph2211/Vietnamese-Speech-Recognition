import glob
import os
from tqdm import tqdm
import csv


def get_text(path_to_script_file):
    with open(path_to_script_file,'rb') as f:
        text = f.read().decode("utf8").strip().upper()
    return text


def make(root='data', name='vlsp2020_train_set_02', save='custom_data'):
    # all_audio_file_paths = glob.glob(os.path.join(root, name,"*.wav"))
    all_script_file_paths = glob.glob(os.path.join(root, name,"*.txt"))
    # for audio_file in tqdm(all_audio_file_paths):
    #     file_name = audio_file.split('\\')[-1]
    #     os.rename(audio_file, os.path.join(root,save,"wavs",file_name))
    with open(os.path.join(root,save,'metadata.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter="|")
        for text_file in tqdm(all_script_file_paths):
            file_name = text_file.split("\\")[-1].split(".")[0]
            text = get_text(text_file)
            row = [file_name,text,text]
            writer.writerow(row)

if __name__ == "__main__":
    make()
