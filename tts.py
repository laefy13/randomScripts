import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import librosa
import webbrowser
import soundfile as sf

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']

sentence_lang = [
    ("Always bring cinnamon buns on a deep sea diving expedition.",lang[2]),
    ("The teens wondered what was kept in the red shed on the far edge of the school grounds.",lang[2]),
    ("Barking dogs and screaming toddlers have the unique ability to turn friendly neighbors into cranky enemies.",lang[2]),
    ("He was the type of guy who liked Christmas lights on his house in the middle of July.",lang[2]),
    ("He knew it was going to be a bad day when he saw mountain lions roaming the streets.",lang[2]),
    ("何か不審なものを見たり聞いたりは？楽器を除いたら静かな人でしたから",lang[0]),
    ("何かおかしいことを見たり聞いたりしたならば、牧師先生に話しなさい。",lang[0]),
    ("物に触る　私たちは、自分が見たり聞いたりするものの外見を身につけるだけではなく、また自分が触るものの感触も身につけます-それを素早い実験をおこなうことで、見ることができます。",lang[0]),
    ("アゴラの職員として劇場の運営面からトイレ掃除までいろいろやりながら、空いた時間に稽古をして、自主公演で発表していました。",lang[0]),
    ("元請負人は、前払金の支払を受けたときは、下請負人に対して、資材の購入、労働者の募集その他建設工事の着手に必要な費用を前払金として支払うよう適切な配慮をしなければならない。",lang[0]),

]
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def tts_fn(text, speaker, language, speed,model, hps, speaker_ids,output_file):
    if language is not None:
        text = language_marks[language] + text + language_marks[language]
    speaker_id = speaker_ids[speaker]
    stn_tst = get_text(text, hps, False)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    sf.write(output_file, audio, samplerate=22050, subtype='PCM_16')

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


if __name__ == "__main__":

    vc_fn=[]
    models_list = []
    for dirs,path,files in os.walk('models'):
        for file in files:
          if file.startswith('G_'):
            models_list.append(f'./models/{file}')
    model_countt = 0
    for model_dir in models_list:

        if os.path.exists(f'/content/drive/MyDrive/sample_outputs/{model_dir[7:]}') or \
        os.path.exists(f'/content/drive/MyDrive/sample_outputs/{model_dir[9:]}') :
          continue

        hps = utils.get_hparams_from_file("./config.json")
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = net_g.eval()
        try:
            _ = utils.load_checkpoint(model_dir, net_g, None)
        except:
            continue   
        speakers = list(hps.speakers.keys())
        for count,(sentence,language) in enumerate(sentence_lang):
            for speaker in speakers:
                tts_fn(sentence, speaker, language, 1,net_g, hps, hps.speakers,f'/content/drive/MyDrive/sample_outputs/{model_dir[9:]}/{speaker}/{language}_{count}.wav')
                print('made one')



