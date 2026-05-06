import os
import json
from glob import glob
import argparse
from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
from math import ceil
import xml.etree.ElementTree as ET
from hs_utils import quantize_hidden_states


def _load_audio_segment_class():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError("pydub is required for TTS validation or audio cutting, but not for hidden-state extraction") from exc
    return AudioSegment


ZH_LEGACY_VOICE_TO_SPK_ID = {
    'zh-CN-XiaoxiaoNeural': 0,
    'zh-CN-XiaoyiNeural': 1,
    'zh-CN-YunjianNeural': 2,
    'zh-CN-YunxiaNeural': 3,
    'zh-CN-YunxiNeural': 4,
    'zh-CN-YunyangNeural': 5
}


def _normalize_whisper_style(style: str) -> str:
    style = str(style or 'auto').strip().lower().replace('_', '-')
    aliases = {
        'v2': 'large-v2',
        'largev2': 'large-v2',
        'v3': 'large-v3',
        'largev3': 'large-v3',
    }
    return aliases.get(style, style)


def _resolve_whisper_ckpt(whisper_ckpt: str, whisper_style: str) -> Tuple[str, str]:
    style = _normalize_whisper_style(whisper_style)
    if style not in {'auto', 'medium', 'large-v2', 'large-v3'}:
        raise ValueError('unsupported whisper_style `{}`, expected auto|medium|large-v2|large-v3'.format(whisper_style))

    ckpt = str(whisper_ckpt or '').strip()
    if ckpt == '':
        if style == 'medium':
            ckpt = 'openai/whisper-medium'
        elif style == 'large-v2':
            ckpt = 'openai/whisper-large-v2'
        elif style == 'large-v3':
            ckpt = 'openai/whisper-large-v3'
        else:
            raise ValueError('please provide --whisper or set --whisper_style to medium|large-v2|large-v3')

    if style == 'auto':
        ckpt_l = ckpt.lower()
        if 'large-v3' in ckpt_l or 'largev3' in ckpt_l:
            style = 'large-v3'
        elif 'large-v2' in ckpt_l or 'largev2' in ckpt_l:
            style = 'large-v2'
        elif 'medium' in ckpt_l:
            style = 'medium'

    return ckpt, style


def _parse_layer_groups(spec: str) -> List[Tuple[int, int]]:
    groups = []
    if spec is None or str(spec).strip() == '':
        return groups
    for token in str(spec).split(','):
        token = token.strip()
        if token == '':
            continue
        if '-' in token:
            start, end = token.split('-', 1)
            start = int(start.strip())
            end = int(end.strip())
        else:
            start = int(token)
            end = int(token)
        if end < start:
            start, end = end, start
        groups.append((start, end))
    return groups


def _parse_group_weights(spec: str, n_groups: int) -> List[float]:
    if n_groups <= 0:
        return []
    if spec is None or str(spec).strip() == '':
        return [1.0 / n_groups for _ in range(n_groups)]
    weights = [float(x.strip()) for x in str(spec).split(',') if x.strip() != '']
    if len(weights) != n_groups:
        raise ValueError('group weights count ({}) does not match groups count ({})'.format(len(weights), n_groups))
    total = sum(weights)
    if total <= 0:
        return [1.0 / n_groups for _ in range(n_groups)]
    return [w / total for w in weights]


def _prepare_hidden_state_fuse(
    hs_fuse: str,
    hs_layer_start: int,
    hs_layer_end: int,
    hs_groups: str,
    hs_group_weights: str,
    max_layer_idx: int
):
    hs_fuse = str(hs_fuse or 'last').strip().lower()
    if hs_fuse not in {'last', 'range_mean', 'grouped_mean'}:
        raise ValueError('unsupported hs_fuse `{}`, expected last|range_mean|grouped_mean'.format(hs_fuse))

    start = max(0, min(int(hs_layer_start), max_layer_idx))
    end = max(0, min(int(hs_layer_end), max_layer_idx))
    if end < start:
        start, end = end, start

    groups = _parse_layer_groups(hs_groups)
    if hs_fuse == 'grouped_mean' and len(groups) == 0:
        groups = [(10, 13), (14, 17), (18, min(22, max_layer_idx))]
    clipped_groups = []
    for g_start, g_end in groups:
        g_start = max(0, min(int(g_start), max_layer_idx))
        g_end = max(0, min(int(g_end), max_layer_idx))
        if g_end < g_start:
            g_start, g_end = g_end, g_start
        clipped_groups.append((g_start, g_end))

    weights = _parse_group_weights(hs_group_weights, len(clipped_groups)) if hs_fuse == 'grouped_mean' else []
    return hs_fuse, start, end, clipped_groups, weights


def _select_hidden_states(
    hidden_states,
    t_len: int,
    hs_fuse: str,
    layer_start: int,
    layer_end: int,
    hs_groups: List[Tuple[int, int]],
    hs_weights: List[float]
) -> torch.Tensor:
    if hs_fuse == 'last':
        selected = hidden_states[-1]
    elif hs_fuse == 'range_mean':
        selected = torch.stack(hidden_states[layer_start:layer_end + 1], dim=0).mean(dim=0)
    else:
        selected = None
        for weight, (g_start, g_end) in zip(hs_weights, hs_groups):
            group_state = torch.stack(hidden_states[g_start:g_end + 1], dim=0).mean(dim=0)
            selected = float(weight) * group_state if selected is None else selected + float(weight) * group_state
    return selected[:, :t_len, :]


def _write_hidden_state_manifest(
    target: str,
    manifest_name: str,
    whisper_ckpt: str,
    whisper_style: str,
    feature_extractor: WhisperFeatureExtractor,
    encoder: torch.nn.Module,
    hs_fuse: str,
    layer_start: int,
    layer_end: int,
    hs_groups: List[Tuple[int, int]],
    hs_weights: List[float],
    device: str
):
    if manifest_name is None or str(manifest_name).strip() == '':
        return
    config = getattr(encoder, 'config', None)
    manifest = {
        'format_version': 1,
        'whisper_ckpt': whisper_ckpt,
        'whisper_style': whisper_style,
        'feature_size': int(getattr(feature_extractor, 'feature_size', 0)),
        'num_mel_bins': int(getattr(config, 'num_mel_bins', 0)) if config is not None else None,
        'd_model': int(getattr(config, 'd_model', 0)) if config is not None else None,
        'encoder_layers': int(getattr(config, 'encoder_layers', 0)) if config is not None else None,
        'hs_fuse': hs_fuse,
        'hs_layer_start': int(layer_start),
        'hs_layer_end': int(layer_end),
        'hs_groups': [[int(s), int(e)] for s, e in hs_groups],
        'hs_group_weights': [float(w) for w in hs_weights],
        'normalized': True,
        'quantized': 'int8_symmetric_127',
        'device': device,
    }
    with open(os.path.join(target, manifest_name), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _infer_tts_language(locale: str, acoustic_model: str = None) -> str:
    if acoustic_model is not None:
        dataset = acoustic_model.rsplit('_', 1)[-1]
        if dataset in {'csmsc', 'aishell3', 'male'}:
            return 'zh'
        if dataset in {'ljspeech', 'vctk'}:
            return 'en'
        if dataset in {'mix', 'canton'}:
            return dataset

    locale = locale.lower()
    if locale.startswith('zh'):
        return 'zh'
    if locale.startswith('en'):
        return 'en'
    raise ValueError(f'could not infer PaddleSpeech language from locale `{locale}`')


def _infer_tts_models(locale: str, acoustic_model: str = None):
    if acoustic_model is None:
        lang = _infer_tts_language(locale)
        if lang == 'zh':
            return lang, 'fastspeech2_aishell3', 'hifigan_aishell3'
        if lang == 'en':
            return lang, 'fastspeech2_ljspeech', 'hifigan_ljspeech'
        raise ValueError(f'please provide `--voice` for locale `{locale}`')

    lang = _infer_tts_language(locale, acoustic_model)
    dataset = acoustic_model.rsplit('_', 1)[-1]
    vocoder_map = {
        'csmsc': 'hifigan_csmsc',
        'ljspeech': 'hifigan_ljspeech',
        'aishell3': 'hifigan_aishell3',
        'vctk': 'hifigan_vctk',
        'male': 'hifigan_male'
    }
    if dataset not in vocoder_map:
        raise ValueError(f'could not infer PaddleSpeech vocoder for acoustic model `{acoustic_model}`')
    return lang, acoustic_model, vocoder_map[dataset]


def _resolve_tts_profile(
    locale: str,
    requested_voice: str = None,
    keyword_index: int = 0
):
    if requested_voice in ZH_LEGACY_VOICE_TO_SPK_ID:
        lang, acoustic_model, vocoder = _infer_tts_models(locale, 'fastspeech2_aishell3')
        return {
            'lang': lang,
            'am': acoustic_model,
            'voc': vocoder,
            'spk_id': ZH_LEGACY_VOICE_TO_SPK_ID[requested_voice],
            'voice_tag': requested_voice
        }

    if requested_voice is None and locale.lower().startswith('zh'):
        voice_names = list(ZH_LEGACY_VOICE_TO_SPK_ID.keys())
        voice_tag = voice_names[keyword_index % len(voice_names)]
        lang, acoustic_model, vocoder = _infer_tts_models(locale, 'fastspeech2_aishell3')
        return {
            'lang': lang,
            'am': acoustic_model,
            'voc': vocoder,
            'spk_id': ZH_LEGACY_VOICE_TO_SPK_ID[voice_tag],
            'voice_tag': voice_tag
        }

    lang, acoustic_model, vocoder = _infer_tts_models(locale, requested_voice)
    return {
        'lang': lang,
        'am': acoustic_model,
        'voc': vocoder,
        'spk_id': 0,
        'voice_tag': acoustic_model
    }


class MOSScorer:
    def __init__(self, model_specifier: str = 'utmos22_strong'):
        self.model_specifier = model_specifier
        self.predictor = None

    def _load(self):
        if self.predictor is None:
            self.predictor = torch.hub.load(
                'tarepan/SpeechMOS:v1.2.0',
                self.model_specifier,
                trust_repo=True
            )
            self.predictor.eval()

    def score(self, audio_file: str) -> float:
        self._load()
        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.size(dim=0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        with torch.inference_mode():
            return float(self.predictor(waveform, sample_rate).item())


def _is_valid_tts_audio(
    output_file: str,
    mos_scorer: Optional[MOSScorer] = None,
    mos_threshold: Optional[float] = None,
    min_duration_ms: int = 250
):
    AudioSegment = _load_audio_segment_class()
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return False, None

    try:
        audio = AudioSegment.from_file(output_file)
    except Exception:
        return False, None

    if len(audio) < min_duration_ms:
        return False, None

    if mos_scorer is None or mos_threshold is None:
        return True, None

    mos_score = mos_scorer.score(output_file)
    return mos_score >= mos_threshold, mos_score


def keyword_tts(
    tts_folder: str,
    keyword_file: str,
    locale: str,
    voice: str = None,
    tts_device: str = 'auto',
    mos_threshold: float = 3.5,
    tts_retries: int = 3,
    log_every: int = 100,
    fail_on_tts: bool = False,
    min_duration_ms: int = 250
):
    # argument check
    assert os.path.isdir(tts_folder), f'the provided folder for storing the synthesized speech does not exist'
    assert os.path.exists(keyword_file), f'there is no file with keywords list'

    # get list of already produced synthesized speech
    tts_file_indices = [int(os.path.splitext(os.path.basename(f_name))[0]) for f_name in glob(os.path.join(tts_folder, '*.wav'))]

    # get keywords
    with open(keyword_file, 'r') as f:
        keywords = [{
            'keyword': line.split('\t')[0].strip(),
            'voice': line.split('\t')[1].strip() if len(line.split('\t')) != 1 else None,
            'idx': idx
        } for idx, line in enumerate(f.readlines())]

    leading_zeros = len(str(len(keywords) - 1))
    # remove indices of the already produced speech
    keywords = [item for item in keywords if item['idx'] not in tts_file_indices]

    # Lazy import so device can be configured first and non-TTS paths don't require PaddleSpeech.
    tts_device = str(tts_device).lower().strip()
    if tts_device not in {'auto', 'cpu', 'gpu'}:
        raise ValueError(f'unsupported tts_device `{tts_device}`, expected one of: auto|cpu|gpu')
    if tts_device == 'cpu':
        # Force PaddleSpeech to run on CPU (useful when latest GPUs are unsupported).
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from paddlespeech.cli.tts.infer import TTSExecutor
    if tts_device in {'cpu', 'gpu'}:
        try:
            import paddle
            paddle.set_device(tts_device)
        except Exception as e:
            print(f'[tts][warn] failed to set paddle device to `{tts_device}` ({e}), fallback to default device.')
    tts_executor = TTSExecutor()
    mos_scorer = MOSScorer() if mos_threshold is not None else None

    # generate audio for each keyword
    for idx, item in enumerate(tqdm(keywords), start=1):
        profile = _resolve_tts_profile(
            locale = locale,
            requested_voice = item['voice'] if item['voice'] is not None else voice,
            keyword_index = item['idx']
        )
        item['voice'] = profile['voice_tag']
        output_file = os.path.join(tts_folder, str(item['idx']).zfill(leading_zeros) + '.wav')
        best_path = None
        best_mos = None
        best_size = -1
        best_passes_threshold = False
        attempts = 0
        max_attempts = max(3, tts_retries)
        local_min_duration_ms = 120 if len(item['keyword']) <= 1 else min_duration_ms
        while attempts < max_attempts:
            attempts += 1
            if log_every > 0 and (idx % log_every == 0 or attempts > 1):
                print(f"[tts] {idx}/{len(keywords)} keyword='{item['keyword']}' attempt={attempts} synthesize...")
            attempt_file = output_file + f".attempt{attempts}.wav"
            if os.path.exists(attempt_file):
                os.remove(attempt_file)
            tts_executor(
                text = item['keyword'],
                lang = profile['lang'],
                am = profile['am'],
                voc = profile['voc'],
                spk_id = profile['spk_id'],
                output = attempt_file
            )
            basic_valid, _ = _is_valid_tts_audio(
                output_file = attempt_file,
                mos_scorer = None,
                mos_threshold = None,
                min_duration_ms = local_min_duration_ms
            )
            mos_score = None
            mos_pass = False
            if basic_valid and mos_scorer is not None:
                mos_score = mos_scorer.score(attempt_file)
                if mos_threshold is not None:
                    mos_pass = mos_score >= mos_threshold
                else:
                    mos_pass = True
            else:
                mos_pass = basic_valid
            if log_every > 0 and (idx % log_every == 0 or attempts > 1):
                if basic_valid:
                    mos_msg = f"mos={mos_score:.3f}" if mos_score is not None else "valid"
                    print(f"[tts] {idx}/{len(keywords)} keyword='{item['keyword']}' attempt={attempts} ok ({mos_msg})")
                else:
                    mos_msg = f"mos={mos_score:.3f}" if mos_score is not None else "invalid audio"
                    print(f"[tts] {idx}/{len(keywords)} keyword='{item['keyword']}' attempt={attempts} failed ({mos_msg})")
            if basic_valid:
                if mos_score is not None:
                    if best_mos is None or mos_score > best_mos:
                        if best_path is not None and os.path.exists(best_path):
                            os.remove(best_path)
                        os.replace(attempt_file, output_file)
                        best_path = output_file
                        best_mos = mos_score
                        best_passes_threshold = mos_pass
                    else:
                        if os.path.exists(attempt_file):
                            os.remove(attempt_file)
                else:
                    # fallback: choose the largest file as "best"
                    size = os.path.getsize(attempt_file)
                    if size > best_size:
                        if best_path is not None and os.path.exists(best_path):
                            os.remove(best_path)
                        os.replace(attempt_file, output_file)
                        best_path = output_file
                        best_size = size
                        best_passes_threshold = True
                    else:
                        if os.path.exists(attempt_file):
                            os.remove(attempt_file)
            else:
                if os.path.exists(attempt_file):
                    os.remove(attempt_file)
        if best_path is None or not os.path.exists(best_path):
            msg = (
                f'failed to synthesize valid audio for keyword `{item["keyword"]}`'
                + (f' after {attempts} attempts, best MOS={best_mos:.3f}' if best_mos is not None else f' after {attempts} attempts')
            )
            if fail_on_tts:
                raise RuntimeError(msg)
            print(f"[tts][warn] {msg}, skipping.")
            continue
        if mos_threshold is not None and best_mos is not None and not best_passes_threshold:
            print(
                f"[tts][warn] keyword='{item['keyword']}' best MOS {best_mos:.3f} "
                f"is below threshold {mos_threshold:.3f}, keeping best anyway"
            )
        if log_every > 0 and (idx % log_every == 0 or attempts > 1):
            mos_msg = f", best_mos={best_mos:.3f}" if best_mos is not None else ""
            print(f"[tts] {idx}/{len(keywords)} keyword='{item['keyword']}' attempts={attempts}{mos_msg}")

    # dump keywords metadata with voice information
    with open(os.path.splitext(keyword_file)[0] + '_voice.txt' if 'voice' not in keyword_file else keyword_file, 'w') as f:
        f.write('\n'.join(['\t'.join([item['keyword'], item['voice']]) for item in keywords]))


def get_keywords_audios(
    wav: str,
    keywords: str,
    keywords_audios: str
):    
    AudioSegment = _load_audio_segment_class()
    # check if dataset folder exists
    assert os.path.isdir(wav), f'the directory for the audios could not be found, got {wav}'

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    # or of folders with folders of only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(wav, '*'))]):
        if all([os.path.isdir(f_name) for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*'))]):
            audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for subsubfolder in glob(os.path.join(subfolder, '*')) for f_name in glob(os.path.join(subsubfolder, '*.mp3'))+glob(os.path.join(subsubfolder, '*.wav'))+glob(os.path.join(subsubfolder, '*.opus'))]
        else:
            audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(wav, '*.mp3'))+glob(os.path.join(wav, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] : f_name
    for f_name in audio_files}

    # load keywords data
    with open(keywords, 'r') as f:
        metadata = [{
            'keyword': line.split('\t')[0].strip(),
            'source': line.split('\t')[1].strip(),
            'start': int(float(line.split('\t')[2].strip()) * 1000),
            'end': int(float(line.split('\t')[3].strip()) * 1000)
        } if len(line.split('\t')) == 4 else None for line in f.readlines()]

    leading_zeros = len(str(len(metadata) - 1))
    for idx, m_ in tqdm(enumerate(metadata)):
        if m_ == None:
            continue
        # skip keywords that were not aligned
        if m_['start'] == m_['end']:
            continue
        # load the audio file
        audio = AudioSegment.from_file(audio_files[m_['source']])
        # cut the audio
        cut_audio = audio[m_['start']:m_['end']]
        # save the cut audio to a new file
        cut_audio.export(os.path.join(keywords_audios, str(idx).zfill(leading_zeros) + '.mp3'), format="mp3")


def extract_hidden_states(
    audios: str,
    whisper_ckpt: str,
    target: str,
    codes: str = None,
    whisper_style: str = 'auto',
    hs_fuse: str = 'last',
    hs_layer_start: int = 10,
    hs_layer_end: int = 22,
    hs_groups: str = '',
    hs_group_weights: str = '',
    manifest_name: str = '_hs_manifest.json'
):
    # check if audio and target folders exist
    assert os.path.isdir(audios), f'the directory for the audios could not be found, got {audios}'
    assert os.path.isdir(target), f'the directory for the target could not be found, got {target}'

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    whisper_ckpt, whisper_style = _resolve_whisper_ckpt(
        whisper_ckpt=whisper_ckpt,
        whisper_style=whisper_style
    )

    # instantiate WhisperProcessor object
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_ckpt)

    # instantiate WhisperModel object and get encoder
    encoder = WhisperModel.from_pretrained(whisper_ckpt).encoder.to(device)
    encoder.eval()

    feature_size = int(getattr(feature_extractor, 'feature_size', 0))
    encoder_config = getattr(encoder, 'config', None)
    num_mel_bins = int(getattr(encoder_config, 'num_mel_bins', 0)) if encoder_config is not None else feature_size
    if whisper_style == 'large-v3' and feature_size != 128:
        raise ValueError('large-v3 hidden-state extraction expects 128 mel bins, got {}'.format(feature_size))
    if whisper_style in {'medium', 'large-v2'} and feature_size != 80:
        raise ValueError('{} hidden-state extraction expects 80 mel bins, got {}'.format(whisper_style, feature_size))

    max_layer_idx = int(getattr(encoder_config, 'encoder_layers', 0)) if encoder_config is not None else 0
    if max_layer_idx <= 0:
        # hidden_states includes the embedding output at index 0, so the last layer index equals encoder_layers.
        max_layer_idx = 32 if whisper_style in {'large-v2', 'large-v3'} else 24
    if str(hs_fuse or '').strip().lower() == 'grouped_mean' and str(hs_groups or '').strip() == '' and whisper_style == 'large-v3':
        hs_groups = '20-23,24-27,28-32'
    hs_fuse, layer_start, layer_end, hs_groups, hs_weights = _prepare_hidden_state_fuse(
        hs_fuse=hs_fuse,
        hs_layer_start=hs_layer_start,
        hs_layer_end=hs_layer_end,
        hs_groups=hs_groups,
        hs_group_weights=hs_group_weights,
        max_layer_idx=max_layer_idx
    )
    _write_hidden_state_manifest(
        target=target,
        manifest_name=manifest_name,
        whisper_ckpt=whisper_ckpt,
        whisper_style=whisper_style,
        feature_extractor=feature_extractor,
        encoder=encoder,
        hs_fuse=hs_fuse,
        layer_start=layer_start,
        layer_end=layer_end,
        hs_groups=hs_groups,
        hs_weights=hs_weights,
        device=device
    )

    # get codes if necessary
    if codes != None:
        with open(codes, 'r') as f:
            codes = [line.split('\t')[0].strip().split(' ')[0].strip() for line in f.readlines()]

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    # or of folders with folders of only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(audios, '*'))]):
        if all([os.path.isdir(f_name) for subfolder in glob(os.path.join(audios, '*')) for f_name in glob(os.path.join(subfolder, '*'))]):
            audio_files = [f_name for subfolder in glob(os.path.join(audios, '*')) for subsubfolder in glob(os.path.join(subfolder, '*')) for f_name in glob(os.path.join(subsubfolder, '*.mp3'))+glob(os.path.join(subsubfolder, '*.wav'))+glob(os.path.join(subsubfolder, '*.opus'))]
        else:
            audio_files = [f_name for subfolder in glob(os.path.join(audios, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(audios, '*.mp3'))+glob(os.path.join(audios, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] : f_name
    for f_name in audio_files}

    # and extract hidden states for each one
    # if code is present in list of codes, if it exists
    for code, audio_file in tqdm(audio_files.items()):
        if codes != None and not any([c_ in code for c_ in codes]):
            continue
        try:
            # load utterance audio and preprocess it
            t_waveform, t_sample_rate = torchaudio.load(audio_file)
            if t_waveform.size(dim=0) > 1:
                t_waveform = torch.mean(torchaudio.functional.resample(t_waveform, t_sample_rate, 16000), dim=0, keepdim=True)
            else:
                t_waveform = torchaudio.functional.resample(t_waveform, t_sample_rate, 16000)
            # extract features and hidden states
            t_features = feature_extractor(t_waveform[0], sampling_rate=16000, return_tensors='pt').input_features
            t_len = ceil(feature_extractor(t_waveform[0], sampling_rate=16000, return_tensors='pt', padding=True).input_features.size(dim=2) / 2.)
            with torch.inference_mode():
                hidden_states = encoder(
                    input_features = t_features.to(device),
                    output_hidden_states = True,
                    return_dict = True
                )['hidden_states']
            t_hidden_states = _select_hidden_states(
                hidden_states=hidden_states,
                t_len=t_len,
                hs_fuse=hs_fuse,
                layer_start=layer_start,
                layer_end=layer_end,
                hs_groups=hs_groups,
                hs_weights=hs_weights
            )

            # normalize hidden states
            t_hidden_states = t_hidden_states / torch.linalg.norm(t_hidden_states, dim=-1, keepdim=True).clamp_min(1e-8)

            # target file name
            f_name = os.path.join(target, os.path.splitext(os.path.basename(audio_file))[0] + '.bin') if 'audio-' not in os.path.splitext(os.path.basename(audio_file))[0] else os.path.join(target, os.path.splitext(os.path.basename(audio_file))[0][6:] + '.bin')
            # and dump hidden_states
            with open(f_name, 'wb') as f:
                torch.save(quantize_hidden_states(t_hidden_states.clone()), f)

        except Exception as e:
            print(e)
            continue


def cut_audios(
    wav: str,
    segments: str,
    segments_audios: str
):    
    AudioSegment = _load_audio_segment_class()
    # check if dataset folder exists
    assert os.path.isdir(wav), f'the directory for the audios could not be found, got {wav}'
    # check if xml segments file exists
    assert os.path.exists(segments), f'the file with segments does not exist'

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(wav, '*'))]):
        audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(wav, '*.mp3'))+glob(os.path.join(wav, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] if 'audio-' not in os.path.splitext(os.path.basename(f_name))[0] else os.path.splitext(os.path.basename(f_name))[0][6:] : f_name
    for f_name in audio_files}

    # parse xml file
    tree = ET.parse(segments)
    root = tree.getroot()
    for doc in tqdm(root):
        code = doc.attrib['code']
        for segment in doc:
            seg_id = segment.attrib['id']
            start = float(segment.attrib['start']) * 1000
            end = float(segment.attrib['end']) * 1000
            transcript = segment.find('current').text

            if transcript.strip() == '':
                continue
            # skip segments that were not aligned
            if start == end:
                continue
            # load the audio file
            audio = AudioSegment.from_file(audio_files[code])
            # cut the audio
            cut_audio = audio[start:end]
            # save the cut audio to a new file
            cut_audio.export(os.path.join(segments_audios, code + '-seg' + str(seg_id) + '.wav'), format='wav')


def main():

    parser = argparse.ArgumentParser(description = 'Utilities for building datasets')

    # command options
    parser.add_argument('--tts', dest='tts', action='store_true', help='use PaddleSpeech to generate the audios for the keywords')
    parser.add_argument('--cut_audios', dest='cut_audios', action='store_true', help='cut audios')
    parser.add_argument('--extract_hs', dest='extract_hs', action='store_true', help='extract the hidden states from the whisper encoder')

    # input options
    parser.add_argument('-a', '--audios', dest='audios', type=str, help='folder with the audios')
    parser.add_argument('-k', '--keywords', dest='keywords', type=str, help='file with keywords and other relevant information')
    parser.add_argument('-t', '--target', dest='target', type=str, help='folder to store results')
    parser.add_argument('-u', '--utterances', dest='utterances', type=str, default='', help='list of utterances ids and other relevant information')
    parser.add_argument('-s', '--segments', dest='segments', type=str, help='xml file with segments from whole audios')
    parser.add_argument('-l', '--locale', dest='locale', type=str, help='locale input used to infer the PaddleSpeech language')
    parser.add_argument('-v', '--voice', dest='voice', type=str, default='', help='optional PaddleSpeech acoustic model, such as `fastspeech2_csmsc`')
    parser.add_argument('-w', '--whisper', dest='whisper', type=str, help='whisper version')
    parser.add_argument('--whisper_style', dest='whisper_style', type=str, default='auto', help='hidden-state extraction profile: auto|medium|large-v2|large-v3')
    parser.add_argument('--hs_fuse', dest='hs_fuse', type=str, default='last', help='hidden states fusion mode: last|range_mean|grouped_mean')
    parser.add_argument('--hs_layer_start', dest='hs_layer_start', type=int, default=10, help='start layer index for range_mean/grouped_mean')
    parser.add_argument('--hs_layer_end', dest='hs_layer_end', type=int, default=22, help='end layer index for range_mean/grouped_mean')
    parser.add_argument('--hs_groups', dest='hs_groups', type=str, default='', help='group spec for grouped_mean, e.g. "10-13,14-17,18-22"')
    parser.add_argument('--hs_group_weights', dest='hs_group_weights', type=str, default='', help='weights for grouped_mean groups, e.g. "0.2,0.3,0.5"')
    parser.add_argument('--manifest_name', dest='manifest_name', type=str, default='_hs_manifest.json', help='metadata file written in the hidden-state target folder; empty disables it')
    parser.add_argument('--tts_device', dest='tts_device', type=str, default='auto', help='tts device: auto|cpu|gpu')
    parser.add_argument('--mos_threshold', dest='mos_threshold', type=float, default=3.5, help='MOS threshold for validating synthesized speech')
    parser.add_argument('--tts_retries', dest='tts_retries', type=int, default=3, help='number of retries when synthesized speech is invalid (minimum 3 attempts)')
    parser.add_argument('--log_every', dest='log_every', type=int, default=100, help='log progress every N keywords (0 disables)')
    parser.add_argument('--fail_on_tts', dest='fail_on_tts', action='store_true', help='fail the whole run if a keyword fails after retries')
    parser.add_argument('--min_duration_ms', dest='min_duration_ms', type=int, default=250, help='minimum audio duration in ms to treat a TTS output as valid')

    args = parser.parse_args()

    if args.tts:
        keyword_tts(
            tts_folder = args.target,
            keyword_file = args.keywords,
            locale = args.locale,
            voice = args.voice if args.voice != '' else None,
            tts_device = args.tts_device,
            mos_threshold = args.mos_threshold,
            tts_retries = args.tts_retries,
            log_every = args.log_every,
            fail_on_tts = args.fail_on_tts,
            min_duration_ms = args.min_duration_ms
        )
    elif args.cut_audios:
        if args.segments != None:
            cut_audios(
                wav = args.audios,
                segments = args.segments,
                segments_audios = args.target
            )
        else:
            get_keywords_audios(
                wav = args.audios,
                keywords = args.keywords,
                keywords_audios = args.target
            )
    elif args.extract_hs:
        extract_hidden_states(
            audios = args.audios,
            whisper_ckpt = args.whisper,
            target = args.target,
            codes = args.utterances if args.utterances != '' else None,
            whisper_style = args.whisper_style,
            hs_fuse = args.hs_fuse,
            hs_layer_start = args.hs_layer_start,
            hs_layer_end = args.hs_layer_end,
            hs_groups = args.hs_groups,
            hs_group_weights = args.hs_group_weights,
            manifest_name = args.manifest_name
        )


if __name__ == '__main__':

    main()
