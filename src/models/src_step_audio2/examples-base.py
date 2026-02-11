# ASR
def asr_test(model):
    messages = [
        "请记录下你所听到的语音内容。",
        {"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.1, do_sample=True)
    print(text)

# S2TT（support: en,zh,ja）
def s2tt_test(model):
    messages = [
        "请仔细聆听这段语音，然后将其内容翻译成中文",
        # "Please listen carefully to this audio and then translate its content into Chinese.",
        {"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.1, do_sample=True)
    print(text)


# audio caption
def audio_caption_test(model):
    messages = [
        "Please briefly explain the important events involved in this audio clip.",
        {"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.1, do_sample=True)
    print(text)

# TTS（support: en,zh,ja)
def tts_test(model, token2wav):
    messages = [
        "以自然的语速读出下面的文字。\n",
        # "Read this paragraph at a natural pace.\n",
        "你好呀，我是你的AI助手，很高兴认识你！"
        "<tts_start>"
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    #print(tokens)
    audio = [x for x in audio if x < 6561] # remove audio padding
    audio = token2wav(audio, prompt_wav='assets/default_male.wav')
    with open('output-tts.wav', 'wb') as f:
        f.write(audio)

# T2ST（support: en,zh,ja)
def t2st_test(model, token2wav):
    messages = [
        "将下面的文本翻译成英文，并用语音播报。\n",
        # "Translate the following text into English and broadcast it with voice.\n",
        "你好呀，我是你的AI助手，很高兴认识你！"
        "<tts_start>"
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    #print(tokens)
    audio = [x for x in audio if x < 6561] # remove audio padding
    audio = token2wav(audio, prompt_wav='assets/default_male.wav')
    with open('output-t2st.wav', 'wb') as f:
        f.write(audio)

# S2ST（support: en,zh）
def s2st_test(model, token2wav):
    messages = [
        "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。",
        # "Please listen carefully to this audio and then translate its content into Chinese speech.",
        {"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"},
        "<tts_start>"
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    #print(tokens)
    audio = [x for x in audio if x < 6561] # remove audio padding
    audio = token2wav(audio, prompt_wav='assets/default_female.wav')
    with open('output-s2st.wav', 'wb') as f:
        f.write(audio)

# multi turn aqta
def multi_turn_aqta_test(model):
    history = []
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append(
            {"type": "audio", "audio": inp_audio}
        )
        tokens, text, _ = model(history, max_new_tokens=256, temperature=0.5, do_sample=True)
        print(text)
        history.append(text)

# multi turn aqaa
def multi_turn_aqaa_test(model, token2wav):
    history = []
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append(
            {"type": "audio", "audio": inp_audio},
        )
        history.append("<tts_start>")
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(text)
        audio = [x for x in audio if x < 6561] # remove audio padding
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open(f'output-round-{round_idx}.wav', 'wb') as f:
            f.write(audio)
        history.append(
            {"type":"token", "token": tokens}
        )

if __name__ == '__main__':
    from stepaudio2 import StepAudio2Base
    from token2wav import Token2wav

    model = StepAudio2Base('Step-Audio-2-mini-Base')
    token2wav = Token2wav('Step-Audio-2-mini-Base/token2wav')
    asr_test(model)
    s2tt_test(model)
    audio_caption_test(model)
    tts_test(model, token2wav)
    t2st_test(model, token2wav)
    s2st_test(model, token2wav)
    multi_turn_aqta_test(model)
    multi_turn_aqaa_test(model, token2wav)
