from stepaudio2vllm import StepAudio2
from token2wav import Token2wav


# ASR
def asr_test(model):
    messages = [
        {"role": "system", "content": "请记录下你所听到的语音内容。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0)
    print(text)


# S2TT（support: en,zh,ja）
def s2tt_test(model):
    messages = [
        {"role": "system", "content":"请仔细聆听这段语音，然后将其内容翻译成中文。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.1)
    print(text)


# audio caption
def audio_caption_test(model):
    messages = [
        {"role": "system", "content":"Please briefly explain the important events involved in this audio clip."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.1)
    print(text)


# S2ST（support: en,zh）
def s2st_test(model, token2wav):
    messages = [
        {"role": "system", "content":"请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese speech."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    _, text, audio = model(messages, max_tokens=2048, temperature=0.7)
    print(text)
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-s2st.wav', 'wb') as f:
            f.write(audio)


# multi turn speech-to-text conversation
def multi_turn_aqta_test(model):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": None})
        _, text, _ = model(history, max_tokens=1024, temperature=0.5)
        print(text)
        history.pop(-1)
        history.append({"role": "assistant", "content": text})


# multi turn speech-to-speech conversation
def multi_turn_aqaa_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": "<tts_start>", "eot": False})
        response, text, audio = model(history, max_tokens=2048, temperature=0.7)
        print(text)
        if audio:
            audio = [x for x in audio if x < 6561]
            audio = token2wav(audio, prompt_wav='assets/default_female.wav')
            with open(f'output-round-{round_idx}.wav', 'wb') as f:
                f.write(audio)
        history.pop(-1)
        history.append({"role": "assistant", "tts_content": response.get("tts_content", {})})


# Tool call & Web search
def tool_call_test(model, token2wav):
    history = [
        {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]
    response, text, audio = model(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7)
    print(text)
    print(response["tool_calls"])
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-tool-call-1.wav', 'wb') as f:
            f.write(audio)

    history.pop(-1)
    with open('assets/search_result.txt') as f:
        search_result = f.read().strip()
    history += [
        {"role": "assistant", "tts_content": response["tts_content"], "tool_calls": response["tool_calls"]},
        {"role": "input", "tool_call_id": response["tool_calls"][0]["id"], "content": [{"type": "text", "text": search_result}, {"type": "text", "text": '\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。'}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    response, text, audio = model(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7)
    print(text)
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-tool-call-2.wav', 'wb') as f:
            f.write(audio)


# Paralinguistic information understanding
def paralinguistic_test(model, token2wav):
    messages = [
        {"role": "system", "content":"请用语音与我交流。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/paralinguistic_information_understanding.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    _, text, audio = model(messages, max_tokens=2048, temperature=0.7)
    print(text)
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-paralinguistic.wav', 'wb') as f:
            f.write(audio)


# Audio understanding
def mmau_test(model):
    messages = [
        {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/mmau_test.wav"}]},
                                      {"type": "text", "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering] Output the final answer in <RESPONSE> </RESPONSE>."},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, best_of=2, use_beam_search=True)
    print(text)


# Universal audio caption
def uac_test(model):
    messages = [
        {"role": "system", "content": "你是一位经验丰富的音频分析专家，擅长对各种语音音频进行深入细致的分析。你的任务不仅仅是将音频内容准确转写为文字，还要对说话人的声音特征（如性别、年龄、情绪状态）、背景声音、环境信息以及可能涉及的事件进行全面描述。请以专业、客观的视角，详细、准确地完成每一次分析和转写。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.5, top_p=0.9)
    print(text)

if __name__ == '__main__':
    api_url = "http://localhost:8000/v1/chat/completions"
    model_name = "step-audio-2-mini"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')

    asr_test(model)
    s2tt_test(model)
    audio_caption_test(model)
    s2st_test(model, token2wav)
    multi_turn_aqta_test(model)
    multi_turn_aqaa_test(model, token2wav)
    tool_call_test(model, token2wav)
    paralinguistic_test(model, token2wav)
    mmau_test(model)
    uac_test(model)
