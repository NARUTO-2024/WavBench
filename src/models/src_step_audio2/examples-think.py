def think_test(model, token2wav):
    history = [{"role": "system", "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append(
            {"role": "human", "content": [{"type": "audio", "audio": inp_audio}]}
        )
        # get think content, stop when "</think>" appears
        history.append({"role": "assistant", "content": "\n<think>\n", "eot": False})
        _, think_content, _ = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True, stop_strings=['</think>'])
        print('<think>' + think_content + '>')
        # get audio response
        history[-1]["content"] += think_content + ">\n\n<tts_start>"
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(text)
        audio = [x for x in audio if x < 6561] # remove audio padding
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open(f'output-round-{round_idx}-think.wav', 'wb') as f:
            f.write(audio)
        # remove think content from history
        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content":[
                    {"type": "text", "text":"<tts_start>"},
                    {"type":"token", "token": tokens}
                ]
            }
        )

def think_test_vllm(model, token2wav):
    history = [{"role": "system", "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": "<think>", "eot": False})
        #_, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=[{"token": "</think>"}])
        _, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=["</think>"])
        print('<think>' + think_content + '</think>')
        history[-1]["content"] += think_content + "</think>" + "\n\n<tts_start>"
        response, text, audio = model(history, max_tokens=2048, temperature=0.7, repetition_penalty=1.05)
        print(text)
        audio = None
        if audio:
            audio = [x for x in audio if x < 6561]
            audio = token2wav(audio, prompt_wav='assets/default_female.wav')
            with open(f'output-round-{round_idx}-think.wav', 'wb') as f:
                f.write(audio)
        history.pop(-1)
        history.append({"role": "assistant", "tts_content": response.get("tts_content", {})})


if __name__ == '__main__':
    from stepaudio2 import StepAudio2
    from token2wav import Token2wav

    model = StepAudio2('Step-Audio-2-mini-Think')
    token2wav = Token2wav('Step-Audio-2-mini-Think/token2wav')
    think_test(model, token2wav)

    from stepaudio2vllm import StepAudio2
    api_url = "http://localhost:8001/v1/chat/completions"
    model_name = "step-audio-2-mini-think"
    token2wav = None

    model = StepAudio2(api_url, model_name)
    think_test_vllm(model, token2wav)
