import tempfile
import traceback
from pathlib import Path

import gradio as gr

def save_tmp_audio(audio, cache_dir):
    with tempfile.NamedTemporaryFile(
        dir=cache_dir, delete=False, suffix=".wav"
    ) as temp_audio:
        temp_audio.write(audio)
    return temp_audio.name

def add_message(chatbot, history, mic, text):
    if not mic and not text:
        return chatbot, history, "Input is empty"

    if text:
        chatbot.append({"role": "user", "content": text})
        history.append({"role": "human", "content": text})
    elif mic and Path(mic).exists():
        chatbot.append({"role": "user", "content": {"path": mic}})
        history.append({"role": "human", "content": [{"type":"audio", "audio": mic}]})

    print(f"{history=}")
    return chatbot, history, None

def reset_state(system_prompt):
    return [], [{"role": "system", "content": system_prompt}]

def predict(chatbot, history, audio_model, token2wav, prompt_wav, cache_dir):
    try:
        history.append({"role": "assistant", "content": [{"type": "text", "text": "<tts_start>"}], "eot": False})
        tokens, text, audio = audio_model(history, max_new_tokens=4096, temperature=0.7, repetition_penalty=1.05, do_sample=True)
        print(f"predict {text=}")
        audio = token2wav(audio, prompt_wav)
        audio_path = save_tmp_audio(audio, cache_dir)
        chatbot.append({"role": "assistant", "content": {"path": audio_path}})
        history[-1]["content"].append({"type": "token", "token": tokens})
        history[-1]["eot"] = True
    except Exception:
        print(traceback.format_exc())
        gr.Warning(f"Some error happend, please try again.")
    return chatbot, history

def _launch_demo(args, audio_model, token2wav):
    with gr.Blocks(delete_cache=(86400, 86400)) as demo:
        gr.Markdown("""<center><font size=8>Step Audio 2 Demo</center>""")
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月29日，星期五\n请用默认女声与用户交流。",
                lines=2
            )
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            #avatar_images=["assets/user.png", "assets/assistant.png"],
            min_height=800,
            type="messages",
        )
        history = gr.State([{"role": "system", "content": system_prompt.value}])
        mic = gr.Audio(type="filepath")
        text = gr.Textbox(placeholder="Enter message ...")

        with gr.Row():
            clean_btn = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            submit_btn = gr.Button("🚀 Submit")

        def on_submit(chatbot, history, mic, text):
            chatbot, history, error = add_message(
                chatbot, history, mic, text
            )
            if error:
                gr.Warning(error)  # 显示警告消息
                return chatbot, history, None, None
            else:
                chatbot, history = predict(chatbot, history, audio_model, token2wav, args.prompt_wav, args.cache_dir)
                return chatbot, history, None, None

        submit_btn.click(
            fn=on_submit,
            inputs=[chatbot, history, mic, text],
            outputs=[chatbot, history, mic, text],
            concurrency_limit=4,
            concurrency_id="gpu_queue",
        )

        clean_btn.click(
            fn=reset_state,
            inputs=[system_prompt],
            outputs=[chatbot, history],
            #show_progress=True,
        )

        def regenerate(chatbot, history):
            while chatbot and chatbot[-1]["role"] == "assistant":
                chatbot.pop()
            while history and history[-1]["role"] == "assistant":
                print(f"discard {history[-1]}")
                history.pop()
            return predict(chatbot, history, audio_model, token2wav, args.prompt_wav, args.cache_dir)

        regen_btn.click(
            regenerate,
            [chatbot, history],
            [chatbot, history],
            #show_progress=True,
            concurrency_id="gpu_queue",
        )

    demo.queue().launch(
        server_port=args.server_port,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from pathlib import Path

    from stepaudio2 import StepAudio2
    from token2wav import Token2wav

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default='Step-Audio-2-mini', help="Model path.")
    parser.add_argument(
        "--server-port", type=int, default=7862, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--prompt-wav", type=str, default="assets/default_female.wav", help="Prompt wave for the assistant."
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/stepaudio2", help="Cache directory."
    )
    args = parser.parse_args()
    os.environ["GRADIO_TEMP_DIR"] = args.cache_dir
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    audio_model = StepAudio2(args.model_path)
    token2wav = Token2wav(f"{args.model_path}/token2wav")
    _launch_demo(args, audio_model, token2wav)
