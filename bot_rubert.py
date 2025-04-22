import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import html
from asr import transcribe_audio
from pydub import AudioSegment
import tempfile
from sentence_transformers import SentenceTransformer, util
import time

TOKEN = ""

# Загрузка модели и токенизатора
model_path = "rubert_fakenews20"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Загрузка датасета с источниками
df = pd.read_csv("Dataset_With_Core_Fake_Claim.csv")
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Список текстов и источников
corpus_texts = []
corpus_sources = []

for _, row in df.iterrows():
    for col in ["CORE_FAKE_CLAIM", "REAL_TEXT"]:
        text = str(row.get(col, "")).strip()
        source = str(row.get("SOURCE", "")).strip()
        if text and source and text != "/" * len(text):
            corpus_texts.append(text)
            corpus_sources.append(source)
corpus_embeddings = sbert_model.encode(corpus_texts, convert_to_tensor=True)

# Функция поиска наиболее похожего текста по смыслу
def find_semantic_source(user_text, threshold=0.7):
    query_embedding = sbert_model.encode(user_text, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0]
    best_hit = hits[0]
    if best_hit["score"] >= threshold:
        matched_index = best_hit["corpus_id"]
        return corpus_sources[matched_index]
    return None



def get_trigger_words(text: str, vectorizer, model, top_n: int = 5):
    text_lower = text.lower()
    vec = vectorizer.transform([text_lower])
    tokens = vectorizer.inverse_transform(vec)[0]

    log_probs = model.feature_log_prob_
    importance = log_probs[1] - log_probs[0]
    feature_names = vectorizer.get_feature_names_out()

    weights = {feature_names[i]: importance[i] for i in range(len(feature_names))}
    matched = [(word, weights[word]) for word in tokens if word in weights]
    sorted_words = sorted(matched, key=lambda x: x[1], reverse=True)[:top_n]

    return [w[0] for w in sorted_words]


# Handle voice/audio messages
async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # First inform the user that processing is starting
    processing_msg = await update.message.reply_text("🎧 Обрабатываю аудио сообщение...")
    
    try:
        # Get voice message file
        voice = update.message.voice or update.message.audio
        voice_file = await context.bot.get_file(voice.file_id)
        
        # Create temp files for processing
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as ogg_file:
            ogg_path = ogg_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name
        
        # Download the voice message
        await voice_file.download_to_drive(ogg_path)
        
        # Convert ogg to wav using pydub
        audio = AudioSegment.from_file(ogg_path, format="ogg")
        audio.export(wav_path, format="wav")
        
        text = transcribe_audio(wav_path)
        print(f"Transcription: {text}")
        
        # Clean up temp files
        os.remove(ogg_path)
        os.remove(wav_path)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            fake_prob = probs[0][1].item()

        percent = round(fake_prob * 100, 2)

        matched_link = find_semantic_source(text)


        reply = f"🤖 <b>Анализ аудиосообщения:</b>\n\n"
        reply += f"🎤 Транскрипция: <i>{text}</i>\n\n"
        reply += f"📊 Вероятность фейка: <b>{percent}%</b>\n"
        reply += f"🔗 Источник: "
        reply += f"<a href='{matched_link}'>{matched_link}</a>" if matched_link else "не найден"
        
        await update.message.reply_text(reply, parse_mode="HTML")
        
    except Exception as e:
        await processing_msg.edit_text(f"❌ Ошибка при обработке аудио: {str(e)}")


# Обработка сообщения
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    tik = time.time()
    inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        fake_prob = probs[0][1].item()
    tak = time.time()
    print(f"Время обработки: {tak - tik:.4f} секунд")
    percent = round(fake_prob * 100, 2)

    tik = time.time()
    matched_link = find_semantic_source(user_text)
    tak = time.time()
    print(f"Время поиска источника: {tak - tik:.4f} секунд")

    reply = f"🤖 <b>Анализ новости:</b>\n\n"
    reply += f"📊 Вероятность фейка: <b>{percent}%</b>\n"
    # result = analyze_text(user_text)
    # reply += f"{result}\n\n"
    reply += f"🔗 Источник: "
    reply += f"<a href='{matched_link}'>{matched_link}</a>" if matched_link else "не найден"

    await update.message.reply_text(reply, parse_mode="HTML")

# Запуск бота
async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен...")

    # Using run_polling with shutdown parameters
    await app.run_polling(
        stop_signals=None,  # Handle shutdown yourself
        close_loop=False,  # Don't close the event loop
    )

if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    # Patch asyncio to allow nested event loops
    nest_asyncio.apply()

    # Now you can safely run
    asyncio.run(main())
