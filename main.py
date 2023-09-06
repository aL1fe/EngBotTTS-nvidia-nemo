from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
import io
import time
import torchaudio

app = FastAPI()

# Load FastPitch
from nemo.collections.tts.models import FastPitchModel
spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")

# Load vocoder
from nemo.collections.tts.models import HifiGanModel
model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

@app.get("/")
def get_audio(query):
    a = time.time()

    parsed = spec_generator.parse(query)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)


    # Преобразование в формат MP3 в памяти
    audio_data = io.BytesIO()
    torchaudio.save(audio_data, audio.squeeze(1), 22050, format='wav')

    # Конвертация в MP3
    audio_data.seek(0)
    audio_segment = AudioSegment.from_wav(audio_data)
    mp3_data = io.BytesIO()
    audio_segment.export(mp3_data, format="mp3")

    # Перемотка указателя в начало данных
    audio_data.seek(0)

    # Отправка MP3-данных в качестве ответа
    return StreamingResponse(audio_data, media_type="audio/mpeg", headers={"Content-Disposition": f'attachment; filename="Pronunciation.mp3"'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)