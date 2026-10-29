# 🎙️ Fish Audio Drama 3 (35B-A3B) + Fish Speech WebUI в Google Colab (G4 96GB GPU)

Готовый к запуску шаблон для работы с флагманской голосовой моделью **[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)** и официальным интерфейсом **[Fish Speech Awesome WebUI](https://github.com/fishaudio/fish-speech)**, оптимизированный под видеокарту **G4 (96 ГБ VRAM)** в Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IjoiK/colab-comfy-a100-template/blob/main/fish-audio-drama-3/colab_fish_drama3_g4_96gb.ipynb)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-fishaudio%2Fdrama--3-orange)](https://huggingface.co/fishaudio/drama-3)
[![GitHub UI](https://img.shields.io/badge/GitHub-fishaudio%2Ffish--speech-blue?logo=github)](https://github.com/fishaudio/fish-speech)

---

## 🌟 Ключевые особенности оптимизации под G4 (96 ГБ VRAM)

- **🔥 Нативный аппаратный Block-FP8 (W8A8 Tensor Cores)**
  - Модель **Fish Audio Drama 3** содержит **35 млрд параметров** (MoE-архитектура, 3 млрд активных параметров на токен).
  - На видеокарте **G4 (96 ГБ VRAM)** задействуются аппаратные инструкции FP8 (`float8_e4m3fn`) и высокоскоростное Triton-ядро `fused_moe_kernel`.
- **⚡ Полный стек CUDA Graphs (`FISH_DRAMA3_CUDA_GRAPH=1`)**
  - Благодаря **96 ГБ видеопамяти** в VRAM предзахватываются все 4 бакета CUDA Graph (`1024`, `2048`, `4096` и `10240` фреймов). Это полностью убирает задержки CPU при декодировании и ускоряет генерацию длинных диалогов.
- **🎛️ Выбор из 3 режимов точности под 96 ГБ VRAM**
  - `FP8_TURBO` (~42 ГБ из 96 ГБ VRAM с графами) — аппаратный W8A8 FP8, максимальная скорость.
  - `BF16_ULTRA` (~76 ГБ из 96 ГБ VRAM) — полная распаковка всех 35B весов в `bfloat16` (доступно только на картах с 80–96 ГБ памяти!).
  - `W8A16_SAFE` (~38 ГБ VRAM) — хранение весов в FP8 с распаковкой активных экспертов на лету в BF16.
- **🖥️ Официальный интерфейс Fish Speech Awesome WebUI**
  - Автоматическая сборка современного фронтенда `awesome_webui` и привязка его как к `/ui`, так и к корневому адресу `/`.
- **💾 Полное сохранение данных на Google Drive**
  - Веса модели (`~38 ГБ`) кэшируются в `MyDrive/FishSpeech_Models/drama-3` (скачиваются только при первом запуске).
  - Все загруженные вами образцы голосов сохраняются в `MyDrive/FishSpeech_References` и доступны при следующих запусках.
- **🚀 3 быстрые ссылки для входа (без тормозов Cloudflare в РФ)**
  - Одновременно поднимаются **Google Colab Direct Proxy**, **Pinggy Tunnel** и **Cloudflare Tunnel (HTTP/2)**.

---

## 🚀 Быстрый старт

1. Нажмите на бейдж **[Open In Colab]** вверху страницы.
2. В меню Colab выберите **«Среда выполнения» (Runtime) → «Сменить среду выполнения» (Change runtime type)** и укажите **«G4 GPU»** (96 ГБ VRAM).
3. Запустите ячейки сверху вниз (`Shift + Enter` или «Выполнить все»).
4. В выводе **Step 4** откройте ссылку **№1 (Google Colab Proxy)** или **№2 (Pinggy Tunnel)** — откроется веб-интерфейс **Fish Speech Awesome WebUI**.

---

## 🎭 Возможности модели Fish Audio Drama 3

### 1. Управление эмоциями и паралингвистикой в тексте
Вы можете вставлять теги эмоций и звуковых реакций прямо внутрь текста (на русском, английском и ещё ~80 языках):
```text
[sighs] Ну ладно... Если мы опоздаем, виноват будешь ты. [laughs] Хотя, пожалуй, это справедливо. [whisper] Только никому ни слова.
```
Поддерживаемые теги: `[sighs]`, `[laughs]`, `[whisper]`, `[angry]`, `[crying]`, `[excited]`, `[pause]` и др.

### 2. Многоголосая озвучка сцен (до 16 персонажей)
Используйте теги `<|speaker:0|>` … `<|speaker:15|>` для озвучки целого диалога за один проход:
```text
<|speaker:0|> Мы наконец-то запустили Drama 3 на G4 96GB!
<|speaker:1|> [laughs] И с полным набором CUDA-графов!
```
Каждому спикеру (`0..15`) можно задать собственный референсный аудиофайл для клонирования тембра.

### 3. Клонирование голоса (Zero-Shot Voice Cloning)
Загрузите короткий чистый аудиофайл (5–30 секунд) и его текстовую расшифровку в разделе **Reference Audio** в WebUI (или передайте через `client.py`). Модель точно скопирует тембр и сможет говорить этим голосом на любом языке с сохранением живых эмоций.

---

## 🤖 Вызов по API из Python или ИИ-агента ([`client.py`](client.py))

Вы можете управлять синтезом речи программно через скрипт [`client.py`](client.py), передав ссылку туннеля из **Step 4**:

```python
from client import FishSpeechColabClient

client = FishSpeechColabClient(base_url="https://xxxx.a.free.pinggy.link")

# Одиночная реплика или многоголосый диалог с эмоциями
client.generate_speech(
    text="<|speaker:0|> [sighs] Какой долгий день... <|speaker:1|> [excited] Зато смотри, какая озвучка получилась!",
    output_path="dialogue_ru.wav"
)

# Клонирование голоса по образцу
client.generate_speech(
    text="[laughs] Привет! Теперь я говорю твоим голосом через Fish Audio Drama 3!",
    references=[{
        "audio_path": "my_voice_sample.wav",
        "text": "Текст, который произносится в образце голоса.",
        "speaker": 0
    }],
    output_path="cloned_voice.wav"
)
```

---

## 📜 Лицензия
- Код `fish-speech` и веса `fishaudio/drama-3` распространяются на условиях [Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE).
