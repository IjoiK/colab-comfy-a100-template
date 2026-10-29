# 🚀 Коллекция оптимизированных ноутбуков для Google Colab (A100 GPU)

Сборник готовых шаблонов «всё в одном» для запуска тяжёлых генеративных моделей (видео, звук, речь) на видеокартах **NVIDIA A100 (40 ГБ / 80 ГБ VRAM)** в Google Colab с использованием пакета **200 Compute Units (CU)** из подписки Google AI Pro.

Все ноутбуки в репозитории оптимизированы по видеопамяти, автоматически кэшируют многогигабайтные веса моделей на **Google Drive** (чтобы не скачивать их повторно), поднимают сразу **3 быстрых туннеля в обход замедлений Cloudflare** и содержат готовые Python-клиенты и инструкции (`SKILL.md`) для управления через ИИ-агентов.

---

## 📚 Каталог ноутбуков

| Направление | Модель и интерфейс | Оптимизации под A100 | Документация | Запуск в 1 клик |
| :--- | :--- | :--- | :--- | :--- |
| 🎬 **Генерация видео со стереозвуком** | **[MiniMax-H3 (Hailuo)](https://huggingface.co/Comfy-Org/MiniMax-H3)** + **ComfyUI** | `int8_convrot` + `NVFP4 Qwen3-VL 32B` + **8-step Turbo LoRA** (рендер в 4 раза быстрее) + `SageAttention` | [📖 Открыть инструкцию](minimax-h3-comfyui/README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IjoiK/colab-comfy-a100-template/blob/main/minimax-h3-comfyui/colab_comfy_minimax_h3_a100.ipynb) |
| 🎙️ **Многоголосая озвучка и клонирование голоса** | **[Fish Audio Drama 3 (35B-A3B)](https://huggingface.co/fishaudio/drama-3)** + **[Fish Speech WebUI](https://github.com/fishaudio/fish-speech)** | **W8A16 FP8-MoE патч для A100 (SM80)** — снижает расход VRAM с ~72 ГБ до **~36 ГБ** (работает и на A100 40GB, и на 80GB!) | [📖 Открыть инструкцию](fish-audio-drama-3/README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IjoiK/colab-comfy-a100-template/blob/main/fish-audio-drama-3/colab_fish_drama3_a100.ipynb) |

---

## 🗂️ Структура репозитория

```text
├── README.md                                  # Главный каталог ноутбуков и общая справка по Colab A100
├── minimax-h3-comfyui/                        # 🎬 Видео + Аудио генерация (MiniMax-H3 + ComfyUI)
│   ├── README.md                              # Инструкция по воркфлоу MiniMax-H3 и Turbo LoRA
│   ├── colab_comfy_minimax_h3_a100.ipynb      # Ноутбук для Google Colab A100
│   ├── client.py                              # Python API-клиент для удалённого рендера и скачивания MP4
│   ├── submit_job.py                          # Пример скрипта со спецификацией промпта MiniMax-H3
│   └── generate_kenmou_video.py               # Пример генерации длинного 15-секундного видео
├── fish-audio-drama-3/                        # 🎙️ Озвучка сцен и клонирование голоса (Fish Audio Drama 3)
│   ├── README.md                              # Инструкция по тегам эмоций, спикерам и WebUI
│   ├── colab_fish_drama3_a100.ipynb           # Ноутбук для Google Colab A100 (с W8A16 FP8 патчем)
│   └── client.py                              # Python API-клиент для синтеза речи и клонирования голоса
└── .agent/skills/                             # 🤖 Навыки (Skills) для автоматизации через ИИ-агентов
    ├── colab-comfy-agent/SKILL.md             # Навык управления видеогенерацией MiniMax-H3
    └── colab-fish-drama3-agent/SKILL.md       # Навык управления озвучкой Fish Audio Drama 3
```

---

## 🌐 3 быстрых способа подключения к WebUI в каждом ноутбуке

Поскольку стандартный бесплатный туннель `trycloudflare.com` часто режется по скорости провайдерами в РФ (ТСПУ) и внутри контейнеров Colab (из-за UDP QUIC), в **Step 4** каждого ноутбука одновременно поднимаются **3 ссылки на выбор**:

1. 🚀 **Google Colab Direct Proxy (`*.googleusercontent.com`)** — встроенный прокси самого Google Colab. Трафик идёт напрямую через серверы Google на максимальной скорости (открывайте в том же браузере, где запущен Colab).
2. ⚡ **Pinggy Tunnel (`https://xxxx.a.free.pinggy.link`)** — быстрый SSH/TCP-туннель без участия Cloudflare. Мгновенно открывается с ПК без VPN и с мобильных устройств.
3. 🌐 **Cloudflare Tunnel в режиме `--protocol http2` (`https://xxxx.trycloudflare.com`)** — классический туннель, переведённый с проблемного UDP QUIC на стабильный TCP HTTP/2.

---

## ⏱️ Памятка по расходу Compute Units (CU) на A100

- **Стандартный A100 (~5.37 CU/час)**: 200 CU из подписки Google AI Pro хватает примерно на **37 часов** чистой работы в месяц.
- **High-RAM / A100 80GB (~7.52 CU/час)**: хватает примерно на **26 часов** работы в месяц.

### 🛑 Как со 100% гарантией остановить списание CU после завершения работы
Когда закончите генерацию, не просто останавливайте ячейку, а удалите виртуальную машину через меню:
1. В верхнем меню Colab нажмите **«Среда выполнения» (Runtime)**.
2. Выберите **«Отключиться и удалить среду выполнения» (Disconnect and delete runtime)**.

> Все скачанные модели и результаты вашей работы сохраняются на вашем **Google Диске** (`MyDrive/ComfyUI_*` и `MyDrive/FishSpeech_*`), поэтому удалять среду выполнения полностью безопасно.
