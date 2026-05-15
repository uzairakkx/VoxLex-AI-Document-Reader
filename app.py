import os
import sys
import atexit
import torch
import numpy as np
import re
import threading
import warnings
import base64
import io
import json
import wave
import gc
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from flask import Flask, request, Response, send_from_directory, stream_with_context
from flask_cors import CORS
from transformers import VitsModel, AutoTokenizer
from num2words import num2words
import ollama

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# torch._dynamo.disable() is kept; torch.compile is removed entirely.
# VITS models use in-place index writes inside rational quadratic spline layers
# that are incompatible with TorchDynamo FakeTensor tracing. Calling
# torch.compile() after torch._dynamo.disable() re-enables dynamo on the model
# and causes: "tensors used as indices must be long, int, byte or bool tensors".
# Eager mode on CUDA with inference_mode() provides sufficient throughput.
torch._dynamo.disable()

# ─── ENVIRONMENT AUTO-DETECT ──────────────────────────────────────────────────
RUNPOD = os.environ.get("RUNPOD_POD_ID") is not None
CUDA   = torch.cuda.is_available()

# Set HF_HOME early so RunPod persists model cache across pod restarts
if RUNPOD:
    os.environ["HF_HOME"] = "/workspace/hf_cache"

if RUNPOD and CUDA:
    MODE         = "runpod_gpu"
    OLLAMA_MODEL = "gemma4:e4b-it-q4_K_M"
    OFFLINE_MODE = False   # RunPod: download on first run, cache after
    print("🌍 Environment: RunPod (GPU)")
elif CUDA:
    MODE         = "local_gpu"
    OLLAMA_MODEL = "gemma4:e2b"
    OFFLINE_MODE = True    # Local GPU: models pre-downloaded, fully offline
    print("🌍 Environment: Local (GPU)")
else:
    MODE         = "local_cpu"
    OLLAMA_MODEL = "gemma4:e2b"
    OFFLINE_MODE = True    # Local CPU: fully offline
    print("🌍 Environment: Local (CPU — fully offline)")

# Apply HuggingFace offline env vars only on local — cloud still needs to download
if OFFLINE_MODE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    print("🔒 Offline mode: HuggingFace network calls disabled")
else:
    print("🌐 Online mode: models will download if not cached")

print(f"🤖 Ollama model  : {OLLAMA_MODEL}")
print(f"⚙️  Run mode      : {MODE}")
# ──────────────────────────────────────────────────────────────────────────────

from allprompts import LANG_CONFIG

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── DEVICE ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if CUDA else "cpu")
print(f"🖥️  TTS Device: {DEVICE}")

# ─── TTS MODEL REGISTRY ───────────────────────────────────────────────────────
tts_models      = {}
tts_tokenizers  = {}
tts_samplerates = {}

current_tts_model_name = None
tts_model     = None
tts_tokenizer = None
SAMPLE_RATE   = None

# Single lock for ALL TTS operations — both model loading and synthesis.
# synthesize_sentence() acquires this lock so it can never read globals
# while a CPU eviction+load is in progress.
tts_lock = threading.Lock()

# Cleanly shut down thread pool on process exit — prevents thread leaks
# under gunicorn worker recycling on RunPod.
TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1 if not CUDA else 2)
atexit.register(TTS_EXECUTOR.shutdown, wait=False)

# 180s — RunPod logs confirm Gemma needs ~136s to load on a cold pod.
# The startup warmup pays this cost once; subsequent requests are fast.
# 180s gives a safe margin for the first real request after any pod restart.
OLLAMA_TIMEOUT = 180

# Cached health probe — avoids blocking a Flask worker for the full Ollama
# cold-start time on every /health poll.
_ollama_health_cache     = {"status": "unknown", "checked_at": 0}
_OLLAMA_HEALTH_CACHE_TTL = 10   # seconds

if CUDA:
    torch.backends.mkldnn.enabled = True
else:
    cpu_threads = os.cpu_count() or 4
    torch.set_num_threads(cpu_threads)
    print(f"🧵 PyTorch CPU threads: {cpu_threads}")


# ─── WARMUP HELPER ────────────────────────────────────────────────────────────
def _warmup_model(model: VitsModel, tokenizer: AutoTokenizer, model_name: str):
    """
    Run one silent forward pass to JIT-compile kernels before the first
    real request.

    ROOT CAUSE OF WARNING:
    AutoTokenizer returns input_ids as torch.int64 (Long) on CPU, but after
    .to(DEVICE) some tokenizer implementations produce float32 tensors on
    certain CUDA versions when the tensor originates from a numpy int32 array.
    VITS embedding lookup requires Long/Int indices; Float indices crash with:
      "Expected tensor for argument #1 'indices' to have scalar types
       Long, Int; but got torch.cuda.FloatTensor"

    FIX: explicitly cast input_ids, attention_mask, and speaker_id to
    torch.long AFTER moving to device. Safe because the embedding layer
    always needs Long regardless of what the tokenizer produces.
    """
    dummy_text = "آزمائش" if "urd" in model_name else "test"
    try:
        dummy = tokenizer(dummy_text, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        # Guarantee integer index tensors are always Long — fixes the
        # FloatTensor index error on Hindi, Arabic, Bengali, Farsi, Tamil.
        for key in ("input_ids", "attention_mask", "speaker_id"):
            if key in dummy:
                dummy[key] = dummy[key].long()
        with torch.inference_mode():
            _ = model(**dummy).waveform
        print(f"  ✅ Warmup OK")
    except Exception as e:
        print(f"  ⚠️  Warmup skipped: {e}")


def load_tts_model(model_name: str):
    """
    Load a VITS TTS model and cache it.

    Thread safety:
    - GPU: all 10 models cached in VRAM at startup; switching is an instant
      pointer swap inside tts_lock.
    - CPU: only one model resident at a time; eviction+load is serialised by
      tts_lock so synthesize_sentence() never reads a None model mid-swap.
    """
    global tts_model, tts_tokenizer, SAMPLE_RATE, current_tts_model_name
    global tts_models, tts_tokenizers, tts_samplerates

    # Entire function inside tts_lock — makes check-then-act atomic and
    # prevents synthesize_sentence() reading globals during a CPU model swap.
    with tts_lock:
        if current_tts_model_name == model_name:
            return

        # GPU fast-path: model already in VRAM cache
        if CUDA and model_name in tts_models:
            tts_model              = tts_models[model_name]
            tts_tokenizer          = tts_tokenizers[model_name]
            SAMPLE_RATE            = tts_samplerates[model_name]
            current_tts_model_name = model_name
            print(f"⚡ TTS switched to (cached): {model_name}")
            return

        # CPU path: evict current model to free RAM before loading the next one
        if not CUDA and current_tts_model_name is not None:
            print(f"🗑️  Evicting TTS model from RAM: {current_tts_model_name}")
            old = current_tts_model_name
            if old in tts_models:
                del tts_models[old]
                del tts_tokenizers[old]
                del tts_samplerates[old]
            # Set to None BEFORE gc.collect() — any concurrent read of globals
            # in synthesize_sentence() gets None and raises a clear error rather
            # than accessing a freed object.
            tts_model     = None
            tts_tokenizer = None
            SAMPLE_RATE   = None
            gc.collect()

        print(f"⏳ Loading TTS model: {model_name} on {DEVICE} ...")

        # local_files_only=True on local machines — guarantees air-gapped inference.
        # On RunPod (OFFLINE_MODE=False) models download on first use and are
        # cached at /workspace/hf_cache for subsequent pod restarts.
        m = VitsModel.from_pretrained(model_name, local_files_only=OFFLINE_MODE)
        t = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE_MODE)
        m = m.to(DEVICE)
        m.eval()

        # NOTE: torch.compile intentionally absent — see top-of-file comment.

        try:
            m.length_scale = 0.85
        except Exception:
            pass

        # Warmup via dedicated helper — fixes the FloatTensor index warning
        _warmup_model(m, t, model_name)

        # Cache on GPU (all models fit in VRAM); on CPU keep only active model
        if CUDA:
            tts_models[model_name]      = m
            tts_tokenizers[model_name]  = t
            tts_samplerates[model_name] = m.config.sampling_rate

        tts_model              = m
        tts_tokenizer          = t
        SAMPLE_RATE            = m.config.sampling_rate
        current_tts_model_name = model_name

        # Guard: surface corrupted/incomplete model config clearly rather than
        # crashing silently in the thread pool with a TypeError on None.
        if SAMPLE_RATE is None:
            raise RuntimeError(
                f"Model {model_name} loaded but SAMPLE_RATE is None — "
                "model config may be corrupted or incomplete."
            )

        print(f"✅ TTS ready: {model_name} (sr={SAMPLE_RATE})")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
TTS_TO_NUM2WORDS = {
    "facebook/mms-tts-urd-script_arabic": "ur",
    "facebook/mms-tts-hin": "hi",
    "facebook/mms-tts-ara": "ar",
    "facebook/mms-tts-ben": "bn",
    "facebook/mms-tts-fas": "fa",
    "facebook/mms-tts-ind": "id",
    # Tamil, Swahili, Tagalog have no num2words support — intentional English
    # fallback. Do NOT change these to their ISO codes.
    "facebook/mms-tts-swh": "en",
    "facebook/mms-tts-tam": "en",
    "facebook/mms-tts-tgl": "en",
    "facebook/mms-tts-tur": "tr",
}

RTL_MODELS = {
    "facebook/mms-tts-urd-script_arabic",
    "facebook/mms-tts-ara",
    "facebook/mms-tts-fas",
}

MONTH_NAMES = {
    "ur": ['جنوری','فروری','مارچ','اپریل','مئی','جون','جولائی','اگست','ستمبر','اکتوبر','نومبر','دسمبر'],
    "ar": ['يناير','فبراير','مارس','أبريل','مايو','يونيو','يوليو','أغسطس','سبتمبر','أكتوبر','نوفمبر','ديسمبر'],
    "fa": ['ژانویه','فوریه','مارس','آوریل','مه','ژوئن','ژوئیه','اوت','سپتامبر','اکتبر','نوامبر','دسامبر'],
    "hi": ['जनवरी','फरवरी','मार्च','अप्रैल','मई','जून','जुलाई','अगस्त','सितंबर','अक्टूबर','नवंबर','दिसंबर'],
    "bn": ['জানুয়ারি','ফেব্রুয়ারি','মার্চ','এপ্রিল','মে','জুন','জুলাই','আগস্ট','সেপ্টেম্বর','অক্টোবর','নভেম্বর','ডিসেম্বর'],
    "tr": ['Ocak','Şubat','Mart','Nisan','Mayıs','Haziran','Temmuz','Ağustos','Eylül','Ekim','Kasım','Aralık'],
    "id": ['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember'],
    "en": ['January','February','March','April','May','June','July','August','September','October','November','December'],
}

CURRENCY_WORD = {
    "ur": "روپے", "ar": "ريال", "fa": "تومان",
    "hi": "रुपये", "bn": "টাকা", "tr": "lira",
    "id": "rupiah", "en": "",
}


def urdu_number(n: int) -> str:
    ones = [
        '', 'ایک', 'دو', 'تین', 'چار', 'پانچ', 'چھ', 'سات', 'آٹھ', 'نو',
        'دس', 'گیارہ', 'بارہ', 'تیرہ', 'چودہ', 'پندرہ', 'سولہ', 'سترہ',
        'اٹھارہ', 'انیس', 'بیس', 'اکیس', 'بائیس', 'تئیس', 'چوبیس', 'پچیس',
        'چھبیس', 'ستائیس', 'اٹھائیس', 'انتیس', 'تیس', 'اکتیس', 'بتیس',
        'تینتیس', 'چونتیس', 'پینتیس', 'چھتیس', 'سینتیس', 'اڑتیس', 'انتالیس',
        'چالیس', 'اکتالیس', 'بیالیس', 'تینتالیس', 'چوالیس', 'پینتالیس',
        'چھیالیس', 'سینتالیس', 'اڑتالیس', 'انچاس', 'پچاس', 'اکاون', 'باون',
        'ترپن', 'چون', 'پچپن', 'چھپن', 'ستاون', 'اٹھاون', 'انسٹھ', 'ساٹھ',
        'اکسٹھ', 'باسٹھ', 'تریسٹھ', 'چوسٹھ', 'پینسٹھ', 'چھیاسٹھ', 'سڑسٹھ',
        'اڑسٹھ', 'انہتر', 'ستر', 'اکہتر', 'بہتر', 'تہتر', 'چوہتر', 'پچھتر',
        'چھہتر', 'ستتر', 'اٹھتر', 'اناسی', 'اسی', 'اکیاسی', 'بیاسی', 'تراسی',
        'چوراسی', 'پچاسی', 'چھیاسی', 'ستاسی', 'اٹھاسی', 'نواسی', 'نوے',
        'اکانوے', 'بانوے', 'ترانوے', 'چورانوے', 'پچانوے', 'چھیانوے',
        'ستانوے', 'اٹھانوے', 'ننانوے'
    ]
    if n == 0: return 'صفر'
    elif n <= 99: return ones[n]
    elif n < 1000:
        r = n % 100
        return ones[n // 100] + ' سو' + (' ' + urdu_number(r) if r else '')
    elif n < 100000:
        r = n % 1000
        return urdu_number(n // 1000) + ' ہزار' + (' ' + urdu_number(r) if r else '')
    elif n < 10000000:
        r = n % 100000
        return urdu_number(n // 100000) + ' لاکھ' + (' ' + urdu_number(r) if r else '')
    else:
        r = n % 10000000
        return urdu_number(n // 10000000) + ' کروڑ' + (' ' + urdu_number(r) if r else '')


def convert_numbers(text: str, model_name: str) -> str:
    lang     = TTS_TO_NUM2WORDS.get(model_name, "en")
    months   = MONTH_NAMES.get(lang, MONTH_NAMES["en"])
    currency = CURRENCY_WORD.get(lang, "")

    def to_words(n):
        if lang == "ur": return urdu_number(n)
        try: return num2words(n, lang=lang)
        except: return num2words(n, lang="en")

    def convert_date(m):
        try:
            parts = re.split(r'[/\-]', m.group())
            d, mo, y = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{to_words(d)} {months[mo - 1] if 1 <= mo <= 12 else mo} {to_words(y)}"
        except: return m.group()

    def convert_currency(m):
        try:
            n = int(float(m.group(1).replace(',', '')))
            return currency + ' ' + to_words(n)
        except: return m.group()

    def convert_plain(m):
        try: return to_words(int(m.group().replace(',', '')))
        except: return m.group()

    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', convert_date, text)
    text = re.sub(r'(?:Rs\.?|PKR)\s*(\d[\d,]*)', convert_currency, text)
    text = re.sub(r'\b\d[\d,]*\b', convert_plain, text)
    return text


def clean_text(text: str, model_name: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'[#\_\:]', '', text)
    text = re.sub(r'•', '', text)
    text = re.sub(r'[<>{}|\[\]\\]', '', text)
    text = re.sub(r'\([a-zA-Z][a-zA-Z\s]*\)', '', text)
    text = re.sub(r'\(\s*\)', '', text)

    # Strip ALL quote variants — unknown token IDs from quotes cause the VITS
    # embedding layer to receive FloatTensor indices instead of Long/Int,
    # producing a hard crash in synthesize_sentence().
    text = re.sub(r'["""\'`\u2018\u2019\u00ab\u00bb\u201e]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    text = convert_numbers(text, model_name)
    return text


def split_sentences(text: str):
    """
    Split on strong sentence boundaries (۔ . ! ? newlines).
    Returns (complete_sentences, remainder).
    Chunks shorter than 20 chars are carried into the next one to avoid
    micro-clips that sound unnatural from TTS.
    """
    pattern = re.compile(r'([^.!?؟۔\n]+[.!?؟۔\n]+)', re.UNICODE)
    matches = pattern.findall(text)

    sentences = []
    pending   = ""

    for m in matches:
        chunk   = (pending + " " + m).strip()
        pending = ""
        if len(chunk) < 20:
            pending = chunk
        else:
            sentences.append(chunk)

    if pending:
        if sentences:
            sentences[-1] = (sentences[-1] + " " + pending).strip()
        else:
            pending = ""   # too short with no prior sentence — falls to remainder

    last_match_end = 0
    for m in re.finditer(r'[^.!?؟۔\n]+[.!?؟۔\n]+', text):
        last_match_end = m.end()
    remainder = text[last_match_end:].strip()

    return sentences, remainder


def merge_short_sentences(sentences: list, model_name: str = "") -> list:
    """
    Merge sentences that are too short for natural-sounding TTS output.
    RTL (Urdu/Arabic/Persian): merge until >= 80 characters.
    LTR: merge until >= 10 words.
    """
    is_rtl = model_name in RTL_MODELS

    def is_long_enough(t: str) -> bool:
        return len(t) >= 80 if is_rtl else len(t.split()) >= 10

    merged  = []
    pending = ""

    for sentence in sentences:
        pending = (pending + " " + sentence).strip() if pending else sentence
        if is_long_enough(pending):
            merged.append(pending)
            pending = ""

    if pending:
        if merged:
            merged[-1] = (merged[-1] + " " + pending).strip()
        else:
            merged.append(pending)

    return merged


def should_flush(text: str, model_name: str = "") -> bool:
    """
    Flush the accumulation buffer mid-stream only when long enough that
    waiting for the next sentence-end would cause noticeable latency.
    RTL: 120 chars (~2 Urdu sentences). LTR: 20 words.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if model_name in RTL_MODELS:
        return len(stripped) >= 120
    return len(stripped.split()) >= 20


def add_silence(audio: np.ndarray, sample_rate: int, ms: int = 80) -> np.ndarray:
    silence = np.zeros(int(sample_rate * ms / 1000), dtype=np.float32)
    return np.concatenate([audio, silence])


def synthesize_sentence(sentence: str) -> bytes:
    """
    Synthesise one sentence to WAV bytes.

    Snapshots tts_model/tts_tokenizer/SAMPLE_RATE under tts_lock so
    this function is mutually exclusive with load_tts_model(). On CPU a model
    swap (evict + load) cannot race with an active synthesis.

    FIX: input_ids, attention_mask, and speaker_id explicitly cast to
    torch.long after moving to device — same fix as _warmup_model — prevents
    the FloatTensor embedding index error during real synthesis on all models.
    """
    with tts_lock:
        model     = tts_model
        tokenizer = tts_tokenizer
        sr        = SAMPLE_RATE

    if model is None or tokenizer is None or sr is None:
        raise RuntimeError("TTS model not loaded — cannot synthesize.")

    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Cast integer index tensors to Long — prevents FloatTensor index crash
    # in VITS embedding lookup on certain CUDA/tokenizer combinations.
    for key in ("input_ids", "attention_mask", "speaker_id"):
        if key in inputs:
            inputs[key] = inputs[key].long()

    with torch.inference_mode():
        try:
            model.length_scale = 0.85
        except Exception:
            pass
        output = model(**inputs).waveform

    audio = output.squeeze().cpu().numpy().astype(np.float32)

    # Single normalisation pass after silence is appended — avoids the old
    # double-normalisation where the silence padding slightly altered peak level.
    if len(audio) > 1:
        audio = np.diff(audio, prepend=audio[0]) * 0.05 + audio * 0.95

    audio = add_silence(audio, sr, ms=80)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def flush_sentence(text: str, tts_model_name: str):
    """
    Clean, synthesise, and stream one TTS chunk.
    Yields NDJSON: text event immediately, audio event when synthesis completes.
    """
    cleaned = clean_text(text, tts_model_name)
    if not cleaned.strip():
        return

    print(f"[TTS] Synthesizing ({MODE}): '{cleaned[:70]}'")
    yield json.dumps({"type": "text", "content": cleaned}) + "\n"

    future = TTS_EXECUTOR.submit(synthesize_sentence, cleaned)

    try:
        wav_bytes = future.result(timeout=30)
        wav_b64   = base64.b64encode(wav_bytes).decode()
        print(f"[TTS] OK — {len(wav_bytes)} bytes")
        yield json.dumps({"type": "audio", "content": wav_b64}) + "\n"
    except FuturesTimeoutError:
        print(f"[TTS] Timeout — skipping audio for this sentence")
        yield json.dumps({"type": "tts_error", "content": "TTS timeout"}) + "\n"
    except Exception as e:
        print(f"[TTS] Error: {type(e).__name__}: {e}")
        yield json.dumps({"type": "tts_error", "content": str(e)}) + "\n"


def ollama_generate_with_timeout(model, prompt, images, timeout=OLLAMA_TIMEOUT):
    """
    Wrap ollama.generate in a daemon thread so the Flask worker is never
    blocked longer than `timeout` seconds.
    """
    result_holder = [None]
    error_holder  = [None]
    ready_event   = threading.Event()

    def _run():
        try:
            result_holder[0] = ollama.generate(
                model=model,
                prompt=prompt,
                images=images,
                stream=True
            )
        except Exception as e:
            error_holder[0] = e
        finally:
            ready_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not ready_event.wait(timeout=timeout):
        raise TimeoutError(f"Ollama did not respond within {timeout}s")
    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


MAX_IMAGE_BYTES   = 10 * 1024 * 1024
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/languages')
def get_languages():
    langs = list(LANG_CONFIG.keys())
    return json.dumps(langs), 200, {'Content-Type': 'application/json'}


@app.route('/health')
def health():
    """
    Non-blocking health endpoint.
    Ollama probed via a background thread with 10s TTL cache — never blocks
    a Flask worker for the full Ollama load time.
    """
    now = time.time()
    if now - _ollama_health_cache["checked_at"] > _OLLAMA_HEALTH_CACHE_TTL:
        def _probe():
            try:
                ollama.generate(model=OLLAMA_MODEL, prompt="hi", stream=False)
                _ollama_health_cache["status"]     = "ok"
            except Exception as e:
                _ollama_health_cache["status"]     = f"error: {e}"
            _ollama_health_cache["checked_at"] = time.time()
        threading.Thread(target=_probe, daemon=True).start()

    status = {
        "mode":             MODE,
        "offline_mode":     OFFLINE_MODE,
        "ollama_model":     OLLAMA_MODEL,
        "tts_device":       str(DEVICE),
        "tts_model_loaded": current_tts_model_name or "none",
        "cuda_available":   CUDA,
        "ollama_status":    _ollama_health_cache["status"],
    }
    return json.dumps(status), 200, {'Content-Type': 'application/json'}


@app.route('/analyze', methods=['POST'])
def analyze():
    req_lang   = request.form.get('language', '')
    image_file = request.files.get('image')

    if not req_lang or not image_file:
        return json.dumps({"error": "missing language or image"}), 400

    content_type = image_file.content_type or ""
    if content_type not in ALLOWED_MIMETYPES:
        return json.dumps({
            "error": f"Unsupported file type '{content_type}'. "
                     f"Allowed: {', '.join(ALLOWED_MIMETYPES)}"
        }), 415

    image_bytes = image_file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return json.dumps({
            "error": f"Image too large ({len(image_bytes) // 1024}KB). "
                     f"Maximum allowed is {MAX_IMAGE_BYTES // 1024 // 1024}MB."
        }), 413

    lang_cfg = LANG_CONFIG.get(req_lang)
    if not lang_cfg:
        return json.dumps({"error": "unknown language"}), 400

    prompt         = lang_cfg["prompt"]
    tts_model_name = lang_cfg["tts_model"]

    # Validate prompt — empty prompt produces garbled Ollama output that passes
    # all downstream checks silently.
    if not prompt or not prompt.strip():
        return json.dumps({"error": f"Empty prompt for language '{req_lang}'"}), 500

    # Catch model-load failures and return a clean HTTP error rather than
    # letting them appear mid-stream with no status code.
    try:
        load_tts_model(tts_model_name)
    except Exception as e:
        return json.dumps({"error": f"TTS model failed to load: {e}"}), 500

    # Encode once; free raw bytes immediately — up to 10 MB otherwise held
    # for the full streaming duration.
    image_b64 = base64.b64encode(image_bytes).decode()
    del image_bytes

    def generate():
        buffer          = ""
        last_chunk_time = time.time()   # FIX: mid-stream stall guard

        try:
            try:
                response = ollama_generate_with_timeout(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    images=[image_b64],
                    timeout=OLLAMA_TIMEOUT
                )
            except TimeoutError as te:
                yield json.dumps({"type": "error", "content": str(te)}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return

            for chunk in response:
                # FIX: detect mid-stream stall (no token for 60 s) and bail
                # cleanly rather than hanging the Flask worker indefinitely.
                if time.time() - last_chunk_time > 60:
                    print("[ERROR] Ollama stream stalled — no token for 60s")
                    yield json.dumps({
                        "type":    "error",
                        "content": "Stream stalled — no response from model for 60s"
                    }) + "\n"
                    break
                last_chunk_time = time.time()

                text_chunk = chunk.get("response", "")
                if not text_chunk:
                    continue

                buffer += text_chunk

                # Stream token to UI immediately so text appears before audio
                yield json.dumps({"type": "token", "content": text_chunk}) + "\n"

                sentences, buffer = split_sentences(buffer)
                sentences = merge_short_sentences(sentences, model_name=tts_model_name)

                for sentence in sentences:
                    if sentence.strip():
                        yield from flush_sentence(sentence, tts_model_name)

                # Flush only the REMAINDER when it grows long enough — already-split
                # sentences are gone so there is no risk of speaking text twice.
                if should_flush(buffer, tts_model_name):
                    yield from flush_sentence(buffer, tts_model_name)
                    buffer = ""

            # Flush whatever is left after the LLM stream ends
            if buffer.strip():
                yield from flush_sentence(buffer.strip(), tts_model_name)

        except Exception as e:
            print(f"[ERROR] {e}")
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# ─── STARTUP ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 VoxLex — Document Reader Server")
    print("━" * 40)
    print(f"📦 Mode        : {MODE}")
    print(f"🤖 Model       : {OLLAMA_MODEL}")
    print(f"🖥️  Device      : {DEVICE}")
    print(f"🔒 Offline mode: {OFFLINE_MODE}")
    print("━" * 40)

    # FIX: warn loudly on RunPod that gunicorn should be used for production
    if RUNPOD:
        print("\n⚠️  RUNPOD DETECTED — for production use gunicorn via start.sh:")
        print("   gunicorn -w 1 --timeout 300 -b 0.0.0.0:5000 app:app\n")

    print("\n⏳ Pre-warming Gemma via Ollama...")
    try:
        # FIX: verify the model is actually pulled before attempting warmup
        available_models = [m["name"] for m in (ollama.list().get("models") or [])]
        if OLLAMA_MODEL not in available_models:
            print(f"⚠️  Model '{OLLAMA_MODEL}' not found in Ollama!")
            print(f"   Run: ollama pull {OLLAMA_MODEL}")
            print(f"   Available models: {available_models or 'none'}")
            _ollama_health_cache["status"]     = f"model not found: {OLLAMA_MODEL}"
            _ollama_health_cache["checked_at"] = time.time()
        else:
            ollama.generate(model=OLLAMA_MODEL, prompt="hi", stream=False)
            print("✅ Gemma ready")
            _ollama_health_cache["status"]     = "ok"
            _ollama_health_cache["checked_at"] = time.time()
    except Exception as e:
        print(f"⚠️  Gemma warmup failed (is Ollama running?): {e}")
        _ollama_health_cache["status"]     = f"error: {e}"
        _ollama_health_cache["checked_at"] = time.time()

    if CUDA:
        print(f"\n⏳ Pre-loading all {len(LANG_CONFIG)} TTS models into GPU VRAM...")
        for _sl, _sc in LANG_CONFIG.items():
            try:
                print(f"  • {_sl}...")
                load_tts_model(_sc["tts_model"])
            except Exception as e:
                print(f"  ⚠️  {_sl} failed: {e}")
        print("✅ All TTS models loaded into GPU!\n")
    else:
        _fl  = next(iter(LANG_CONFIG))
        _fm  = LANG_CONFIG[_fl]["tts_model"]
        print(f"\n⏳ CPU mode — pre-loading default TTS model ({_fl})...")
        try:
            load_tts_model(_fm)
            print(f"✅ Default TTS model ready. Others load on first use.\n")
        except Exception as e:
            print(f"⚠️  Default TTS load failed: {e}\n")

    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print("📱 Open on your phone  :", f"http://{local_ip}:5000")
    print("💻 Or on this computer :", "http://localhost:5000")
    if RUNPOD:
        pod_id = os.environ.get("RUNPOD_POD_ID", "YOUR_POD_ID")
        print(f"🌐 RunPod public URL   : https://{pod_id}-5000.proxy.runpod.net")
    print("─" * 40)
    print("⚠️  Dev server active. On RunPod, use start.sh with gunicorn.")
    print("━" * 40 + "\n")

    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
