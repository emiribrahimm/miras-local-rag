"""Tüm ayarlar tek yerde. `miras.toml` dosyasıyla ve ortam değişkenleriyle (MIRAS_*) ezilebilir.

Öncelik: komut satırı > MIRAS_* > miras.toml > bu dosyadaki varsayılan.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MIRAS_<AD> ortam değişkeniyle ve miras.toml'daki <ad> anahtarıyla ezilebilen alanlar; geri
# kalanı yalnızca bu dosyadan değişir.
ENV_FIELDS = ("docs_dir", "db_path", "chat_model", "embed_model", "chat_device", "top_k",
              "min_similarity", "bm25_weight", "followup_min_similarity", "answer_cache",
              "query_log")

PATH_FIELDS = {"docs_dir", "db_path"}

CONFIG_PATH = Path(os.environ.get("MIRAS_CONFIG", str(ROOT / "miras.toml")))


def load_config_file(path: Path) -> dict[str, str]:
    """miras.toml'daki ayarlar, ortam değişkenleriyle aynı biçimde (metin). Dosya yoksa boş.

    Tanınmayan bir anahtar ("top-k", "esik") sessizce yok sayılmasın diye hata verir.
    """
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
        raise ValueError(f"{path} okunamadı: {err}") from None
    unknown = sorted(set(data) - set(ENV_FIELDS))
    if unknown:
        raise ValueError(f"{path}: bilinmeyen ayar {', '.join(unknown)}. "
                         f"Kullanılabilenler: {', '.join(ENV_FIELDS)}")
    values = {name: ("1" if value else "0") if isinstance(value, bool) else str(value)
              for name, value in data.items()}
    # Göreli yollar dosyanın klasörüne göre: komut hangi dizinden çalışırsa çalışsın aynı yer.
    for name in PATH_FIELDS & values.keys():
        values[name] = str((path.parent / values[name]).resolve())
    return values


FILE_VALUES = load_config_file(CONFIG_PATH)


def _source(name: str) -> str:
    """Bir ayarın değeri nereden okunduysa onun adı; hata mesajlarında gösterilir."""
    if f"MIRAS_{name}" in os.environ or name.lower() not in FILE_VALUES:
        return f"MIRAS_{name}"
    return f"{CONFIG_PATH.name} içindeki {name.lower()}"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MIRAS_{name}", FILE_VALUES.get(name.lower(), default))


def _env_number(name: str, default: str, kind: type[int] | type[float]) -> int | float:
    """Sayısal ayar; "abc" gibi değerlerde hangi değişkenin (ya da dosya anahtarının) bozuk olduğunu söyler."""
    value = _env(name, default)
    try:
        return kind(value.replace(",", ".") if kind is float else value)
    except ValueError:
        expected = "tam sayı" if kind is int else "sayı"
        raise ValueError(f"{_source(name)} bir {expected} olmalı, '{value}' verildi.") from None


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path(_env("DOCS_DIR", str(ROOT / "data" / "docs")))
    db_path: Path = Path(_env("DB_PATH", str(ROOT / "data" / "miras.db")))

    # Foundry Local model takma adları (alias).
    chat_model: str = _env("CHAT_MODEL", "qwen2.5-7b")
    embed_model: str = _env("EMBED_MODEL", "qwen3-embedding-0.6b")
    # Sohbet modelinin çalışacağı donanım: "cpu" her makinede çalışır, "gpu" (WebGPU)
    # destekleyen makinelerde çok daha hızlıdır. Embedding modeli her zaman CPU'da çalışır.
    chat_device: str = _env("CHAT_DEVICE", "cpu")

    # Parçalama: hedef ve üst sınır karakter cinsinden.
    chunk_target_chars: int = 900
    chunk_max_chars: int = 1400

    # Arama: kaç parça getirilecek ve en iyi parçanın benzerliği bu eşiğin
    # altındaysa model hiç çağrılmadan "bilmiyorum" cevabı verilir.
    # Eşik elle seçilmedi; `python main.py calibrate` ile ölçülüp yazıldı.
    top_k: int = _env_number("TOP_K", "4", int)
    min_similarity: float = _env_number("MIN_SIMILARITY", "0.53", float)
    # Hibrit arama: BM25 anahtar kelime skorunun (0–1) sıralamadaki ağırlığı. 0 = yalnızca
    # embedding. Kapı eşiği her durumda ham kosinüs skoruna bakar. Değer `python main.py
    # tune-hybrid` ile ölçülmeden açılmamalı; varsayılan bu yüzden kapalı.
    bm25_weight: float = _env_number("BM25_WEIGHT", "0.0", float)
    # Takip soruları: alan adı geçmeyen soru önceki turun konusuyla sorulur, ama sorunun
    # kendisinin en iyi benzerliği bu değerin altındaysa (konu dışı) konu taşınmaz; aksi halde
    # alan adı benzerliği şişirip konu dışı soruyu kapıdan geçirir. 0 = kontrol kapalı.
    # Konu dışı sorular 0.31–0.44 aldı; `python main.py calibrate` ile ölçülmesi önerilir.
    followup_min_similarity: float = _env_number("FOLLOWUP_MIN_SIMILARITY", "0.40", float)

    # Cevap önbelleği: aynı model, ayarlar ve aynı prompt (soru + getirilen parçalar) için ham
    # model çıktısı SQLite'ta saklanır ve tekrar sorulduğunda model çağrılmaz. "0" = kapalı.
    answer_cache: bool = _env("ANSWER_CACHE", "1") != "0"
    # Soru günlüğü: sorulan her soru (cevap metni olmadan) reddetme sebebi ve en iyi benzerlikle
    # aynı veritabanına yazılır; `log` komutu bilgi tabanındaki boşlukları gösterir. "0" = kapalı.
    query_log: bool = _env("QUERY_LOG", "1") != "0"

    temperature: float = 0.0
    max_output_tokens: int = 400
    seed: int = 42

    def __post_init__(self) -> None:
        """Anlamsız değerler (top_k=0, eşik 5) aramayı sessizce bozmasın, baştan reddedilsin."""
        problems = []
        if self.top_k < 1:
            problems.append(f"top_k en az 1 olmalı ({self.top_k})")
        for name in ("min_similarity", "followup_min_similarity"):
            if not 0 <= getattr(self, name) <= 1:
                problems.append(f"{name} 0 ile 1 arasında olmalı ({getattr(self, name)})")
        if self.bm25_weight < 0:
            problems.append(f"bm25_weight negatif olamaz ({self.bm25_weight})")
        if self.chat_device not in ("cpu", "gpu"):
            problems.append(f"chat_device cpu ya da gpu olmalı ({self.chat_device!r})")
        if not 0 < self.chunk_target_chars <= self.chunk_max_chars:
            problems.append("chunk_target_chars 0'dan büyük ve chunk_max_chars'tan küçük olmalı")
        if problems:
            raise ValueError("Geçersiz ayar: " + "; ".join(problems))


def env_name(field: str) -> str:
    return f"MIRAS_{field.upper()}"


SETTINGS = Settings()
