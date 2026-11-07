"""Tüm ayarlar tek yerde. Ortam değişkenleriyle (MIRAS_*) ezilebilir."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MIRAS_{name}", default)


def _env_number(name: str, default: str, kind: type[int] | type[float]) -> int | float:
    """Sayısal ortam değişkeni; "abc" gibi değerlerde hangi değişkenin bozuk olduğunu söyler."""
    value = _env(name, default)
    try:
        return kind(value.replace(",", ".") if kind is float else value)
    except ValueError:
        expected = "tam sayı" if kind is int else "sayı"
        raise ValueError(f"MIRAS_{name} bir {expected} olmalı, '{value}' verildi.") from None


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


SETTINGS = Settings()
