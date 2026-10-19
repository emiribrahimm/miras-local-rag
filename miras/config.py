"""Tüm ayarlar tek yerde. Ortam değişkenleriyle (MIRAS_*) ezilebilir."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MIRAS_{name}", default)


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
    top_k: int = int(_env("TOP_K", "4"))
    min_similarity: float = float(_env("MIN_SIMILARITY", "0.53"))
    # Hibrit arama: BM25 anahtar kelime skorunun (0–1) sıralamadaki ağırlığı. 0 = yalnızca
    # embedding. Kapı eşiği her durumda ham kosinüs skoruna bakar. Değer `python main.py
    # tune-hybrid` ile ölçülmeden açılmamalı; varsayılan bu yüzden kapalı.
    bm25_weight: float = float(_env("BM25_WEIGHT", "0.0"))
    # Takip soruları: alan adı geçmeyen soru önceki turun konusuyla sorulur, ama sorunun
    # kendisinin en iyi benzerliği bu değerin altındaysa (konu dışı) konu taşınmaz; aksi halde
    # alan adı benzerliği şişirip konu dışı soruyu kapıdan geçirir. 0 = kontrol kapalı.
    # Konu dışı sorular 0.31–0.44 aldı; `python main.py calibrate` ile ölçülmesi önerilir.
    followup_min_similarity: float = float(_env("FOLLOWUP_MIN_SIMILARITY", "0.40"))

    temperature: float = 0.0
    max_output_tokens: int = 400
    seed: int = 42


SETTINGS = Settings()
