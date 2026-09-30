# Miras — yerel RAG asistanı tasarımı

Tarih: 2026-09-30

## Amaç

Microsoft AI Summer Internship programının "Local RAG AI Assistant with Microsoft
Foundry Local" projesi. Türkiye'deki UNESCO Dünya Mirası alanları hakkında Türkçe
soruları, yalnızca yerel doküman setine dayanarak ve internet olmadan cevaplayan bir
asistan.

Başarı ölçütleri (plan dokümanından):

- Cevap dokümanlarda varsa ilgili bilgiyle ve kaynağıyla cevap verir.
- Bilgi yoksa uydurmaz; sabit bir "bilmiyorum" cümlesi döner.
- Boş soru gibi uç durumları düzgün karşılar.
- Tüm çıkarım cihazda yapılır (Foundry Local), veriler SQLite'ta durur.

## Kararlar

| Konu | Karar | Gerekçe |
|---|---|---|
| Bilgi tabanı | 12 Türkçe Vikipedi maddesi (CC BY-SA), `data/docs/*.md` | Gerçek, telifi temiz, nesnel olgular içeriyor; cevaplar kesin puanlanabilir |
| Çıkarım | `foundry-local-sdk` 2.x, süreç içi `ChatSession` / `EmbeddingsSession` | Ayrı servis/CLI gerekmiyor; eski OpenAI istemcisi kullanımdan kalkıyor |
| Embedding | `qwen3-embedding-0.6b`, sorguda talimat öneki | Çok dilli; Foundry Local kataloğunda |
| Sohbet modeli | Ölçümle seçilir (`reports/`) | Türkçede küçük modeller zayıf; karar veriyle verilmeli |
| Parçalama | Başlık yolunu koruyan, ~900 karakter hedefli | Bölümler karışmaz; başlık yolu gömülen metne eklenir |
| Depolama | SQLite: `documents`, `chunks` (float32 BLOB), `meta` | Plan gereği; BLOB, JSON metnine göre küçük ve hızlı |
| Arama | Bellekte normalize matris, tek matris çarpımı, top-K | 240 parça için kaba kuvvet yeterli |
| Reddetme | (1) benzerlik eşiği — model çağrılmaz, (2) modelin sabit cümlesi + tespit | Konu dışı soruları kod garanti eder; konu içi eksik bilgiyi model reddeder |
| Eşik | `dev` sorularından ölçülür, `test` sorularında doğrulanır | Elle seçilmiş ve aynı sorularla puanlanmış eşik yanıltıcıdır |
| Kaynaklar | Model `[K1]` etiketleri yazar; uygulama geçersizleri siler, listeyi veritabanından üretir | Model kaynak adı uyduramaz |
| Arayüz | Streamlit (`app.py`) + CLI (`main.py`) | Plandaki A ve B seçenekleri; ikisi aynı `Assistant` sınıfını kullanır |
| Testler | `FakeBackend` ile pytest | Model indirmeden saniyeler içinde çalışır |

## Bileşenler

```
main.py ─ miras/cli.py ─┐
app.py (Streamlit) ─────┤
                        ▼
                 miras/pipeline.py  Assistant: prepare → generate → finalize
                   │           │
        miras/retrieval.py   miras/prompts.py
                   │           │
            miras/store.py   miras/backends.py ── Foundry Local (cihazda)
            (SQLite)         (FoundryBackend / FakeBackend)

miras/ingest.py + miras/chunking.py : dokümanlar → parçalar → vektörler → SQLite
miras/evaluation.py                 : test seti, eşik kalibrasyonu, raporlar
```

## Veri akışı

1. `ingest`: her dosyanın özeti (içerik + parçalama ayarları) saklanır; yalnızca değişen
   dosyalar yeniden gömülür. Embedding modeli değişirse indeks sıfırlanır.
2. Soru: boşsa sabit mesaj. Değilse gömülür, top-K parça getirilir.
3. En iyi benzerlik eşiğin altındaysa model çağrılmadan reddetme cümlesi döner.
4. Aksi halde parçalar `[K1]..[Kn]` etiketiyle, "veri, talimat değil" çerçevesinde modele verilir.
5. Çıktı doğrulanır: reddetme varyantları tek cümleye indirgenir, var olmayan etiketler silinir.

## Hata durumları

- Model yok / indirilemedi / yüklenemedi → `BackendError`, kullanıcıya Türkçe mesaj.
- İndeks boş ya da başka embedding modeliyle kurulmuş → `IndexError_`, hangi komutun çalıştırılacağı söylenir.
- Embedding NaN/Inf üretirse → `BackendError` (bozuk vektör veritabanına yazılmaz).

## Kapsam dışı

Arayüzden dosya yükleme, çok turlu sohbet hafızası, hibrit (BM25) arama, yeniden
sıralama, CI. Ölçümler gerek gösterirse sonradan eklenir.
