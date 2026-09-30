# 🏛️ Miras — Yerel RAG Asistanı (Microsoft Foundry Local)

Türkiye'deki UNESCO Dünya Mirası alanları hakkında Türkçe soruları **tamamen bilgisayarınızda,
internet olmadan** cevaplayan bir soru-cevap asistanı. Microsoft AI Summer Internship programının
*"Local RAG AI Assistant with Microsoft Foundry Local"* projesi kapsamında geliştirildi.

- **Retrieval-Augmented Generation (RAG):** Asistan önce yerel dokümanlardan ilgili parçaları bulur,
  sonra yalnızca bu parçalara dayanarak cevap üretir.
- **Microsoft Foundry Local:** Hem embedding hem sohbet modeli cihazda çalışır; hiçbir ağ çağrısı yapılmaz.
- **SQLite:** Doküman parçaları ve vektörleri tek bir dosyada saklanır.
- **Kaynak gösterir, uydurmaz:** Her bilginin yanında `[K1]` gibi bir kaynak etiketi bulunur.
  Cevap dokümanlarda yoksa asistan *"Bu bilgi elimdeki dokümanlarda yer almıyor."* der.

```
Soru> Göbeklitepe hangi yıl UNESCO Dünya Mirası kalıcı listesine girdi?
Göbeklitepe 2018'de UNESCO Dünya Mirası kalıcı listesine girdi [K1].

Kaynaklar:
  [K1] Göbeklitepe  (gobeklitepe.md, benzerlik 0.86)

Soru> Efes'e giriş ücreti ne kadar?
Bu bilgi elimdeki dokümanlarda yer almıyor.
```

## İçindekiler

- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Bilgi tabanı](#bilgi-tabanı)
- [Değerlendirme sonuçları](#değerlendirme-sonuçları)
- [Tasarım kararları](#tasarım-kararları)
- [Sınırlamalar](#sınırlamalar)
- [Proje yapısı](#proje-yapısı)
- [Kaynaklar ve lisans](#kaynaklar-ve-lisans)

## Mimari

Tüm bileşenler tek makinede çalışır:

```
             ┌───────────────────────────────────────────────────────────┐
             │                      Bu bilgisayar                        │
             │                                                           │
  Kullanıcı ─┼─► Arayüz: Streamlit (app.py) veya CLI (main.py)           │
             │        │                                                  │
             │        ▼                                                  │
             │   Assistant (miras/pipeline.py)                           │
             │    1. Soruyu göm ──────────────► qwen3-embedding-0.6b ─┐  │
             │    2. En benzer 4 parçayı getir ◄── SQLite (miras.db)  │  │
             │    3. Benzerlik eşiğin altında mı? → "bilmiyorum"      │  │
             │    4. Parçalar + soru ─────────► qwen2.5-7b ───────────┤  │
             │    5. Çıktıyı doğrula (reddetme, kaynak etiketleri)    │  │
             │                                   Foundry Local SDK ◄──┘  │
             └───────────────────────────────────────────────────────────┘
```

**Veri alma (ingest):** `data/docs/*.md` → başlık yapısını koruyarak parçalama → her parça için
embedding → SQLite'a kayıt. Yalnızca değişen dosyalar yeniden işlenir.

**Soru-cevap:** soru gömülür → tüm parçalarla kosinüs benzerliği hesaplanır → en iyi 4 parça
`[K1]..[K4]` etiketleriyle modele verilir → model cevabı akış halinde üretir → uygulama cevabı
doğrular ve kaynak listesini veritabanından ekler.

## Kurulum

**Gereksinimler**

- macOS (Apple silicon), Windows ya da Linux
- Python 3.12 ve [uv](https://docs.astral.sh/uv/). `pip` ile kurmak isterseniz `requirements.txt` da var.
- Yaklaşık 7 GB boş disk (modeller) ve 16 GB RAM
- İnternet yalnızca ilk çalıştırmada, modelleri indirmek için gerekir.

Ayrıca Foundry Local CLI kurmanıza gerek yok: `foundry-local-sdk` paketi çalışma zamanını kendisi içerir.

```bash
git clone <repo-adresi> miras && cd miras
uv sync                             # sanal ortam + bağımlılıklar (Python 3.12)
uv run python main.py ingest        # embedding modelini indirir, dokümanları indeksler
```

İlk çalıştırmada modeller `~/.miras-rag/cache/models/` altına indirilir:

| Model | Boyut |
|---|---|
| `qwen3-embedding-0.6b` | ~0.5 GB |
| `qwen2.5-7b` | ~6 GB |

İndeksleme bu makinede (Apple M5 Pro) 240 parça için yaklaşık 100 saniye sürdü.

> **Windows notu:** Windows'ta Foundry Local'in `foundry-local-sdk-winml` paketiyle
> kullanıldığı biliniyor; bu proje yalnızca macOS'ta test edildi.

## Kullanım

### Web arayüzü (Streamlit)

```bash
uv run streamlit run app.py
```

Sohbet ekranındaki her cevabın altında, atıf yapılan parçalar açılır kutular halinde görünür:
bölüm yolu, benzerlik skoru ve Vikipedi bağlantısıyla birlikte. Kenar çubuğunda örnek sorular,
bilgi tabanındaki alanlar ve "getirilen tüm parçaları göster" seçeneği bulunur.

### Komut satırı

```bash
uv run python main.py chat                      # etkileşimli sohbet ("çık" ile biter)
uv run python main.py ask "Selimiye Camii'nin mimarı kimdir?"
uv run python main.py ask -v "Ani hangi ilde?"  # parça metinleri ve süreler de gösterilir
uv run python main.py ingest --rebuild          # indeksi sıfırdan kurar
uv run python main.py calibrate                 # benzerlik eşiğini test setinden ölçer
uv run python main.py eval                      # test setini çalıştırır, reports/ altına yazar
uv run pytest                                   # birim testleri (model gerektirmez, ~1 sn)
```

### Ayarlar

Ayarlar `miras/config.py` dosyasındadır ve `MIRAS_*` ortam değişkenleriyle değiştirilebilir:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MIRAS_CHAT_MODEL` | `qwen2.5-7b` | Foundry Local sohbet modeli |
| `MIRAS_EMBED_MODEL` | `qwen3-embedding-0.6b` | Embedding modeli (değişirse `ingest --rebuild` gerekir) |
| `MIRAS_TOP_K` | `4` | Modele verilen parça sayısı |
| `MIRAS_MIN_SIMILARITY` | `0.53` | Bunun altında model çağrılmadan reddedilir |
| `MIRAS_DOCS_DIR` | `data/docs` | Doküman klasörü (`.md` / `.txt`) |

Kendi dokümanlarınızı kullanmak için dosyaları `data/docs/` klasörüne koyup `ingest` çalıştırmanız
yeterli. Bu durumda `calibrate` ile eşiği yeniden ölçmeniz ve sistem prompt'undaki konu tanımını
(`miras/prompts.py`) güncellemeniz önerilir.

## Bilgi tabanı

`data/docs/` klasöründe 12 Türkçe Vikipedi maddesi var:

Afrodisias, Ani, Çatalhöyük, Divriği Ulu Camii ve Darüşşifası, Efes, Göbeklitepe, Hattuşa,
Nemrut Dağı, Pamukkale, Safranbolu, Selimiye Camii, Troya.

- Maddeler `scripts/fetch_corpus.py` ile bir kez indirildi ve repoya eklendi; asistanın çalışması
  için internet gerekmez.
- Her dosyanın başında kaynak adresi, revizyon numarası ve lisans bilgisi bulunur.
- Kaynakça ve "dış bağlantılar" gibi bilgi taşımayan bölümler çıkarıldı.
- Toplam yaklaşık 158 bin karakter, 240 parça (ortanca 644 karakter).

## Değerlendirme sonuçları

Test seti `data/eval/questions.json` dosyasında, 44 Türkçe sorudan oluşuyor:

| Tür | Adet | Beklenen davranış |
|---|---|---|
| Cevaplanabilir | 26 | Doğru cevap. Anahtar kelime gruplarıyla puanlanır (ör. `[["Mellaart"], ["1958"]]`). |
| Konu içi ama dokümanda yok | 10 | Reddetme. Ör. *"Efes'e giriş ücreti ne kadar?"*, *"Sümela Manastırı hangi ilde?"* |
| Konu dışı | 8 | Reddetme. Ör. *"Python'da liste nasıl sıralanır?"* |

Sorular `dev` ve `test` diye ikiye ayrıldı. Benzerlik eşiği ve prompt yalnızca `dev` sorularına
bakılarak ayarlandı; `test` sonucu ayrıca raporlanıyor.

**Sonuçlar:** `qwen2.5-7b` (CPU) + `qwen3-embedding-0.6b`, Apple M5 Pro. Tam rapor, soru bazında
cevaplarla birlikte [`reports/eval-qwen2.5-7b-cpu.md`](reports/eval-qwen2.5-7b-cpu.md) dosyasında.

| Metrik | Sonuç |
|---|---|
| **Genel** | **44/44** (yalnızca test bölümü: 22/22) |
| Cevaplanabilir sorularda doğru cevap | 26/26 (yanlışlıkla reddedilen: 0) |
| Verilen cevaplarda doğru dokümana atıf | 26/26 |
| Konu içi ama dokümanda olmayan sorular reddedildi | 10/10 |
| Konu dışı sorular reddedildi | 8/8 (hepsi eşikte, model çağrılmadan) |
| Arama: doğru doküman 1. sırada / ilk 4 içinde | %96 / %100 (MRR 0.981) |
| Arama: beklenen cevap getirilen parçalarda | %100 |
| Model çağrılan sorularda ortanca / en uzun süre | 13.6 sn / 20.0 sn |

**Eşik kalibrasyonu (`main.py calibrate`)**

| Soru türü | En iyi parçanın benzerliği |
|---|---|
| Cevaplanabilir | 0.637 – 0.900 |
| Konu dışı | 0.312 – 0.439 |
| Konu içi ama dokümanda olmayan | 0.489 – 0.799 |

- Seçilen eşik 0.53: `dev` sorularında mükemmel ayrım veren 0.42–0.63 aralığının ortası.
  `test` sorularında da ayrım %100.
- Konu içi ama dokümanda olmayan soruların skorları cevaplanabilir sorularla örtüşüyor; eşik
  bunların yalnızca %20'sini yakalıyor. Kalanını modelin reddetmesi gerekiyor ve test setinde reddetti.

**Bu sonuçları nasıl okumalı:**

- 44 soru küçük bir set ve soruları biz yazdık. %100 sonuç, sistemin bu soru tipinde güvenilir
  olduğunu gösterir; her soruda doğru cevap vereceğini garanti etmez.
- Prompt ve başlık eşleştirme `dev` sorularındaki hatalara bakılarak geliştirildi. Bu yüzden asıl
  bağımsız ölçüm `test` bölümüdür (22/22).
- Puanlama anahtar kelimeye dayanır: doğru kelimeyi içeren ama yanlış bağlamda kurulmuş bir cümle
  de geçer sayılabilir. Soru bazındaki cevaplar raporda elle incelenebilir.

## Tasarım kararları

**1. Foundry Local SDK 2.x, süreç içi oturumlar**

- `ChatSession` ve `EmbeddingsSession` kullanılıyor. Ayrı bir servis, port ya da OpenAI istemcisi
  yok; SDK'daki eski OpenAI uyumlu istemci 2026 sonunda kaldırılacak.
- Her soru için yeni bir sohbet oturumu açılıyor, böylece cevap önceki sorulardan etkilenmiyor.
- Model varyantı (CPU) açıkça seçiliyor; varsayılan varyant makineye göre değişebiliyor.

**2. Türkçe için 7B model.** 7B'den küçük modeller Türkçede zayıf kaldı. Geliştirme sırasında
`qwen2.5-1.5b` ile yapılan bir ön denemede model, cevabı getirilen parçada olan soruların bir
kısmını da reddetti ve atıfların çoğunu atladı (bu deneme artık kullanılmayan bir sıralama
ayarıyla yapıldı). `qwen2.5-7b` bu sorunları giderdi; bedeli yanıt süresi.

**3. İki katmanlı reddetme**

- *Benzerlik kapısı:* Hiçbir parça yeterince benzer değilse model hiç çağrılmaz. Konu dışı
  sorularda uydurma riski böylece sıfıra iner ve cevap anında döner.
- *Model reddi:* Konu içi ama dokümanda olmayan sorular yüksek benzerlik alır. Bunları, iki
  örnekli (biri cevaplanan, biri reddedilen) sistem prompt'u ile model reddeder.
- Modelin reddetme cümlesini farklı yazdığı durumlar (*"kaynaklarda bulunmamaktadır"* gibi)
  yakalanıp tek standart cümleye indirgenir.

**4. Eşik ölçülerek seçildi.** Elle seçilmiş ve aynı sorularla puanlanmış bir eşik yanıltıcıdır.
Bu yüzden eşik `dev` sorularından seçilip `test` sorularında doğrulandı.

**5. Kaynaklar uygulama tarafından doğrulanır**

- Model yalnızca `[K1]` gibi etiketler yazar.
- Getirilmemiş bir parçaya işaret eden etiketler (ör. `[K9]`) metinden silinir.
- Kaynak listesi (başlık, bölüm, dosya, Vikipedi bağlantısı) veritabanından üretilir; model kaynak
  adı uyduramaz.

**6. Başlığı koruyan parçalama**

- Parçalar bölüm sınırlarını aşmaz.
- Her parçanın gömülen metnine başlık yolu eklenir (ör. `Selimiye Camii › Tarihçe › Minareler`).
  Böylece "Tarihçe" gibi genel bölümler de hangi alana ait olduklarını taşır.

**7. Başlık eşleştirmeli sıralama**

- *Sorun:* "Ani" gibi kısa özel isimler embedding'de zayıf temsil ediliyor. *"Ani hangi yıl tescil
  edildi?"* sorusunda ilk 4 parçanın hiçbiri Ani'ye ait değildi.
- *Çözüm:* Soruda bir alanın adı geçiyorsa (`Ani'deki`, `Efes'e`) o dokümanın parçalarına
  sıralamada +0.1 puan veriliyor.
- Eşik kontrolü ham benzerlikle yapıldığı için kalibrasyon bundan etkilenmiyor.

**8. SQLite + NumPy**

- Vektörler `float32 BLOB` olarak saklanıyor ve açılışta tek bir normalize matrise yükleniyor.
  Her sorgu tek bir matris çarpımı; 240 parça için milisaniyenin altında.
- `meta` tablosu indeksi oluşturan embedding modelini saklar. Model değişirse eski vektörlerle
  yanlış sonuç üretmek yerine açık bir hata verilir.
- Dosya özetiyle artımlı güncelleme yapılır: yalnızca değişen dokümanlar yeniden gömülür.

**9. Prompt enjeksiyonuna karşı çerçeveleme.** Parçalar `<<< >>>` arasında, "yalnızca veri"
olarak verilir ve sistem prompt'u içlerindeki talimatların uygulanmamasını söyler. Bu bir önlem;
tam koruma değildir.

**10. Model gerektirmeyen testler.** `FakeBackend` kelime özetlerinden vektör üretir ve sabit cevap
döndürür. Böylece 29 test (parçalama, SQLite, artımlı ingest, arama, reddetme, atıf temizleme,
değerlendirme metrikleri ve Streamlit arayüzü) model indirmeden yaklaşık 1 saniyede çalışır.

## Sınırlamalar

- **Yanıt süresi uzun.** 7B model CPU'da bir cevabı ortalama 14 saniyede üretiyor; plandaki 1–3
  saniye hedefinin çok üstünde. Cevap akış halinde geldiği için ilk kelimeler daha erken görünür.
  - Kodda sohbet modelini GPU (WebGPU) üzerinde çalıştırma seçeneği var
    (`--device gpu` / `MIRAS_CHAT_DEVICE=gpu`), ancak **test edilmedi** ve ölçülmedi.
- **Tek turlu soru-cevap.** Önceki sorulara referans veren takip soruları (*"peki orası ne zaman
  keşfedildi?"*) desteklenmiyor.
- **Başlık eşleştirme basit.** Yalnızca başlığın ilk kelimesine bakıyor; *"Truva"* yazılırsa
  (dosyadaki ad *"Troya"*) eşleşmez. Bu durumda yalnızca embedding benzerliği kullanılır.
- **Vikipedi içeriği hatasız değil.** Asistan dokümanda yazanı aktarır; dokümandaki bir hata cevaba
  da yansır. Kullanılan revizyonlar dosya başlıklarında kayıtlı.
- **Arayüzden dosya yüklenemiyor.** Yeni doküman eklemek için `data/docs/` klasörüne koyup
  `ingest` çalıştırmak gerekiyor.

## Proje yapısı

```
main.py                 CLI giriş noktası
app.py                  Streamlit web arayüzü
miras/
  config.py             ayarlar (model, top_k, eşik, yollar)
  backends.py           FoundryBackend (Foundry Local SDK) ve testler için FakeBackend
  chunking.py           üst bilgi ayrıştırma, başlığı koruyan parçalama
  store.py              SQLite şeması ve okuma/yazma
  ingest.py             artımlı veri alma hattı
  retrieval.py          vektör araması, başlık eşleştirmeli sıralama
  prompts.py            sistem prompt'u, reddetme tespiti, atıf ayrıştırma
  pipeline.py           Assistant: prepare → generate (akış) → finalize
  evaluation.py         metrikler, eşik kalibrasyonu, raporlar
  cli.py                ingest / ask / chat / calibrate / eval komutları
data/docs/              bilgi tabanı (12 Vikipedi maddesi)
data/eval/questions.json  44 soruluk test seti
reports/                değerlendirme raporları (JSON + Markdown)
scripts/fetch_corpus.py bilgi tabanını Vikipedi'den indiren betik
tests/                  pytest testleri
docs/superpowers/specs/ tasarım dokümanı
```

## Kaynaklar ve lisans

- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968): Microsoft Tech Community
- [Foundry Local belgeleri](https://learn.microsoft.com/azure/ai-foundry/foundry-local/): Microsoft Learn
- Bilgi tabanı: [Türkçe Vikipedi](https://tr.wikipedia.org) katkıcıları,
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Her dosyanın başında
  kaynak adresi ve revizyon numarası bulunur. `data/docs/` içeriği aynı lisansla paylaşılır.
