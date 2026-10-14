# Miras

A small RAG assistant that answers Turkish questions about Turkey's UNESCO World Heritage sites,
running entirely on your own machine with [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/).
No API keys, no network calls once the models are downloaded.

I built it as my project for the Microsoft AI Summer Internship ("Local RAG AI Assistant with
Microsoft Foundry Local"). The knowledge base is 12 Turkish Wikipedia articles, so questions and
answers are in Turkish.

[Türkçe için aşağı kaydırın](#türkçe)

```
Soru> Göbeklitepe hangi yıl UNESCO Dünya Mirası kalıcı listesine girdi?
Göbeklitepe 2018'de UNESCO Dünya Mirası kalıcı listesine girdi [K1].

Kaynaklar:
  [K1] Göbeklitepe  (gobeklitepe.md, benzerlik 0.86)

Soru> Efes'e giriş ücreti ne kadar?
Bu bilgi elimdeki dokümanlarda yer almıyor.
```

## How it works

Markdown files in `data/docs/` are split into chunks along their headings (each chunk keeps its
heading path, e.g. `Selimiye Camii › Tarihçe`), embedded with `qwen3-embedding-0.6b` and stored in
a single SQLite file. At query time the question is embedded, compared against every chunk with
NumPy, and the top 4 go to `qwen2.5-7b` with `[K1]..[K4]` labels.

A few things I added after looking at where it went wrong:

- If even the best chunk is below a similarity threshold (0.53, measured with `calibrate`), the
  model isn't called at all and it just says it doesn't know. Off-topic questions never reach the LLM.
- The model only writes `[K1]`-style tags. The source list comes from the database, and tags
  pointing at chunks that weren't retrieved get dropped, so it can't invent a source.
- Short names like "Ani" embed poorly, so if a site's name or alias shows up in the question its
  chunks get a small boost. Spellings without Turkish characters work too ("Gobeklitepe").

The design notes from the start of the project are in `docs/superpowers/specs/`.

## Setup

You need Python 3.12, [uv](https://docs.astral.sh/uv/), about 7 GB of disk for the models and
ideally 16 GB of RAM. Internet is only needed the first time, to download the models. There's a
`requirements.txt` if you'd rather use pip. You don't need the Foundry Local CLI, the
`foundry-local-sdk` package ships the runtime.

```bash
git clone https://github.com/emiribrahimm/miras-local-rag.git
cd miras-local-rag
uv sync
uv run python main.py ingest
```

Models end up in `~/.miras-rag/cache/models/`. Indexing the 240 chunks took around 100 seconds on
my machine (Apple M5 Pro). I've only tested it on macOS. On Windows, Foundry Local is supposed to be
used with the `foundry-local-sdk-winml` package instead, I haven't tried that.

## Usage

```bash
uv run streamlit run app.py                 # web UI

uv run python main.py chat                  # interactive, type "çık" to quit
uv run python main.py ask "Selimiye Camii'nin mimarı kimdir?"
uv run python main.py ask -v "Ani hangi ilde?"   # also prints chunks and timings
uv run python main.py ingest --rebuild
uv run python main.py calibrate             # re-measure the similarity threshold
uv run python main.py eval                  # run the question set, writes to reports/
uv run pytest                               # no models needed, ~1s
```

Settings live in `miras/config.py` and can be overridden with `MIRAS_*` env vars, e.g.
`MIRAS_CHAT_MODEL`, `MIRAS_TOP_K`, `MIRAS_MIN_SIMILARITY` or `MIRAS_DOCS_DIR`. If you change the
embedding model you'll need `ingest --rebuild`.

To use your own documents, drop `.md` or `.txt` files into `data/docs/` and run `ingest`. Only
changed files get re-embedded. You'll probably want to rerun `calibrate` and update the topic
description in `miras/prompts.py` too. An optional header at the top of a file sets the title and
aliases:

```
---
title: Troya
aliases: Truva, Troia, İlion
source: https://tr.wikipedia.org/wiki/Troya
---
```

## Results

The eval set in `data/eval/questions.json` has 44 questions: 26 answerable, 10 on-topic but not
covered by the docs (should be refused) and 8 off-topic. They're split into `dev` and `test`, and
I only looked at `dev` while tuning the threshold and the prompt.

With `qwen2.5-7b` on CPU it got 44/44 (22/22 on the test half), cited the right document every
time, and took about 14 seconds per answer (median, when the model is called). The full report
with every answer is in [`reports/eval-qwen2.5-7b-cpu.md`](reports/eval-qwen2.5-7b-cpu.md).

Take that with a grain of salt though. It's a small set I wrote myself, and scoring is keyword
based, so it says more about this kind of question than about the system in general.

## Limitations

- It's slow. ~14 s per answer on CPU, although streaming helps a bit. There's a GPU option
  (`--device gpu` or `MIRAS_CHAT_DEVICE=gpu`) but I haven't tested or measured it.
- No follow-up questions, every question is answered on its own.
- Name matching is dictionary based. A typo like "Göbeklitpe" falls back to plain embedding search.
- Answers are only as good as the Wikipedia revisions in `data/docs/`.
- You can't upload files from the UI yet.

## Credits and license

- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Microsoft Tech Community)
- [Foundry Local docs](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
- The articles in `data/docs/` come from [Turkish Wikipedia](https://tr.wikipedia.org) and are
  shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each file has its
  source URL and revision number at the top.

---

## Türkçe

Miras, Türkiye'deki UNESCO Dünya Mirası alanları hakkındaki Türkçe soruları cevaplayan küçük bir
RAG asistanı. Her şey Microsoft Foundry Local ile bilgisayarın kendisinde çalışıyor; modeller bir
kez indirildikten sonra internete ya da API anahtarına gerek yok.

Microsoft AI Summer Internship kapsamındaki "Local RAG AI Assistant with Microsoft Foundry Local"
projesi için yaptım. Bilgi tabanı 12 Türkçe Vikipedi maddesinden oluşuyor.

### Nasıl çalışıyor

`data/docs/` altındaki dosyalar başlıklarına göre parçalanıyor. Her parça hangi başlığın altında
olduğunu da taşıyor (`Selimiye Camii › Tarihçe` gibi), böylece "Tarihçe" diye genel bir bölüm bile
hangi alana ait olduğunu kaybetmiyor. Parçalar `qwen3-embedding-0.6b` ile gömülüp tek bir SQLite
dosyasına yazılıyor. Soru gelince en benzer 4 parça bulunuyor ve `[K1]..[K4]` etiketleriyle
`qwen2.5-7b`'ye veriliyor.

Hatalara bakarak sonradan eklediğim birkaç şey:

- En iyi parçanın benzerliği eşiğin (0.53) altındaysa model hiç çağrılmıyor, direkt "bilmiyorum"
  deniyor. Konu dışı sorular modele hiç ulaşmıyor. Eşiği elle seçmedim, `calibrate` ile ölçtüm.
- Model sadece `[K1]` gibi etiket yazıyor; kaynak listesini uygulama veritabanından çıkarıyor.
  Getirilmemiş bir parçaya işaret eden etiket metinden siliniyor, yani kaynak uyduramıyor.
- "Ani" gibi kısa adlar embedding'de iyi temsil edilmiyor. Soruda bir alanın adı ya da takma adı
  geçiyorsa (`Truva`, `Ephesos`) o alanın parçaları biraz öne alınıyor. "Gobeklitepe" gibi Türkçe
  karaktersiz yazımlar da tanınıyor.

### Kurulum

Python 3.12 ve [uv](https://docs.astral.sh/uv/) lazım. Modeller için yaklaşık 7 GB disk, rahat
çalışması için 16 GB RAM öneririm. İnternet sadece ilk çalıştırmada modelleri indirmek için
gerekiyor. pip kullanmak isteyen için `requirements.txt` de var. Foundry Local CLI'ı ayrıca kurmaya
gerek yok, `foundry-local-sdk` paketi çalışma zamanını içinde getiriyor.

```bash
git clone https://github.com/emiribrahimm/miras-local-rag.git
cd miras-local-rag
uv sync
uv run python main.py ingest
```

Modeller `~/.miras-rag/cache/models/` altına iniyor. 240 parçanın indekslenmesi benim makinemde
(Apple M5 Pro) 100 saniye kadar sürdü. Sadece macOS'ta denedim. Windows'ta Foundry Local'in
`foundry-local-sdk-winml` paketiyle kullanılması gerekiyormuş, onu test etmedim.

### Kullanım

```bash
uv run streamlit run app.py                 # web arayüzü

uv run python main.py chat                  # sohbet, çıkmak için "çık"
uv run python main.py ask "Selimiye Camii'nin mimarı kimdir?"
uv run python main.py ask -v "Ani hangi ilde?"   # parçaları ve süreleri de gösterir
uv run python main.py ingest --rebuild
uv run python main.py calibrate             # benzerlik eşiğini yeniden ölç
uv run python main.py eval                  # test setini çalıştır, reports/ altına yazar
uv run pytest                               # model gerekmez, ~1 sn
```

Ayarlar `miras/config.py` içinde; `MIRAS_CHAT_MODEL`, `MIRAS_TOP_K`, `MIRAS_MIN_SIMILARITY`,
`MIRAS_DOCS_DIR` gibi ortam değişkenleriyle değiştirilebiliyor. Embedding modelini değiştirirseniz
`ingest --rebuild` çalıştırmak gerekiyor.

Kendi dokümanlarınızla denemek için `.md` ya da `.txt` dosyalarını `data/docs/` klasörüne atıp
`ingest` çalıştırmanız yeterli, sadece değişen dosyalar yeniden işleniyor. Sonrasında `calibrate`
ile eşiği tekrar ölçmek ve `miras/prompts.py` içindeki konu tanımını güncellemek iyi olur. Dosyanın
başına başlık ve takma adlar için şöyle bir üst bilgi eklenebilir:

```
---
title: Troya
aliases: Truva, Troia, İlion
source: https://tr.wikipedia.org/wiki/Troya
---
```

### Sonuçlar

`data/eval/questions.json` içinde 44 soru var: 26'sı dokümanlardan cevaplanabilir, 10'u konuyla
ilgili ama dokümanlarda yok (reddedilmeli), 8'i tamamen konu dışı. Sorular `dev` ve `test` olarak
ikiye ayrılmış durumda; eşiği ve prompt'u ayarlarken sadece `dev` sorularına baktım.

CPU'da `qwen2.5-7b` ile 44/44 aldı (test kısmında 22/22), atıfların hepsi doğru dokümana gidiyordu.
Model çağrılan sorularda ortanca süre 14 saniye civarı. Soru soru cevapların olduğu rapor
[`reports/eval-qwen2.5-7b-cpu.md`](reports/eval-qwen2.5-7b-cpu.md) dosyasında.

Bu sonucu fazla büyütmemek lazım: set küçük, soruları ben yazdım ve puanlama anahtar kelimeye
dayanıyor. Bu tip sorularda iyi çalıştığını gösteriyor, her soruda doğru cevap vereceğini değil.

### Eksikler

- Yavaş. CPU'da cevap başına ~14 sn, akış halinde geldiği için biraz daha az hissediliyor.
  GPU seçeneği var (`--device gpu` ya da `MIRAS_CHAT_DEVICE=gpu`) ama test etmedim.
- Takip sorusu desteklenmiyor, her soru tek başına cevaplanıyor.
- Ad eşleştirme sözlüğe dayalı. "Göbeklitpe" gibi bir yazım hatasında sadece embedding araması kalıyor.
- Cevaplar `data/docs/` içindeki Vikipedi revizyonları kadar doğru.
- Arayüzden dosya yüklenemiyor, şimdilik klasöre koymak gerekiyor.

### Kaynaklar ve lisans

- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Microsoft Tech Community)
- [Foundry Local belgeleri](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
- `data/docs/` içindeki maddeler [Türkçe Vikipedi](https://tr.wikipedia.org) katkıcılarına ait ve
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisansıyla paylaşılıyor. Her
  dosyanın başında kaynak adresi ve revizyon numarası var.
