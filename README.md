# Miras

[![tests](https://github.com/emiribrahimm/miras-local-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/emiribrahimm/miras-local-rag/actions/workflows/tests.yml)

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
  chunks get a small boost. Spellings without Turkish characters work too ("Gobeklitepe"), and
  so do one-letter typos in longer names ("Göbeklitpe", "Selimye") and two-letter ones in names of
  9 letters or more ("Gbeklitpe").
- In the chat (CLI and web), a question that doesn't name any site is asked about the previous
  one's site: "Peki ne zaman yapılmış?" becomes "Selimiye Camii hakkında: Peki ne zaman
  yapılmış?". No extra model call, just a rule. Naming a site switches the topic; `yeni` in the
  CLI or "Sohbeti temizle" in the web UI resets it. The site name alone was enough to push
  off-topic questions ("Python'da liste nasıl sıralanır?") over the threshold, so the topic is
  only carried if the question on its own scores at least 0.40 against some chunk
  (`MIRAS_FOLLOWUP_MIN_SIMILARITY`, `calibrate` suggests a value). Off-topic questions in the eval
  set scored 0.31–0.44, so 0.40 is a cautious first guess, not a measured one yet. Short
  follow-ups can score below that too, so if the question points back with a word like "orası",
  "oradaki", "onun" or "hangisi" the topic is carried without checking the floor. "Peki" alone
  doesn't count, otherwise "Peki ekmek hamuru nasıl mayalanır?" would get the site name too. A
  follow-up without such a word is also carried if every word in it (besides question words) shows
  up in the previous site's own document: "Peki yüksekliği kaç metre?" after Nemrut Dağı, but not
  "Peki kahve nasıl demlenir?" after Efes. It has to be that site's document and not the whole
  knowledge base, because "Türk kahvesi" does appear in another article.
- There's an optional keyword score (BM25 over word stems, the first 5 letters of each word) that
  can be added to the ranking, for dates and rare names the embeddings miss. It's off by default
  because I haven't measured it on the real models yet: `tune-hybrid` tries a few weights on the
  eval questions (no chat model needed) and suggests one for `MIRAS_BM25_WEIGHT`. The threshold
  still looks at the plain embedding similarity.
- Answers are cached in the same SQLite file. The key is the full prompt (question plus the
  retrieved chunks) together with the model and generation settings, so asking the same thing
  again returns instantly, and changing a document, the prompt or the model just misses the cache.
  The raw output is stored and still goes through the citation checks. `eval` never uses it,
  `ingest --rebuild` clears it and `MIRAS_ANSWER_CACHE=0` turns it off. The vectors of the last 256
  questions are kept in memory as well, so a repeated question (or the same follow-up with its
  topic) doesn't wait for the embedding model either.
- Chunks are up to 1400 characters, so under each cited source the sentence that backs the answer
  is picked out: highlighted in the web UI, quoted in the Markdown export and printed with `»` in
  the CLI. Each answer sentence goes to the chunks its `[K1]` tags point at, and the chunk sentence
  sharing the most word stems with it wins (numbers count double, words from the heading path don't
  count). No model call. If nothing shares at least two stems or one number, nothing is highlighted.
- A question naming two sites ("Efes mi Göbeklitepe mi listeye daha önce girdi?") used to get all
  4 chunks from whichever site was closer, so the model could only answer half of it. Now when two
  or more sites are named, each gets its best chunk first and the rest are filled by ranking as
  usual (with more sites than `top_k`, the best ranked ones win). The prompt also asks for each
  site's facts to be cited from its own chunk. `search` says when this kicks in.

## Setup

You need Python 3.12, [uv](https://docs.astral.sh/uv/), about 7 GB of disk for the models and
ideally 16 GB of RAM. Internet is only needed the first time, to download the models. There's a
`requirements.txt` if you'd rather use pip. You don't need the Foundry Local CLI, the
`foundry-local-sdk` package ships the runtime.

```bash
git clone https://github.com/emiribrahimm/miras-local-rag.git
cd miras-local-rag
uv sync
uv run python main.py doctor   # optional: checks the setup without loading the models
uv run python main.py ingest
```

`doctor` checks the Python version, whether the Foundry Local SDK is installed, whether the models
have been downloaded (and if not, whether there's room for them), the docs folder (files that
aren't UTF-8), the index (missing, built with another embedding model, files not indexed yet) and
settings that switch off a safeguard, like `MIRAS_MIN_SIMILARITY=0`. It exits with 1 if anything
would stop the app from working.

Models end up in `~/.miras-rag/cache/models/`. Indexing the 240 chunks took around 100 seconds on
my machine (Apple M5 Pro). I've only tested it on macOS. On Windows, Foundry Local is supposed to be
used with the `foundry-local-sdk-winml` package instead, I haven't tried that.

## Usage

```bash
uv run streamlit run app.py                 # web UI

uv run python main.py chat                  # interactive, type "çık" to quit
uv run python main.py ask "Selimiye Camii'nin mimarı kimdir?"
uv run python main.py ask -v "Ani hangi ilde?"   # also prints chunks and timings
uv run python main.py ask --json "Ani hangi ilde?"   # answer, sources and timings as JSON
uv run python main.py search "Ani hangi ilde?"   # retrieved chunks and the gate, no chat model
uv run python main.py ask --refresh "Ani hangi ilde?"   # skip the cached answer, ask the model again
uv run python main.py ask --doc efes "Tiyatro kaç kişilik?"   # only search some documents
uv run python main.py ask-batch sorular.txt --out cevaplar.jsonl   # one question per line
uv run python main.py ingest --rebuild
uv run python main.py add-wiki "Sümela Manastırı" --aliases "Sumela"   # needs internet
uv run python main.py update-wiki --check   # which articles changed on Wikipedia, needs internet
uv run python main.py calibrate             # re-measure the similarity threshold
uv run python main.py tune-hybrid           # measure the BM25 weight for hybrid search
uv run python main.py eval                  # run the question set, writes to reports/
uv run python main.py eval --split dev      # only one half (or --ids a01,f03 for a few questions)
uv run python main.py eval --retrieval-only # search and threshold only, no chat model, seconds
uv run python main.py compare-reports reports/old.json reports/eval-qwen2.5-7b-cpu.json
uv run python main.py feedback              # thumbs down answers from the web UI
uv run python main.py log --refused         # questions that were refused, from every interface
uv run python main.py status                # index, cache, votes, files not indexed yet
uv run python main.py docs                  # what's in the index
uv run python main.py remove-doc efes.md    # delete from data/docs/ and the index
uv run python main.py cache --clear         # empty the answer cache
uv run python main.py config                # effective settings and where each one comes from
uv run python main.py serve                 # local HTTP API on 127.0.0.1:8765
uv run pytest                               # no models needed, ~1s
```

In the web UI, "Sohbeti indir (.md)" in the sidebar saves the conversation as a Markdown file,
with the cited chunks and their Wikipedia links under each answer, and "Sohbeti indir (.json)"
saves the same turns with similarity scores and supporting sentences. In `chat`, `kaydet` does the
Markdown version (`kaydet notlar/ani.md` to pick the file), `kaynak` prints the full chunks behind
the last answer (`kaynak K2` for one of them), and the up arrow brings back questions from earlier
sessions (kept in `~/.miras-rag/chat_history`).

Since answers are cached, a bad one would keep coming back. "Yeniden üret" under the last answer
in the web UI (or `ask --refresh`) asks the model again and replaces the cached answer; a vote
given to the old answer is kept separately.

`ask-batch` asks every line of a file as a separate question (empty lines and lines starting with
`#` are skipped, `-` reads from stdin) and writes one JSON object per answer, in the same shape as
`ask --json`. Questions don't carry topics from one line to the next.

`serve` starts a small HTTP API (standard library only, no extra dependency) so other tools can use
the assistant. It listens on `127.0.0.1:8765` by default (`--host`, `--port`). With `--token` (or
`MIRAS_API_TOKEN`) every endpoint except `/health` needs `Authorization: Bearer <token>`; without
one it warns when bound to anything other than localhost. `--cors http://localhost:3000` lets a
page on that origin call it from the browser. Each request is printed to stderr with its status and
duration (`--quiet` turns that off). Requests reach the model one at a time.

```bash
curl -s localhost:8765/ask -d '{"question": "Ani hangi ilde?"}'
curl -s localhost:8765/ask -d '{"question": "Peki ne zaman yapılmış?", "topic": ["Ani"]}'
curl -s localhost:8765/ask -d '{"question": "Tiyatro kaç kişilik?", "docs": ["efes"], "refresh": true}'
curl -s localhost:8765/search -d '{"question": "Efes mi Troya mı daha eski?"}'
curl -sN localhost:8765/ask -d '{"question": "Ani hangi ilde?", "stream": true}'
curl -s localhost:8765/docs -H "Authorization: Bearer $MIRAS_API_TOKEN"
curl -s localhost:8765/health
```

`POST /ask` answers in the same shape as `ask --json`, plus `docs` and `topic`. With
`"stream": true` it sends NDJSON instead: one `{"delta": "..."}` line per piece as the model writes,
then `{"done": true, ...}` with the checked answer (citations cleaned, sources), or an `{"error": ...}`
line if the model fails halfway. `/health` also reports the cached answers and logged questions. The server keeps no
conversation: to ask a follow-up, send back the `topic` of the previous answer. `POST /search`
returns the chunks, their similarity, the site names it spotted and whether the question passes the
threshold, without calling the chat model. Errors come back as `{"error": "..."}` with 400 (bad
JSON, empty question, unknown document), 401 (missing or wrong token), 404/405, 413 (body over
64 KB) or 503 (model error).

`search` is for looking at retrieval on its own: it loads only the embedding model and prints the
chunks that would go to the chat model, their similarity, which site names it spotted in the
question and whether the question passes the threshold. `-k` changes how many chunks come back.

`--doc` on `ask`, `search`, `ask-batch` and `chat` keeps the search inside some documents. It takes
a file name (`efes.md` or `efes`), a title (`"Nemrut Dağı"`) or an alias (`Truva`), can be repeated
or comma separated, and an unknown name stops with the list of indexed files. Inside `chat`,
`sadece efes, troya` sets it and `sadece` on its own clears it; in the web UI it's "Sadece şu
alanlarda ara" in the sidebar. The threshold still applies, so a question about a site outside the
selection is refused without calling the model. The Markdown and JSON exports note the filter.

Each answer also gets a 👍/👎. Votes go into the same SQLite file with the question, the answer
and the retrieved chunks (`ingest --rebuild` keeps them). `feedback` lists the 👎 ones (`--all`
for everything), and `feedback --export reports/feedback-candidates.json` writes them in the
`questions.json` format so they can be turned into eval questions. The vote doesn't say whether
the right answer was a different one or a refusal, so `type` and `expected` are left to fill in;
questions that carried a topic are marked with `carried` and belong in `followups.json`.

Votes only cover answers someone bothered to rate, so every question is also logged (CLI, web UI
and API, but not `eval`): the question as typed, the carried topic, the document filter, why it
was refused if it was, the best similarity and which files were cited. Answer texts aren't kept.
`log` shows the latest ones (`-n 50`), `log --refused` only the refusals, which is the quickest way
to see what people ask that the documents don't cover, and `log --export reports/log-candidates.json`
writes those in the `questions.json` format. The latest refusals are also under "Reddedilen sorular"
in the web UI sidebar. `log --clear` empties it, `MIRAS_QUERY_LOG=0` turns it off and
`ingest --rebuild` leaves it alone.

The tests use a fake backend instead of the models, so they also run on GitHub Actions on every
push (Python 3.12 and 3.13, without the Foundry Local SDK).

In the web UI, "Arama ayarları" in the sidebar has sliders for `top_k`, the threshold and the BM25
weight. They only change the current browser session (the loaded models and the index are shared),
which makes it easy to see what a different threshold would do to a question before changing the
default. Answers given with changed settings say so under the answer and in the exports.

Settings live in `miras/config.py` and can be overridden with `MIRAS_*` env vars, e.g.
`MIRAS_CHAT_MODEL`, `MIRAS_TOP_K`, `MIRAS_MIN_SIMILARITY`, `MIRAS_BM25_WEIGHT`,
`MIRAS_FOLLOWUP_MIN_SIMILARITY`, `MIRAS_ANSWER_CACHE`, `MIRAS_QUERY_LOG` or `MIRAS_DOCS_DIR`. If you change the
embedding model you'll need `ingest --rebuild`. Values that don't make sense (`MIRAS_TOP_K=0`, a
threshold above 1, a typo like `MIRAS_TOP_K=dört`) stop it at startup with a message naming the
setting, instead of quietly breaking the search.

HTML files (`.html`, `.htm`, e.g. a saved web page) are converted to Markdown before chunking,
with the standard library: the first `<h1>` (or `<title>`) becomes the title, `<h2>` and below
become sections, and menus, scripts, headers and footers are dropped. The file itself stays as it is.

Files have to be UTF-8 (Windows line endings are fine). One that isn't is skipped by `ingest` with
a warning and its previous version stays in the index; a file emptied down to its header is taken
out of the index.

To use your own documents, drop `.md`, `.txt` or `.html` files into `data/docs/` and run `ingest`, or
upload them from the web UI sidebar ("Doküman ekle"), which saves them there and indexes them
right away. Only changed files get re-embedded. "Doküman sil" in the same sidebar (or
`remove-doc`) removes a file from `data/docs/` and the index (nothing to re-embed), as long as at
least one document is left. Files edited or deleted by hand while the web UI is open show up as a
warning in the sidebar with an "İndeksi güncelle" button, and `status` lists them too. The answer
cache can be emptied with "Önbelleği temizle" there or `cache --clear`. "Dokümanı incele" shows
the chunks a document was split into, with their heading paths, and the web UI shows a progress
bar while new chunks are being embedded.

Turkish Wikipedia articles can be added by name or link, with "Vikipedi'den ekle" in the sidebar
or `add-wiki`. The article is downloaded, converted the same way as the ones in `data/docs/`
(reference sections dropped, header with source and revision) and indexed. This is the only
thing that goes online, and only when you press the button or run the command. If the name was a
redirect ("Truva" -> Troya) it's added as an alias, disambiguation pages are refused, and adding
an article that's already there updates it and keeps its aliases.

`update-wiki` (or "Vikipedi güncellemeleri" in the sidebar) compares the revision number in each
Wikipedia doc's header with the latest one, in a single request for all of them, and lists the
ones that changed. Without `--check` those get downloaded again under the same file name, keeping
their aliases, and re-indexed. An article that can't be found anymore is only reported, its file
is left alone. Answers to repeated questions about an updated article miss the cache on their own,
since the chunks in the prompt changed.

After adding or removing documents you'll probably want to rerun `calibrate` and update the topic
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

The eval set in `data/eval/questions.json` has 48 questions: 30 answerable (4 of them compare two
sites, their `doc` is a list and the citation only counts if both documents are cited), 10
on-topic but not covered by the docs (should be refused) and 8 off-topic. They're split into `dev` and `test`, and
I only looked at `dev` while tuning the threshold and the prompt.

With `qwen2.5-7b` on CPU it got 44/44 (22/22 on the test half), cited the right document every
time, and took about 14 seconds per answer (median, when the model is called). The full report
with every answer is in [`reports/eval-qwen2.5-7b-cpu.md`](reports/eval-qwen2.5-7b-cpu.md). That run
was on the first 44 questions, before the comparison ones were added.

Take that with a grain of salt though. It's a small set I wrote myself, and scoring is keyword
based, so it says more about this kind of question than about the system in general.

There's also a small follow-up set in `data/eval/followups.json`: 29 two-turn conversations where
only the second question is scored (19 answerable, 10 that should be refused, 7 of them off-topic). `eval` runs it after
the main set and adds a section to the report that also shows where the right document would have
ranked if the follow-up had been asked on its own. The report above predates it, so rerun `eval`
to get those numbers, including how many off-topic follow-ups were refused without calling the
model.

The report also has a table per document (answerable questions only: correct, refused, cited the
right document), since a total like 25/26 doesn't say which article the misses come from. While
tuning, `eval --split dev` keeps the test questions out of sight and `eval --ids a07,f12` reruns
just the ones being worked on; both write to a separate report (`...-dev`, `...-secili`) so the
full one isn't overwritten. `compare-reports old.json new.json` lists the settings that changed,
the scores before and after, and which questions got fixed or broken in between.

A full `eval` takes around ten minutes on CPU, most of it in the chat model. `eval
--retrieval-only` skips it and only loads the embedding model: the right document at rank 1 and in
the top k, MRR, whether the answer is in the retrieved chunks, a table per document, what the
threshold would let through or stop, and for the follow-up set whether the topic was carried. It
writes `reports/arama-<embedding model>.md/.json` and takes the same `--split` and `--ids`. That's
the quick check after changing chunking, `top_k`, the BM25 weight or the follow-up rules.

## Limitations

- It's slow. ~14 s per answer on CPU, although streaming helps a bit (repeated questions come from the cache). There's a GPU option
  (`--device gpu` or `MIRAS_CHAT_DEVICE=gpu`) but I haven't tested or measured it.
- Follow-ups only carry over the site, not the rest of the conversation. Whether a follow-up is
  off-topic is decided by one similarity floor, a short list of words that point back and whether
  all of its words appear in the site's document. A short follow-up that uses a word the article
  doesn't ("Peki ne zaman?" when the article never says "zaman") can still score below the floor
  and lose the topic, and an off-topic one that happens to point back ("Orada yarın hava nasıl?")
  gets the site name and has to be refused by the model.
- Comparison questions only get one guaranteed chunk per named site (the rest go by ranking). That's
  enough for a year or a province, but if one site's answer sits in a chunk that ranks low it can
  still be missed. The 4 comparison questions in the eval set haven't been run on the real models
  yet.
- Name matching is dictionary based. Short names (Ani, Efes, Troya) need to be spelled right, and
  typos bigger than two letters ("Gbklitpe") fall back to plain embedding search.
- Answers are only as good as the Wikipedia revisions in `data/docs/` (`update-wiki` brings them
  up to date, but the eval set was written against the current ones).

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
  karaktersiz yazımlar, uzun adlardaki tek harflik hatalar ("Göbeklitpe", "Selimye") ve 9 harf ve
  üstü adlarda iki harflik hatalar ("Gbeklitpe") da tanınıyor.
- Sohbette (CLI ve web) hiçbir alan adı geçmeyen soru bir önceki sorunun alanıyla soruluyor:
  "Peki ne zaman yapılmış?" sorusu "Selimiye Camii hakkında: Peki ne zaman yapılmış?" oluyor. Ek
  model çağrısı yok, basit bir kural. Başka bir alanın adı geçince konu değişiyor; CLI'da `yeni`,
  web arayüzünde "Sohbeti temizle" sıfırlıyor. Alan adı tek başına konu dışı soruları ("Python'da
  liste nasıl sıralanır?") eşiğin üstüne taşıyabildiği için konu ancak soru kendi başına bir
  parçayla en az 0.40 benzerlik alıyorsa taşınıyor (`MIRAS_FOLLOWUP_MIN_SIMILARITY`, `calibrate` bir
  değer öneriyor). Değerlendirmedeki konu dışı sorular 0.31–0.44 aldı; 0.40 henüz ölçülmüş değil,
  temkinli bir başlangıç değeri. Kısa takip soruları da bu değerin altında kalabildiği için soru
  "orası", "oradaki", "onun", "hangisi" gibi bir kelimeyle öncekine gönderme yapıyorsa konu eşiğe
  bakılmadan taşınıyor. Tek başına "Peki" sayılmıyor; yoksa "Peki ekmek hamuru nasıl mayalanır?"
  sorusuna da alan adı eklenirdi. Böyle bir kelime içermeyen takip sorusu da, soru kelimeleri
  dışındaki bütün kelimeleri önceki alanın kendi dokümanında geçiyorsa taşınıyor: Nemrut Dağı'ndan
  sonra "Peki yüksekliği kaç metre?" taşınıyor, Efes'ten sonra "Peki kahve nasıl demlenir?"
  taşınmıyor. Bütün bilgi tabanına değil o alanın dokümanına bakılıyor, çünkü "Türk kahvesi" başka
  bir maddede geçiyor.
- İsteğe bağlı bir anahtar kelime skoru var (kelimelerin ilk 5 harfi üzerinden BM25). Embedding'in
  kaçırdığı tarih ve nadir isimler için sıralamaya eklenebiliyor. Gerçek modellerle henüz
  ölçmediğim için varsayılan olarak kapalı: `tune-hybrid` birkaç ağırlığı test sorularında deneyip
  (sohbet modeli gerekmez) `MIRAS_BM25_WEIGHT` için bir değer öneriyor. Eşik yine sadece embedding
  benzerliğine bakıyor.
- Cevaplar aynı SQLite dosyasında önbelleğe alınıyor. Anahtar modele giden prompt'un tamamı (soru
  ve getirilen parçalar) ile model ve üretim ayarları; yani aynı soru tekrar sorulunca cevap anında
  geliyor, doküman, prompt ya da model değişince de önbellek kendiliğinden ıskalıyor. Ham çıktı
  saklandığı için atıf kontrolleri yine çalışıyor. `eval` önbelleği kullanmıyor, `ingest --rebuild`
  temizliyor, `MIRAS_ANSWER_CACHE=0` kapatıyor. Son 256 sorunun vektörleri de bellekte tutuluyor;
  tekrar sorulan bir soru (ya da konusuyla aynı takip sorusu) embedding modelini de beklemiyor.
- Parçalar 1400 karaktere kadar çıkabildiği için atıf yapılan her kaynağın altında cevabı
  destekleyen cümle ayrıca gösteriliyor: web arayüzünde vurgulu, Markdown dışa aktarımında alıntı,
  CLI'da `»` ile. Cevabın her cümlesi `[K1]` etiketinin gösterdiği parçaya bağlanıyor ve parçada
  onunla en çok kelime kökü paylaşan cümle seçiliyor (sayılar iki kat, başlık yolundaki kelimeler
  hiç sayılmıyor). Model çağrısı yok. En az iki ortak kök ya da bir ortak sayı yoksa vurgu yapılmıyor.
- İki alanı birden anan bir soruda ("Efes mi Göbeklitepe mi listeye daha önce girdi?") 4 parçanın
  hepsi hangi alan daha yakınsa ondan geliyordu, model sorunun yarısını cevaplayabiliyordu. Artık
  soruda iki ya da daha fazla alan geçince önce her alanın en iyi parçası alınıyor, kalan yerler
  her zamanki sıralamayla doluyor (alan sayısı `top_k`'yı aşarsa en iyi sıradakiler kalıyor).
  Prompt da her alanın bilgisinin kendi parçasından etiketlenmesini istiyor. `search` bunun devreye
  girdiğini yazıyor.

### Kurulum

Python 3.12 ve [uv](https://docs.astral.sh/uv/) lazım. Modeller için yaklaşık 7 GB disk, rahat
çalışması için 16 GB RAM öneririm. İnternet sadece ilk çalıştırmada modelleri indirmek için
gerekiyor. pip kullanmak isteyen için `requirements.txt` de var. Foundry Local CLI'ı ayrıca kurmaya
gerek yok, `foundry-local-sdk` paketi çalışma zamanını içinde getiriyor.

```bash
git clone https://github.com/emiribrahimm/miras-local-rag.git
cd miras-local-rag
uv sync
uv run python main.py doctor   # isteğe bağlı: modelleri yüklemeden kurulumu denetler
uv run python main.py ingest
```

`doctor` Python sürümünü, Foundry Local SDK'nın kurulu olup olmadığını, modellerin inip inmediğini
(inmediyse yer olup olmadığını), doküman klasörünü (UTF-8 olmayan dosyalar), indeksi (yok, başka bir
embedding modeliyle kurulmuş, indekslenmemiş dosyalar) ve `MIRAS_MIN_SIMILARITY=0` gibi bir güvenlik
önlemini kapatan ayarları denetliyor. Uygulamanın çalışmasını engelleyecek bir şey varsa 1 ile
bitiyor.

Modeller `~/.miras-rag/cache/models/` altına iniyor. 240 parçanın indekslenmesi benim makinemde
(Apple M5 Pro) 100 saniye kadar sürdü. Sadece macOS'ta denedim. Windows'ta Foundry Local'in
`foundry-local-sdk-winml` paketiyle kullanılması gerekiyormuş, onu test etmedim.

### Kullanım

```bash
uv run streamlit run app.py                 # web arayüzü

uv run python main.py chat                  # sohbet, çıkmak için "çık"
uv run python main.py ask "Selimiye Camii'nin mimarı kimdir?"
uv run python main.py ask -v "Ani hangi ilde?"   # parçaları ve süreleri de gösterir
uv run python main.py ask --json "Ani hangi ilde?"   # cevap, kaynaklar ve süreler JSON olarak
uv run python main.py search "Ani hangi ilde?"   # getirilen parçalar ve eşik, sohbet modeli yok
uv run python main.py ask --refresh "Ani hangi ilde?"   # önbelleği atla, modele yeniden sor
uv run python main.py ask --doc efes "Tiyatro kaç kişilik?"   # yalnızca bazı dokümanlarda ara
uv run python main.py ask-batch sorular.txt --out cevaplar.jsonl   # satır başına bir soru
uv run python main.py ingest --rebuild
uv run python main.py add-wiki "Sümela Manastırı" --aliases "Sumela"   # internet gerekir
uv run python main.py update-wiki --check   # Vikipedi'de değişen maddeler, internet gerekir
uv run python main.py calibrate             # benzerlik eşiğini yeniden ölç
uv run python main.py tune-hybrid           # hibrit arama için BM25 ağırlığını ölç
uv run python main.py eval                  # test setini çalıştır, reports/ altına yazar
uv run python main.py eval --split dev      # yalnızca bir yarısı (birkaç soru için --ids a01,f03)
uv run python main.py eval --retrieval-only # yalnızca arama ve eşik, sohbet modeli yok, saniyeler
uv run python main.py compare-reports reports/eski.json reports/eval-qwen2.5-7b-cpu.json
uv run python main.py feedback              # web arayüzünde 👎 verilen cevaplar
uv run python main.py log --refused         # reddedilen sorular, bütün arayüzlerden
uv run python main.py status                # indeks, önbellek, oylar, indekslenmemiş dosyalar
uv run python main.py docs                  # indeksteki dokümanlar
uv run python main.py remove-doc efes.md    # data/docs/'tan ve indeksten sil
uv run python main.py cache --clear         # cevap önbelleğini boşalt
uv run python main.py config                # geçerli ayarlar ve her birinin nereden geldiği
uv run python main.py serve                 # 127.0.0.1:8765 üzerinde yerel HTTP API
uv run pytest                               # model gerekmez, ~1 sn
```

Web arayüzünde kenar çubuğundaki "Sohbeti indir (.md)" sohbeti Markdown dosyası olarak kaydediyor;
her cevabın altında atıf yapılan parçalar ve Vikipedi bağlantıları da yer alıyor. "Sohbeti indir
(.json)" aynı turları benzerlik skorları ve destekleyen cümlelerle birlikte yazıyor. `chat` içinde
`kaydet` Markdown halini kaydediyor (dosyayı seçmek için `kaydet notlar/ani.md`), `kaynak` son
cevabın dayandığı parçaları tam metinle basıyor (tek bir parça için `kaynak K2`); yukarı ok tuşu da
önceki oturumlardaki soruları getiriyor (`~/.miras-rag/chat_history` içinde tutuluyor).

Cevaplar önbelleğe alındığı için kötü bir cevap da her seferinde geri gelirdi. Web arayüzünde son
cevabın altındaki "Yeniden üret" (ya da `ask --refresh`) modele yeniden soruyor ve önbellekteki
cevabı değiştiriyor; eski cevaba verilen oy ayrı olarak kalıyor.

`ask-batch` bir dosyanın her satırını ayrı bir soru olarak soruyor (boş ve `#` ile başlayan satırlar
atlanıyor, `-` verilirse standart girdiden okuyor) ve her cevap için `ask --json` biçiminde bir JSON
satırı yazıyor. Satırlar arasında konu taşınmıyor.

`serve` başka araçlar da asistanı kullanabilsin diye küçük bir HTTP API başlatıyor (yalnızca
standart kütüphane, ek bağımlılık yok). Varsayılan olarak `127.0.0.1:8765` dinliyor (`--host`,
`--port`). `--token` (ya da `MIRAS_API_TOKEN`) verilirse `/health` dışındaki her uç nokta
`Authorization: Bearer <token>` istiyor; token yoksa localhost dışında bir adrese bağlanınca
uyarıyor. `--cors http://localhost:3000` o adresteki bir sayfanın tarayıcıdan çağırabilmesini
sağlıyor. Her istek durum ve süresiyle stderr'e yazılıyor (`--quiet` kapatıyor). İstekler modele
sırayla gidiyor.

```bash
curl -s localhost:8765/ask -d '{"question": "Ani hangi ilde?"}'
curl -s localhost:8765/ask -d '{"question": "Peki ne zaman yapılmış?", "topic": ["Ani"]}'
curl -s localhost:8765/ask -d '{"question": "Tiyatro kaç kişilik?", "docs": ["efes"], "refresh": true}'
curl -s localhost:8765/search -d '{"question": "Efes mi Troya mı daha eski?"}'
curl -sN localhost:8765/ask -d '{"question": "Ani hangi ilde?", "stream": true}'
curl -s localhost:8765/docs -H "Authorization: Bearer $MIRAS_API_TOKEN"
curl -s localhost:8765/health
```

`POST /ask` cevabı `ask --json` biçiminde veriyor, ek olarak `docs` ve `topic` alanlarıyla.
`"stream": true` verilirse NDJSON gönderiyor: model yazdıkça parça başına bir `{"delta": "..."}`
satırı, sonda da doğrulanmış cevapla (temizlenmiş atıflar, kaynaklar) `{"done": true, ...}`; model
yarıda hata verirse son satır `{"error": ...}` oluyor. `/health` önbellekteki cevap ve günlükteki
soru sayılarını da veriyor. Sunucu
sohbet tutmuyor; takip sorusu için önceki cevabın `topic` alanı geri gönderiliyor. `POST /search`
sohbet modelini çağırmadan parçaları, benzerliklerini, tanıdığı alan adlarını ve sorunun eşiği geçip
geçmediğini döndürüyor. Hatalar `{"error": "..."}` olarak geliyor: 400 (bozuk JSON, boş soru,
bilinmeyen doküman), 401 (eksik ya da yanlış token), 404/405, 413 (64 KB'den büyük gövde) ya da 503
(model hatası).

`search` aramayı tek başına incelemek için: yalnızca embedding modelini yükleyip sohbet modeline
gidecek parçaları, benzerliklerini, soruda tanıdığı alan adlarını ve sorunun eşiği geçip
geçmediğini yazıyor. Kaç parça getirileceği `-k` ile değişiyor.

`ask`, `search`, `ask-batch` ve `chat` için `--doc` aramayı bazı dokümanlarla sınırlıyor. Dosya adı
(`efes.md` ya da `efes`), başlık (`"Nemrut Dağı"`) ya da takma ad (`Truva`) kabul ediyor; tekrar
verilebiliyor ya da virgülle ayrılabiliyor, tanınmayan bir ad indeksteki dosyaların listesiyle
birlikte hata veriyor. `chat` içinde `sadece efes, troya` kısıtı koyuyor, tek başına `sadece`
kaldırıyor; web arayüzünde kenar çubuğundaki "Sadece şu alanlarda ara". Eşik yine geçerli, yani
seçimin dışındaki bir alan sorulursa model çağrılmadan reddediliyor. Markdown ve JSON dışa aktarımı
kısıtı da yazıyor.

Her cevabın altında bir de 👍/👎 var. Oylar soru, cevap ve getirilen parçalarla birlikte aynı SQLite
dosyasına yazılıyor (`ingest --rebuild` silmiyor). `feedback` 👎 verilenleri listeliyor (hepsi için
`--all`), `feedback --export reports/feedback-candidates.json` ise bunları `questions.json`
biçiminde yazıyor; böylece test setine soru olarak eklenebiliyorlar. Oy, doğru davranışın başka bir
cevap mı yoksa reddetmek mi olduğunu söylemediği için `type` ve `expected` elle dolduruluyor; konu
taşınmış sorular `carried` alanıyla işaretli, onların yeri `followups.json`.

Oylar yalnızca birinin değerlendirdiği cevapları kapsadığı için her soru ayrıca günlüğe yazılıyor
(CLI, web arayüzü ve API; `eval` hariç): yazıldığı haliyle soru, taşınan konu, doküman kısıtı,
reddedildiyse sebebi, en iyi benzerlik ve atıf yapılan dosyalar. Cevap metinleri saklanmıyor. `log`
son soruları gösteriyor (`-n 50`), `log --refused` yalnızca reddedilenleri; dokümanların kapsamadığı
ne sorulduğunu görmenin en hızlı yolu bu. `log --export reports/log-adaylar.json` bunları
`questions.json` biçiminde yazıyor. Son reddedilenler web arayüzünde kenar çubuğundaki "Reddedilen
sorular" bölümünde de görünüyor. `log --clear` boşaltıyor, `MIRAS_QUERY_LOG=0` kapatıyor,
`ingest --rebuild` dokunmuyor.

Testler modeller yerine sahte bir arka uç kullandığı için her push'ta GitHub Actions'ta da
çalışıyor (Python 3.12 ve 3.13, Foundry Local SDK olmadan).

Web arayüzünde kenar çubuğundaki "Arama ayarları" bölümünde `top_k`, eşik ve BM25 ağırlığı için
kaydırıcılar var. Yalnızca o tarayıcı oturumunu değiştiriyorlar (yüklü modeller ve indeks
paylaşılıyor); varsayılanı değiştirmeden önce farklı bir eşiğin bir soruya ne yapacağını görmek
kolaylaşıyor. Değişmiş ayarlarla verilen cevaplar bunu cevabın altında ve dışa aktarımda belirtiyor.

Ayarlar `miras/config.py` içinde; `MIRAS_CHAT_MODEL`, `MIRAS_TOP_K`, `MIRAS_MIN_SIMILARITY`,
`MIRAS_BM25_WEIGHT`, `MIRAS_FOLLOWUP_MIN_SIMILARITY`, `MIRAS_ANSWER_CACHE`, `MIRAS_QUERY_LOG`, `MIRAS_DOCS_DIR` gibi ortam değişkenleriyle değiştirilebiliyor. Embedding modelini değiştirirseniz
`ingest --rebuild` çalıştırmak gerekiyor. Anlamsız değerler (`MIRAS_TOP_K=0`, 1'den büyük bir eşik,
`MIRAS_TOP_K=dört` gibi bir yazım hatası) aramayı sessizce bozmak yerine açılışta hangi ayarın
hatalı olduğunu söyleyen bir mesajla durduruyor.

HTML dosyaları (`.html`, `.htm`, ör. kaydedilmiş bir web sayfası) parçalanmadan önce standart
kütüphaneyle Markdown'a çevriliyor: ilk `<h1>` (ya da `<title>`) başlık, `<h2>` ve altı bölüm
oluyor; menüler, betikler, üst ve alt bilgiler atılıyor. Dosyanın kendisine dokunulmuyor.

Dosyaların UTF-8 olması gerekiyor (Windows satır sonları sorun değil). UTF-8 olmayan bir dosyayı
`ingest` uyarı vererek atlıyor ve önceki hali indekste kalıyor; üst bilgisi dışında boşaltılan bir
dosya indeksten çıkarılıyor.

Kendi dokümanlarınızla denemek için `.md`, `.txt` ya da `.html` dosyalarını `data/docs/` klasörüne atıp
`ingest` çalıştırmanız ya da web arayüzünün kenar çubuğundan ("Doküman ekle") yüklemeniz yeterli.
Yüklenen dosyalar aynı klasöre kaydedilip hemen indeksleniyor, sadece değişen dosyalar yeniden
işleniyor. Aynı yerdeki "Doküman sil" (ya da `remove-doc`) bir dosyayı `data/docs/`'tan ve indeksten
kaldırıyor (yeniden gömme gerekmiyor); en az bir doküman kalması gerekiyor. Web arayüzü açıkken elle
düzenlenen ya da silinen dosyalar kenar çubuğunda "İndeksi güncelle" düğmesiyle birlikte uyarı
olarak çıkıyor, `status` da bunları listeliyor. Cevap önbelleği oradaki "Önbelleği temizle" ya da
`cache --clear` ile boşaltılabiliyor. "Dokümanı incele" bir dokümanın hangi parçalara bölündüğünü
başlık yollarıyla gösteriyor; yeni parçalar gömülürken web arayüzünde bir ilerleme çubuğu çıkıyor.

Türkçe Vikipedi maddeleri adıyla ya da bağlantısıyla da eklenebiliyor: kenar çubuğunda
"Vikipedi'den ekle" ya da `add-wiki`. Madde indirilip `data/docs/` içindekilerle aynı şekilde
dönüştürülüyor (kaynakça gibi bölümler atılıyor, kaynak ve revizyonlu üst bilgi ekleniyor) ve
indeksleniyor. İnternete çıkan tek şey bu, o da sadece düğmeye basınca ya da komut çalışınca.
Yazılan ad bir yönlendirmeyse ("Truva" -> Troya) takma ad olarak ekleniyor, anlam ayrımı sayfaları
reddediliyor; zaten ekli bir madde tekrar eklenirse güncelleniyor ve takma adları korunuyor.

`update-wiki` (ya da kenar çubuğundaki "Vikipedi güncellemeleri") Vikipedi kaynaklı her dokümanın
üst bilgisindeki revizyon numarasını, hepsi için tek bir istekle, en son revizyonla karşılaştırıp
değişenleri listeliyor. `--check` verilmezse bunlar aynı dosya adıyla ve takma adları korunarak
yeniden indirilip indeksleniyor. Artık bulunamayan bir madde yalnızca bildiriliyor, dosyasına
dokunulmuyor.

Doküman ekleyip sildikten sonra `calibrate`
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

`data/eval/questions.json` içinde 48 soru var: 30'u dokümanlardan cevaplanabilir (4'ü iki alanı
karşılaştırıyor; bunların `doc` alanı liste ve atıf ancak iki dokümana da yapılırsa sayılıyor),
10'u konuyla ilgili ama dokümanlarda yok (reddedilmeli), 8'i tamamen konu dışı. Sorular `dev` ve `test` olarak
ikiye ayrılmış durumda; eşiği ve prompt'u ayarlarken sadece `dev` sorularına baktım.

CPU'da `qwen2.5-7b` ile 44/44 aldı (test kısmında 22/22), atıfların hepsi doğru dokümana gidiyordu.
Model çağrılan sorularda ortanca süre 14 saniye civarı. Soru soru cevapların olduğu rapor
[`reports/eval-qwen2.5-7b-cpu.md`](reports/eval-qwen2.5-7b-cpu.md) dosyasında. O çalıştırma
karşılaştırma soruları eklenmeden önceki 44 soruyla yapıldı.

Bu sonucu fazla büyütmemek lazım: set küçük, soruları ben yazdım ve puanlama anahtar kelimeye
dayanıyor. Bu tip sorularda iyi çalıştığını gösteriyor, her soruda doğru cevap vereceğini değil.

Takip soruları için ayrı küçük bir set de var (`data/eval/followups.json`): 29 iki turluk sohbet,
sadece ikinci soru puanlanıyor (19'u cevaplanabilir, 10'u reddedilmeli, bunların 7'si konu dışı). `eval` bunları ana setten
sonra çalıştırıp rapora ayrı bir bölüm ekliyor; takip sorusu tek başına sorulsaydı doğru dokümanın
kaçıncı sırada geleceği de orada. Yukarıdaki rapor bu setten önce üretildi, rakamlar için `eval`
yeniden çalıştırılmalı. Konu dışı takip sorularından kaçının model çağrılmadan reddedildiği de
raporda.

Raporda doküman bazında bir tablo da var (yalnızca cevaplanabilir sorular: doğru, reddedilen, doğru
dokümana atıf); 25/26 gibi bir toplam, kaçırılan sorunun hangi maddeden geldiğini söylemiyor. Ayar
yaparken `eval --split dev` test sorularını görmeden çalıştırıyor, `eval --ids a07,f12` de yalnızca
üzerinde çalışılan soruları yeniden soruyor; ikisi de ayrı bir rapora yazıyor (`...-dev`,
`...-secili`), tam rapor ezilmiyor. `compare-reports eski.json yeni.json` değişen ayarları, önceki
ve sonraki skorları ve arada düzelen ya da bozulan soruları listeliyor.

Tam bir `eval` CPU'da on dakika kadar sürüyor, çoğu sohbet modelinde geçiyor. `eval
--retrieval-only` onu atlayıp yalnızca embedding modelini yüklüyor: doğru dokümanın 1. sırada ve
ilk k içinde olması, MRR, cevabın getirilen parçalarda olup olmadığı, doküman bazında tablo, eşiğin
neyi geçirip neyi durduracağı ve takip setinde konunun taşınıp taşınmadığı. `reports/arama-<embedding
modeli>.md/.json` dosyasına yazıyor, `--split` ve `--ids` de çalışıyor. Parçalamayı, `top_k`'yı,
BM25 ağırlığını ya da takip kurallarını değiştirdikten sonra hızlı kontrol için bu.

### Eksikler

- Yavaş. CPU'da cevap başına ~14 sn, akış halinde geldiği için biraz daha az hissediliyor (tekrar eden sorular önbellekten geliyor).
  GPU seçeneği var (`--device gpu` ya da `MIRAS_CHAT_DEVICE=gpu`) ama test etmedim.
- Takip sorularında sadece alan taşınıyor, sohbetin geri kalanı değil. Sorunun konu dışı olup
  olmadığına tek bir benzerlik eşiği, kısa bir gönderme kelimeleri listesi ve kelimelerinin alanın
  dokümanında geçip geçmediği karar veriyor. Maddede geçmeyen bir kelime kullanan kısa bir soru
  ("Peki ne zaman?", madde hiç "zaman" demiyorsa) yine eşiğin altında kalıp konuyu kaybedebilir;
  gönderme kelimesi içeren konu dışı bir soru ("Orada yarın hava nasıl?") ise alan adını alıp
  modelin reddetmesine kalır.
- Karşılaştırma sorularında adı geçen her alan için yalnızca bir parça garanti (geri kalanı
  sıralamaya göre). Bir yıl ya da il için yetiyor, ama bir alanın cevabı sıralamada geride kalan bir
  parçadaysa yine kaçabilir. Değerlendirmedeki 4 karşılaştırma sorusu henüz gerçek modellerle
  çalıştırılmadı.
- Ad eşleştirme sözlüğe dayalı. Kısa adların (Ani, Efes, Troya) doğru yazılması gerekiyor; iki
  harften büyük hatalarda ("Gbklitpe") sadece embedding araması kalıyor.
- Cevaplar `data/docs/` içindeki Vikipedi revizyonları kadar doğru (`update-wiki` onları
  güncelliyor, ama değerlendirme seti şu anki revizyonlara göre yazıldı).

### Kaynaklar ve lisans

- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Microsoft Tech Community)
- [Foundry Local belgeleri](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
- `data/docs/` içindeki maddeler [Türkçe Vikipedi](https://tr.wikipedia.org) katkıcılarına ait ve
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisansıyla paylaşılıyor. Her
  dosyanın başında kaynak adresi ve revizyon numarası var.
