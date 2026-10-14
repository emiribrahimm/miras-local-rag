# Miras

A small RAG assistant that answers Turkish questions about Turkey's UNESCO World Heritage sites,
running entirely on your own machine with [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/).
No API keys, no network calls once the models are downloaded.

I built it as my project for the Microsoft AI Summer Internship ("Local RAG AI Assistant with
Microsoft Foundry Local"). The knowledge base is 12 Turkish Wikipedia articles, so questions and
answers are in Turkish.

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
