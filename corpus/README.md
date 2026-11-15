# Corpus

Nothing in this folder is committed (see `.gitignore`).

* `standin-threemen.txt`, `standin-bummel.txt`: the public-domain stand-in
  (Jerome K. Jerome, *Three Men in a Boat* and *Three Men on the Bummel*, from
  Project Gutenberg), downloaded by `tools/get_standin.sh`.
* `hhgttg.txt`: put your own plain-text copy of *The Hitchhiker's Guide to the
  Galaxy* here (UTF-8 or ASCII, any line breaks) and run `./retrain.sh`.
  It is used instead of the stand-in automatically.

A model trained on your own copy stays in `build/` (`build/beeb-llm.ssd`,
`build/BEEB-LLM.MMB.zip`). `retrain.sh` deliberately does not copy it into
`models/`, `discs/` or `web/`, so neither the book nor weights trained on it
end up in the repository or on GitHub Pages.
