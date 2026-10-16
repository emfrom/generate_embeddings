# generate_embeddings

A small command-line wrapper for generating text embeddings with
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The first run uses `uv` to prepare the daemon's Python dependencies and model.

```sh
./generate_embeddings "hello world"
echo "hello world" | ./generate_embeddings -
./generate_embeddings --json "hello world"
```

A Unix-socket daemon is started automatically when needed. Manage it explicitly
with `./generate_embeddings --daemon [MINUTES]`; use
`./generate_embeddings.daemon --stop` to stop it. `--daemon` without a timeout
runs until stopped.

Licensed under the MIT License; see [LICENSE](LICENSE).
