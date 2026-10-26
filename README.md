# generate_embeddings

A small command-line wrapper for generating text embeddings with
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The first run uses `uv` to prepare the daemon's Python dependencies and model.

```sh
./generate_embeddings "hello world"
echo "hello world" | ./generate_embeddings -
printf 'first\\0second\\0' | ./generate_embeddings --json -0 -
./generate_embeddings --json "hello world"
```

NUL-delimited stdin (`-0 -`) is streamed in 512-text requests. A one-batch read-ahead buffer overlaps input reading with embedding, bounding buffered work to two batches; each completed batch is emitted immediately. The daemon loads the CPU model at startup for one-shot requests; the CUDA model is loaded on the first batch request. Model instances are shared by concurrent request threads; CPU inference uses an internal batch size of 16 for low latency, while GPU inference uses 128 for throughput. A Unix-socket daemon is started automatically when needed. Manage it explicitly
with `./generate_embeddings --daemon [MINUTES]`; this explicit launch runs
until stopped if no timeout is given. Automatic startup uses a 5-minute timeout.
Use `./generate_embeddings.daemon --stop` to stop it. Directly launching
`generate_embeddings.daemon` without a timeout also runs until stopped.

### Throughput benchmark

On this machine (Python 3.13.14, CPU), 12,541 representative short texts were embedded using the same daemon/model: individual Unix-socket requests took 91.3 s; two large batch requests (10,000 + 2,541) took 26.8 s (3.4× faster). That earlier comparison established the value of batching; larger 512-text requests reduce socket overhead while keeping read-ahead bounded. Model warm-up was excluded.
