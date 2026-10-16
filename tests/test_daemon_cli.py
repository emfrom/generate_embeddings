import json
import os
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "generate_embeddings"


class EmbeddingHandler(socketserver.StreamRequestHandler):
    requests = []

    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        request = json.loads(line)
        self.requests.append(request)
        self.wfile.write(b'{"embedding":[1.0,2.0]}\n')


class DaemonCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.bin = self.home / "bin"
        self.bin.mkdir()
        self.pythonpath = self.home / "python"
        self.pythonpath.mkdir()
        (self.pythonpath / "sentence_transformers.py").write_text(
            "class SentenceTransformer:\n"
            "    def __init__(self, *args, **kwargs): pass\n"
            "    def encode(self, texts, **kwargs): return [[1.0, 2.0] for _ in texts]\n"
        )
        (self.pythonpath / "numpy.py").write_text("")
        uv = self.bin / "uv"
        uv.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "[[ $1 == run && $2 == --script && $3 == - ]]\n"
            "shift 3\n"
            "script=$(mktemp)\n"
            "trap 'rm -f \"$script\"' EXIT\n"
            "cat > \"$script\"\n"
            "exec python3 \"$script\" \"$@\"\n"
        )
        uv.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            HOME=str(self.home),
            PATH=f"{self.bin}:{os.environ['PATH']}",
            PYTHONPATH=str(self.pythonpath),
        )
        self.sock = self.home / ".cache/generate_embeddings/daemon.sock"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            [str(SCRIPT), *args], env=self.env, text=True,
            capture_output=True, timeout=15,
        )

    def test_existing_daemon_receives_text_request(self):
        self.sock.parent.mkdir(parents=True)
        server = socketserver.UnixStreamServer(str(self.sock), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        EmbeddingHandler.requests.clear()
        try:
            result = self.run_cli("hello")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "1.0 2.0")
            self.assertEqual(EmbeddingHandler.requests, [{"text": "hello"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def daemon_pid(self):
        inode = None
        with open("/proc/net/unix") as unix_sockets:
            for line in unix_sockets:
                fields = line.split()
                if len(fields) >= 8 and fields[-1] == str(self.sock):
                    inode = fields[6]
                    break
        if inode is None:
            return None
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                fds = os.listdir(f"/proc/{entry}/fd")
                if any(os.readlink(f"/proc/{entry}/fd/{fd}") == f"socket:[{inode}]" for fd in fds):
                    return int(entry)
            except OSError:
                continue
        return None

    def test_missing_daemon_starts_and_handles_request(self):
        try:
            result = self.run_cli("hello")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "1.0 2.0")
            self.assertTrue(self.sock.exists())
        finally:
            pid = self.daemon_pid()
            if pid:
                os.kill(pid, 15)
                deadline = time.monotonic() + 5
                while self.sock.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.assertFalse(self.sock.exists(), "autostart daemon should remove its socket on shutdown")


if __name__ == "__main__":
    unittest.main()
