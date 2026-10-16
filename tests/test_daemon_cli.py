import json
import os
import pty
import socket
import socketserver
import shutil
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
            "import os\n"
            "class SentenceTransformer:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        with open(os.environ['MODEL_INIT_MARKER'], 'a') as marker: marker.write('loaded\\n')\n"
            "    def encode(self, texts, **kwargs): return [[1.0, 2.0] for _ in texts]\n"
        )
        (self.pythonpath / "numpy.py").write_text("")
        uv = self.bin / "uv"
        uv.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "[[ $1 == run && $2 == --script ]]\n"
            "shift 2\n"
            "script=$1; shift\n"
            "echo \"$script\" >> \"$UV_SEEN\"\n"
            "if [[ $script == - ]]; then\n"
            "  script=$(mktemp)\n"
            "  trap 'rm -f \"$script\"' EXIT\n"
            "  cat > \"$script\"\n"
            "fi\n"
            "exec python3 \"$script\" \"$@\"\n"
        )
        uv.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            HOME=str(self.home),
            PATH=f"{self.bin}:{os.environ['PATH']}",
            PYTHONPATH=str(self.pythonpath),
            UV_SEEN=str(self.home / "uv-seen"),
            MODEL_INIT_MARKER=str(self.home / "model-init"),
        )
        self.sock = self.home / ".cache/generate_embeddings/daemon.sock"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, input=None):
        return subprocess.run(
            [str(SCRIPT), *args], env=self.env, text=True, input=input,
            capture_output=True, timeout=15,
        )

    def run_daemon(self, *args):
        return subprocess.run(
            [str(ROOT / "generate_embeddings.daemon"), *args], env=self.env,
            text=True, capture_output=True, timeout=15,
        )

    def run_daemon_in_terminal(self, *args):
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(ROOT / "generate_embeddings.daemon"), *args], env=self.env,
            stdout=subprocess.PIPE, stderr=slave,
        )
        os.close(slave)
        stdout, _ = process.communicate(timeout=15)
        chunks = []
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        os.close(master)
        return process.returncode, stdout.decode(), b"".join(chunks).decode()

    def test_cli_executes_through_uv_script_shebang(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(SCRIPT), Path(self.env["UV_SEEN"]).read_text())

    def test_version_switch_prints_version(self):
        result = self.run_cli("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1.0.0")

    def test_json_switch_formats_embedding_as_json(self):
        self.sock.parent.mkdir(parents=True)
        server = socketserver.UnixStreamServer(str(self.sock), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        EmbeddingHandler.requests.clear()
        try:
            result = self.run_cli("--json", "hello")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [1.0, 2.0])
            self.assertEqual(EmbeddingHandler.requests, [{"text": "hello"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_stdin_switch_sends_stdin_text(self):
        self.sock.parent.mkdir(parents=True)
        server = socketserver.UnixStreamServer(str(self.sock), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        EmbeddingHandler.requests.clear()
        try:
            result = self.run_cli("-", input="hello")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(EmbeddingHandler.requests, [{"text": "hello"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_nul_switch_embeds_each_stdin_record(self):
        self.sock.parent.mkdir(parents=True)
        server = socketserver.UnixStreamServer(str(self.sock), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        EmbeddingHandler.requests.clear()
        try:
            result = self.run_cli("-", "-0", input="first" + chr(0) + "second" + chr(0))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 2)
            self.assertEqual(EmbeddingHandler.requests, [{"text": "first"}, {"text": "second"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_double_dash_treats_control_tokens_as_text(self):
        self.sock.parent.mkdir(parents=True)
        server = socketserver.UnixStreamServer(str(self.sock), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        EmbeddingHandler.requests.clear()
        try:
            result = self.run_cli("--", "--json", "-")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "1.0 2.0")
            self.assertEqual(EmbeddingHandler.requests, [{"text": "--json -"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_daemon_switch_defaults_to_unlimited(self):
        isolated = self.home / "isolated"
        isolated.mkdir()
        cli = isolated / "generate_embeddings"
        shutil.copy2(SCRIPT, cli)
        daemon = isolated / "generate_embeddings.daemon"
        daemon.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "open(os.environ['DAEMON_ARGS_FILE'], 'w').write(json.dumps(sys.argv[1:]))\n"
        )
        daemon.chmod(0o755)
        env = self.env.copy()
        env["DAEMON_ARGS_FILE"] = str(self.home / "daemon-args")
        result = subprocess.run([str(cli), "--daemon"], env=env, text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(env["DAEMON_ARGS_FILE"]).exists(), result.stderr)
        self.assertEqual(json.loads(Path(env["DAEMON_ARGS_FILE"]).read_text()), ["--timeout", "0"])

    def test_automatic_start_uses_five_minute_timeout(self):
        isolated = self.home / "autostart"
        isolated.mkdir()
        cli = isolated / "generate_embeddings"
        shutil.copy2(SCRIPT, cli)
        daemon = isolated / "generate_embeddings.daemon"
        daemon.write_text("""#!/usr/bin/env python3
import json, os, socket, sys
from pathlib import Path
Path(os.environ['DAEMON_ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))
if os.fork():
    sys.exit(0)
os.setsid()
path = Path.home() / '.cache/generate_embeddings/daemon.sock'
path.parent.mkdir(parents=True)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
    server.bind(str(path))
    server.listen()
    while True:
        connection, _ = server.accept()
        with connection, connection.makefile('rwb') as channel:
            line = channel.readline()
            if not line:
                continue
            channel.write(b'{\"embedding\":[1.0,2.0]}\\n')
            break
path.unlink()
""")
        daemon.chmod(0o755)
        env = self.env.copy()
        env["DAEMON_ARGS_FILE"] = str(self.home / "autostart-args")
        result = subprocess.run([str(cli), "hello"], env=env, text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1.0 2.0")
        self.assertEqual(json.loads(Path(env["DAEMON_ARGS_FILE"]).read_text()), ["--timeout", "5"])

    def test_daemon_switch_launches_daemon_with_timeout(self):
        result = self.run_cli("--daemon", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Starting daemon...", result.stderr)
        self.assertNotIn("✓", result.stderr)
        self.assertTrue(self.sock.exists())
        stopped = self.run_daemon("--stop")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse(self.sock.exists())

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

    def test_cli_daemon_start_propagates_failure_status(self):
        started = self.run_daemon("--timeout", "10")
        self.assertEqual(started.returncode, 0, started.stderr)
        try:
            failed = self.run_cli("--daemon", "10")
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Error:", failed.stderr)
        finally:
            self.run_daemon("--stop")

    def test_daemon_actions_report_progress_and_success(self):
        code, _, start_message = self.run_daemon_in_terminal("--timeout", "10")
        self.assertEqual(code, 0, start_message)
        try:
            self.assertIn("Starting daemon...", start_message)
            self.assertIn("✓ Daemon started", start_message)
            code, _, stop_message = self.run_daemon_in_terminal("--stop")
            self.assertEqual(code, 0, stop_message)
            self.assertIn("Stopping daemon...", stop_message)
            self.assertIn("✓ Daemon stopped", stop_message)
            self.assertFalse(self.sock.exists())
        finally:
            if self.sock.exists():
                self.run_daemon("--stop")

    def test_failed_duplicate_start_reports_red_cross(self):
        first = self.run_daemon("--timeout", "10")
        self.assertEqual(first.returncode, 0, first.stderr)
        try:
            code, _, failure_message = self.run_daemon_in_terminal("--timeout", "10")
            self.assertNotEqual(code, 0)
            self.assertIn("Starting daemon...", failure_message)
            self.assertIn("✗", failure_message)
        finally:
            self.run_daemon("--stop")

    def test_daemon_preloads_model_before_reporting_ready(self):
        result = self.run_daemon("--timeout", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            self.assertTrue(Path(self.env["MODEL_INIT_MARKER"]).exists())
            self.assertEqual(Path(self.env["MODEL_INIT_MARKER"]).read_text(), "loaded\n")
        finally:
            self.run_daemon("--stop")

    def test_force_replaces_running_daemon_and_timeout_flag_is_accepted(self):
        first = self.run_daemon("--timeout", "10")
        self.assertEqual(first.returncode, 0, first.stderr)
        first_pid = self.daemon_pid()
        self.assertIsNotNone(first_pid)
        try:
            forced = self.run_daemon("--force", "--timeout", "10")
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertTrue(self.sock.exists())
            self.assertNotEqual(self.daemon_pid(), first_pid)
        finally:
            pid = self.daemon_pid()
            if pid:
                os.kill(pid, 15)
                deadline = time.monotonic() + 5
                while self.sock.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)

    def test_stop_stops_running_daemon_and_removes_socket(self):
        started = self.run_daemon("--timeout", "10")
        self.assertEqual(started.returncode, 0, started.stderr)
        pid = self.daemon_pid()
        self.assertIsNotNone(pid)
        stopped = self.run_daemon("--stop")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse(self.sock.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

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
