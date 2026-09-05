#!/usr/bin/env python3
"""Lightweight local web server for the DeepCycloNet Model Performance Dashboard."""
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        # API Endpoints
        if self.path == "/api/benchmarks":
            self.serve_json(DATA_DIR / "benchmarks.json")
        elif self.path == "/api/storms":
            self.serve_json(DATA_DIR / "sample_storms.json")
        elif self.path == "/api/status":
            ckpts = list(Path("experiments/checkpoints").glob("*/best.pt"))
            status = {
                "active_checkpoints": [str(p.parent.name) for p in ckpts],
                "count": len(ckpts),
                "status": "ready"
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            # Serve static files from static directory
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def serve_json(self, file_path: Path):
        if not file_path.exists():
            self.send_error(404, f"Data file {file_path.name} not found")
            return
        
        with open(file_path, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress spammy log outputs for clean CLI experience
        pass


def run():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"\n=======================================================")
    print(f"  DeepCycloNet Model Performance Dashboard Active!")
    print(f"  Open in Browser: http://localhost:{PORT}")
    print(f"=======================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
