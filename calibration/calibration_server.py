import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_DIR / '.venv'
sys.path.insert(0, str(MODULE_DIR))

from empymod_anis_harmonic_calibration import (  # noqa: E402
    CHANNELS,
    TILT_NAMES,
    estimate_tilts_from_harmonics,
    tilt_coeff_to_angle_deg,
)

HOST = '127.0.0.1'
PORT = 8765


class CalibrationHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != '/calibrate':
            self.send_json({'error': 'Unknown endpoint'}, status=404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            heights = np.asarray(payload['heights'], dtype=np.float64)
            harmonics = np.asarray(payload['harmonics'], dtype=np.float64)
            expected_shape = (len(heights), len(CHANNELS), 3)
            if harmonics.shape != expected_shape:
                raise ValueError(f'harmonics shape must be {expected_shape}, got {harmonics.shape}')
            if np.any(~np.isfinite(heights)) or np.any(heights <= 0.0):
                raise ValueError('heights must be finite positive numbers')
            if np.any(~np.isfinite(harmonics)) or np.any(harmonics < 0.0):
                raise ValueError('harmonics must be finite non-negative numbers')
            estimated_tilts, result = estimate_tilts_from_harmonics(harmonics, heights)
            response = {
                'tilts': {
                    name: {
                        'angle_deg': float(tilt_coeff_to_angle_deg(estimated_tilts[name])),
                        'mx_over_mz': float(estimated_tilts[name]),
                    }
                    for name in TILT_NAMES
                },
                'residual_norm': float(np.linalg.norm(result.fun)),
                'success': bool(result.success),
                'message': result.message,
            }
            self.send_json(response)
        except Exception as exc:
            self.send_json({'error': str(exc)}, status=400)

    def log_message(self, format, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    server = ThreadingHTTPServer((HOST, PORT), CalibrationHandler)
    print(f'Calibration server: http://{HOST}:{PORT}')
    print('Open calibration_input_form.html in browser and press "Запустить калибровку".')
    server.serve_forever()
