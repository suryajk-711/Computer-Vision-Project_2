import os
import sys
import tempfile
from pathlib import Path
from flask import Flask, request, render_template_string, send_file

app = Flask(__name__)

SRC_DIR       = Path(__file__).parent.resolve()
PROJECT_ROOT  = SRC_DIR.parent.resolve()
TEMPLATES_DIR = PROJECT_ROOT / "templates"
OUTPUT_DIR    = SRC_DIR / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_on_image

_last_output = {}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Sign Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: monospace;
            background: #f7f7f5;
            color: #1a1a1a;
            min-height: 100vh;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 48px 20px;
        }

        .card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 36px;
            width: 100%;
            max-width: 560px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        h1 {
            font-size: 1.25rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border-bottom: 2px solid #1a1a1a;
            padding-bottom: 10px;
            margin-bottom: 6px;
        }

        .subtitle {
            font-size: 0.78rem;
            color: #888;
            margin-bottom: 28px;
        }

        /* Upload area */
        .upload-area {
            border: 1.5px dashed #bbb;
            border-radius: 4px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            margin-bottom: 14px;
            position: relative;
        }
        .upload-area:hover { border-color: #555; }
        .upload-area input[type=file] {
            position: absolute; inset: 0;
            opacity: 0; cursor: pointer;
            width: 100%; height: 100%;
        }
        .upload-area .icon   { font-size: 1.6rem; margin-bottom: 6px; }
        .upload-area .hint   { font-size: 0.8rem; color: #888; }
        .upload-area .chosen { font-size: 0.82rem; color: #333; margin-top: 6px; font-weight: bold; }

        button {
            width: 100%;
            padding: 10px;
            background: #1a1a1a;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
            cursor: pointer;
            letter-spacing: 0.05em;
            transition: background 0.2s;
        }
        button:hover    { background: #333; }
        button:disabled { background: #999; cursor: not-allowed; }

        .loading {
            display: none;
            font-size: 0.78rem;
            color: #888;
            margin-top: 10px;
            text-align: center;
        }

        /* Result section */
        .result {
            margin-top: 28px;
            padding-top: 24px;
            border-top: 1px solid #e0e0e0;
        }

        .result-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            padding: 6px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .result-row .label { color: #888; }
        .result-row .value { font-weight: bold; }

        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 3px;
            font-size: 0.75rem;
        }
        .badge-ok  { background: #e6f4ea; color: #2d7a3a; }
        .badge-err { background: #fdecea; color: #b71c1c; }

        .predicted-label {
            font-size: 1.1rem;
            font-weight: bold;
            margin: 16px 0 10px;
        }

        .annotated-img {
            width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-top: 4px;
        }

        details { margin-top: 16px; }
        summary {
            font-size: 0.78rem;
            color: #888;
            cursor: pointer;
            user-select: none;
        }
        pre {
            background: #f4f4f2;
            border: 1px solid #e8e8e8;
            border-radius: 4px;
            padding: 12px;
            font-size: 0.74rem;
            white-space: pre-wrap;
            overflow-x: auto;
            margin-top: 8px;
            max-height: 200px;
            overflow-y: auto;
        }

        .error-msg {
            background: #fdecea;
            border: 1px solid #f5c6c6;
            color: #b71c1c;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.82rem;
            margin-top: 12px;
        }
    </style>

    <script>
        function onFileChange(input) {
            const label = document.getElementById('chosen-name');
            label.textContent = input.files[0] ? input.files[0].name : '';
        }
        function onSubmit() {
            const btn = document.getElementById('submit-btn');
            btn.disabled = true;
            btn.textContent = 'Running...';
            document.getElementById('loading').style.display = 'block';
        }
    </script>
</head>
<body>
<div class="card">

    <h1>Traffic Sign Detection</h1>
    <p class="subtitle">Classical CV pipeline - SIFT feature matching</p>

    <form method="POST" enctype="multipart/form-data"
          action="/predict" onsubmit="onSubmit()">
        <div class="upload-area">
            <input type="file" name="image" accept="image/*"
                   required onchange="onFileChange(this)">
            <div class="icon">📷</div>
            <div class="hint">Click to choose an image</div>
            <div class="chosen" id="chosen-name"></div>
        </div>
        <button type="submit" id="submit-btn">Run Detection</button>
        <div class="loading" id="loading">Processing - this may take a moment…</div>
    </form>

    {% if result %}
    <div class="result">

        {% if result.error %}
            <div class="error-msg">{{ result.error }}</div>

        {% else %}
            <div class="result-row">
                <span class="label">File</span>
                <span class="value">{{ result.filename }}</span>
            </div>
            <div class="result-row">
                <span class="label">Status</span>
                <span class="value">
                    <span class="badge {{ 'badge-ok' if result.status == 'ok' else 'badge-err' }}">
                        {{ result.status }}
                    </span>
                </span>
            </div>

            <div class="predicted-label">
                Predicted: {{ result.predicted or "no match" }}
            </div>

            {% if result.has_output_image %}
                <img class="annotated-img" src="/output_image" alt="annotated output">
            {% endif %}



        {% endif %}
    </div>
    {% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template_string(HTML, result={
            "error": "No file uploaded.", "filename": ""
        })

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        pipeline_result = run_on_image(
            image_path=tmp_path,
            templates_dir=str(TEMPLATES_DIR),
            output_dir=str(OUTPUT_DIR),
        )

        if pipeline_result is None:
            result = {
                "filename":         file.filename,
                "predicted":        None,
                "status":           "no_match",
                "has_output_image": False,
                "error":            None,
            }
        else:
            predicted = pipeline_result.get("predicted")
            status    = pipeline_result.get("status", "no_match")

            has_output_image = False
            if predicted:
                class_dir = OUTPUT_DIR / predicted
                if class_dir.exists():
                    images = sorted(class_dir.glob("*.jpg"))
                    if images:
                        _last_output["path"] = str(images[-1])
                        has_output_image = True

            result = {
                "filename":         file.filename,
                "predicted":        predicted,
                "status":           status,
                "has_output_image": has_output_image,
                "error":            None,
            }

    except Exception as e:
        result = {
            "filename": file.filename,
            "error":    str(e),
            "predicted": None,
        }
    finally:
        os.unlink(tmp_path)

    return render_template_string(HTML, result=result)


@app.route("/output_image")
def output_image():
    """Serve the latest annotated output image produced by main.py."""
    path = _last_output.get("path")
    if path and Path(path).exists():
        return send_file(path, mimetype="image/jpeg")
    return "Image not found", 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)