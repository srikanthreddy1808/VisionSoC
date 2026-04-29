import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

import RRDBNet_arch as arch

# ── App config ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']  = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# ── Load ESRGAN model once at startup ────────────────────────────────────────
MODEL_PATH = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
model.eval()
model = model.to(device)
print(f"✅ Model loaded on {device}")


# ── Helper functions ─────────────────────────────────────────────────────────
def allowed_file(filename):
    """Check that the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def enhance_image(input_path, output_path):
    """
    Core ESRGAN pipeline:
    1. Read image with OpenCV (BGR format)
    2. Normalise to [0,1] float32
    3. Convert BGR → RGB, HWC → CHW, add batch dim
    4. Run through model
    5. Reverse all transforms and save
    """
    # 1. Read
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    # 2. Normalise: uint8 [0,255] → float32 [0,1]
    img = img.astype(np.float32) / 255.0
    # 3. HWC (H,W,C) → CHW → add batch dim → tensor
    img_tensor = torch.from_numpy(
        np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))   # BGR→RGB, HWC→CHW
    ).float().unsqueeze(0).to(device)                    # shape: (1,3,H,W)

    # 4. Inference (no gradient needed)
    with torch.no_grad():
        output = model(img_tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # 5. CHW → HWC, RGB → BGR, [0,1] → [0,255]
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB→BGR, CHW→HWC
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, output)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/enhance', methods=['POST'])
def enhance():
    """
    POST /enhance
    Expects: multipart/form-data with field 'image'
    Returns: JSON { success, original, enhanced, message }
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'File type not allowed'}), 400

    filename   = secure_filename(file.filename)
    input_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'input_'  + filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)

    file.save(input_path)

    try:
        enhance_image(input_path, output_path)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

    return jsonify({
        'success':  True,
        'original': f'/static/uploads/input_{filename}',
        'enhanced': f'/static/uploads/output_{filename}',
        'message':  'Image enhanced successfully!'
    })


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)