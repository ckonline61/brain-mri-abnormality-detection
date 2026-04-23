"""
Brain MRI Abnormality Detection - Flask Web Application
Author : Manoj Kumar Sao | Enrollment: 2452448072 | MCA 4th Sem | IGNOU
"""

import base64
import os
import secrets
import time
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from utils.database import (
    clear_analysis_author,
    count_patient_analyses,
    create_patient,
    create_user,
    delete_patient,
    delete_user,
    get_all_users,
    get_analyses,
    get_analysis_by_id,
    get_dashboard_stats,
    get_patient_by_user_id,
    get_patient_options,
    get_user_by_id,
    get_user_by_username,
    get_user_with_patient_profile,
    init_db,
    log_action,
    save_analysis,
    update_patient,
    update_user,
)
from utils.preprocessing import (
    compute_anomaly_score,
    compute_reconstruction_error,
    generate_heatmap,
    load_image_from_bytes,
    overlay_heatmap,
    skull_strip,
    threshold_anomaly,
)
from utils.reporting import build_pdf_report
from utils.security import (
    create_encrypted_backup,
    restore_encrypted_backup,
    validate_age,
    validate_full_name,
    validate_gender,
    validate_image_bytes,
    validate_password_strength,
    validate_role,
    validate_username,
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-local-dev-secret')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ROLES = {'admin', 'doctor', 'patient'}

MODEL_LOADED = False
autoencoder = None
BINARY_LABELS = ['Normal', 'Abnormal']
RATE_LIMIT_STORE = {}
MODEL_INPUT_SIZE = (224, 224)
AUTOENCODER_THRESHOLD_SCORE = 0.04
AUTOENCODER_THRESHOLD_LOSS = 0.003
AUTOENCODER_THRESHOLD_CLUSTER_PCT = 3.75
init_db()


def try_load_models():
    global MODEL_LOADED, autoencoder, MODEL_INPUT_SIZE
    ae_path = os.path.join(BASE_DIR, 'models', 'autoencoder.h5')
    try:
        import tensorflow as tf

        if os.path.exists(ae_path):
            autoencoder = tf.keras.models.load_model(ae_path, compile=False)
            ae_shape = autoencoder.input_shape
            MODEL_INPUT_SIZE = (int(ae_shape[1]), int(ae_shape[2]))
            MODEL_LOADED = True
            print(f"[INFO] Autoencoder loaded successfully with input size {MODEL_INPUT_SIZE}.")

        if not autoencoder:
            print("[INFO] No saved weights found. Running in demo mode.")
    except Exception as exc:
        autoencoder = None
        MODEL_LOADED = False
        print(f"[WARN] Could not load models: {exc}. Running in demo mode.")


try_load_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def demo_inference(img_array):
    np.random.seed(int(time.time()) % 1000)
    recon = img_array + np.random.normal(0, 0.05, img_array.shape)
    recon = np.clip(recon, 0, 1)
    error = compute_reconstruction_error(img_array, recon)
    anomaly_score = compute_anomaly_score(error)
    heatmap = generate_heatmap(error)
    mask, _ = threshold_anomaly(error)
    overlay = overlay_heatmap(img_array, heatmap)
    probs = np.random.dirichlet(np.ones(2) * 2)
    pred_label = BINARY_LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))
    return recon, error, heatmap, mask, overlay, anomaly_score, pred_label, confidence, probs


def summarize_anomaly_mask(mask):
    mask_uint8 = mask.squeeze().astype(np.uint8)
    total_pixels = float(mask_uint8.size) if mask_uint8.size else 1.0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask_uint8 > 0).astype(np.uint8), 8)
    cluster_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    largest_cluster = max(cluster_areas) if cluster_areas else 0
    largest_cluster_pct = round((largest_cluster / total_pixels) * 100, 2)
    return {
        'cluster_count': len(cluster_areas),
        'largest_cluster_area': largest_cluster,
        'largest_cluster_pct': largest_cluster_pct,
    }


def autoencoder_label_from_metrics(anomaly_score, recon_loss, mask):
    mask_summary = summarize_anomaly_mask(mask)
    is_abnormal = (
        anomaly_score >= AUTOENCODER_THRESHOLD_SCORE
        or recon_loss >= AUTOENCODER_THRESHOLD_LOSS
        or mask_summary['largest_cluster_pct'] >= AUTOENCODER_THRESHOLD_CLUSTER_PCT
    )
    label = 'Abnormal' if is_abnormal else 'Normal'
    confidence = min(
        0.99,
        max(
            anomaly_score / AUTOENCODER_THRESHOLD_SCORE,
            recon_loss / AUTOENCODER_THRESHOLD_LOSS,
            mask_summary['largest_cluster_pct'] / AUTOENCODER_THRESHOLD_CLUSTER_PCT,
        ),
    )
    if not is_abnormal:
        confidence = max(0.55, 1.0 - confidence * 0.5)
    return label, float(confidence), mask_summary


def real_inference(img_array):
    inp = img_array[np.newaxis, ...]
    recon = autoencoder.predict(inp, verbose=0)[0] if autoencoder is not None else img_array.copy()
    error = compute_reconstruction_error(img_array, recon)
    anomaly_score = compute_anomaly_score(error)
    heatmap = generate_heatmap(error)
    mask, _ = threshold_anomaly(error)
    overlay = overlay_heatmap(img_array, heatmap)
    pred_label, confidence, _ = autoencoder_label_from_metrics(anomaly_score, float(np.mean(error)), mask)
    probs = np.array([1.0, 0.0]) if pred_label == 'Normal' else np.array([0.0, 1.0])
    return recon, error, heatmap, mask, overlay, anomaly_score, pred_label, confidence, probs


def run_inference(img_array):
    if MODEL_LOADED:
        return real_inference(img_array)
    return demo_inference(img_array)


def file_to_b64(path, grayscale=False):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        return None
    if not grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode('.png', img if grayscale else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return get_user_by_id(user_id)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not g.user:
            flash('Please login to continue.', 'error')
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)

    return wrapped


def roles_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not g.user:
                flash('Please login to continue.', 'error')
                return redirect(url_for('login', next=request.path))
            if g.user['role'] not in roles:
                flash('You are not authorized to access that page.', 'error')
                return redirect(url_for('index'))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def rate_limit(limit, window_seconds, scope='global'):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            identifier = f"{request.remote_addr or 'local'}:{scope}:{request.endpoint}"
            now = time.time()
            history = RATE_LIMIT_STORE.get(identifier, [])
            history = [stamp for stamp in history if now - stamp < window_seconds]
            if len(history) >= limit:
                flash('Too many requests. Please wait a moment and try again.', 'error')
                return redirect(request.referrer or url_for('index'))
            history.append(now)
            RATE_LIMIT_STORE[identifier] = history
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def render_analysis_report(analysis):
    asset_paths = get_report_asset_paths(analysis)
    upload_path = asset_paths['original_path']
    heatmap_path = os.path.join(RESULTS_FOLDER, analysis['heatmap_path']) if analysis.get('heatmap_path') else None
    overlay_path = asset_paths['overlay_path']
    mask_path = asset_paths['mask_path']
    report_data = {
        'id': analysis['id'],
        'patient_name': analysis['patient_name'],
        'patient_age': analysis.get('patient_age'),
        'patient_gender': analysis.get('patient_gender'),
        'patient_code': analysis.get('patient_code'),
        'filename': analysis['image_filename'],
        'pred_label': analysis['result_label'],
        'confidence': round((analysis['confidence'] or 0) * 100, 2),
        'anomaly_score': round(analysis['anomaly_score'] or 0, 4),
        'recon_loss': round(analysis['reconstruction_loss'] or 0, 6),
        'is_abnormal': analysis['result_label'] != 'Normal',
        'orig_b64': file_to_b64(upload_path, grayscale=True),
        'heatmap_b64': file_to_b64(heatmap_path),
        'overlay_b64': file_to_b64(overlay_path),
        'mask_b64': file_to_b64(mask_path, grayscale=True),
        'analysed_at': analysis['analysed_at'],
        'analysed_by_username': analysis.get('analysed_by_username'),
        'model_mode': 'Unsupervised Autoencoder Detection' if autoencoder is not None else 'Demo Detection Mode',
    }
    return render_template('result.html', r=report_data)


def get_report_asset_paths(analysis):
    return {
        'original_path': os.path.join(UPLOAD_FOLDER, analysis['image_filename']),
        'overlay_path': os.path.join(RESULTS_FOLDER, analysis['overlay_path']) if analysis.get('overlay_path') else None,
        'mask_path': os.path.join(RESULTS_FOLDER, analysis['mask_path']) if analysis.get('mask_path') else None,
    }


def flash_validation_errors(errors):
    for error in errors:
        if error:
            flash(error, 'error')


def validate_user_form(full_name, username, role=None, age=None, gender=None, password=None, require_password=True):
    errors = [
        validate_full_name(full_name),
        validate_username(username),
    ]
    if role is not None:
        errors.append(validate_role(role))
    if role == 'patient':
        errors.append(validate_age(age))
        errors.append(validate_gender(gender))
    if require_password:
        errors.append(validate_password_strength(password))
    elif password:
        errors.append(validate_password_strength(password))
    return [error for error in errors if error]


def get_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


@app.before_request
def load_user():
    if request.method == 'POST':
        token = session.get('_csrf_token')
        submitted_token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
        if not token or not submitted_token or token != submitted_token:
            flash('Security validation failed. Please refresh and try again.', 'error')
            return redirect(request.referrer or url_for('index'))
    g.user = current_user()
    g.patient_profile = get_patient_by_user_id(g.user['id']) if g.user and g.user['role'] == 'patient' else None


@app.context_processor
def inject_template_context():
    return {
        'current_user': g.user,
        'patient_profile': g.patient_profile,
        'csrf_token': get_csrf_token,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
@rate_limit(limit=5, window_seconds=300, scope='auth')
def login():
    if g.user:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        login_errors = [validate_username(username)]
        if login_errors[0]:
            flash_validation_errors(login_errors)
            return render_template('login.html')
        user = get_user_by_username(username)

        if not user or not check_password_hash(user['password_hash'], password):
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        session['user_id'] = user['id']
        log_action(user['id'], 'login', 'User logged in', request.remote_addr or '')
        flash(f"Welcome back, {user['full_name'] or user['username']}!", 'success')
        next_url = request.args.get('next')
        return redirect(next_url or url_for('index'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
@rate_limit(limit=5, window_seconds=300, scope='auth')
def register():
    if g.user:
        return redirect(url_for('index'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', 'Unknown')

        form_errors = validate_user_form(
            full_name=full_name,
            username=username,
            role='patient',
            age=age,
            gender=gender,
            password=password,
            require_password=True,
        )
        if form_errors:
            flash_validation_errors(form_errors)
            return render_template('register.html')
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        if get_user_by_username(username):
            flash('Username already exists. Please choose another one.', 'error')
            return render_template('register.html')

        user_id = create_user(
            username=username,
            password_hash=generate_password_hash(password),
            role='patient',
            full_name=full_name,
        )
        create_patient(
            name=full_name,
            age=age,
            gender=gender,
            user_id=user_id,
        )
        flash('Patient account created successfully. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    log_action(g.user['id'], 'logout', 'User logged out', request.remote_addr or '')
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


@app.route('/dashboard')
@roles_required('admin', 'doctor')
def dashboard():
    stats = get_dashboard_stats()
    analyses = get_analyses()
    return render_template('dashboard.html', stats=stats, analyses=analyses)


@app.route('/analyze', methods=['GET', 'POST'])
@roles_required('admin', 'doctor', 'patient')
@rate_limit(limit=12, window_seconds=300, scope='analysis')
def analyze():
    patient_options = get_patient_options() if g.user['role'] in {'admin', 'doctor'} else []

    if request.method == 'POST':
        selected_patient_id = request.form.get('selected_patient_id', type=int)

        if g.user['role'] == 'patient':
            patient_record = g.patient_profile
            if not patient_record:
                flash('Patient profile not found. Please contact admin.', 'error')
                return redirect(url_for('index'))
            pat_name = patient_record['name']
            pat_age = patient_record['age']
            pat_gender = patient_record['gender']
            pat_db_id = patient_record['id']
            update_patient(pat_db_id, pat_name, pat_age, pat_gender)
        elif selected_patient_id:
            existing = next((p for p in patient_options if p['id'] == selected_patient_id), None)
            if not existing:
                flash('Selected patient was not found.', 'error')
                return redirect(url_for('analyze'))
            pat_name = existing['name']
            pat_age = existing['age']
            pat_gender = existing['gender']
            pat_db_id = existing['id']
        else:
            pat_name = request.form.get('patient_name', 'Anonymous').strip()
            pat_age = request.form.get('patient_age', type=int)
            pat_gender = request.form.get('patient_gender', 'Unknown')
            patient_errors = [
                validate_full_name(pat_name),
                validate_age(pat_age),
                validate_gender(pat_gender),
            ]
            if any(patient_errors):
                flash_validation_errors([error for error in patient_errors if error])
                return redirect(url_for('analyze'))
            pat_db_id = create_patient(pat_name, pat_age, pat_gender)

        if 'mri_file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)
        file = request.files['mri_file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Unsupported file format. Use PNG, JPG, JPEG, BMP, or GIF.', 'error')
            return redirect(request.url)

        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file_bytes = file.read()
        image_error = validate_image_bytes(file_bytes)
        if image_error:
            flash(image_error, 'error')
            return redirect(request.url)
        with open(upload_path, 'wb') as file_obj:
            file_obj.write(file_bytes)

        try:
            img = load_image_from_bytes(file_bytes, target_size=MODEL_INPUT_SIZE)
        except Exception:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            flash('Unable to process the uploaded image.', 'error')
            return redirect(request.url)
        img = skull_strip(img)
        img_array = img[..., np.newaxis]
        _, error, heatmap, mask, overlay, anomaly_score, pred_label, confidence, _ = run_inference(img_array)

        heatmap_name = f"heatmap_{filename}"
        mask_name = f"mask_{filename}"
        overlay_name = f"overlay_{filename}"
        cv2.imwrite(os.path.join(RESULTS_FOLDER, heatmap_name), heatmap)
        cv2.imwrite(os.path.join(RESULTS_FOLDER, mask_name), mask.squeeze().astype(np.uint8))
        cv2.imwrite(os.path.join(RESULTS_FOLDER, overlay_name), overlay)

        recon_loss = float(np.mean(error))
        analysis_id = save_analysis(
            patient_db_id=pat_db_id,
            image_filename=filename,
            result_label=pred_label,
            confidence=confidence,
            anomaly_score=anomaly_score,
            recon_loss=recon_loss,
            heatmap_path=heatmap_name,
            mask_path=mask_name,
            overlay_path=overlay_name,
            model_version='demo' if not MODEL_LOADED else '1.0',
            analysed_by_user_id=g.user['id'],
        )

        log_action(
            g.user['id'],
            'analysis_created',
            f"Analysis #{analysis_id} created for patient #{pat_db_id} using {'unsupervised autoencoder detection' if autoencoder is not None else 'demo detection'} mode",
            request.remote_addr or '',
        )
        flash('MRI analysis completed successfully.', 'success')
        return redirect(url_for('view_report', analysis_id=analysis_id))

    return render_template(
        'analyze.html',
        patient_options=patient_options,
    )


@app.route('/history')
@roles_required('admin', 'doctor')
def history():
    analyses = get_analyses()
    return render_template('history.html', analyses=analyses, page_title='Analysis History')


@app.route('/my-reports')
@roles_required('patient')
def my_reports():
    if not g.patient_profile:
        flash('No linked patient profile found.', 'error')
        return redirect(url_for('index'))
    analyses = get_analyses(g.patient_profile['id'])
    return render_template('history.html', analyses=analyses, page_title='My Reports')


@app.route('/reports/<int:analysis_id>')
@login_required
def view_report(analysis_id):
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        abort(404)

    if g.user['role'] == 'patient' and analysis.get('patient_user_id') != g.user['id']:
        flash('You can only view your own reports.', 'error')
        return redirect(url_for('my_reports'))

    if g.user['role'] not in {'admin', 'doctor', 'patient'}:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))

    return render_analysis_report(analysis)


@app.route('/reports/<int:analysis_id>/download')
@login_required
def download_report_pdf(analysis_id):
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        abort(404)

    if g.user['role'] == 'patient' and analysis.get('patient_user_id') != g.user['id']:
        flash('You can only download your own reports.', 'error')
        return redirect(url_for('my_reports'))

    asset_paths = get_report_asset_paths(analysis)
    pdf_buffer = build_pdf_report(
        analysis=analysis,
        original_path=asset_paths['original_path'],
        abnormality_path=asset_paths['overlay_path'],
        mask_path=asset_paths['mask_path'],
    )
    filename = f"mri_report_{analysis.get('patient_code') or analysis['id']}.pdf"
    return send_file(pdf_buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')


@app.route('/account', methods=['GET', 'POST'])
@login_required
@rate_limit(limit=10, window_seconds=300, scope='self-account')
def my_account():
    if request.method == 'POST':
        action = request.form.get('action', '').strip()
        current_password = request.form.get('current_password', '')

        if not check_password_hash(g.user['password_hash'], current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('my_account'))

        if action == 'username':
            new_username = request.form.get('username', '').strip()
            username_error = validate_username(new_username)
            if username_error:
                flash(username_error, 'error')
                return redirect(url_for('my_account'))

            existing = get_user_by_username(new_username)
            if existing and existing['id'] != g.user['id']:
                flash('Username already exists.', 'error')
                return redirect(url_for('my_account'))

            update_user(
                user_id=g.user['id'],
                username=new_username,
                role=g.user['role'],
                full_name=g.user.get('full_name'),
                password_hash=None,
            )
            session['user_id'] = g.user['id']
            log_action(
                g.user['id'],
                'self_username_updated',
                f"User changed username to {new_username}",
                request.remote_addr or '',
            )
            flash('Username updated successfully.', 'success')
            return redirect(url_for('my_account'))

        if action == 'password':
            new_password = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')
            if new_password != confirm_password:
                flash('New password and confirm password do not match.', 'error')
                return redirect(url_for('my_account'))
            password_error = validate_password_strength(new_password)
            if password_error:
                flash(password_error, 'error')
                return redirect(url_for('my_account'))

            update_user(
                user_id=g.user['id'],
                username=g.user['username'],
                role=g.user['role'],
                full_name=g.user.get('full_name'),
                password_hash=generate_password_hash(new_password),
            )
            log_action(
                g.user['id'],
                'self_password_updated',
                'User changed own password',
                request.remote_addr or '',
            )
            flash('Password updated successfully.', 'success')
            return redirect(url_for('my_account'))

        flash('Invalid account action.', 'error')
        return redirect(url_for('my_account'))

    return render_template('account.html')


@app.route('/users', methods=['GET', 'POST'])
@roles_required('admin')
@rate_limit(limit=20, window_seconds=300, scope='admin-users')
def manage_users():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'patient').strip().lower()
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', 'Unknown')

        form_errors = validate_user_form(
            full_name=full_name,
            username=username,
            role=role,
            age=age,
            gender=gender,
            password=password,
            require_password=True,
        )
        if form_errors:
            flash_validation_errors(form_errors)
            return redirect(url_for('manage_users'))
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('manage_users'))
        if get_user_by_username(username):
            flash('Username already exists.', 'error')
            return redirect(url_for('manage_users'))

        user_id = create_user(
            username=username,
            password_hash=generate_password_hash(password),
            role=role,
            full_name=full_name,
        )

        if role == 'patient':
            create_patient(
                name=full_name,
                age=age,
                gender=gender,
                user_id=user_id,
            )

        log_action(
            g.user['id'],
            'user_created',
            f"Created {role} user #{user_id} ({username})",
            request.remote_addr or '',
        )
        flash(f'{role.title()} account created successfully.', 'success')
        return redirect(url_for('manage_users'))

    users = get_all_users()
    search_query = request.args.get('q', '').strip().lower()
    role_filter = request.args.get('role', '').strip().lower()

    if search_query:
        users = [
            user for user in users
            if search_query in (user.get('full_name') or '').lower()
            or search_query in (user.get('username') or '').lower()
            or search_query in (user.get('linked_patient_code') or '').lower()
        ]

    if role_filter in ROLES:
        users = [user for user in users if user.get('role') == role_filter]

    edit_user_id = request.args.get('edit', type=int)
    edit_user = get_user_with_patient_profile(edit_user_id) if edit_user_id else None
    if edit_user_id and not edit_user:
        flash('Selected user was not found.', 'error')
        return redirect(url_for('manage_users'))
    return render_template(
        'users.html',
        users=users,
        edit_user=edit_user,
        search_query=request.args.get('q', ''),
        role_filter=request.args.get('role', ''),
    )


@app.route('/users/<int:user_id>/edit', methods=['POST'])
@roles_required('admin')
@rate_limit(limit=20, window_seconds=300, scope='admin-users')
def edit_user_account(user_id):
    target_user = get_user_with_patient_profile(user_id)
    if not target_user:
        flash('User not found.', 'error')
        return redirect(url_for('manage_users'))
    if user_id == g.user['id']:
        flash('You cannot change your own admin account from this page.', 'error')
        return redirect(url_for('manage_users'))

    full_name = request.form.get('full_name', '').strip()
    username = request.form.get('username', '').strip()
    role = request.form.get('role', '').strip().lower()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')
    age = request.form.get('age', type=int)
    gender = request.form.get('gender', 'Unknown')

    form_errors = validate_user_form(
        full_name=full_name,
        username=username,
        role=role,
        age=age,
        gender=gender,
        password=password,
        require_password=False,
    )
    if form_errors:
        flash_validation_errors(form_errors)
        return redirect(url_for('manage_users', edit=user_id))

    existing = get_user_by_username(username)
    if existing and existing['id'] != user_id:
        flash('Username already exists.', 'error')
        return redirect(url_for('manage_users', edit=user_id))

    if password or confirm_password:
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('manage_users', edit=user_id))
        password_hash = generate_password_hash(password)
    else:
        password_hash = None

    linked_patient_db_id = target_user.get('linked_patient_db_id')
    if target_user['role'] == 'patient' and role != 'patient' and linked_patient_db_id:
        if count_patient_analyses(linked_patient_db_id) > 0:
            flash('Cannot change a patient with existing reports to another role.', 'error')
            return redirect(url_for('manage_users', edit=user_id))
        delete_patient(linked_patient_db_id)

    update_user(
        user_id=user_id,
        username=username,
        role=role,
        full_name=full_name,
        password_hash=password_hash,
    )

    if role == 'patient':
        patient_db_id = linked_patient_db_id or create_patient(
            name=full_name,
            age=age,
            gender=gender,
            user_id=user_id,
        )
        update_patient(patient_db_id, full_name, age, gender)

    log_action(
        g.user['id'],
        'user_updated',
        f"Updated user #{user_id} to role {role}",
        request.remote_addr or '',
    )
    flash('User updated successfully.', 'success')
    return redirect(url_for('manage_users'))


@app.route('/users/<int:user_id>/delete', methods=['POST'])
@roles_required('admin')
@rate_limit(limit=10, window_seconds=300, scope='admin-users')
def delete_user_account(user_id):
    target_user = get_user_with_patient_profile(user_id)
    if not target_user:
        flash('User not found.', 'error')
        return redirect(url_for('manage_users'))
    if user_id == g.user['id']:
        flash('You cannot delete your own admin account.', 'error')
        return redirect(url_for('manage_users'))

    linked_patient_db_id = target_user.get('linked_patient_db_id')
    if linked_patient_db_id and count_patient_analyses(linked_patient_db_id) > 0:
        flash('Cannot delete a patient account that already has reports.', 'error')
        return redirect(url_for('manage_users'))

    if linked_patient_db_id:
        delete_patient(linked_patient_db_id)
    clear_analysis_author(user_id)
    delete_user(user_id)
    log_action(
        g.user['id'],
        'user_deleted',
        f"Deleted user #{user_id} ({target_user['username']})",
        request.remote_addr or '',
    )
    flash('User deleted successfully.', 'success')
    return redirect(url_for('manage_users'))


@app.route('/users/<int:user_id>/reset-password', methods=['POST'])
@roles_required('admin')
@rate_limit(limit=10, window_seconds=300, scope='admin-users')
def reset_user_password(user_id):
    target_user = get_user_with_patient_profile(user_id)
    if not target_user:
        flash('User not found.', 'error')
        return redirect(url_for('manage_users'))
    if user_id == g.user['id']:
        flash('Use the edit flow if you want to change your own password.', 'error')
        return redirect(url_for('manage_users'))

    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    if not new_password:
        flash('New password is required.', 'error')
        return redirect(url_for('manage_users'))
    if new_password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('manage_users'))
    password_error = validate_password_strength(new_password)
    if password_error:
        flash(password_error, 'error')
        return redirect(url_for('manage_users'))

    update_user(
        user_id=user_id,
        username=target_user['username'],
        role=target_user['role'],
        full_name=target_user.get('full_name'),
        password_hash=generate_password_hash(new_password),
    )
    log_action(
        g.user['id'],
        'password_reset',
        f"Reset password for user #{user_id} ({target_user['username']})",
        request.remote_addr or '',
    )
    flash(f"Password reset successfully for {target_user['username']}.", 'success')
    return redirect(url_for('manage_users'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/admin/backup', methods=['POST'])
@roles_required('admin')
@rate_limit(limit=5, window_seconds=300, scope='backup')
def download_backup():
    backup_path = create_encrypted_backup(os.path.join(os.path.dirname(__file__), 'brain_mri.db'))
    log_action(
        g.user['id'],
        'backup_created',
        f"Created encrypted backup {os.path.basename(backup_path)}",
        request.remote_addr or '',
    )
    return send_file(backup_path, as_attachment=True, download_name=os.path.basename(backup_path))


@app.route('/admin/restore-backup', methods=['POST'])
@roles_required('admin')
@rate_limit(limit=3, window_seconds=600, scope='backup')
def restore_backup():
    if 'backup_file' not in request.files:
        flash('Please select an encrypted backup file.', 'error')
        return redirect(url_for('manage_users'))

    backup_file = request.files['backup_file']
    if not backup_file.filename or not backup_file.filename.endswith('.enc'):
        flash('Only encrypted .enc backup files are allowed.', 'error')
        return redirect(url_for('manage_users'))

    backup_bytes = backup_file.read()
    if not backup_bytes:
        flash('Backup file is empty.', 'error')
        return redirect(url_for('manage_users'))

    try:
        snapshot_path = restore_encrypted_backup(
            backup_bytes,
            os.path.join(os.path.dirname(__file__), 'brain_mri.db'),
        )
        init_db()
    except Exception:
        flash('Backup restore failed. Please verify the encrypted file and key.', 'error')
        return redirect(url_for('manage_users'))

    log_action(
        g.user['id'],
        'backup_restored',
        f"Restored encrypted backup; snapshot saved as {os.path.basename(snapshot_path)}",
        request.remote_addr or '',
    )
    flash('Encrypted backup restored successfully.', 'success')
    return redirect(url_for('manage_users'))


@app.route('/api/stats')
@roles_required('admin', 'doctor')
def api_stats():
    return jsonify(get_dashboard_stats())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
