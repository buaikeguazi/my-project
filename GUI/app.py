import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    normalized_image = image / 255.0
    return normalized_image

class LearnableWavelet(tf.keras.layers.Layer):
    def __init__(self, filters, wavelet='haar', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.wavelet = wavelet
        self.conv_low = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.conv_high = tf.keras.layers.Conv2D(filters, 3, padding='same')

    def call(self, x):
        x_low = self.conv_low(x)
        x_high = x - x_low
        x_high = self.conv_high(x_high)
        return x_low + x_high

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "wavelet": self.wavelet
        })
        return config

class_counts = [1456, 295, 589, 153, 236]
class BalancedSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, class_counts, name='balanced_softmax_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_counts = tf.constant(class_counts, dtype=tf.float32)

    def call(self, y_true, logits):
        y_true_converted = tf.cond(
            tf.equal(tf.rank(y_true), 2),
            lambda: tf.argmax(y_true, axis=-1),
            lambda: tf.cast(y_true, tf.int64)
        )
        adjusted_logits = logits + tf.math.log(self.class_counts)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_converted, logits=adjusted_logits)
        return loss

    def get_config(self):
        return {'class_counts': self.class_counts.numpy().tolist()}


model = load_model('model_best.h5', custom_objects={
    'BalancedSoftmaxLoss': BalancedSoftmaxLoss,
    'LearnableWavelet': LearnableWavelet
})


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.Variable(img_tensor)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

def make_saliency_map(img_array, model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.Variable(img_tensor)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    gradients = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = saliency[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def save_and_display_saliency_map(img_path, saliency_map, saliency_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
    saliency_map = np.uint8(255 * saliency_map)
    colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    superimposed_img = colored * alpha + img
    cv2.imwrite(saliency_path, cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

def compute_entropy(probabilities):
    epsilon = 1e-8
    return -np.sum(probabilities * np.log(probabilities + epsilon))

def assess_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 15:
        return "Too blurry", lap_var
    elif lap_var < 25:
        return "Blurry", lap_var
    elif lap_var < 35:
        return "Slightly blurry", lap_var
    elif lap_var < 45:
        return "Acceptable", lap_var
    else:
        return "Good", lap_var

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Invalid image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image)
    return np.expand_dims(processed, axis=0)


class_dict = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = file.filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return jsonify({'filename': filename})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        original_image = cv2.imread(file_path)
        quality_label, quality_score = assess_image_quality(original_image)
        processed_image = process_image(file_path)

        logits = model.predict(processed_image)
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        pred_index = int(np.argmax(probabilities))
        pred_label = class_dict.get(pred_index, "Unknown")
        uncertainty = float(compute_entropy(probabilities))

        # GradCAM
        last_conv_layer = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer, pred_index)
        gradcam_filename = "gradcam_" + filename
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        save_and_display_gradcam(file_path, heatmap, gradcam_path)

        # Saliency
        saliency_map = make_saliency_map(processed_image, model, pred_index)
        saliency_filename = "saliency_" + filename
        saliency_path = os.path.join(app.config['UPLOAD_FOLDER'], saliency_filename)
        save_and_display_saliency_map(file_path, saliency_map, saliency_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'prediction': pred_label,
        'probabilities': probabilities.tolist(),
        'true_label': filename.rsplit('.', 1)[0],
        'uncertainty': uncertainty,
        'image_quality': quality_label,
        'quality_score': quality_score,
        'gradcam_image': gradcam_filename,
        'saliency_image': saliency_filename
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
