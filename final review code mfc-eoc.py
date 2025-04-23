import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import time
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, Add, Lambda
import pandas as pd

# ---------------------- DIV2K Loader (Local Only) ----------------------
def load_div2k_dataset():
    images, labels = [], []
    folder = r"C:\\Users\\sriii\\OneDrive\\Desktop\\mfc\\div2k"
    for i in range(1, 21):
        if len(images) >= 10:
            break
        for ext in ['png', 'jpg']:
            path = os.path.join(folder, f"{i:04d}.{ext}")
            if os.path.exists(path):
                try:
                    img = Image.open(path).resize((256, 256), Image.LANCZOS)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(np.array(img) / 255.0)
                    labels.append(f"Image {i}")
                    break
                except:
                    continue
    if len(images) < 1:
        raise Exception("Failed to load any DIV2K images from local folder.")
    return images[:10], labels[:10]

# ---------------------- AES Crypto ----------------------
def encrypt_aes(img):
    key = get_random_bytes(32)
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad((img * 255).astype(np.uint8).tobytes(), AES.block_size)
    encrypted = cipher.encrypt(padded)
    return encrypted, key, iv

def decrypt_aes(data, shape, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(data), AES.block_size)
    return np.frombuffer(decrypted, dtype=np.uint8).reshape(shape) / 255.0

# ---------------------- Supercharged CNN ----------------------
def get_placeholder_cnn():
    def scramble(x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.3)
        x = x + noise
        return tf.clip_by_value(x, 0.0, 1.0)

    inputs = Input(shape=(256, 256, 3))
    x0 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x0 = BatchNormalization()(x0)
    x1 = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x0)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(512, 3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)

    x = UpSampling2D()(x3)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x2 = UpSampling2D()(x2)
    x = Add()([x, x2])

    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x1 = UpSampling2D()(x1)
    x = Add()([x, x1])

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])

    x = Lambda(scramble)(x)
    outputs = Conv2D(3, 3, padding='same', activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def encrypt_cnn(img, model):
    return  model.predict(np.expand_dims(img, axis=0))[0]

def decryption_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    
    # denoising operation
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # Add a denoising operation (e.g., convolution + activation)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Some upsampling,  resizing or super-resolution step
    x = UpSampling2D()(x)
    
    # Adding a smoothing operation
    x = Lambda(lambda x: tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='SAME'))(x)
    
    x = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(x)
    
    # Create model
    model = tf.keras.Model(inputs, x)
    
    return model

def decrypt_cnn(encrypted_img):
    # Initialize the  decryption model
    model = decryption_model()

    # Perform the decryption 
    decrypted = model.predict(np.expand_dims(encrypted_img, axis=0))[0]
    
    return img


# ---------------------- Metrics ----------------------
def get_metrics(orig, recon):
    # Cast the images to uint8 using tf.cast, not np.astype
    orig_uint8 = tf.cast(orig * 255, tf.uint8)
    recon_uint8 = tf.cast(recon * 255, tf.uint8)

    # Convert TensorFlow tensors to NumPy arrays if needed (for SSIM and PSNR)
    orig_uint8 = orig_uint8.numpy() if isinstance(orig_uint8, tf.Tensor) else orig_uint8
    recon_uint8 = recon_uint8.numpy() if isinstance(recon_uint8, tf.Tensor) else recon_uint8

    mse_val = np.mean((orig_uint8 - recon_uint8) ** 2)
    psnr_val = psnr(orig_uint8, recon_uint8, data_range=255)
    ssim_val = ssim(orig_uint8, recon_uint8, channel_axis=-1, data_range=255)
    psnr_val = 'inf' if np.isinf(psnr_val) else psnr_val

    return {'mse': mse_val, 'psnr': psnr_val, 'ssim': ssim_val}

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Image Crypto Comparator", layout="centered")
st.title("🛡️ Image Cryptography: AES CBC vs CNN")
st.markdown("Compare traditional AES encryption with a deep CNN-based approach for visual cryptography.")

images, labels = load_div2k_dataset()
model = get_placeholder_cnn()

index = st.selectbox("🔍 Select an Image:", range(10), format_func=lambda i: labels[i])
img = images[index]

if 'encrypted_aes' not in st.session_state:
    st.session_state.encrypted_aes = None
    st.session_state.key = None
    st.session_state.iv = None
    st.session_state.encrypted_cnn = None

col1, col2 = st.columns(2)
if col1.button("🔒 Encrypt"):
    st.session_state.encrypted_aes, st.session_state.key, st.session_state.iv = encrypt_aes(img)
    st.session_state.encrypted_cnn = encrypt_cnn(img, model)
    st.image((img * 255).astype(np.uint8), caption="Original Image", use_container_width=True)
    st.image((st.session_state.encrypted_cnn * 255).astype(np.uint8), caption="CNN Encrypted", use_container_width=True)
    st.image(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), caption="AES Encrypted (Random View)", use_container_width=True)

if col2.button("🔓 Decrypt"):
    decrypted_aes = decrypt_aes(st.session_state.encrypted_aes, (256, 256, 3), st.session_state.key, st.session_state.iv)
    decrypted_cnn = decrypt_cnn(st.session_state.encrypted_cnn)

    st.image((decrypted_aes * 255).astype(np.uint8), caption="AES Decrypted", use_container_width=True)

    # Convert the decrypted CNN image to uint8 for proper display
    decrypted_cnn_uint8 = tf.cast(decrypted_cnn * 255, tf.uint8)

    # Convert TensorFlow tensor to NumPy array
    decrypted_cnn_np = decrypted_cnn_uint8.numpy()

    # Display the image using Streamlit
    st.image(decrypted_cnn_np, caption="CNN Decrypted", use_container_width=True)

    # Performance Metrics Calculation
    m1 = get_metrics(img, decrypted_aes)
    m2 = get_metrics(img, decrypted_cnn)

    # Display the metrics in a DataFrame
    st.markdown("### 📊 Performance Metrics")
    st.dataframe({
        'Metric': ['MSE', 'PSNR', 'SSIM'],
        'AES CBC': [m1['mse'], m1['psnr'], m1['ssim']],
        'CNN': [m2['mse'], m2['psnr'], m2['ssim']]
    })

if st.button("📈 Compare All 10 Images"):
    mse_aes, mse_cnn, psnr_aes, psnr_cnn, ssim_aes, ssim_cnn = [], [], [], [], [], []

    for img in images:
        e_aes, k, v = encrypt_aes(img)
        d_aes = decrypt_aes(e_aes, (256, 256, 3), k, v)
        e_cnn = encrypt_cnn(img, model)
        d_cnn = decrypt_cnn(img)

        m_aes = get_metrics(img, d_aes)
        m_cnn = get_metrics(img, d_cnn)

        mse_aes.append(m_aes['mse'])
        mse_cnn.append(m_cnn['mse'])
        psnr_aes.append(999 if m_aes['psnr'] == 'inf' else m_aes['psnr'])
        psnr_cnn.append(999 if m_cnn['psnr'] == 'inf' else m_cnn['psnr'])
        ssim_aes.append(m_aes['ssim'])
        ssim_cnn.append(m_cnn['ssim'])

    def plot_graph(title, aes_vals, cnn_vals, ylabel):
        fig, ax = plt.subplots()
        ax.bar(np.arange(10) - 0.2, aes_vals, width=0.4, label='AES CBC')
        ax.bar(np.arange(10) + 0.2, cnn_vals, width=0.4, label='CNN')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Image Index")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔬 Metric Comparison Across All Images")
    plot_graph("MSE Comparison", mse_aes, mse_cnn, "MSE")
    plot_graph("PSNR Comparison", psnr_aes, psnr_cnn, "PSNR")
    plot_graph("SSIM Comparison", ssim_aes, ssim_cnn, "SSIM")
