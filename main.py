mport numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Configuration
img_name = "image1.jpg"
wm_name = "watermark2.jpg"
watermarked_img = "Watermarked_Image.jpg"
watermarked_extracted = "watermarked_extracted.jpg"
key = 50
bs = 8
w1, w2 = 64, 64
fact = 8
indx, indy = 0, 0
b_cut = 50
val1, val2 = [], []

# ================== Metrics ==================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def NCC(img1, img2):
    return abs(np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / (np.std(img1) * np.std(img2)))

# ========== DCT Utilities ==========
def dct2(a):
    return cv.dct(a.astype(np.float32))

def idct2(a):
    return cv.idct(a.astype(np.float32))

# ========== AES Encryption Utilities ==========
def generate_aes_key(password):
    return hashlib.sha256(password.encode()).digest()[:16]

def encrypt_watermark(wm, password):
    wm_flat = wm.flatten()
    wm_bytes = bytes(wm_flat.tolist())
    key = generate_aes_key(password)
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(wm_bytes, AES.block_size))
    encrypted_bits = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))
    return encrypted_bits[:w1 * w2]

def decrypt_watermark(bits, password):
    key = generate_aes_key(password)
    bits = np.pad(bits, (0, 8 - len(bits) % 8), 'constant')
    byte_data = np.packbits(bits).tobytes()
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = unpad(cipher.decrypt(byte_data), AES.block_size)
    decrypted_array = np.frombuffer(decrypted, dtype=np.uint8)
    return decrypted_array.reshape((w1, w2))

def get_hash(wm):
    return hashlib.sha256(wm.tobytes()).hexdigest()

# ========== Watermark Embedding ==========
def watermark_image(img, wm, password):
    c1, c2 = img.shape
    c1 -= b_cut * 2
    c2 -= b_cut * 2
   
    if (c1 * c2) // (bs * bs) < w1 * w2:
        print("Watermark too large.")
        return img

    blocks = (c1 // bs) * (c2 // bs)
    blocks_needed = w1 * w2
    imf = img.astype(np.float32).copy()
    st = set()
    final = imf.copy()
    random.seed(key)

    encrypted_bits = encrypt_watermark(wm, password)

    for i in range(blocks_needed):
        to_embed = encrypted_bits[i]
        x = random.randint(1, blocks)
        if x in st:
            continue
        st.add(x)
        n, m = c1 // bs, c2 // bs
        ind_i = (x // m) * bs + b_cut
        ind_j = (x % m) * bs + b_cut

        dct_block = dct2(imf[ind_i:ind_i + bs, ind_j:ind_j + bs])
        elem = dct_block[indx][indy] / fact

        if to_embed % 2 == 1:
            elem = math.ceil(elem) if math.ceil(elem) % 2 == 1 else math.ceil(elem) - 1
        else:
            elem = math.ceil(elem) if math.ceil(elem) % 2 == 0 else math.ceil(elem) - 1

        dct_block[indx][indy] = elem * fact
        val1.append((elem * fact, to_embed))
        imf[ind_i:ind_i + bs, ind_j:ind_j + bs] = idct2(dct_block)

    print("PSNR is:", psnr(imf, img))
    watermarked = np.uint8(np.clip(imf, 0, 255))
    cv.imwrite(watermarked_img, watermarked)
    return watermarked

# ========== Watermark Extraction ==========
def extract_watermark(img, ext_name, password):
    img = cv.resize(img, (1000, 1000))
    c1, c2 = 1000 - b_cut * 2, 1000 - b_cut * 2
    blocks = (c1 // bs) * (c2 // bs)
    blocks_needed = w1 * w2

    wm = np.zeros((w1, w2), dtype=np.uint8)
    st = set()
    random.seed(key)

    for i in range(blocks_needed):
        x = random.randint(1, blocks)
        if x in st:
            continue
        st.add(x)
        n, m = c1 // bs, c2 // bs
        ind_i = (x // m) * bs + b_cut
        ind_j = (x % m) * bs + b_cut

        dct_block = dct2(img[ind_i:ind_i + bs, ind_j:ind_j + bs])
        elem = round(dct_block[indx][indy] / fact)
        wm[i // w2][i % w2] = 0 if elem % 2 == 0 else 255
        val2.append((elem, elem % 2 != 0))

    bits = np.array([1 if x == 255 else 0 for x in wm.flatten()])
    decrypted = decrypt_watermark(bits, password)
    cv.imwrite(ext_name, decrypted)
    return decrypted

# ========== Attacks ==========
def ScalingBigger(img): return cv.resize(img, (1100, 1100))
def ScalingHalf(img): return cv.resize(img, (0, 0), fx=0.1, fy=0.1)
def Cut100Rows(img): return np.delete(img, slice(400, 500), axis=0)
def AverageFilter(img): return cv.filter2D(img, -1, np.ones((5, 5), np.float32) / 25)
def MedianFilter(img): return cv.medianBlur(img, 3)
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return np.clip(image + noise, 0, 255)
    elif noise_typ == "s&p":
        output = image.copy()
        prob = 0.05
        rnd = np.random.rand(*image.shape)
        output[rnd < prob] = 0
        output[rnd > 1 - prob] = 255
        return output
    elif noise_typ == "speckle":
        noise = np.random.randn(*image.shape).astype(np.uint8)
        return np.clip(image + image * noise, 0, 255)

# ========== GUI ==========
def run_gui():
    def embed():
        img_path = filedialog.askopenfilename(title="Select Host Image")
        wm_path = filedialog.askopenfilename(title="Select Watermark")
        password = key_entry.get()

        img = cv.imread(img_path, 0)
        wm = cv.imread(wm_path, 0)
        wm = cv.resize(wm, (w1, w2), interpolation=cv.INTER_CUBIC)
        watermarked = watermark_image(img, wm, password)
        cv.imshow("Watermarked Image", watermarked)
        cv.waitKey(0)

    def extract():
        img_path = filedialog.askopenfilename(title="Select Watermarked Image")
        password = key_entry.get()
        img = cv.imread(img_path, 0)
        extracted = extract_watermark(img, "extracted_gui.jpg", password)
        cv.imshow("Extracted Watermark", extracted)
        cv.waitKey(0)

    window = tk.Tk()
    window.title("Digital Watermarking GUI")
    tk.Label(window, text="Secret Key:").pack()
    key_entry = tk.Entry(window, show="*")
    key_entry.pack()
    tk.Button(window, text="Embed Watermark", command=embed).pack()
    tk.Button(window, text="Extract Watermark", command=extract).pack()
    window.mainloop()

# ========== Main ==========
if __name__ == "__main__":
    password = input("Enter secret key: ")

    img = cv.imread(img_name, 0)
    wm = cv.imread(wm_name, 0)
    wm = cv.resize(wm, (w1, w2), interpolation=cv.INTER_CUBIC)

    print("\n===== EMBEDDING WATERMARK =====")
    watermarked = watermark_image(img, wm, password)
    print("\nWatermarking Done!\n")

    print("\n===== EXTRACTING WATERMARK =====")
    extracted = extract_watermark(watermarked, watermarked_extracted, password)

    ncc_value = NCC(cv.resize(wm, (64, 64)), extracted)
    print(f"NCC between original and extracted watermark: {ncc_value:.4f}")

    print("\n===== WATERMARK AUTHENTICATION =====")
    if get_hash(wm) == get_hash(extracted):
        print("Authentication successful: Watermark is intact.")
    else:
        print("Authentication failed: Watermark is altered.")

    print("\n===== ROBUSTNESS TESTING =====")
    attacks = {
        "Scaled Up": ScalingBigger,
        "Scaled Down": ScalingHalf,
        "Cropped": Cut100Rows,
        "Average Filter": AverageFilter,
        "Median Filter": MedianFilter,
        "Gaussian Noise": lambda img: noisy("gauss", img),
        "Salt & Pepper": lambda img: noisy("s&p", img),
        "Speckle Noise": lambda img: noisy("speckle", img)
    }

    for name, func in attacks.items():
        print(f"\n-- Attack: {name} --")
        attacked = func(watermarked)
        attacked = cv.resize(attacked, (1000, 1000))
        extracted_attack = extract_watermark(attacked, f"extracted_{name}.jpg", password)
        ncc = NCC(cv.resize(wm, (64, 64)), extracted_attack)
        print(f"NCC after {name}: {ncc:.4f}")

    # Optional: Run GUI
    # run_gui()