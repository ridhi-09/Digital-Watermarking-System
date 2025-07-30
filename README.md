# Digital-Watermarking-System
This project presents a robust and secure digital watermarking system that combines image processing and cryptographic techniques to protect intellectual property and enable secure image communication.

Using Discrete Cosine Transform (DCT), the system invisibly embeds a binary watermark into the frequency components of a grayscale image. This approach ensures imperceptibility, meaning the watermark remains hidden to the human eye, while maintaining high robustness against common image distortions such as compression, noise, and scaling.

To enhance security, the watermarked image is encrypted using AES encryption (Fernet) with a password-derived key. This ensures that even if the image is intercepted, the embedded information remains protected unless decrypted with the correct password.
