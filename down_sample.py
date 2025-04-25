from PIL import Image
import cv2
import numpy as np

image = Image.open("images/high_res.jpg").convert("RGB")

low_res_image = image.resize((image.width // 4, image.height // 4))

low_res_image.save("images/input.jpg")

low_res = cv2.imread("images/input.jpg")
high_res = cv2.imread("images/high_res.jpg")

high_res_resized = cv2.resize(high_res, (low_res.shape[1], low_res.shape[0]))

low_res_gray = cv2.cvtColor(low_res, cv2.COLOR_BGR2GRAY)
high_res_gray = cv2.cvtColor(high_res_resized, cv2.COLOR_BGR2GRAY)

psnr = cv2.PSNR(low_res_gray, high_res_gray)
print(f"PSNR: {psnr} dB")
