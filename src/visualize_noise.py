import matplotlib.pyplot as plt
import numpy as np

img_path = "BSD68/test002.png"
img = plt.imread(img_path)
print("type:", type(img), "shape:", img.shape, "dtype:", img.dtype)
# print basic information about the image

sigma = 25 / 255.0
# define the max noise level
# sigma controls the intensity of the Gaussian noise
noise = np.random.randn(*img.shape) * sigma
# generate Gaussian noise with zero mean and standard deviation sigma
noisy = np.clip(img + noise, 0, 1)
# add the noise to the clean image

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Clean Image")
plt.axis("off")
# display clean image

plt.subplot(1,2,2)
plt.imshow(noisy, cmap='gray')
plt.title(f"Noisy σ=25")
plt.axis("off")
plt.tight_layout()
plt.show()
# display noise image
