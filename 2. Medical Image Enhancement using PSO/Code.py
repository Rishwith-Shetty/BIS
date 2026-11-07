
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from google.colab import files
from IPython.display import display

# STEP 1: Upload Medical Image
print("Upload your X-ray / Medical Image")
uploaded = files.upload()

filename = next(iter(uploaded))

image = cv2.imdecode(
    np.frombuffer(uploaded[filename], np.uint8),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError("Unable to read image!")

# STEP 2: Display Original Image
plt.figure(figsize=(6, 5))
plt.imshow(image, cmap='gray')
plt.title("Original Medical Image")
plt.axis("off")
plt.show()


# STEP 3: Image Enhancement Function
def enhance_image(alpha, beta):
    return cv2.convertScaleAbs(
        image, alpha=alpha, beta=beta
    )


# STEP 4: Fitness Function
def fitness(position):
    alpha, beta = position

    enhanced = enhance_image(alpha, beta)

    hist = cv2.calcHist(
        [enhanced], [0], None, [256], [0, 256]
    )

    hist = hist / hist.sum()

    entropy = -np.sum(
        hist[hist > 0] * np.log2(hist[hist > 0])
    )

    return entropy


# STEP 5: Initialize PSO Parameters
n_particles = 10
iterations = 30

w = 0.5
c1 = 1.5
c2 = 1.5

positions = [
    [random.uniform(0.5, 2.0),
     random.uniform(-50, 50)]
    for _ in range(n_particles)
]

velocities = [
    [0.0, 0.0] for _ in range(n_particles)
]

pbest = [p.copy() for p in positions]

gbest = max(pbest, key=fitness).copy()


# STEP 6: Apply PSO Algorithm
for iteration in range(iterations):

    for i in range(n_particles):

        r1 = random.random()
        r2 = random.random()

        # Update velocity
        for j in range(2):
            velocities[i][j] = (
                w * velocities[i][j]
                + c1 * r1 * (pbest[i][j] - positions[i][j])
                + c2 * r2 * (gbest[j] - positions[i][j])
            )

        # Update position
        for j in range(2):
            positions[i][j] += velocities[i][j]

        # Limit brightness and contrast
        positions[i][0] = np.clip(
            positions[i][0], 0.5, 2.0
        )

        positions[i][1] = np.clip(
            positions[i][1], -50, 50
        )

        # Update personal best
        if fitness(positions[i]) > fitness(pbest[i]):
            pbest[i] = positions[i].copy()

        # Update global best
        if fitness(pbest[i]) > fitness(gbest):
            gbest = pbest[i].copy()

    print(
        "Iteration:", iteration + 1,
        "Best Fitness:", round(fitness(gbest), 4)
    )


# STEP 7: Generate Enhanced Medical Image
best_alpha, best_beta = gbest

enhanced = enhance_image(best_alpha, best_beta)


# STEP 8: Display Original and Enhanced Images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Medical Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced Medical Image using PSO")
plt.axis("off")

plt.show()


# STEP 9: Display Optimal Parameters
print("\nOPTIMAL RESULTS")
print("----------------------")
print("Optimal Contrast:", round(best_alpha, 4))
print("Optimal Brightness:", round(best_beta, 4))
print("Best Fitness:", round(fitness(gbest), 4))


# STEP 10: Save and Download Enhanced Image
cv2.imwrite("enhanced_medical.jpg", enhanced)

print("\nEnhanced image saved successfully!")

files.download("enhanced_medical.jpg")
