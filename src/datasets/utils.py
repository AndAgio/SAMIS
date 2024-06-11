import numpy as np


# Introduce Gaussian noise to noise_percentage of image pixels
def noisy(image, noise_percentage, noise_std):
    row, col, ch = image.shape
    num_corrupt = int(np.floor(noise_percentage * row * col / 100))

    # Randomly choose pixels to add noise to
    xy_coords = np.random.choice(row * col, num_corrupt, replace=False)
    chan_coords = np.random.choice(ch, num_corrupt, replace=True)
    xy_coords = np.unravel_index(xy_coords, (row, col))

    out = np.copy(image)

    mean = 120

    # Add randomly generated Gaussian noise to pixels
    for coord in range(num_corrupt):
        noise = np.random.normal(mean, noise_std, 1)
        out[xy_coords[0][coord], xy_coords[1][coord],
            chan_coords[coord]] += noise

    return out