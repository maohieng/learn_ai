import math
import sys

from PIL import Image, ImageFilter

# Ensure correct usage
if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")

# Open image
# Convert image to RGB if it is a .png file
image = Image.open(sys.argv[1]).convert("RGB")

# Filter image according to edge detection kernel
# Kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
# Scale: 1
# Size: 3x3

filtered = image.filter(ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
    scale=1
))

# Show resulting image
filtered.show()
