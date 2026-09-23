
import os
from collections import Counter


from torchvision import datasets, transforms

folder_dir = "/Users/tsampikakiaourtzi/Documents/FL_VIP/knee_osteo/data/raw/knee_images"
print("Exists:", os.path.exists(folder_dir))
print("folder_dir =", folder_dir)

knee_images = datasets.ImageFolder(root=folder_dir, transform=transforms.ToTensor())
print(type(knee_images[0]))

print(f"Classes:{knee_images.classes}")
print(f"Class and Index: {knee_images.class_to_idx}")
print(f"Total images: {len(knee_images)}")

counts = Counter(knee_images.targets)
for class_idx, count in sorted(counts.items()):
    print(f"Class {knee_images.classes[class_idx]}: {count} images")

image, label = knee_images[0]
print("Image shape:", image.shape)
print("Class:", knee_images.classes[label])

class_names = knee_images.classes
values = [counts[i] for i in range(len(class_names))]


import matplotlib.pyplot as plt

# Plotting images per class (bar chart)
plt.figure(figsize=(8, 5))
bars = plt.bar(class_names, values, color='purple', edgecolor='black')

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             str(val), ha='center', va='bottom', fontsize=11)

plt.title('Number of Images per Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.show()
