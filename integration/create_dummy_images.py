
import os
import numpy as np
from PIL import Image
import argparse

def create_images(root_dir, num_episodes, num_frames):
    os.makedirs(root_dir, exist_ok=True)
    print(f'Generating dummy images in {os.path.abspath(root_dir)}')
    for i in range(num_episodes):
        ep_dir = os.path.join(root_dir, f'ep_{i}')
        os.makedirs(ep_dir, exist_ok=True)
        for j in range(num_frames):
            img_array = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(ep_dir, f'frame_{j:04d}.png'))
        print(f'...created episode in {os.path.abspath(ep_dir)}')
    print('Dummy image files created.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--num_frames', type=int, default=10)
    args = parser.parse_args()
    create_images(args.root, args.num_episodes, args.num_frames)
