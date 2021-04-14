import sys

from PIL import Image

'''
Example:
python resizer.py data/igor.png 512 512
'''
img = Image.open(sys.argv[1])
img.resize((int(sys.argv[2]), int(sys.argv[3]))).save(sys.argv[1])
