import dataset_tool
from training.dataset import load_dataset

tfrecord_dir = "../databases/replay-attack/blink_dataset_replay"
image_dir = '../databases/replay-attack/faces/test_faces/**'
shuffle = True
dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle)
