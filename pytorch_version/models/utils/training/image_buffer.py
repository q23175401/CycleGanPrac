from collections import deque
import random


class ImageBuffer:
    def __init__(self, buffer_size=50) -> None:
        self.buffer_size = buffer_size
        self.deque = deque(maxlen=buffer_size)

    def addImages(self, images):
        self.deque.append(images)

    def sampleImages(self):
        images = random.sample(self.deque, 1)
        return images[0]

    def __len__(self):
        return len(self.deque)


def usage_test():
    ib = ImageBuffer(50)
    ib.addImages(range(200))
    print(len(ib))
    for i in range(10):
        print(ib.sampleImages(2))


if __name__ == "__maine__":
    usage_test()

