import grpc
import argparse
import os

from proto.classification_pb2 import Input, SampleInput
from proto.classification_pb2_grpc import ClassificationStub

import logging


# Text
TEXT_CHUNK_SIZE = 8
CHUNK_SIZE = 64  # bytes

# Image
IMAGE_CHUNK_SIZE = 4096

_CLASS_DICT = {
    0: "AIRPLANE",
    1: "AUTOMOBILE",
    2: "BIRD",
    3: "CAT",
    4: "DEER",
    5: "DOG",
    6: "FROG",
    7: "HORSE",
    8: "SHIP",
    9: "TRUCK"
}

def generate_text_iterator(text):
    for idx in range(0, len(text), TEXT_CHUNK_SIZE):
        yield SampleInput(text=text[idx:idx + TEXT_CHUNK_SIZE])

def generate_image_iterator(image_path):
    with open(image_path, mode="rb") as f:
        while True:
            chunk = f.read(IMAGE_CHUNK_SIZE)
            if chunk:
                yield Input(image=chunk)
            else:  # The chunk was empty, which means we're at the end of the file\
                break
    
class ClassificationClient:
    def __init__(self, remote):
        self.channel = grpc.insecure_channel(remote)
        self.stub = ClassificationStub(self.channel)

    def get_text_result(self, text):
        """Example client function for text input

        Args:
            text (str): input text

        Returns:
            int: result value (in this case, text length)
        """

        bytes_text = bytes(text, encoding="UTF-8")
        binary_iterator = generate_text_iterator(bytes_text)
        result = self.stub.GetTextResult(binary_iterator)
        return result.result

    def get_result(self, image_path):
        """TODO: skeletal function for stub calling

        Args:
            image_path (str): image path
            
        Returns:
            str: predicted class
        """
        binary_iterator = generate_image_iterator(image_path)
        result = self.stub.GetResult(binary_iterator)
        return result.result
                
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', type=str, default="127.0.0.1:6010",
                        help='Remote IP and Port')
    parser.add_argument('--text', type=str, default="Hello, World",
                        help='Input text')
    parser.add_argument('--img_root_dir', type=str, default="./image/",
                        help='Folder location for image load')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    images_dir = os.listdir(args.img_root_dir)

    client = ClassificationClient(args.remote)

    for i in images_dir:
        result = client.get_result(args.img_root_dir + i)
        print(_CLASS_DICT[result])