from concurrent import futures
import argparse
import time
import logging
import grpc

from resnet import CifarModelHandler
from proto.classification_pb2_grpc import ClassificationServicer, add_ClassificationServicer_to_server
from proto.classification_pb2 import SampleOutput, Output, AIRPLANE, AUTOMOBILE, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_CLASS_DICT = {
    0: AIRPLANE,
    1: AUTOMOBILE,
    2: BIRD,
    3: CAT,
    4: DEER,
    5: DOG,
    6: FROG,
    7: HORSE,
    8: SHIP,
    9: TRUCK
}

def get_encoded_text(chunks):
    result = bytearray()
    for chunk in chunks:
        result.extend(chunk.text)
    return bytes(result)


def get_encoded_image(chunks):
    result = bytearray()
    for chunk in chunks:
        result.extend(chunk.image)
    return bytes(result)


class ClassificationServer(ClassificationServicer):
    def __init__(self):
        self.model = CifarModelHandler()

    def GetTextResult(self, request, context):
        """Example stub definition for text input

        Args:
            request: gRPC chunk or gRPC chunk iterator
            context: gRPC context

        Returns:
            gRPC chunk: chunk output with int result
        """
        binary = get_encoded_text(request)
        text = binary.decode('utf-8')

        logging.info(f"Request Text: {text}")

        try:
            length = len(text)
            return SampleOutput(result=length)
        except Exception as e:
            logging.exception(e)
            context.set_code(grpc.StatusCode.ABORTED)
            context.set_details(str(e))

    def GetResult(self, request, context):
        """TODO: skeletal code for image classification service

        Args:
            request: gRPC chunk or gRPC chunk iterator
            context: gRPC context

        Returns:
            gRPC chunk: chunk output with class index result
        """
        logging.info(f"Request image")

        image_data = get_encoded_image(request)

        image_data_list = []
        image_data_list.append(image_data)

        data = self.model.handle(image_data_list)

        try:
            return Output(result = _CLASS_DICT[data])
        except Exception as e:
            logging.exception(e)
            context.set_code(grpc.StatusCode.ABORTED)
            context.set_details(str(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default="6010",
                        help='Remote IP and Port')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), )
    service_server = ClassificationServer()
    servicer = add_ClassificationServicer_to_server(service_server, server)
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()

    print(f"Classification service starts... | PORT: {args.port}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
