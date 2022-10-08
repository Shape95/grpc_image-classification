"""
    CIFAR pretrained ResNet sample
    Author: Mr. Balsdnim
    Date:   1st Dec. 2021
"""

import numpy as np
import torch
import logging
import os

from torchvision import transforms
import io
from PIL import Image

# Model Handler

class CifarModelHandler():
    def __init__(self):
        """TODO: Initialize classification service with the given torch model.
        """
        try:
            self.model = torch.jit.load("script/resnet_scripted_quantized.pt")
        except:
            self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

    
    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
            First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logging.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logging.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()
        # if ipex_enabled:
        #     self.model = self.model.to(memory_format=torch.channels_last)
        #     self.model = ipex.optimize(self.model)

        logging.debug("Model file %s loaded successfully", model_pt_path)

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # self.mapping = load_label_mapping(mapping_file_path)

        self.initialized = True

    def preprocess_one_image(self, image_data):
        # Create image value from byte encoded image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = self.transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)
        return image

    def preprocess(self, image_datas):
        images = [self.preprocess_one_image(image) for image in image_datas]
        images = torch.cat(images)
        return images

    def inference(self, image_tensor):
        # Skip gradient calculation in inference
        image_tensor = image_tensor.float()
        
        # No Gradient, DropOut, etc.. for speed
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)  # [1, 10]
            prediction = predictions[0].argmax()  # [1] prediction result
            return int(prediction)
    
    def handle(self, data):
        """ Entry point for default handler. It takes the data from the input request and returns
            the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.

            Returns:
                list : Returns a list of dictionary with the predicted response.
        """
        logging.info(f"Request handle")

        if data is None:
            return None
        
        data = self.preprocess(data)
        data = self.inference(data)
        
        logging.info(f"Request data: {data}")
        
        return data
