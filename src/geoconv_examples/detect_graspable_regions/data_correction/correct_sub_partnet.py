from geoconv_examples.detect_graspable_regions.deepview.correction_pipeline import correction_pipeline
from geoconv_examples.detect_graspable_regions.partnet_grasp.dataset import (
    PartNetGraspDataset, processed_partnet_grasp_generator
)
from geoconv_examples.detect_graspable_regions.training.imcnn import SegImcnn

import torch
import scipy as sp


def pred_wrapper(data, model):
    """Get the predicted probabilities of an IMCNN."""
    return sp.special.softmax(model.model.output_dense(torch.tensor(data).float()).detach().numpy(), axis=-1)


def embed(imcnn, inputs):
    """Retrieves the output of the last ISC-layer of an IMCNN."""
    #################
    # Handling Input
    #################
    signal, bc = inputs
    signal = imcnn.model.normalize(signal)
    signal = imcnn.model.downsize_dense(signal)
    signal = imcnn.model.downsize_bn(signal)

    ###############
    # Forward pass
    ###############
    for idx in range(len(imcnn.model.do_layers)):
        signal = imcnn.model.do_layers[idx](signal)
        signal = imcnn.model.isc_layers[idx]([signal, bc])
        signal = imcnn.model.amp_layers[idx](signal)
        signal = imcnn.model.bn_layers[idx](signal)

    #########
    # Output
    #########
    return signal.detach().numpy()


def correct_sub_partnet(data_path, model_path, correction_csv_path=None):
    """Runs DeepView correction with the given IMCNN."""
    if correction_csv_path is None:
        correction_csv_path = "./partnet_correction.csv"

    model = SegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=0, only_signal=True))
    model.load_state_dict(torch.load(model_path))

    # Load the dataset
    dataset = processed_partnet_grasp_generator(data_path, set_type=3)

    # Create class descriptions
    class_dict = {
        0: "non-graspable",
        1: "graspable"
    }

    # Start correcting
    correction_pipeline(
        model=model,
        dataset=dataset,
        embedding_shape=(96,),
        embed_fn=embed,
        pred_fn=pred_wrapper,
        class_dict=class_dict,
        signals_are_coordinates=True,
        correction_file_name=correction_csv_path,
        max_samples=10_000
    )
