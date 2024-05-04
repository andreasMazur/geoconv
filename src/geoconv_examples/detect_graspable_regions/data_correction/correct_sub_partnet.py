from geoconv_examples.improving_segmentation.deepview.correction_pipeline import correction_pipeline
from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset, processed_data_generator
from geoconv_examples.detect_graspable_regions.models.imcnn import SegImcnn

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


def correct_sub_partnet(data_path, model_path):
    """Runs DeepView correction with the given IMCNN."""
    model = SegImcnn(adapt_data=PartNetDataset(data_path, set_type=0, only_signal=True))
    model.load_state_dict(torch.load(model_path))

    # Load the dataset
    dataset = processed_data_generator(data_path, set_type=3)

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
        correction_file_name="partnet_correction.csv"
    )
