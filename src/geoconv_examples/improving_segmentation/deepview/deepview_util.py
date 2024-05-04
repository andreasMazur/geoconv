import numpy as np
import scipy as sp


def embed(imcnn, inputs):
    """Retrieves the output of the last ISC-layer of an IMCNN."""
    #################
    # Handling Input
    #################
    signal, bc = inputs
    signal = imcnn.normalize(signal)
    signal = imcnn.downsize_dense(signal)
    signal = imcnn.downsize_bn(signal)

    ###############
    # Forward pass
    ###############
    for idx in range(len(imcnn.do_layers)):
        signal = imcnn.do_layers[idx](signal)
        signal = imcnn.isc_layers[idx]([signal, bc])
        signal = imcnn.amp_layers[idx](signal)
        signal = imcnn.bn_layers[idx](signal)

    #########
    # Output
    #########
    return np.array(signal)


def pred_wrapper(data, model):
    """Get the predicted probabilities of an IMCNN."""
    return sp.special.softmax(model.output_dense(data), axis=-1)
