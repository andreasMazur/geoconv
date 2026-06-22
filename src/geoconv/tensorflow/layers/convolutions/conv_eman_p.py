from geoconv.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP


class ConvEMANP(ConvEMAN, ConvGEMP):
    """Implements EMANs with radially dependent weights and learnable phase offsets."""
    pass
