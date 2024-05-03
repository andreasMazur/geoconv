import numpy as np


class BoundingBox:
    """A class for axis-aligned bounding boxes.

    Attributes
    ----------
    anchor: tuple
        The anchor point of the bounding box (low, left, deep) corner.
    width: float
        The width of the bounding box.
    height: float
        The height of the bounding box.
    depth: float
        The depth of the bounding box.
    """

    def __init__(self, anchor, width, height, depth):
        self.anchor = anchor
        self.width = width
        self.height = height
        self.depth = depth

        self.x_min = anchor[0]
        self.x_max = anchor[0] + width

        self.y_min = anchor[1]
        self.y_max = anchor[1] + height

        self.z_min = anchor[2]
        self.z_max = anchor[2] + depth

    def is_within(self, query_point):
        """Checks whether the given query point is within the bounding box.

        Parameters
        ----------
        query_point: np.ndarray
            The point to check.

        Returns
        -------
        bool:
            Whether the query-point is within the bounding box.
        """
        x_okay = self.x_min <= query_point[0] <= self.x_max
        y_okay = self.y_min <= query_point[1] <= self.y_max
        z_okay = self.z_min <= query_point[2] <= self.z_max
        return x_okay and y_okay and z_okay

    def corners(self):
        """Returns the 3D coordinates of the bounding box.

        Returns
        -------
        np.ndarray:
            Returns the 3D coordinates of the bounding box.
        """
        return np.array([
            [self.x_min, self.y_min, self.z_min],  # lower left
            [self.x_max, self.y_min, self.z_min],  # lower right
            [self.x_min, self.y_max, self.z_min],  # upper left
            [self.x_max, self.y_max, self.z_min],  # upper right
            [self.x_min, self.y_min, self.z_max],  # deep lower left
            [self.x_max, self.y_min, self.z_max],  # deep lower right
            [self.x_min, self.y_max, self.z_max],  # deep upper left
            [self.x_max, self.y_max, self.z_max],  # deep upper right
        ])
