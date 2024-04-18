from deepview.DeepView import DeepView

import numpy as np


class DeepViewSubClass(DeepView):

    @property
    def distances(self):
        '''
        Combines euclidian with discriminative fisher distances.
        Here the two distance measures are weighted with lambda
        to emphasise structural properties (lambda > 0.5) or
        to emphasise prediction properties (lambda < 0.5).
        '''
        eucl_scale = 1. / self.eucl_distances.max()
        discr_dist_max = (self.discr_distances.max() + 1 if self.discr_distances.max() == 0 else self.discr_distances.max())
        fisher_scale = 1. / discr_dist_max
        eucl = self.eucl_distances * eucl_scale * self.lam
        fisher = self.discr_distances * fisher_scale * (1. - self.lam)
        stacked = np.dstack((fisher, eucl))
        return stacked.sum(-1)

    def get_artist_sample(self, point):
        """Maps the location of an embedded point to it's image.

		# TODO: Pass proper data_viz function as argument to DeepView instantitation

        Parameters
        ----------
        point: tuple
        	Coorinates of click in image
        """
        sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
        yp, yt = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
        return sample_id, yp, yt
