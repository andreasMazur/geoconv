from geoconv_examples.detect_graspable_regions.deepview.select_collection import SelectFromCollection

from deepview.DeepView import DeepView

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import warnings


class DeepViewSubClass(DeepView):

    def __init__(self, *args, class_dict, selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_dict = class_dict
        self.selector = selector

    @property
    def distances(self):
        """Combines euclidian with discriminative fisher distances.

        Here the two distance measures are weighted with lambda
        to emphasise structural properties (lambda > 0.5) or
        to emphasise prediction properties (lambda < 0.5).
        """
        eucl_scale = 1. / self.eucl_distances.max()
        discr_dist_max = (
            self.discr_distances.max() + 1 if self.discr_distances.max() == 0 else self.discr_distances.max()
        )
        fisher_scale = 1. / discr_dist_max
        eucl = self.eucl_distances * eucl_scale * self.lam
        fisher = self.discr_distances * fisher_scale * (1. - self.lam)
        stacked = np.dstack((fisher, eucl))
        return stacked.sum(-1)

    def get_artist_sample(self, sample_ids):
        """Maps the location of embedded points to their image.

        Parameters
        ----------
        sample_ids: tuple
            The ids of selected vertices.
        """
        # sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
        yps = []
        yts = []
        for sample_id in sample_ids:
            yp, yt = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
            yps.append(yp)
            yts.append(yt)
        return sample_ids, yps, yts

    def show(self):
        """Shows the current plot."""
        if not hasattr(self, 'fig'):
            self._init_plots()

        x_min, y_min, x_max, y_max = self._get_plot_measures()

        self.cls_plot.set_data(self.classifier_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
        desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
        self.desc.set_text(desc)

        for c in range(self.n_classes):
            data = self.embedded[self.y_true == c]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.n_classes):
            data = self.embedded[np.logical_and(self.y_pred == c, self.background_at != c)]
            self.sample_plots[self.n_classes + c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.selector = SelectFromCollection(self.ax, self.embedded)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def _init_plots(self):
        '''
        Initialises matplotlib artists and plots.
        '''
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
                                       interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        for c in range(self.n_classes):
            color = self.cmap(c / (self.n_classes - 1))
            plot = self.ax.plot([], [], 'o', label=self.class_dict[c],
                                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        for c in range(self.n_classes):
            color = self.cmap(c / (self.n_classes - 1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                                fillstyle='none', ms=12, mew=2.5, zorder=1)
            self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.legend()

    def show_sample(self, event):
        '''
        Invoked when the user clicks on the plot. Determines the
        embedded or synthesised sample at the click location and
        passes it to the data_viz method, together with the prediction,
        if present a groun truth label and the 2D click location.
        '''

        # when there is an artist attribute, a
        # concrete sample was clicked, otherwise
        # show the according synthesised image

        if event.key == "enter":
            indices = self.selector.ind
            sample, p, t = self.get_artist_sample(indices)
            # title = '%s <-> %s' if p != t else '%s --- %s'
            # title = title % (self.classes[p], self.classes[t])
            self.disable_synth = True
        elif not self.disable_synth:
            # workaraound: inverse embedding needs more points
            # otherwise it doens't work --> [point]*5
            point = np.array([[event.xdata, event.ydata]] * 5)

            # if the outside of the plot was clicked, points are None
            if None in point[0]:
                return

            sample = self.inverse(point)[0]
            sample += abs(sample.min())
            sample /= sample.max()
            # title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
            p, t = self.get_mesh_prediction_at(*point[0]), None
        else:
            self.disable_synth = False
            return

        if self.data_viz is not None:
            self.data_viz(sample, p, t, self.cmap)
            return
        else:
            warnings.warn("Data visualization not possible, as the partnet_grasp points have"
                          "no image shape. Pass a function in the data_viz argument,"
                          "to enable custom partnet_grasp visualization.")
            return
