# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

import matplotlib.pyplot as plt
from cycler import cycler


def plot_graph(
    scene,
    fscore,
    dist_threshold,
    edges_source,
    cum_source,
    edges_target,
    cum_target,
    plot_stretch,
    mvs_outpath,
    show_figure=False,
):
    f = plt.figure()
    plt_size = [14, 7]
    pfontsize = "medium"

    ax = plt.subplot(111)
    label_str = "precision"
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "recall"
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)
    plt.rcParams["figure.figsize"] = plt_size
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))
    plt.title("Precision and Recall: " + scene + ", " + "%02.2f f-score" %
              (fscore * 100))
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)

    plt.ylabel("# of points (%)", fontsize=15)
    plt.xlabel("Meters", fontsize=15)
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)
    plt.legend(loc=4)
    leg = plt.legend(loc="lower right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)
    png_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.png".format(
        scene, "%04d" % (dist_threshold * 10000))
    pdf_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.pdf".format(
        scene, "%04d" % (dist_threshold * 10000))

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()
