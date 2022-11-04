"""
http://colorizer.org/
"""
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from psd_tools import PSDImage
import numpy as np
import pandas as pd
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.util.shape import view_as_windows, view_as_blocks
from skimage.transform import rescale, resize, downscale_local_mean
import math
import ternary


def load_image(x):
    img = io.imread(x)
    hs = color.rgb2hsv(img)
    rgb = img/255
    return hs, rgb

def get_mask(x):
    psd = PSDImage.open(x)
    for layer in psd:
        print(dir(psd))
        #print(psd._record)

def pan_hue_saturation(x, names, radius=5):
    # fig, axes = plt.subplots(1, len(x), subplot_kw=dict(polar=True))
    fig = plt.figure(figsize=(10,10))
    for i, (arr, n) in enumerate(zip(x, names)):
        np.random.seed(19680801)
        X = arr[:, :, 0]
        Y = arr[:, :, 1]
        Z = arr[:, :, 2]
        X = downscale_local_mean(X, (radius, radius))
        Y = downscale_local_mean(Y, (radius, radius))
        Z = downscale_local_mean(Z, (radius, radius))
        ax = fig.add_subplot(3, len(x), i+1)
        ax.imshow(X, cmap='hsv')
        ax.set_title(n)

        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)

        selection = (X!=0) & (Y!=0) & (Z!=0)
        X = X[selection]
        Y = Y[selection]
        Z = Z[selection]

        hist = np.histogram(X, np.arange(0, 1.1, 0.05))
        peak = hist[1][np.argsort(hist[0])]

        ax = fig.add_subplot(3, len(x), i+len(x)+1)
        ax.hist(X, bins=20)
        ax.axvline(x=peak[-1]-0.1, color='b', linestyle='dashed', linewidth=2)
        ax.axvline(x=peak[-1]+0.1, color='b', linestyle='dashed', linewidth=2)
        ax.axvline(x=peak[-2]-0.1, color='r', linestyle='dashed', linewidth=2)
        ax.axvline(x=peak[-2]+0.1, color='r', linestyle='dashed', linewidth=2)
        ax.axvline(x=peak[-3]-0.1, color='g', linestyle='dashed', linewidth=2)
        ax.axvline(x=peak[-3]+0.1, color='g', linestyle='dashed', linewidth=2)

        # X_sel = ((X > peak[-1]-0.1) & (X < peak[-1] + 0.1)) | ((X > peak[-2]-0.1) & (X < peak[-2] + 0.1)) | ((X > peak[-3]-0.1) & (X < peak[-3]+0.1))
        # X = X[X_sel]
        # Y = Y[X_sel]
        # Z = Z[X_sel]

        # X_sel = np.random.choice(np.arange(len(X)), 10000, replace=False)
        # X = X[X_sel]
        # Y = Y[X_sel]
        # Z = Z[X_sel]

        ax = fig.add_subplot(3, len(x), i+len(x)+6, projection='polar')
        ax.scatter(np.arange(0, 1, 0.001)*2*np.pi, [1]*1000, marker='.', c=np.arange(0, 1, 0.001), cmap='hsv',
                   vmin=0, vmax=1,
                   alpha=1)

        ## https://stackoverflow.com/questions/9071084/polar-contour-plot-in-matplotlib-best-modern-way-to-do-it
        # ax.scatter(X*2*np.pi, Y, c=X, cmap='hsv', alpha=0.65, marker='.', s=[0.05]*len(X), #s=5*(Z**2),
        #            ##edgecolors='black',
        #            vmin=0, vmax=1) #  s=2*(2*Y)**2
        ax.contourf(X*2*np.pi, Y, X, cmap='hsv', alpha=0.65)  ## Z should be 2d array
        ax.set_xticklabels(['0', '', '90', '', '180', '', '270', ''])
        ax.set_rmax(1.02)
        plt.tight_layout()
    plt.show()


def polar_heatmap(x):
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                     [[0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]]])
    data = x
    print(data.shape)
    # data = np.repeat(data, 25, axis=1)
    # print(data.shape)
    ax = plt.subplot(111, polar=True)

    # get coordinates:
    phi = np.linspace(0, 2 * np.pi, data.shape[1] + 1)
    r = np.linspace(0, 1, data.shape[0] + 1)
    Phi, R = np.meshgrid(phi, r)
    # get color
    color = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

    # plot colormesh with Phi, R as coordinates,
    # and some 2D array of the same shape as the image, except the last dimension
    # provide colors as `color` argument
    m = plt.pcolormesh(Phi, R, data[:, :, 0], color=color, linewidth=0)
    # This is necessary to let the `color` argument determine the color
    m.set_array(None)
    plt.show()


## Generate Data
import random
def random_points(num_points=25, scale=40):
    points = []
    for i in range(num_points):
        x = random.randint(1, scale)
        y = random.randint(0, scale - x)
        z = scale - x - y
        points.append((x,y,z))
    return points

def color_point(x, y, z, scale):
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 0.5)

def ternary_plot(x):
    # scale = 30
    # figure, tax = ternary.figure(scale=scale)
    # tax.set_title("RGB ternary scatter plot", fontsize=20)
    # tax.boundary(linewidth=2.0)
    # tax.boundary(linewidth=1.5)
    # tax.gridlines(color="black", multiple=6)
    # tax.gridlines(color="blue", multiple=2, linewidth=0.5)
    # points = random_points(30, scale=scale)
    # tax.scatter(points, marker='s', color='red', label="Red Squares")

    def generate_heatmap_data(scale=5):
        from ternary.helpers import simplex_iterator
        d = dict()
        for (i, j, k) in simplex_iterator(scale):
            d[(i, j, k)] = color_point(i, j, k, scale)
        return d

    scale = 80
    data = generate_heatmap_data(scale)
    print(data)
    print(len(data))
    figure, tax = ternary.figure(scale=scale)
    tax.heatmap(data, style="hexagonal", use_rgba=True, colorbar=False)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.boundary()
    tax.set_title("Tenary plots")
    tax.show()
    return


def test_color():
    ## limit color range: https://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib

    # Fixing random state for reproducibility
    from matplotlib import colors

    c = colors.LinearSegmentedColormap
    np.random.seed(19680801)
    # Compute areas and colors
    N = 150
    r = np.array([0.5, 0.2, 0.6])
    theta = 2 * np.pi * np.array([1/16, 0.2, 2.5/4])
    area = 200 * r ** 2
    colors = theta

    v = plt.get_cmap('hsv')
    print(dir(v))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    # print(colors.min(), colors.max())
    ax.scatter(np.arange(0, 0.5, 0.001) * 2 * np.pi,
               [1.1] * 500, c=np.arange(0, 0.5, 0.001), cmap='hsv',
               vmin=0, vmax=1,
               alpha=1)
    # c = plt.scatter(theta, r, c=np.array([1/16, 0.2, 2.5/4]), s=area, cmap='hsv', alpha=0.75)
    plt.show()

def main():
    # img1 = load_image('/home/alvin/Dropbox (Partners HealthCare)/Alvin - Tiffany/Demo images/(1 clone) zebrabow_HC_Cre5pg_kRAS30pg_10x_83dpf_2_6_5_1tumour-normal Maximum intensity projection.tif')
    # # mask = get_mask(
    # #     '/home/alvin/Dropbox (Partners HealthCare)/Alvin - Tiffany/Demo images/(1 clone, masked) zebrabow_HC_Cre5pg_kRAS30pg_10x_83dpf_2_6_5_1tumour-normal Maximum intensity projection.psd')
    # img2 = load_image('/home/alvin/Dropbox (Partners HealthCare)/Alvin - Tiffany/Demo images/(2 obvious clones) zebrabow_HC_Cre5pg_kRAS30pg_10x_55dpf_2_3_2_tumour1-2')
    # img3 = load_image('/home/alvin/Dropbox (Partners HealthCare)/Alvin - Tiffany/Demo images/(maybe 2 clones) zebrabow_HC_Cre5pg_kRAS30pg_MDM2-15pg_10x_79dpf_3_6_3_1tumour-normal_2019_08_27__16_24_43_Maximum intensity projection.tif')
    img1 = load_image('/home/alvin/Desktop/1_cut.tif')
    img2 = load_image('/home/alvin/Desktop/2_1_cut.tif')
    img3 = load_image('/home/alvin/Desktop/2_2_cut.tif')
    img4 = load_image('/home/alvin/Desktop/3_cut.tif')

    img_merge = img2[0] + img3[0]
    # pan_hue_saturation([img1[0], img_merge, img4[0]],
    #                    names=['tumor 1 1 clone', 'tumor 2 2 clones', 'tumor 3 perhaps 2 clone'])
    pan_hue_saturation([img1[0], img_merge, img2[0], img3[0], img4[0]],
                       names=['tumor 1 1 clone', 'tumor 2 merge', 'tumor 2 1st clone', 'tumor 2 2nd clone', 'tumor 3 perhaps 2 clone'])

    # pan_hue_saturation([img1[0], img_merge, img4[0]],
    #                    names=['tumor 1 1 clone', 'tumor 2 1st clone', 'tumor 2 2nd clone', 'tumor 3 perhaps 2 clone'])

    #polar_heatmap(img2)

    # ternary_plot(img1)
    # test_color()

if __name__ == '__main__':
    main()
