from pathlib import Path
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageStat

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


class FloatOrPercent:
    def __init__(self, strng):
        if isinstance(strng, str) and strng[-1] == '%':
            self.value = float(strng[:-1]) / 100.0
            self.type = '%'
        else:
            self.value = float(strng)
            self.type = '#'
    
    def __str__(self):
        if self.type == '#':
            return str(self.value)
        elif self.type == '%':
            return str(self.value * 100.0) + '%'
    
    def __repr__(self):
        if self.type == '#':
            return str(self.value)
        elif self.type == '%':
            return str(self.value * 100.0) + '%'
    
    def __float__(self):
        return self.value


class Colors:
    red, blue, green, purple, orange, brown, ppink, grey, yellow, black = (
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
        '#A65628', '#F781BF', '#999999', '#FFFF33', '#000000')

parser.add_argument("input",
    help="Input file. Should be a tif with a sequence of embedded images.",
    type=Path, metavar="<file>.tif")
parser.add_argument("output", default=None,
    help="""Output file basename (without extension). Defaults to match input.
        Example: 'imgs/Jan01_2015_abc'
        """, nargs="?",
    type=Path, metavar="<file>")
parser.add_argument("-e", "--extension", default='pdf',
    help="Extension of image type to use for output images.")
parser.add_argument("--limits", default=None, metavar='UPPER,LEFT,LOWER,RIGHT',
    help="""Limits for cropping, from the upper left corner.
        Example: '--limit 192,224,320,288' for a box centered at (256,256) of
        width 128 and height 64. Leave blank for no cropping.""",
    type=lambda s: [int(n) for n in s.strip().split(',')])
parser.add_argument('-t', '--threshold',
    type=FloatOrPercent, default=FloatOrPercent('99%'),
    help="""Set threshold for cyan. Value should either be in pixel value, e.g.
        '20', or a percentage value, e.g. '99.9%%'. Percentage values will be
        applied to the histogram to extract a threshold.""")
# TODO: add yellow threshold
parser.add_argument('-u', '--upperthreshold', metavar='THRESHOLD',
    type=FloatOrPercent, default=FloatOrPercent('99.9%'),
    help="""
Set upper threshold for normalization, with the same format as '--threshold'.
""")
# TODO: add yellow upper threshold
# parser.add_argument('-b', '--binarize', action='store_true',
#     help="""
# Use a binary input for the cyan channel. That is, for all pixels
# in the cyan channel that are greater than the threshold, treat them all
# as fully white.""")
parser.add_argument('--loghistogram', action='store_true',
    help="""Use a logarithmic scale for the histograms.""")

args = parser.parse_args()
if args.output is None:
    args.output = args.input.parent / args.input.stem


def output_str(name, base=args.output, extension=args.extension):
    name_component = '_' + name if name else ''
    ext_component = extension if extension[0] == '.' else '.' + extension
    path = base.parent / (
        base.name + name_component + ext_component)
    return str(path)

# img_filename = 'data/PKIM-98-w1-s8.tif'
# threshold = 20
# expand_thresh = 0.999


def image_iter(img):
    """
    Given a PIL.Image with multiple embedded images, this iterates over them.
    """
    n = 0
    while True:
        try:
            img.seek(n)
        except EOFError:
            img.seek(0)
            raise StopIteration
        yield img.copy()
        n += 1

try:
    img = Image.open(str(args.input))
except FileNotFoundError as e:
    print(e)
    exit(2)
except OSError as e:
    print(e)
    exit(3)
if not img.mode == 'P':
    print("Image type recognized, but not a 0-255 grayscale image.")
    print("Unsure how to proceed; exiting.")
    exit(1)
MAX_PIXEL = 255

img_list = (
    list(image_iter(img))
    if args.limits is None else
    [im.crop(args.limits) for im in image_iter(img)])
cyans_unnormalized = img_list[::2]
yellows_unnormalized = img_list[1::2]

hists = cyan_hist, yellow_hist = [
    np.sum([im.histogram() for im in img_list], axis=0)
    for img_list in (cyans_unnormalized, yellows_unnormalized)
]


def hist_to_threshold(hist, threshold):
    if not isinstance(threshold, FloatOrPercent) or threshold.type == '#':
        return float(threshold)
    ixs, = np.nonzero(np.cumsum(hist) / np.sum(hist) >= float(threshold))
    return ixs[0]


def renormalize_all(img_list, threshold=FloatOrPercent(0),
        maximum=args.upperthreshold):
    """
    Renormalize an image set to go from (0-upper) to (0-MAX).
    """
    hist = np.sum([im.histogram() for im in img_list], axis=0)
    threshold = hist_to_threshold(hist, threshold)
    maximum = hist_to_threshold(hist, maximum)
    
    def renorm(pt):
        return min(MAX_PIXEL, pt*MAX_PIXEL/maximum) if pt >= threshold else 0
    
    def binarize(pt):
        return MAX_PIXEL if pt >= threshold else 0
        
    return (
        threshold,
        maximum,
        [im.point(renorm) for im in img_list],
        [im.point(binarize) for im in img_list]
    )


def to_arr_lists(img_list, thresh, upper_thresh):
    thresh, upper_thresh, lst, bin_lst = renormalize_all(
        img_list, thresh, upper_thresh)
    arrs = np.asarray([np.asarray(im) for im in lst], dtype=float) / MAX_PIXEL
    bins = np.asarray([np.asarray(im) > 0 for im in bin_lst], dtype=bool)
    return thresh, upper_thresh, arrs, bins


def arr_to_img(arr):
    return Image.fromarray(np.uint8(arr * MAX_PIXEL))

cyans_threshold, cyans_upper_threshold, cyans, cyans_bin = to_arr_lists(
    cyans_unnormalized, args.threshold, args.upperthreshold)
# cyans_threshold, cyans_upper_threshold, cyans, cyans_bin = renormalize_all(
#     cyans_unnormalized, args.threshold, args.upperthreshold)
# cyan_arrs = np.asarray([np.asarray(im) for im in cyans], dtype=float)
# cyan_bins = np.asarray([np.asarray(im) > 0 for im in cyans_bin], dtype=bool)

yellows_threshold, yellows_upper_threshold, yellows, yellows_bin = (
    to_arr_lists(yellows_unnormalized, 0, args.upperthreshold))
# yellows_threshold, yellows_upper_threshold, yellows, yellows_bin = (
#     renormalize_all(yellows_unnormalized, 0, args.upperthreshold))
# cyan_arrs = np.asarray([np.asarray(im) for im in cyans], dtype=float)
# cyan_bins = np.asarray([np.asarray(im) > 0 for im in cyans_bin], dtype=bool)

img_lists = (cyans, yellows)
thresholds = cyans_threshold, yellows_threshold
upper_thresholds = cyans_upper_threshold, yellows_upper_threshold
names = ('Cyan Channel', 'Yellow Channel')

for img_list, thresh, upper, name in zip(
        img_lists, thresholds, upper_thresholds, names):
    print("Using threshold %d and maximum %d for %s" %
        (thresh, upper, name.lower()))
    out_name = output_str(name.split(' ')[0].lower() + '0')
    arr_to_img(img_list[0]).save(out_name)

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
(ax1, ax2) = axs
for ax, thresh, upper, hist, name in zip(
        axs, thresholds, upper_thresholds, hists, names):
    ax.plot(hist, color=Colors.red, linewidth=2)
    ax.axvline(thresh, color='k', linewidth=2, linestyle=':')
    ax.axvline(upper, color='k', linewidth=2)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.set_xlim(None, 256)
    if args.loghistogram:
        ax.set_yscale('symlog', linthreshy=1)
    ax.set_title(name)
    
fig.tight_layout()
fig.savefig(output_str('histogram'))
plt.close(fig)

areas = np.asarray([np.sum(c) for c in cyans])
areas_bin = np.asarray([np.sum(c) for c in cyans_bin])
multips = [c*y for c, y in zip(cyans, yellows)]
arr_to_img(multips[0]).save(output_str('multiplied'))
y_normed = np.asarray([np.sum(m) for m in multips]) / areas
cyan_total = np.asarray([np.sum(c*b) for c, b in zip(cyans, cyans_bin)])
cyan_mean = cyan_total / areas_bin
yellow_total = np.asarray([np.sum(y*b) for y, b in zip(yellows, cyans_bin)])
yellow_mean = cyan_total / areas_bin

yellow_cyan = yellow_total / cyan_total
frame_values_by_total = np.asarray([0] + [
    np.mean(yellow_cyan[n:]) / np.mean(yellow_cyan[:n])
    for n in range(1, len(yellow_cyan))
])
frame_values_by_norm = np.asarray([0] + [
    np.mean(y_normed[n:]) / np.mean(y_normed[:n])
    for n in range(1, len(yellow_cyan))
])


columns = [
    ('Normalized Yellow', y_normed),
    ('Normalized Area', areas),
    ('Yellow Total', yellow_total),
    ('Yellow Mean', yellow_mean),
    ('Cyan Total', cyan_total),
    ('Cyan Mean', cyan_mean),
    ('Relative Average Yellow / Cyan by Total', frame_values_by_total),
    ('Relative Average Yellow / Cyan by Norm', frame_values_by_norm),
]

headers, rows = zip(*columns)
headerrow = ','.join(headers)

csvname = output_str('', extension='.csv')
# Columns wanted were:
# yellow total, cyan total, yellow/cyan, area
np.savetxt(csvname, np.asarray(rows).T,
    delimiter=',', header=headerrow, fmt='%.6f')
