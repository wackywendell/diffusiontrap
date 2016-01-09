from pathlib import Path
import argparse

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
parser.add_argument('-b', '--binarize', action='store_true',
    help="""
Use a binary input for the cyan channel. That is, for all pixels
in the cyan channel that are greater than the threshold, treat them all
as fully white.""")
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
    return threshold, maximum, [im.point(renorm) for im in img_list]


def binarize(img, threshold):
    hist = img.histogram()
    threshold = hist_to_threshold(hist, threshold)
    
    def cur_binarize(pt):
        return MAX_PIXEL if pt >= threshold else 0
    
    return threshold, [im.point(cur_binarize) for im in img_list]

if args.binarize:
    cyans_threshold, cyans = [binarize(img, args.threshold)
        for img in cyans_unnormalized]
    cyans_upper_threshold = MAX_PIXEL
else:
    cyans_threshold, cyans_upper_threshold, cyans = renormalize_all(
        cyans_unnormalized, args.threshold, args.upperthreshold)

yellows_threshold, yellows_upper_threshold, yellows = renormalize_all(
    yellows_unnormalized, 0, args.upperthreshold)

img_lists = (cyans, yellows)
thresholds = cyans_threshold, yellows_threshold
upper_thresholds = cyans_upper_threshold, yellows_upper_threshold
names = ('Cyan Channel', 'Yellow Channel')

for img_list, thresh, upper, name in zip(
        img_lists, thresholds, upper_thresholds, names):
    print("Using threshold %d and maximum %d for %s" %
        (thresh, upper, name.lower()))
    out_name = output_str(name.split(' ')[0].lower() + '0')
    img_list[0].save(out_name)

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

areas = np.asarray([ImageStat.Stat(c).sum[0] / float(MAX_PIXEL) for c in cyans])
multips = [ImageChops.multiply(c, y) for c, y in zip(cyans, yellows)]
multips[0].save(output_str('multiplied'))
multip_values = np.asarray([
    ImageStat.Stat(i).sum[0] / float(MAX_PIXEL)
    for i in multips])
means = multip_values / areas
