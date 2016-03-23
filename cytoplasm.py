from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from util import to_list_func, FloatOrPercent, Colors, image_iter, hist_to_threshold

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("inputcytoplasm",
    help="Input cytoplasm file. Should be a tif.",
    type=Path, metavar="<cytoplasm_file>.tif")
parser.add_argument("inputnucleus",
    help="Input nucleus file. Should be a tif.",
    type=Path, metavar="<nucleus_file>.tif")
parser.add_argument("inputmeasure",
    help="Input measure file. Should be a tif.",
    type=Path, metavar="<measure_file>.tif")
parser.add_argument("output", default=None,
    help="""Output file basename (without extension). Defaults to match input.
        Example: 'imgs/Jan01_2015_abc'
        """, nargs="?",
    type=Path, metavar="<file>")
parser.add_argument("-e", "--extension", default='pdf',
    help="Extension of image type to use for output images.")
parser.add_argument("--limits", default=None, metavar='UPPER_LEFT_LOWER_RIGHT',
    help="""Limits for cropping, from the upper left corner.
        Example: '--limit 192,224,320,288' for a box centered at (256,256) of
        width 128 and height 64. Leave blank for no cropping.""",
    type=to_list_func(4, int))
parser.add_argument('-t', '--thresholds',
    type=to_list_func(3, FloatOrPercent),
    default=(FloatOrPercent('90%'), FloatOrPercent('80%'), FloatOrPercent('90%')),
    help="""Set three thresholds. Value should either be in pixel value, e.g.
        '20', or a percentage value, e.g. '99.9%%'. Percentage values will be
        applied to the histogram to extract a threshold. Use a comma to separate.
        Example: -t 99%%,10%%,99%%""")
parser.add_argument('-u', '--upperthresholds', metavar='THRESHOLD',
    type=to_list_func(3, FloatOrPercent),
    default=(FloatOrPercent('99.99%'), FloatOrPercent('99%'), FloatOrPercent('99.99%')),
    help="""
Set upper thresholds for normalization, with the same format as '--threshold'.
""")
parser.add_argument('--loghistogram', action='store_true',
    help="""Use a logarithmic scale for the histograms.""")
parser.add_argument('--pictures', '-p', action='count', default=0,
    help="Also produce images of various transformations. " +
    "Repeat up to three times to get more detailed images (e.g. '-ppp').")

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

inputs = []
for name in (args.inputcytoplasm, args.inputnucleus, args.inputmeasure):
    try:
        img = Image.open(str(name))
    except FileNotFoundError as e:
        print("Errored on", name)
        print(e)
        exit(2)
    except OSError as e:
        print("Errored on", name)
        print(e)
        exit(3)
    if img.mode == 'I;16B':
        MAX_PIXEL = (1 << 16) - 1
        # print("Image type %r recognized." % img.mode)
    elif img.mode in 'LP':
        print("Errored on", name)
        print(("Image type %r recognized, but this program is not written" +
         "for an 8-bit image.") % img.mode)
        print("Unsure how to proceed; exiting.")
        exit(1)
    else:
        print("Errored on", name)
        print("Image type %r recognized, but not a 0-255 grayscale image." % img.mode)
        print("Unsure how to proceed; exiting.")
        exit(4)
    
    if args.limits is None:
        inputs.append(list(image_iter(img)))
    else:
        inputs.append([im.crop(args.limits) for im in image_iter(img)])


def renorm_bin(arr, thresh, upper):
    float_arr = np.asarray(arr, dtype=np.float)
    normed = float_arr / upper_thresh
    np.clip(normed, 0, 1, out=normed)
    binned = arr > thresh
    return normed, binned


def arr_to_img(arr):
    return Image.fromarray(np.uint8(arr * 255))

names = ('Cytoplasm', 'Nucleus', 'Measurements')
if args.pictures >= 1:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
else:
    axs = [None for _ in names]
normeds, binneds = [], []
for img_list, threshold, upper_threshold, name, ax in (
        zip(inputs, args.thresholds, args.upperthresholds, names, axs)):
    # thresh, upper_thresh, imgs, imgbins = to_arr_lists(img_list, threshold, upper_threshold)
    bins = np.linspace(0, (1 << 16), (1 << 16) + 1) - 0.5
    mids = (bins[:-1] + bins[1:]) / 2
    
    arrs = [np.asarray(im) for im in img_list]
    arr = np.asarray([np.asarray(im) for im in img_list])
    # hist, _, _ = ax.hist(arr.flatten(),
    #     bins=bins, histtype='step',
    #     color=Colors.red, linewidth=2)
    hist, _ = np.histogram(arr.flatten(), bins=bins)
    
    lower_thresh_ix = hist_to_threshold(hist, threshold)
    upper_thresh_ix = hist_to_threshold(hist, upper_threshold)
    lower_thresh, upper_thresh = (
        np.ceil(bins[lower_thresh_ix]),
        np.ceil(bins[upper_thresh_ix])
    )
    
    print(name, 'Threshold:', lower_thresh, 'Upper:', upper_thresh)
    
    normed, binned = zip(*[renorm_bin(a, lower_thresh, upper_thresh) for a in arrs])
    normeds.append(np.asarray(normed))
    binneds.append(np.asarray(binned))
    if args.pictures >= 1:
        ax.plot(mids/1e3, hist, color=Colors.red, linewidth=2)
        ax.axvline(lower_thresh/1e3, color='k', linewidth=2, linestyle=':')
        ax.axvline(upper_thresh/1e3, color='k', linewidth=2)
        ax.set_xlabel('Pixel Value (1000s)')
        ax.set_ylabel('Count')
        ax.set_xlim(0, np.ceil(np.amax(bins)/1000))
        if args.loghistogram:
            ax.set_yscale('symlog', linthreshy=1)
        ax.set_title(name)

if args.pictures >= 1:
    fig.tight_layout()
    fig.savefig(output_str('histogram'))
    plt.close(fig)

cyto_norm, nuc_norm, measure_norm = normeds
cyto_bin, nuc_bin, measure_bin = binneds
nuc_area = np.sum(np.sum(nuc_bin, axis=2), axis=1)
cyto_area = np.sum(np.sum(cyto_bin, axis=2), axis=1)
cyto_not_nuc_area = np.sum(np.sum(cyto_bin & (~nuc_bin), axis=2), axis=1)
measure_area = np.sum(np.sum(measure_bin, axis=2), axis=1)
measure_not_nuc_area = np.sum(np.sum(measure_bin & (~nuc_bin), axis=2), axis=1)

measure_nuc = measure_norm
measure_not_nuc = measure_norm * (~nuc_bin)
measure_cyto_not_nuc = measure_norm * (cyto_bin & (~nuc_bin))

measure_total = np.sum(np.sum(measure_norm, axis=2), axis=1)
measure_nuc_total = np.sum(np.sum(measure_norm * nuc_bin, axis=2), axis=1)
measure_not_nuc_total = np.sum(np.sum(measure_not_nuc, axis=2), axis=1)
measure_cyto_not_nuc_total = np.sum(np.sum(measure_cyto_not_nuc, axis=2), axis=1)

if args.pictures >= 1:
    arr_to_img(measure_norm[0]).save(output_str('measure_frame0'))
    arr_to_img(measure_nuc[0]).save(output_str('measure_nuc_frame0'))
    arr_to_img(measure_not_nuc[0]).save(output_str('measure_not_nuc_frame0'))
    arr_to_img(measure_cyto_not_nuc[0]).save(output_str('measure_cyto_not_nuc_frame0'))
if args.pictures >= 2:
    arr_to_img(measure_norm[-1]).save(output_str('measure_end'))
    arr_to_img(measure_nuc[-1]).save(output_str('measure_nuc_end'))
    arr_to_img(measure_not_nuc[-1]).save(output_str('measure_not_nuc_end'))
    arr_to_img(measure_cyto_not_nuc[-1]).save(output_str('measure_cyto_not_nuc_end'))
if args.pictures >= 3:
    arr_to_img(nuc_bin[0]).save(output_str('nuc_bin_frame0'))
    arr_to_img(cyto_bin[0]).save(output_str('cyto_bin_frame0'))
    arr_to_img(measure_bin[0]).save(output_str('measure_bin_frame0'))
    arr_to_img(nuc_bin[-1]).save(output_str('nuc_bin_end'))
    arr_to_img(cyto_bin[-1]).save(output_str('cyto_bin_end'))
    arr_to_img(measure_bin[-1]).save(output_str('measure_bin_end'))

N_frames = len(measure_total)
frame_no = np.arange(N_frames) + 1

if args.pictures >= 1:
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
    kw = dict(linewidth=2, marker='o')
    ax1.plot(frame_no, measure_total, color=Colors.red, label='Total', **kw)
    ax1.plot(frame_no, measure_nuc_total, color=Colors.green, label='Nucleus', **kw)
    ax1.plot(frame_no, measure_not_nuc_total, color=Colors.purple, label='Not Nucleus', **kw)
    ax1.plot(frame_no, measure_cyto_not_nuc_total,
        color=Colors.orange, label='Cytoplasm, Not Nucleus', **kw)
    
    ax2.plot(frame_no, measure_area, 'o-', color=Colors.red, label='Total Measure', **kw)
    ax2.plot(frame_no, nuc_area, color=Colors.green, label='Nucleus', **kw)
    ax2.plot(frame_no, measure_not_nuc_area, color=Colors.purple,
        label='Measure, Not Nucleus', **kw)
    ax2.plot(frame_no, cyto_not_nuc_area, color=Colors.orange,
        label='Cytoplasm, Not Nucleus', **kw)

    for fig, ax, title, name in (
            (fig1, ax1, 'Measure Total', 'measure'),
            (fig2, ax2, 'Areas Measured', 'area')):
        ax.set_xlabel('Frame')
        ax.set_ylabel('Sums')
        ax.set_title(title)
        ax.set_ylim(0, None)
        ax.legend(ncol=2, loc='lower center', fontsize='x-small',
            framealpha=0.6, fancybox=True, frameon=True)
        fig.tight_layout()
        
        fig.savefig(output_str(name))
        plt.close(fig)


columns = [
    ('Frame', frame_no),
    ('Nucleus Area', nuc_area),
    ('Cytoplasm Area', cyto_area),
    ('Measure Area', measure_area),
    ('Cytoplasm Not Nucleus Area', cyto_not_nuc_area),
    ('Measure minus Nucleus Area', measure_not_nuc_area),
    ('Measure', measure_total),
    ('Measure Nucleus', measure_nuc_total),
    ('Measure Not Nucleus', measure_not_nuc_total),
    ('Measure Cytoplasm Not Nucleus', measure_cyto_not_nuc_total),
]

headers, rows = zip(*columns)
headerrow = ','.join(headers)

csvname = output_str('', extension='.csv')
np.savetxt(csvname, np.asarray(rows).T,
    delimiter=',', header=headerrow, fmt='%.6g')
