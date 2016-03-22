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
    type=Path, metavar="<file>.tif")
parser.add_argument("inputnucleus",
    help="Input nucleus file. Should be a tif.",
    type=Path, metavar="<file>.tif")
parser.add_argument("inputmeasure",
    help="Input measure file. Should be a tif.",
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
    type=to_list_func(4, int))
parser.add_argument('-t', '--thresholds',
    type=to_list_func(3, FloatOrPercent),
    default=(FloatOrPercent('90%'), FloatOrPercent('80%'), FloatOrPercent('90%')),
    help="""Set three thresholds. Value should either be in pixel value, e.g.
        '20', or a percentage value, e.g. '99.9%%'. Percentage values will be
        applied to the histogram to extract a threshold. Use a comma to separate.
        Example: -t 99%,10,99%""")
parser.add_argument('-u', '--upperthresholds', metavar='THRESHOLD',
    type=to_list_func(3, FloatOrPercent),
    default=(FloatOrPercent('99.99%'), FloatOrPercent('99.99%'), FloatOrPercent('99.99%')),
    help="""
Set upper thresholds for normalization, with the same format as '--threshold'.
""")
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
        print("Image type %r recognized." % img.mode)
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
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
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
    ax.plot(mids/1e3, hist, color=Colors.red, linewidth=2)
    
    lower_thresh_ix = hist_to_threshold(hist, threshold)
    upper_thresh_ix = hist_to_threshold(hist, upper_threshold)
    lower_thresh, upper_thresh = (
        np.ceil(bins[lower_thresh_ix]),
        np.ceil(bins[upper_thresh_ix])
    )
    
    normed, binned = zip(*[renorm_bin(a, lower_thresh, upper_thresh) for a in arrs])
    normeds.append(np.asarray(normed))
    binneds.append(np.asarray(binned))
    
    ax.axvline(lower_thresh/1e3, color='k', linewidth=2, linestyle=':')
    ax.axvline(upper_thresh/1e3, color='k', linewidth=2)
    ax.set_xlabel('Pixel Value (1000s)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, np.ceil(np.amax(bins)/1000))
    if args.loghistogram:
        ax.set_yscale('symlog', linthreshy=1)
    ax.set_title(name)
    
fig.tight_layout()
fig.savefig(output_str('histogram'))
plt.close(fig)

cyto_norm, nuc_norm, measure_norm = normeds
cyto_bin, nuc_bin, measure_bin = binneds
just_cyto = cyto_norm * (1.0 - nuc_norm)
just_cyto_bin = cyto_norm * (~nuc_bin)
cyto_area = np.sum(np.sum(just_cyto, axis=2), axis=1)
cyto_area_bin = np.sum(np.sum(just_cyto_bin, axis=2), axis=1)
measured_cyto = measure_norm * just_cyto
measured_cyto_bin = measure_norm * just_cyto_bin
measured_nuc = measure_norm * nuc_norm
measured_nuc_bin = measure_norm * nuc_bin
measured_not_nuc = measure_norm * (1.0 - nuc_norm)
measured_not_nuc_bin = measure_norm * (~nuc_bin)

# arr_to_img(nuc_norm[0]).save(output_str('nucleus0'))
# arr_to_img(nuc_norm[-1]).save(output_str('nucleus1'))
# arr_to_img(measure_bin[0]).save(output_str('measure0'))
# arr_to_img(measure_bin[-1]).save(output_str('measure1'))
arr_to_img(measured_cyto[0]).save(output_str('measure_cyto0'))
arr_to_img(measured_cyto[-1]).save(output_str('measure_cyto1'))
arr_to_img(measured_cyto_bin[0]).save(output_str('measure_cyto_bin0'))
arr_to_img(measured_cyto_bin[-1]).save(output_str('measure_cyto_bin1'))
arr_to_img(measured_nuc[0]).save(output_str('measure_nuc0'))
arr_to_img(measured_nuc[-1]).save(output_str('measure_nuc1'))
arr_to_img(measured_not_nuc[0]).save(output_str('measure_not_nuc0'))
arr_to_img(measured_not_nuc[-1]).save(output_str('measure_not_nuc1'))
arr_to_img(measured_nuc_bin[0]).save(output_str('measure_nuc_bin0'))
arr_to_img(measured_nuc_bin[-1]).save(output_str('measure_nuc_bin1'))
arr_to_img(measured_not_nuc_bin[0]).save(output_str('measure_not_nuc_bin0'))
arr_to_img(measured_not_nuc_bin[-1]).save(output_str('measure_not_nuc_bin1'))
arr_to_img(just_cyto[0]).save(output_str('just_cytoplasm0'))
arr_to_img(just_cyto[-1]).save(output_str('just_cytoplasm1'))
arr_to_img(just_cyto_bin[0]).save(output_str('just_cytoplasm_bin0'))
arr_to_img(just_cyto_bin[-1]).save(output_str('just_cytoplasm_bin1'))
# arr_to_img(measure_bin[0]).save(output_str('measure0'))
# arr_to_img(measure_bin[-1]).save(output_str('measure1'))

measured_nuc_sum = np.sum(np.sum(measured_nuc, axis=2), axis=1)
measured_nuc_bin_sum = np.sum(np.sum(measured_nuc_bin, axis=2), axis=1)
measured_not_nuc_sum = np.sum(np.sum(measured_not_nuc, axis=2), axis=1)
measured_not_nuc_bin_sum = np.sum(np.sum(measured_not_nuc_bin, axis=2), axis=1)
measured_cyto_sum = np.sum(np.sum(measured_cyto, axis=2), axis=1)
measured_cyto_bin_sum = np.sum(np.sum(measured_cyto_bin, axis=2), axis=1)

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
N_frames = len(measured_nuc_sum)
frame_no = np.arange(N_frames) + 1
kw = dict(linewidth=2, marker='o')
ax1.plot(frame_no, measured_nuc_sum, color=Colors.orange, label='Nucleus', **kw)
ax1.plot(frame_no, measured_cyto_sum, color=Colors.purple, label='Cytoplasm', **kw)
ax1.plot(frame_no, measured_not_nuc_sum, 'o-', color=Colors.green, label='Not Nucleus', **kw)
ax2.plot(frame_no, measured_nuc_bin_sum, 'o-', color=Colors.orange, label='Nucleus', **kw)
ax2.plot(frame_no, measured_cyto_bin_sum, 'o-', color=Colors.purple, label='Cytoplasm', **kw)
ax2.plot(frame_no, measured_not_nuc_bin_sum, color=Colors.green, label='Not Nucleus', **kw)

for fig, ax, title, name in (
        (fig1, ax1, 'Areas Measured', 'norm'),
        (fig2, ax2, 'Areas Measured (Binary)', 'binary')):
    ax.set_xlabel('Frame')
    ax.set_ylabel('Sums')
    ax.set_title(title)
    ax.set_ylim(0, None)
    ax.legend(ncol=3, loc='lower center', fontsize='x-small',
        framealpha=0.6, fancybox=True, frameon=True)
    fig.tight_layout()
    
    fig.savefig(output_str(name))
    plt.close(fig)


columns = [
    ('Frame', frame_no),
    ('Nucleus', measured_nuc_sum),
    ('Nucleus (Binary)', measured_nuc_bin_sum),
    ('Cytoplasm', measured_cyto_sum),
    ('Cytoplasm (Binary)', measured_cyto_bin_sum),
    ('Not Nucleus', measured_not_nuc_sum),
    ('Not Nucleus (Binary)', measured_not_nuc_bin_sum),
]

headers, rows = zip(*columns)
headerrow = ','.join(headers)

csvname = output_str('', extension='.csv')
np.savetxt(csvname, np.asarray(rows).T,
    delimiter=',', header=headerrow, fmt='%.6g')
