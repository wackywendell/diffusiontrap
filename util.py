import numpy as np

MAX_PIXEL = 255


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


def to_list_func(n=None, totype=str, delimiter=','):
    def to_list(input_str):
        items = [totype(i) for i in input_str.split(delimiter)]
        if n is not None and len(items) != n:
            raise ValueError("Expected {} items, found {} in {}".format(n, len(items), input_str))
        return items
    return to_list


class Colors:
    red, blue, green, purple, orange, brown, ppink, grey, yellow, black = (
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
        '#A65628', '#F781BF', '#999999', '#FFFF33', '#000000')


def hist_to_threshold(hist, threshold):
    if not isinstance(threshold, FloatOrPercent) or threshold.type == '#':
        return float(threshold)
    ixs, = np.nonzero(np.cumsum(hist) / np.sum(hist) >= float(threshold))
    return ixs[0]


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


def renormalize_all(img_list, threshold, maximum):
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
        img_list,
        img_list
        #[im.point(renorm) for im in img_list],
        #[im.point(binarize) for im in img_list]
    )
