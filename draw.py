import sys


def draw_by_pixels(np_array, width=28):
    result = "\n"
    left = 0
    thisline = ""
    for pixel in np_array:
        pixel = pixel[0]
        left += 1
        thisline += "#"*(pixel > 0.66)+" "*(pixel < 0.33)\
                    + "="*(0.66 >= pixel >= 0.33)
        if left >= width:
            left = 0
            result += thisline + "\n"
            thisline = ""

    sys.stdout.write(result)
    return result
