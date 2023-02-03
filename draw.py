import sys


def draw_by_pixels(np_array, width=28):
    result = "\n"
    left = 0
    this_line = ""
    for pixel in np_array:
        pixel = pixel[0]
        left += 1
        this_line += "#" * (pixel > 0.66) + " " * (pixel < 0.33) \
                     + "=" * (0.66 >= pixel >= 0.33)
        if left >= width:
            left = 0
            result += this_line + "\n"
            this_line = ""

    sys.stdout.write(result)
    return result
