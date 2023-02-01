import sys
def draw_by_pixels(np_array):
    result="\n"
    j=0
    thisline = ""
    for pixel in np_array:
        i = pixel[0]
        j += 1
        thisline += "1"*(i>0.8)+" "*(i<=0.8)
        if j >= 28:
            j = 0
            print(thisline)
            thisline = ""

        #result+"\n"
    #sys.stdout.write(result)
    return result