import sys
def draw_by_pixels(np_array):
    result="\n"
    j=0
    thisline = ""
    for pixel in np_array:
        i = pixel[0]
        j += 1
        thisline += "#"*(i>0.66)+" "*(i<0.33)+"="*(i<=0.66 and i>=0.33)
        if j >= 28:
            j = 0
            print(thisline)
            thisline = ""

        #result+"\n"
    #sys.stdout.write(result)
    return result