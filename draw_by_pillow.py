#from PIL import Image, ImageDraw
def checkcorreckt(mayanswers,answer):
    if answer in mayanswers:
        return answer
    else:
        return "error2"
def show(string):
    bb=string.split("|")
    mayanswers = bb[0].split("=")[1].split(",")
    print(mayanswers)
    aa=bb[1].replace(")","").replace("(","").split()
    maxy=0
    maxx=0
    miny=999999
    minx=999999
    for i in range (len(aa)):
        aa[i]=list(map(int, aa[i].split(",")))
        #print(aa[i])
        maxy = max(int(maxy),int(aa[i][1]))
        maxx = max(maxx,aa[i][0])
        miny = min(int(miny),int(aa[i][1]))
        minx = min(minx,aa[i][0])
    maxx-=minx-1
    maxy-=miny-1
    lefttop=[]
    leftbot=[]
    righttop=[]
    rightbot=[]
    bot=[]
    right=[]
    left=[]
    top=[]
    lenaa=len(aa)
    for i in range (lenaa):
        aa[i][0]=aa[i][0]-minx
        aa[i][1]=aa[i][1]-miny
        if aa[i][0]==0:
            left.append(aa[i])
        if aa[i][1]==0:
            top.append(aa[i])
        if aa[i][0]==maxx-1:
            right.append(aa[i])
        if aa[i][1]==maxy-1:
            bot.append(aa[i])
    sample = Image.new("RGB",(maxy,maxx))
    pencil = ImageDraw.Draw(sample)

    for a in aa:
        pencil.point((a[1],a[0]),fill="green")
    sample.show()
    if lenaa>=maxx*maxy*0.8:
        if maxy==maxx:
            return checkcorreckt(mayanswers,"square")
        else:
            return checkcorreckt(mayanswers,"rectangle")
    elif len(right)>=maxy//2 and len(left)>=maxy//2 and len(top)>=maxy//2 and len(bot)>=maxy//2:
        if maxy==maxx:
            return checkcorreckt(mayanswers,"square")
        else:
            return checkcorreckt(mayanswers,"rectangle")
    elif len(bot)==len(top)==len(left)==len(right)>=5:
        if checkcorreckt(mayanswers,"circle")!="error2":
            return checkcorreckt(mayanswers,"circle")
        else:
            return checkcorreckt(mayanswers,"ellipse")
    elif len(left)==len(right)>=5 and len(bot)==len(top)>=5:
        return checkcorreckt(mayanswers,"ellipse")
    elif len(left)>=5 and len(bot)>=5 and abs(len(left)-len(right))<3 and abs(len(top)-len(bot))<3:
        return checkcorreckt(mayanswers,"ellipse")
    elif len(left)==len(right)>=5 and len(bot)==len(top)>=1:
        return checkcorreckt(mayanswers,"ellipse")
    elif len(left)==len(right)>=1 and len(bot)==len(top)>=5:
        return checkcorreckt(mayanswers,"ellipse")
    elif len(left)==len(right)<5 and len(bot)==len(top)<5:
        return checkcorreckt(mayanswers,"rectangle")
    else:
        if len(right)==1:
            if right[0] in top or right[0] in bot:
                if len(top)==maxx or len(bot)==maxx:
                    return checkcorreckt(mayanswers,"equilateraltriangle")
                return checkcorreckt(mayanswers,"triangle")
        if len(left)==1:
            if left[0] in top or left[0] in bot:
                if len(top)==maxx or len(bot)==maxx:
                    return checkcorreckt(mayanswers,"equilateraltriangle")
                return checkcorreckt(mayanswers,"triangle")
        if len(top)==1:
            if top[0] in left or top[0] in right:
                if len(left)==maxy or len(right)==maxy:
                    return checkcorreckt(mayanswers,"equilateraltriangle")
                return checkcorreckt(mayanswers,"triangle")
        if len(bot)==1:
            if bot[0] in left or bot[0] in right:
                if len(left)==maxy or len(right)==maxy:
                    return checkcorreckt(mayanswers,"equilateraltriangle")
                return checkcorreckt(mayanswers,"triangle")
        else:
            return "error"
#print(show(input()))
