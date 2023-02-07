# from PIL import Image, ImageDraw
import numpy as np
import json
import string
import time

possible_answers = ['ellipse', 'rectangle', 'triangle',
                    'circle', 'square', 'equilateraltriangle']

"""def checkcorreckt(mayanswers,answer):
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


def show2(string):
    bb=string.split("|")
    mayanswers = bb[0].split("=")[1].split(",")
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
    bot=[]
    right=[]
    left=[]
    top=[]
    lenaa=len(aa)
    for i in range (lenaa):
        aa[i][0]=aa[i][0]-minx
        aa[i][1]=aa[i][1]-miny
    width=maxx
    height=maxy
    kx=28/maxx
    ky=28/maxy
    kmax=min(kx,ky)
    for i in range (lenaa):
        aa[i][0]=int(aa[i][0]*kmax)
        aa[i][1]=int(aa[i][1]*kmax)
    sample = Image.new("RGB",(28,28))
    pencil = ImageDraw.Draw(sample)
    
    for a in aa:
        pencil.point((a[1],a[0]),fill="green")
    sample.show()"""


def save_images(tasks_shape):
    file1 = open("shape_tasks_no_prepared1.txt", "w")
    for task_shape in tasks_shape:
        file1.write(str(task_shape))
        # file1.write(str(task_shape[teamAnswer]))
        file1.write("\n\n")
    file1.close()


def open_json():
    file1 = open("shape_tasks_no_prepared3.txt", "r")
    lines = file1.readlines()
    json_lines = []
    vectors_for_numpy = []
    for line in lines:
        if len(line) < 10:
            continue
        line = line.replace("'", '"')
        json_lines.append(json.loads(line))
        vectors_for_numpy.append(translate_json_to_vector(json_lines[-1]))
        # show2(json_lines[-1]['question'])
        # print(json_lines[-1]['teamAnswer'])
    file1.close()
    print(time.process_time())
    return vectors_for_numpy


def open_json_test_data():
    file1 = open("shape_tasks_no_prepared3.txt", "r")
    lines = file1.readlines()
    json_lines = []
    vectors_for_numpy = []
    for line in lines:
        if len(line) < 10:
            continue
        line = line.replace("'", '"')
        json_lines.append(json.loads(line))
        vectors_for_numpy.append(translate_json_to_vector(json_lines[-1], True))
        # show2(json_lines[-1]['question'])
        # print(json_lines[-1]['teamAnswer'])
    file1.close()
    print(time.process_time())
    return vectors_for_numpy


def translate_json_to_vector(json_object, is_for_test=False):
    global possible_answers
    string = json_object['question']
    correct_answer = json_object['teamAnswer']
    bb = string.split("|")
    # mayanswers = bb[0].split("=")[1].split(",")
    aa = bb[1].replace(")", "").replace("(", "").split()
    maxy = 0
    maxx = 0
    miny = 999999
    minx = 999999
    for i in range(len(aa)):
        aa[i] = list(map(int, aa[i].split(",")))
        # print(aa[i])
        maxy = max(maxy, aa[i][1])
        maxx = max(maxx, aa[i][0])
        miny = min(miny, aa[i][1])
        minx = min(minx, aa[i][0])
    maxx -= minx - 1
    maxy -= miny - 1
    lenaa = len(aa)
    for i in range(lenaa):
        aa[i][0] = aa[i][0] - minx
        aa[i][1] = aa[i][1] - miny
    # width = maxx
    # height = maxy
    kx = 28 / maxx
    ky = 28 / maxy
    kmax = min(kx, ky)
    for i in range(lenaa):
        aa[i][0] = int(aa[i][0] * kmax)
        aa[i][1] = int(aa[i][1] * kmax)
    vector_x = [[0] for i in range(784)]
    for a in aa:
        vector_x[a[1] * 28 + a[0]] = [1]

    if is_for_test:
        vector_y = possible_answers.index(correct_answer) % 3
        return [np.array(vector_x), vector_y]
    else:
        #vector_y = [[int(correct_answer == j)] for j in possible_answers]
        vector_y = [[int(correct_answer in [j,k])] for k,j in zip(possible_answers[:3],possible_answers[3:])]
        #print(vector_y)
    # print(vector_y)
    return [np.array(vector_x), np.array(vector_y)]

# print(show(input()))
# open_json()
