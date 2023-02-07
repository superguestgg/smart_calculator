import requests
import json
import math
import string
import time
import shape
#import threading

team_token = 'z3XC3y5hOqWfL5A2Whe5uOiYqGcPaDs'
round = 'projects-course-2'
taskType = 'shape'

url = "https://www.urfav-challenge.ru/api/tasks/"
print("+=========================+")
print("Client for Challenge is ON!")
print("+=========================+")

tasks_shape=[]
for offset in range (0,100,10):
    print("autogetting task")
    #input()
    print("I`ll get 10 tasks after 0.1 sec")
    time.sleep(0.1)
    response = requests.get(url, params = {'secret': team_token, 'round': round, 'type': taskType,
                                            'status': 1, 'offset': offset, 'count':10})
    json_response = response.json()
    print("get successfully")
    #print("You hava a new task:")
    #print(response)
    #print(json_response)
    #print(f"TaskId:		{json_response['id']}")
    #print(f"TaskType:		{json_response['typeId']}")
    #print(f"Question:		{json_response['question']}")
    
    #task_type = json_response['typeId']
    #question = json_response['question']
    #task_id = json_response['id']
    for one_json_response in json_response:
        print(f"TaskId:		{one_json_response['id']}")
        #print(f"TaskType:		{one_json_response['typeId']}")
        #print(f"Question:		{one_json_response['question']}")
        task_type = one_json_response['typeId']
        question = one_json_response['question']
        task_id = one_json_response['id']
        print(offset)
        tasks_shape.append(one_json_response)
    

    """
    value = shape.show(question)
    
    if (value!="error" and value!="error2" and value!=None):
        pass
    else:
        print(f"Answer:		{value}")
        print(f"write new answer to send this to server!")
        
        value = str(input())
        print(f"Answer:		{value}")
        print("I`m sending your answer to server!")
        print("==================================================================\n\n")
        print(f"Answer:		{value}")
        input()
    print(f"autosend:   {value}")
    answer = json.dumps({"answer": str(value)})
    header = {'Content-Type':'application/json'}

    response = requests.post(url + task_id, headers = header, params = {'secret': team_token}, data = answer)
    json_response = response.json()
    if json_response['status'] == 1:
        print(f"Good work status was {json_response['status']}")
        print("==================================================================\n\n")
    else:
        print(f"Err0r")
        break
    """
shape.save_images(tasks_shape)
