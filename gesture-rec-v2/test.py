import requests

BASE = "http://127.0.0.1:5000/"
response = requests.put(BASE + "config", {"num_readings_per_sample_per_mpu": 10, "num_mpus": 2})
response.raise_for_status()  # raises exception when not a 2xx response
if response.status_code != 204:
    print(response.json())

isAdding = True
while isAdding:
    print("Add Gesture: ")
    gesture = input()
    if gesture:
        response = requests.put(BASE + "add", {"path": r"C:\Users\olusa\Documents\VS Code\Node\MVP\ges-rec-api\gesture_recordings\{}\{}_parsed.csv".format(gesture, gesture)})
        response.raise_for_status()
        if response.status_code != 204:
            print(response.json())
    else:
        isAdding = False

print("Building...")
response = requests.get(BASE + "build")
response.raise_for_status()
if response.status_code != 204:
    print(response.json())