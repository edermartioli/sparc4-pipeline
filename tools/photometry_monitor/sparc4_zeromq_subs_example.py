import zmq
import json

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
# Connect to the publisher
subscriber.connect("tcp://localhost:5556")
subscriber.subscribe("")
print("Waiting for messages...")
while True:
    
    message = subscriber.recv_string()
    print(f"Received: {message}")
    try :
        phot = json.loads(message)
    except :
        continue
    
    print(phot.keys())
