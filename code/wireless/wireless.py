# client

import socket
from datetime import datetime
import time

ip = "10.50.72.111"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, 8080))
print("[LOCAL] Connected to bike.")

while True:
    now = datetime.now()
    current_time = now.strftime("%M%S")
    print(current_time)
    msg = current_time + " 10"
    print("[SENT] " + msg)
    client.send(msg.encode())
    time.sleep(.5)

client.close()