from datetime import datetime

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
