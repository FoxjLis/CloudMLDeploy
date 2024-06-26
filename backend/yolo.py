from flask import *
from flask_cors import *
from ultralytics import YOLO

model = YOLO("best_x.pt")

results = model('bus.jpg', save=True, project="backend")
print(results)