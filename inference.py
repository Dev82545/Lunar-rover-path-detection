
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import heapq
import cv2


IMG_PATH = r"C:\Users\HP\MoonNav2\dataset\images\val\1_jpg.rf.0666e8c8eb5e13993eb61573d42cc3ec.jpg"  
YOLO_WEIGHTS = r"C:\Users\HP\MoonNav2\runs\detect\moon_navigation2\weights\best.pt"
DEEPLAB_WEIGHTS = "deeplabv3_lunar.pth"
RESIZE_SIZE = (256, 256)


meters_per_pixel = 0.5  

# === LOAD IMAGE ===
img_orig = read_image(IMG_PATH).float() / 255.0
resize = Resize(RESIZE_SIZE)
img_resized = resize(img_orig)
img_np = (img_resized.permute(1,2,0).numpy() * 255).astype(np.uint8) 

model_yolo = YOLO(YOLO_WEIGHTS)
results = model_yolo.predict(img_np, conf=0.5, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

crater_mask = np.zeros(RESIZE_SIZE, dtype=np.uint8)
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    crater_mask[y1:y2, x1:x2] = 1

model_dl = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model_dl.load_state_dict(torch.load(DEEPLAB_WEIGHTS, map_location='cpu'))
model_dl.eval()
img_dl = img_resized.unsqueeze(0)
with torch.no_grad():
    pred = model_dl(img_dl)
    obstacle_mask = torch.sigmoid(pred.squeeze(0)).squeeze(0) > 0.5
obstacle_mask = obstacle_mask.cpu().numpy().astype(np.uint8)

img_gray = img_resized.permute(1,2,0).numpy()
if img_gray.shape[2] == 3:
    img_gray = cv2.cvtColor((img_gray*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img_gray)
shadow_mask = cv2.adaptiveThreshold(
    enhanced, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_mask, connectivity=8)
min_area = 50
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        shadow_mask[labels == i] = 0

final_mask = np.logical_or(crater_mask, obstacle_mask)
final_mask = np.logical_or(final_mask, shadow_mask).astype(np.uint8)

def find_nearest_free(point, mask, max_radius=40):
    x, y = point
    h, w = mask.shape
    if mask[y, x] == 0:
        return (x, y)
    for r in range(1, max_radius+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0:
                    return (nx, ny)
    free_pixels = np.argwhere(mask == 0)
    if len(free_pixels) > 0:
        idx = np.random.choice(len(free_pixels))
        return tuple(free_pixels[idx][::-1]) 
    raise ValueError('No free pixel found in the image!')


start_point = (20, 20) 
goal_point = (220, 220)  

start_point = find_nearest_free(start_point, final_mask)
goal_point = find_nearest_free(goal_point, final_mask)
print('Checked start:', start_point)
print('Checked goal:', goal_point)

num_stops = 10
points = np.linspace(start_point, goal_point, num=num_stops+2)
points = [tuple(map(int, pt)) for pt in points]
stops = points[1:-1]

checked_stops = []
for stop in stops:
    try:
        checked_stops.append(find_nearest_free(stop, final_mask))
    except Exception as e:
        print(f"Stop {stop} is unreachable and will be skipped.")

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(mask, start, goal):
    h, w = mask.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    heapq.heappush(open_set, (g+1+heuristic(neighbor, goal), g+1, neighbor, path+[neighbor]))
    return None

route_points = [start_point] + checked_stops + [goal_point]
full_path = []
for i in range(len(route_points) - 1):
    segment = astar(final_mask, route_points[i], route_points[i+1])
    if segment is None:
        print(f'No path from {route_points[i]} to {route_points[i+1]}. Skipping this segment.')
        continue
    if i > 0:
        segment = segment[1:]
    full_path.extend(segment)


def path_length(path):
    return np.sum([np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path))])

pixel_length = path_length(full_path)
real_length_m = pixel_length * meters_per_pixel
print(f"Path length: {real_length_m:.2f} meters")
if real_length_m >= 100:
    print(" Path covers at least 100 meters.")
else:
    print(" Path is shorter than 100 meters. Consider adjusting start/goal or image scale.")

img_to_show = img_orig
if img_orig.shape[1:] != RESIZE_SIZE:
    img_to_show = resize(img_orig)

plt.figure(figsize=(7,7))
plt.title('Lunar Surface with Multi-Stop Path and Shadows')
plt.imshow(img_to_show.permute(1,2,0))
plt.imshow(final_mask, alpha=0.4, cmap='Reds')
plt.scatter([start_point[0]], [start_point[1]], c='lime', s=120, label='Start')
plt.scatter([goal_point[0]], [goal_point[1]], c='blue', s=120, label='Goal')
if checked_stops:
    stop_x, stop_y = zip(*checked_stops)
    plt.scatter(stop_x, stop_y, c='orange', s=80, label='Stops')
if full_path:
    path_x, path_y = zip(*full_path)
    plt.plot(path_x, path_y, c='yellow', linewidth=2, label='A* Path')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()