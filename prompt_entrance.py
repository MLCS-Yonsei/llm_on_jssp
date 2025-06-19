import json
import cv2
import numpy as np
from collections import deque
import itertools
from llm_module import llm_parse_input, llm_generate_final_output

path = #"File path of 'llm_on_jssp'/" 

def parse_grid_and_colors(png_path, grid_shape=(13, 27)):
    img = cv2.imread(png_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    rows, cols = grid_shape
    cell_h, cell_w = h // rows, w // cols

    grid = np.zeros((rows, cols), dtype=np.uint8)
    color_pos = {}
    color_labels = {
        "red":    ([200,   0,   0], [255,  60,  60]),         
        "blue":   ([  0, 140, 200], [ 80, 220, 255]),         
        "yellow": ([200, 200,   0], [255, 255, 100]),        
        "green":  ([  0, 120,  40], [ 80, 220, 120]),         
        "purple": ([ 90,  30, 120], [150,  80, 200]),       
        "pink":   ([200,   0, 200], [255, 120, 255]),       
        "gray":   ([100, 100, 100], [200, 200, 200]),       
    }


    for row in range(rows):
        for col in range(cols):
            cx = int(col * cell_w + cell_w // 2)
            cy = int(row * cell_h + cell_h // 2)
            rgb = img_rgb[cy, cx]
            #print(f"row={row}, col={col}, center RGB={rgb}") 
            if np.all(rgb < 50):
                grid[row, col] = 1
            else:
                for cname, (low, high) in color_labels.items():
                    if np.all(rgb >= low) and np.all(rgb <= high):
                        color_pos[cname] = (row, col)
    return grid, color_pos

def bfs_shortest_path(grid, start):
    h, w = grid.shape
    visited = -np.ones((h, w), dtype=int)
    q = deque()
    q.append(start)
    visited[start] = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < h and 0 <= ny < w:
                if grid[nx, ny]==0 and visited[nx, ny]==-1:
                    visited[nx, ny] = visited[x, y]+1
                    q.append((nx, ny))
    return visited

def color_distances_from_png(png_path, grid_shape=(13,27)):
    grid, color_pos = parse_grid_and_colors(png_path, grid_shape)
    color_order = ['red', 'green', 'blue', 'yellow', 'purple', 'pink', 'gray']
    
    result = {}
    for i, c1 in enumerate(color_order):
        for j, c2 in enumerate(color_order):
            if i < j:  
                if c1 in color_pos and c2 in color_pos:
                    visited = bfs_shortest_path(grid, color_pos[c1])
                    dist = visited[color_pos[c2]]
                    result[(c1, c2)] = int(dist)
    print("\n=== Detected color positions ===")
    for c, pos in color_pos.items():
        print(f"{c}: {pos}")
    print("\n=== Color distances ===")
    for (a, b), d in result.items():
        print(f"{a} <-> {b}: {d}")
    if not result:
        print("No color pairs found! (Check grid or color detection.)")
    return result

def add_move_operations(problem_json, distances):
    color_order = ['red', 'green', 'blue', 'yellow', 'purple', 'pink', 'gray']
    idx2color = {i: c for i, c in enumerate(color_order)}

    matrix = problem_json['matrix']
    new_matrix = []
    for job_idx, job in enumerate(matrix):
        new_job = []
        move_count = 1
        for op_idx, (machine_idx, duration) in enumerate(job):
            new_job.append([machine_idx, duration])
            if op_idx < len(job) - 1:
                curr_color = idx2color[machine_idx]
                next_machine_idx = job[op_idx + 1][0]
                next_color = idx2color[next_machine_idx]
                if (curr_color, next_color) in distances:
                    move_time = distances[(curr_color, next_color)]
                elif (next_color, curr_color) in distances:
                    move_time = distances[(next_color, curr_color)]
                else:
                    move_time = 0
                move_machine_idx = int(f"9{job_idx}{move_count}")
                move_count += 1
                new_job.append([move_machine_idx, move_time])
        new_matrix.append(new_job)
    return {'matrix': new_matrix, 'label': problem_json['label']}




def make_input_matrix(prompt_path, image_path, al_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_json = json.load(f)
    instruction = prompt_json['instruction']
    eval_crit = prompt_json['evaluation_criterion']
    job_descs = '\n'.join(prompt_json['job_descriptions'])
    full_prompt = (
        f"{instruction}\n"
        f"Evaluation criterion: {eval_crit}\n\n"
        f"Job descriptions:\n{job_descs}\nOutput:"
    )
    
    
    problem_json = llm_parse_input(full_prompt, al_path)
    print("\n=== LLM OUTPUT ===\n", problem_json)

    distances = color_distances_from_png(image_path)
    print("\n=== Color Distances ===")
    for (a, b), d in distances.items():
        print(f"{a} <-> {b}: {d}")

    result = add_move_operations(problem_json, distances)
    print("\n=== Final results ===\n", result)

    return result


# 사용 예시:
make_input_matrix(path + "input/problem_prompt.json", path + "input/img3.png", path)
# 입력 예시
# problem_json = {'matrix': [[[2, 2], [0, 3], [3,4]], [[1, 4],[2,2]]], 'label': 'best_makespan'}
# 변형된 json, 최종 출력
# {'matrix': [[[2, 2], [901, 26], [0, 3], [902, 31], [3, 4]], [[1, 4], [911, 25], [2, 2]]], 'label': 'best_makespan'}

'''
# 확장된 문제
입력
problem_json = {'matrix': [[[2, 10], [0, 5], [3,20], [5, 2], [1, 12]], 
[[1, 15],[2,7], [6, 20]], 
[[6, 10], [2, 15], [5, 20]], 
[[0, 20],[4, 5], [1, 2], [3, 10]]], 
'label': 'best_makespan'}
출력
 {'matrix': [[[2, 10], [901, 20], [0, 5], [902, 25], [3, 20], [903, 0], [5, 2], [904, 0], [1, 12]], 
 [[1, 15], [911, 25], [2, 7], [912, 23], [6, 20]], 
 [[6, 10], [921, 23], [2, 15], [922, 0], [5, 20]], 
 [[0, 20], [931, 33], [4, 5], [932, 10], [1, 2], [933, 18], [3, 10]]], 
 'label': 'best_makespan'}
'''
