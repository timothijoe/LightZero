import math
import numpy as np 
  
    
def check_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # 判断线段是否相交
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    # 计算向量叉积
    cross_product1 = dx1 * (y3 - y1) - dy1 * (x3 - x1)
    cross_product2 = dx1 * (y4 - y1) - dy1 * (x4 - x1)
    cross_product3 = dx2 * (y1 - y3) - dy2 * (x1 - x3)
    cross_product4 = dx2 * (y2 - y3) - dy2 * (x2 - x3)

    # 判断是否相交
    if cross_product1 * cross_product2 < 0 and cross_product3 * cross_product4 < 0:
        return True  # 相交
    else:
        return False  # 不相交

def calculate_point_to_segment_distance(px, py, x1, y1, x2, y2):
    # 计算点到线段的最短距离
    dx, dy = x2 - x1, y2 - y1
    # 计算线段的长度
    segment_length = math.sqrt(dx ** 2 + dy ** 2)
    # 计算点到线段的投影长度
    projection = ((px - x1) * dx + (py - y1) * dy) / segment_length
    # 判断投影是否在线段内
    if projection < 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    elif projection > segment_length:
        return math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
    else:
        # 计算点到线段的垂直距离
        distance = abs((px - x1) * dy - (py - y1) * dx) / segment_length
        return distance

def calculate_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    # 检查线段是否相交
    if check_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        return 0  # 线段相交，最短距离为0
    # 计算点A到线段CD的最短距离
    dist_a_cd = calculate_point_to_segment_distance(x1, y1, x3, y3, x4, y4)
    # 计算点B到线段CD的最短距离
    dist_b_cd = calculate_point_to_segment_distance(x2, y2, x3, y3, x4, y4)
    # 计算点C到线段AB的最短距离
    dist_c_ab = calculate_point_to_segment_distance(x3, y3, x1, y1, x2, y2)
    # 计算点D到线段AB的最短距离
    dist_d_ab = calculate_point_to_segment_distance(x4, y4, x1, y1, x2, y2)
    # 返回四个距离中的最小值作为最短距离
    shortest_distance = min(dist_a_cd, dist_b_cd, dist_c_ab, dist_d_ab)
    return shortest_distance

def calculate_seg_dist(line1, line2):
    p1 = line1[0]
    p2 = line1[1]
    p3 = line2[0]
    p4 = line2[1]
    return calculate_distance(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])

def calculate_line_set_distance(l1, l2):
    min_distance = math.inf
    for line1 in l1:
        for line2 in l2:
            distance = calculate_seg_dist(line1, line2)
            min_distance = min(min_distance, distance)
    return min_distance  

def calculate_matrix_distance(matrix1, matrix2):
    A1 = matrix1[0]
    B1 = matrix1[1]
    C1 = matrix1[2]
    D1 = matrix1[3]
    A2 = matrix2[0]
    B2 = matrix2[1]
    C2 = matrix2[2]
    D2 = matrix2[3]
    l1 = np.array([[A1, B1], [B1, C1], [C1, D1], [D1, A1]])
    l2 = np.array([[A2, B2], [B2, C2], [C2, D2], [D2, A2]])
    return calculate_line_set_distance(l1, l2)
    



# 示例使用
x1, y1 = 0, 0
x2, y2 = 3, 0
x3, y3 = 2, 2
x4, y4 = 4, 2

shortest_distance = calculate_distance(x1, y1, x2, y2, x3, y3, x4, y4)
print("线段AB到线段CD的最短距离为:", shortest_distance)

matrix1 = [np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([0,1])]
matrix2 = [np.array([4,4]), np.array([5,4]), np.array([5,5]), np.array([4,5])]

shortest_distance = calculate_matrix_distance(matrix1, matrix2)
print("矩阵的的最短距离为:", shortest_distance)


def calculate_vehicle_vertices(vehicle, position=None):
    if position is None:
        position = vehicle.position
    x = position[0]
    y = position[1]
    theta = vehicle.heading_theta
    length = vehicle.top_down_length
    width = vehicle.top_down_width 
    dx = length / 2
    dy = width / 2
    theta_rad = theta
    vertex1_x = x + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    vertex1_y = y + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
    vertex2_x = x + dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    vertex2_y = y + dx * math.sin(theta_rad) - dy * math.cos(theta_rad)
    vertex3_x = x - dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    vertex3_y = y - dx * math.sin(theta_rad) - dy * math.cos(theta_rad)
    vertex4_x = x - dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    vertex4_y = y - dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
    return [np.array([vertex1_x, vertex1_y]), np.array([vertex2_x, vertex2_y]), np.array([vertex3_x, vertex3_y]), np.array([vertex4_x, vertex4_y])]


def calculate_fine_collision(ego_vehicle, other_vechile, ego_pos=None, other_pos=None):
    if ego_pos is None:
        ego_pos = ego_vehicle.position
    if other_pos is None:
        other_pos = other_vechile.position
    ego_vertices = calculate_vehicle_vertices(ego_vehicle, ego_pos)
    other_vertices = calculate_vehicle_vertices(other_vechile, other_pos)
    shortest_distance = calculate_matrix_distance(ego_vertices, other_vertices)
    if shortest_distance < 1:
        return True 
    else:
        return False

    
    

import math

def calculate_car_vertices(x, y, theta, length, width):
    # 计算车身中心点到四个顶点的偏移量
    dx = length / 2
    dy = width / 2

    # 计算角度的弧度值
    theta_rad = math.radians(theta)

    # 计算四个顶点的位置
    vertex1_x = x + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    vertex1_y = y + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)

    vertex2_x = x + dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    vertex2_y = y + dx * math.sin(theta_rad) - dy * math.cos(theta_rad)

    vertex3_x = x - dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    vertex3_y = y - dx * math.sin(theta_rad) - dy * math.cos(theta_rad)

    vertex4_x = x - dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    vertex4_y = y - dx * math.sin(theta_rad) + dy * math.cos(theta_rad)

    return [(vertex1_x, vertex1_y), (vertex2_x, vertex2_y), (vertex3_x, vertex3_y), (vertex4_x, vertex4_y)]

# 示例使用
car_x = 0
car_y = 0
car_theta = 30  # 角度单位为度
car_theta = 45
car_length = 10
car_width = 10

vertices = calculate_car_vertices(car_x, car_y, car_theta, car_length, car_width)
print("汽车四个顶点的位置：", vertices)