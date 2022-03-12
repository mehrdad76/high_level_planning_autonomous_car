import math
import numpy as np
from Geometry import Point, LineSegment
import Geometry

MAX_LASER_DISTANCE = 5


def get_in_range_angle(angle):
    angle = angle % 360
    if angle > 180:
        return angle - 360
    return angle


class IntersectToConnectionLine:

    def __init__(self, relative_point, relative_angle, connection_number, remaining_laser):
        self.relative_point = relative_point
        self.relative_angle = relative_angle
        self.connection_number = connection_number
        self.remaining_laser = remaining_laser

    def __str__(self):
        return 'relative point: ' + str(self.relative_point) + ', remaining laser: ' + str(self.remaining_laser)


class Segment:

    def __init__(self, segment_angle, number_of_connections):
        self.number_of_connections = number_of_connections
        self.segment_angle = segment_angle

    def scan_lidar_one_laser(self, relative_position, relative_angle, max_laser_distance):
        raise NotImplementedError("Please Implement this method")

    def step(self):
        raise NotImplementedError("Please Implement this method")

    def scan_lidar(self, relative_car_position, number_of_rays, car_relative_angle, viewing_angle_range,
                   max_laser_distance):  # viewing_angle_range is a number e.g. 45
        ray_relative_angles = np.linspace(car_relative_angle - viewing_angle_range,
                                          car_relative_angle + viewing_angle_range, number_of_rays)
        ray_relative_angles = [get_in_range_angle(x_) for x_ in ray_relative_angles]
        return [(-car_relative_angle + r_, self.scan_lidar_one_laser(relative_car_position, r_, max_laser_distance))
                for r_ in ray_relative_angles]


class PolygonSegment(Segment):

    def __init__(self, global_segment_angle, number_of_connections, relative_vertices,
                 relative_main_line_segments, relative_connection_line_segments):
        super().__init__(global_segment_angle, number_of_connections)
        self.relative_vertices = relative_vertices
        self.relative_main_line_segments = relative_main_line_segments
        self.relative_connection_line_segments = relative_connection_line_segments

    def point_is_inside_or_on(self, point):
        raise NotImplementedError("Please Implement this method")

    def scan_lidar_one_laser(self, relative_position, relative_angle, max_laser_distance):
        assert self.point_is_inside_or_on(relative_position)
        relative_angle = math.radians(get_in_range_angle(relative_angle))
        laser = LineSegment(relative_position, Point(relative_position.x +
                                                     max_laser_distance * math.cos(relative_angle),
                                                     relative_position.y + max_laser_distance *
                                                     math.sin(relative_angle)))
        res = None
        min_distance = math.inf
        for line_segment in self.relative_main_line_segments:
            intersection = line_segment.get_intersection_to(laser)
            if intersection is None:
                continue
            distance = relative_position.get_distance_to(intersection)
            if distance < min_distance:
                min_distance = distance
                res = distance
        for index in range(2):
            line_segment = self.relative_connection_line_segments[index]
            intersection = line_segment.get_intersection_to(laser)
            if intersection is None:
                continue
            distance = relative_position.get_distance_to(intersection)
            if distance < min_distance:
                min_distance = distance
                res = IntersectToConnectionLine(intersection, relative_angle, index, max_laser_distance - distance)

        if min_distance <= max_laser_distance:
            return res
        return None

    def step(self):
        raise NotImplementedError("Please Implement this method")


# for Rectangle segments we could use polygon but following implementation is faster
class RectangleSegment(Segment):

    def __init__(self, length, width, global_segment_angle):
        super().__init__(global_segment_angle, 2)
        self.length = length
        self.width = width

    def scan_lidar_one_laser(self, relative_position, relative_angle, max_laser_distance):
        relative_angle = get_in_range_angle(relative_angle)
        relative_x, relative_y = relative_position.x, relative_position.y
        assert 0 <= relative_y <= self.width and 0 <= relative_x <= self.length
        if relative_angle == 0:
            distance_to_right = self.length - relative_x
            if distance_to_right > max_laser_distance:
                return None
            return IntersectToConnectionLine(Point(self.length, relative_y), relative_angle,
                                             1, max_laser_distance - distance_to_right)
        if relative_angle == 180:
            if relative_x > max_laser_distance:
                return None
            return IntersectToConnectionLine(Point(0, relative_y), relative_angle,
                                             0, max_laser_distance - relative_x)
        if relative_angle == 90:
            distance_to_top = self.width - relative_y
            if distance_to_top > max_laser_distance:
                return None
            return distance_to_top
        if relative_angle == -90:
            if relative_y > max_laser_distance:
                return None
            return relative_y
        if -180 < relative_angle < -90:
            laser_to_horizontal = relative_y / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = relative_x / abs(math.cos(math.radians(relative_angle)))
        elif -90 < relative_angle < 0:
            laser_to_horizontal = relative_y / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = (self.length - relative_x) / abs(math.cos(math.radians(relative_angle)))
        elif 0 < relative_angle < 90:
            laser_to_horizontal = (self.width - relative_y) / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = (self.length - relative_x) / abs(math.cos(math.radians(relative_angle)))
        else:
            laser_to_horizontal = (self.width - relative_y) / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = relative_x / abs(math.cos(math.radians(relative_angle)))
        if min(laser_to_vertical, laser_to_horizontal) > max_laser_distance:
            return None
        if laser_to_horizontal <= laser_to_vertical:
            return laser_to_horizontal
        if -180 < relative_angle < -90:
            return IntersectToConnectionLine(
                Point(0, relative_y - laser_to_vertical * abs(math.sin(math.radians(relative_angle)))),
                relative_angle, 0, max_laser_distance - laser_to_vertical)
        elif -90 < relative_angle < 0:
            return IntersectToConnectionLine(
                Point(self.length, relative_y - laser_to_vertical * abs(math.sin(math.radians(relative_angle)))),
                relative_angle, 1, max_laser_distance - laser_to_vertical)
        elif 0 < relative_angle < 90:
            return IntersectToConnectionLine(
                Point(self.length, relative_y + laser_to_vertical * abs(math.sin(math.radians(relative_angle)))),
                relative_angle, 1, max_laser_distance - laser_to_vertical)
        else:
            return IntersectToConnectionLine(
                Point(0, relative_y + laser_to_vertical * abs(math.sin(math.radians(relative_angle)))),
                relative_angle, 0, max_laser_distance - laser_to_vertical)

    def step(self):
        pass


class RightTurnSegment(PolygonSegment):

    def __init__(self, length_of_smaller_wall, width, alpha, global_segment_angle):
        self.length_of_smaller_wall = length_of_smaller_wall
        self.width = width
        alpha = math.radians(alpha)
        self.alpha = alpha

        relative_vertices = [Point(0, 0), Point(width, 0), Point(width, length_of_smaller_wall),
                             Point(width + length_of_smaller_wall * math.sin(alpha),
                                   length_of_smaller_wall - length_of_smaller_wall * math.cos(alpha))]
        tmp_point = Point(0, length_of_smaller_wall + width / math.tan(alpha / 2))
        relative_vertices.append(Point(tmp_point.y * math.sin(alpha),
                                       tmp_point.y - tmp_point.y * math.cos(alpha)))
        relative_vertices.append(tmp_point)

        relative_main_line_segments = [LineSegment(relative_vertices[1], relative_vertices[2]),
                                       LineSegment(relative_vertices[2], relative_vertices[3]),
                                       LineSegment(relative_vertices[4], relative_vertices[5]),
                                       LineSegment(relative_vertices[5], relative_vertices[0])]

        relative_connection_line_segments = [LineSegment(relative_vertices[0], relative_vertices[1]),
                                             LineSegment(relative_vertices[3], relative_vertices[4])]
        super().__init__(global_segment_angle, 2, relative_vertices, relative_main_line_segments,
                         relative_connection_line_segments)

    def point_is_inside_or_on(self, point):
        for line_segment in self.relative_main_line_segments:
            if line_segment.is_on_segment(point):
                return True
        for line_segment in self.relative_connection_line_segments:
            if line_segment.is_on_segment(point):
                return True
        tmp_segment = LineSegment(self.relative_vertices[2], self.relative_vertices[5])
        if tmp_segment.is_on_segment(point):
            return True
        if Geometry.orientation(tmp_segment.start_point, tmp_segment.end_point, point) == -1:
            if Geometry.orientation(self.relative_vertices[5], self.relative_vertices[0], point) == \
                    Geometry.orientation(self.relative_vertices[0], self.relative_vertices[1], point) == \
                    Geometry.orientation(self.relative_vertices[1], self.relative_vertices[2], point) == -1:
                return True
            return False
        elif Geometry.orientation(tmp_segment.start_point, tmp_segment.end_point, point) == 1:
            if Geometry.orientation(self.relative_vertices[5], self.relative_vertices[4], point) == \
                    Geometry.orientation(self.relative_vertices[4], self.relative_vertices[3], point) == \
                    Geometry.orientation(self.relative_vertices[3], self.relative_vertices[2], point) == 1:
                return True
            return False
        return False

    def step(self):
        pass


# this class just uses LeftTurnSegment and uses symmetry
class LeftTurnSegment(Segment):

    def __init__(self, length_of_smaller_wall, width, alpha, global_segment_angle):
        super().__init__(global_segment_angle, 2)
        self.right_turn_helper = RightTurnSegment(length_of_smaller_wall, width, alpha,
                                                  get_in_range_angle(180 - global_segment_angle))
        self.length_of_smaller_wall = length_of_smaller_wall
        self.width = width
        alpha = math.radians(alpha)
        self.alpha = alpha

    def scan_lidar_one_laser(self, relative_position, relative_angle, max_laser_distance):
        tmp_res = self.right_turn_helper.scan_lidar_one_laser(Point(-relative_position.x, relative_position.y),
                                                              get_in_range_angle(180 - relative_angle),
                                                              max_laser_distance)
        if isinstance(tmp_res, int):
            return tmp_res
        elif isinstance(tmp_res, float):
            return tmp_res
        elif isinstance(tmp_res, IntersectToConnectionLine):
            return IntersectToConnectionLine(relative_point=Point(-tmp_res.relative_point.x, tmp_res.relative_point.y),
                                             relative_angle=relative_angle,
                                             connection_number=tmp_res.connection_number,
                                             remaining_laser=tmp_res.remaining_laser)
        elif tmp_res is None:
            return None
        raise Exception('unexpected result')

    def step(self):
        pass


class SegmentHolder:

    def __init__(self, segment):
        self.segment = segment
        self.connections = [(None, -1) for _ in range(segment.number_of_connections)]

    def set_segment_connection(self, self_connection_number, other_segment_holder, other_connection_number):
        self.connections[self_connection_number] = (other_segment_holder, other_connection_number)


class Map:

    # segments is a list of StraitSegment, ...
    # connections is a list of Connection
    def __init__(self, segment_holders):
        self.segment_holders = segment_holders
        self.check_map()

    def check_map(self):
        pass  # TODO: check if the map is well connected and widths are the same


# tt = RightTurnSegment(3, 3, 120, 3)
# print(tt.scan_lidar_one_laser(Point(1, 1), 120, 40))
# tt = LeftTurnSegment(3, 3, 120, 3)
# print(tt.scan_lidar_one_laser(Point(-1, 1), 60, 40))
