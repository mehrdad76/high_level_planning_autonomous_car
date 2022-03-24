import math
import numpy as np
from Geometry import Point, LineSegment
import Geometry

MAX_LASER_DISTANCE = 50
ZERO_THRESHOLD = 0.0001
NUMBER_OF_LASERS = 11
VIEWING_RANGE = 120


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

    def __init__(self, number_of_connections, relative_connection_line_segments):
        self.number_of_connections = number_of_connections
        self.relative_connection_line_segments = relative_connection_line_segments
        self.connection_angles = []
        for index in range(number_of_connections):
            p1 = relative_connection_line_segments[index].start_point
            p2 = relative_connection_line_segments[index].end_point
            length = relative_connection_line_segments[index].length()
            sin = (p2.y - p1.y) / length
            cos = (p2.x - p1.x) / length
            a_acos = math.acos(cos)
            if sin < 0:
                angle = math.degrees(-a_acos) % 360
            else:
                angle = math.degrees(a_acos)
            self.connection_angles.append(angle)

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

    def __init__(self, number_of_connections, relative_vertices,
                 relative_main_line_segments, relative_connection_line_segments):
        super().__init__(number_of_connections, relative_connection_line_segments)
        self.relative_vertices = relative_vertices
        self.relative_main_line_segments = relative_main_line_segments

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
        for index in range(self.number_of_connections):
            line_segment = self.relative_connection_line_segments[index]
            intersection = line_segment.get_intersection_to(laser)
            if intersection is None:
                continue
            distance = relative_position.get_distance_to(intersection)
            if distance < min_distance:
                min_distance = distance
                res = IntersectToConnectionLine(intersection, math.degrees(relative_angle), index,
                                                max_laser_distance - distance)

        if min_distance <= max_laser_distance:
            return res
        return None

    def step(self):
        raise NotImplementedError("Please Implement this method")


# for Rectangle segments we could use polygon but following implementation is faster
class RectangleSegment(Segment):

    def __init__(self, length, width):
        super().__init__(2, [LineSegment(Point(0, 0), Point(width, 0)),
                             LineSegment(Point(width, length), Point(0, length))])
        self.length = length
        self.width = width

    def scan_lidar_one_laser(self, relative_position, relative_angle, max_laser_distance):
        relative_angle = get_in_range_angle(relative_angle)
        relative_x, relative_y = relative_position.x, relative_position.y
        assert 0 <= relative_y <= self.length and 0 <= relative_x <= self.width
        if relative_angle == 0:
            distance_to_right = self.width - relative_x
            if distance_to_right > max_laser_distance:
                return None
            return distance_to_right
        if relative_angle == 180:
            if relative_x > max_laser_distance:
                return None
            return relative_x
        if relative_angle == 90:
            distance_to_top = self.length - relative_y
            if distance_to_top > max_laser_distance:
                return None
            return IntersectToConnectionLine(Point(relative_x, self.length), relative_angle,
                                             1, max_laser_distance - distance_to_top)
        if relative_angle == -90:
            if relative_y > max_laser_distance:
                return None
            return IntersectToConnectionLine(Point(relative_x, 0), relative_angle,
                                             0, max_laser_distance - relative_y)
        if -180 < relative_angle < -90:
            laser_to_horizontal = relative_y / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = relative_x / abs(math.cos(math.radians(relative_angle)))
        elif -90 < relative_angle < 0:
            laser_to_horizontal = relative_y / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = (self.width - relative_x) / abs(math.cos(math.radians(relative_angle)))
        elif 0 < relative_angle < 90:
            laser_to_horizontal = (self.length - relative_y) / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = (self.width - relative_x) / abs(math.cos(math.radians(relative_angle)))
        else:
            laser_to_horizontal = (self.length - relative_y) / abs(math.sin(math.radians(relative_angle)))
            laser_to_vertical = relative_x / abs(math.cos(math.radians(relative_angle)))
        if min(laser_to_vertical, laser_to_horizontal) > max_laser_distance:
            return None
        if laser_to_vertical <= laser_to_horizontal:
            return laser_to_vertical
        if -180 < relative_angle < -90:
            return IntersectToConnectionLine(
                Point(relative_x - laser_to_horizontal * abs(math.sin(math.radians(relative_angle))), 0),
                relative_angle, 0, max_laser_distance - laser_to_horizontal)
        elif -90 < relative_angle < 0:
            return IntersectToConnectionLine(
                Point(relative_x + laser_to_horizontal * abs(math.sin(math.radians(relative_angle))), 0),
                relative_angle, 0, max_laser_distance - laser_to_horizontal)
        elif 0 < relative_angle < 90:
            return IntersectToConnectionLine(
                Point(relative_x + laser_to_horizontal * abs(math.sin(math.radians(relative_angle))), self.length),
                relative_angle, 1, max_laser_distance - laser_to_horizontal)
        else:
            return IntersectToConnectionLine(
                Point(relative_x - laser_to_horizontal * abs(math.sin(math.radians(relative_angle))), 0),
                relative_angle, 1, max_laser_distance - laser_to_horizontal)

    def step(self):
        pass


class RightTurnSegment(PolygonSegment):

    def __init__(self, length_of_smaller_wall, width1, width2, alpha):
        self.length_of_smaller_wall = length_of_smaller_wall
        self.width1 = width1
        self.width2 = width2
        alpha = math.radians(alpha)
        self.alpha = alpha

        relative_vertices = [Point(0, 0), Point(width1, 0), Point(width1, length_of_smaller_wall),
                             Point(width1 + length_of_smaller_wall * math.sin(alpha),
                                   length_of_smaller_wall - length_of_smaller_wall * math.cos(alpha))]
        tmp_point = Point(0, length_of_smaller_wall + width1 / math.tan(alpha) + width2 / math.sin(alpha))
        relative_vertices.append(Point(tmp_point.y * math.sin(alpha),
                                       tmp_point.y - tmp_point.y * math.cos(alpha)))
        relative_vertices.append(tmp_point)

        relative_main_line_segments = [LineSegment(relative_vertices[1], relative_vertices[2]),
                                       LineSegment(relative_vertices[2], relative_vertices[3]),
                                       LineSegment(relative_vertices[4], relative_vertices[5]),
                                       LineSegment(relative_vertices[5], relative_vertices[0])]

        relative_connection_line_segments = [LineSegment(relative_vertices[0], relative_vertices[1]),
                                             LineSegment(relative_vertices[3], relative_vertices[4])]
        super().__init__(2, relative_vertices,
                         relative_main_line_segments,
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


class SymmetricSplitterSegment(PolygonSegment):

    def __init__(self, length_of_main_walls, width):
        self.length_of_main_walls = length_of_main_walls
        self.width = width
        relative_vertices = [Point(0, 0), Point(width, 0), Point(width, length_of_main_walls),
                             Point(width + length_of_main_walls * math.cos(math.radians(30)),
                                   length_of_main_walls + length_of_main_walls * math.sin(math.radians(30)))]
        tmp = length_of_main_walls + width * math.cos(math.radians(30))
        relative_vertices.append(Point(length_of_main_walls * math.cos(math.radians(30)) + width / 2,
                                       tmp + length_of_main_walls * math.sin(math.radians(30))))
        relative_vertices.append(Point(width / 2, tmp))
        relative_vertices.append(Point(-length_of_main_walls * math.cos(math.radians(30)) + width / 2,
                                       tmp + length_of_main_walls * math.sin(math.radians(30))))
        relative_vertices.append(Point(-length_of_main_walls * math.cos(math.radians(30)),
                                       length_of_main_walls + length_of_main_walls * math.sin(math.radians(30))))
        relative_vertices.append(Point(0, length_of_main_walls))

        relative_main_line_segments = [LineSegment(relative_vertices[1], relative_vertices[2]),
                                       LineSegment(relative_vertices[2], relative_vertices[3]),
                                       LineSegment(relative_vertices[4], relative_vertices[5]),
                                       LineSegment(relative_vertices[5], relative_vertices[6]),
                                       LineSegment(relative_vertices[7], relative_vertices[8]),
                                       LineSegment(relative_vertices[8], relative_vertices[0])]

        relative_connection_line_segments = [LineSegment(relative_vertices[0], relative_vertices[1]),
                                             LineSegment(relative_vertices[3], relative_vertices[4]),
                                             LineSegment(relative_vertices[6], relative_vertices[7])]

        super().__init__(3, relative_vertices,
                         relative_main_line_segments,
                         relative_connection_line_segments)

    def point_is_inside_or_on(self, point):
        if -self.width / 2 <= point.x <= self.width / 2 and 0 <= point.y <= self.relative_vertices[5].y:
            return True
        if ((self.relative_vertices[3].x <= point.x <= self.relative_vertices[5].x) or
            (self.relative_vertices[5].x <= point.x <= self.relative_vertices[3].x)) and \
                ((self.relative_vertices[3].y <= point.y <= self.relative_vertices[5].y) or
                 (self.relative_vertices[5].y <= point.y <= self.relative_vertices[3].y)):
            return True
        if ((self.relative_vertices[7].x <= point.x <= self.relative_vertices[5].x) or
            (self.relative_vertices[5].x <= point.x <= self.relative_vertices[7].x)) and \
                ((self.relative_vertices[7].y <= point.y <= self.relative_vertices[5].y) or
                 (self.relative_vertices[5].y <= point.y <= self.relative_vertices[7].y)):
            return True
        return False

    def step(self):
        pass


class SegmentHolder:

    def __init__(self, segment):
        self.segment = segment
        self.global_angle = None
        self.origin_coordinates = None
        self.connections = [(None, -1) for _ in range(segment.number_of_connections)]

    def set_segment_connection(self, self_connection_number, other_segment_holder, other_connection_number):
        self.connections[self_connection_number] = (other_segment_holder, other_connection_number)

    def __str__(self):
        return 'segment type:' + str(type(self.segment)) + ', origin_coordinates: ' + str(self.origin_coordinates)


class Car:

    def __init__(self, position, angle, hysteresis_constant, car_accelerator_constant, car_motor_constant, car_length):
        self.hysteresis_constant = hysteresis_constant
        self.car_accelerator_constant = car_accelerator_constant
        self.car_motor_constant = car_motor_constant
        self.car_length = car_length
        self.position = position
        self.angle = angle

    def bicycle_dynamics_no_beta(self, x, u, delta, turn):

        if turn < 0:  # right turn
            # -V * sin(theta_local)
            ds_dt = -x[2] * np.sin(x[3])
        else:
            # V * sin(theta_local)
            ds_dt = x[2] * np.sin(x[3])

        # -V * cos(theta_local)
        df_dt = -x[2] * np.cos(x[3])

        if u > self.hysteresis_constant:
            # a * u - V
            dVdt = self.car_accelerator_constant * self.car_motor_constant * \
                   (u - self.hysteresis_constant) - self.car_accelerator_constant * x[2]
        else:
            dVdt = - self.car_accelerator_constant * x[2]

        # V * tan(delta) / l
        d_theta_ldt = x[2] * np.tan(delta) / self.car_length

        # V * cos(theta_global)
        dx_dt = x[2] * np.cos(x[6])

        # V * sin(theta_global)
        dy_dt = x[2] * np.sin(x[6])

        # V * tan(delta) / l
        d_theta_gdt = x[2] * np.tan(delta) / self.car_length

        dXdt = [ds_dt, df_dt, dVdt, d_theta_ldt, dx_dt, dy_dt, d_theta_gdt]

        return dXdt


class Map:  # origin of map is always origin of segment 1

    # segments is a list of StraitSegment, ...
    # connections is a list of Connection
    def __init__(self):
        self.segment_holders = []
        self.current_segment_holder_index = 0

    def add_segment(self, segment):
        self.segment_holders.append(SegmentHolder(segment))

    def add_connection(self, segment_holder_1, connection_number_1, segment_holder_2, connection_number_2):
        self.segment_holders[segment_holder_1].set_segment_connection(connection_number_1,
                                                                      self.segment_holders[segment_holder_2],
                                                                      connection_number_2)
        self.segment_holders[segment_holder_2].set_segment_connection(connection_number_2,
                                                                      self.segment_holders[segment_holder_1],
                                                                      connection_number_1)

    def scan_lidar(self, car):
        position = car.position
        angle = car.angle
        current_segment_holder = self.segment_holders[self.current_segment_holder_index]
        lidar_res = current_segment_holder.segment.scan_lidar(Point(position.x -
                                                                    current_segment_holder.origin_coordinates.x,
                                                                    position.y -
                                                                    current_segment_holder.origin_coordinates.y).
                                                              get_rotated(-current_segment_holder.global_angle),
                                                              NUMBER_OF_LASERS,
                                                              angle - current_segment_holder.global_angle,
                                                              VIEWING_RANGE,
                                                              MAX_LASER_DISTANCE)

        for index in range(len(lidar_res)):
            lidar = lidar_res[index]
            add = 0
            current_segment_holder = self.segment_holders[self.current_segment_holder_index]
            while isinstance(lidar[1], IntersectToConnectionLine):
                tmp = lidar[1].relative_point.get_rotated(current_segment_holder.global_angle)
                global_point = Point(tmp.x + current_segment_holder.origin_coordinates.x,
                                     tmp.y + current_segment_holder.origin_coordinates.y)
                global_angle = lidar[1].relative_angle + current_segment_holder.global_angle
                connection_number = lidar[1].connection_number
                remaining_laser = lidar[1].remaining_laser
                new_segment_holder, new_connection_number = current_segment_holder.connections[connection_number]
                if new_segment_holder is None:
                    break
                new_segment = new_segment_holder.segment
                add = MAX_LASER_DISTANCE - remaining_laser
                lidar = (lidar_res[index][0], new_segment
                         .scan_lidar_one_laser(Point(global_point.x - new_segment_holder.origin_coordinates.x,
                                                     global_point.y - new_segment_holder.origin_coordinates.y).
                                               get_rotated(-new_segment_holder.global_angle),
                                               global_angle - new_segment_holder.global_angle, remaining_laser))
                current_segment_holder = new_segment_holder

            if isinstance(lidar[1], float) or isinstance(lidar[1], int):
                lidar_res[index] = (lidar[0], lidar[1] + add)
            else:
                lidar_res[index] = lidar

        return lidar_res

    def check_map(self):
        disconnect_count = 0
        for segment_holder in self.segment_holders:
            for index in range(len(segment_holder.connections)):
                connection = segment_holder.connections[index]
                if connection[0] is None:
                    disconnect_count += 1
                    continue
                if abs(segment_holder.segment.relative_connection_line_segments[index].length() -
                       connection[0].segment.relative_connection_line_segments[connection[1]].length()) >= \
                        ZERO_THRESHOLD:
                    raise Exception('width of segments mismatched')

        print('WARNING: There are ' + str(disconnect_count) + ' disconnects in segments.')

    def set_global_origin_coordinates_and_angles(self, angle_of_first_segment):
        if self.segment_holders is None or len(self.segment_holders) == 0:
            return
        self.segment_holders[0].origin_coordinates = Point(0, 0)
        self.segment_holders[0].global_angle = get_in_range_angle(angle_of_first_segment)
        if len(self.segment_holders) == 1:
            return
        q = []
        seen = [self.segment_holders[0]]
        for connection in self.segment_holders[0].connections:
            if connection[0] is not None and connection[0] not in seen:
                q.append(connection)
                seen.append(connection[0])
        while q:
            segment_holder, connection_number = q.pop()
            segment = segment_holder.segment
            other_segment_holder, other_connection_number = segment_holder.connections[connection_number]
            other_segment = other_segment_holder.segment
            global_angle = other_segment_holder.global_angle + 180 + \
                           other_segment.connection_angles[other_connection_number] - \
                           segment.connection_angles[connection_number]
            global_angle = get_in_range_angle(global_angle)

            other_segment_shift = other_segment.relative_connection_line_segments[other_connection_number].end_point
            segment_shift = segment.relative_connection_line_segments[connection_number].start_point
            other_segment_shift = other_segment_shift.get_rotated(other_segment_holder.global_angle)
            segment_shift = segment_shift.get_rotated(global_angle)
            global_position_x = other_segment_holder.origin_coordinates.x + \
                                other_segment_shift.x - segment_shift.x
            global_position_y = other_segment_holder.origin_coordinates.y + \
                                other_segment_shift.y - segment_shift.y

            if segment_holder.origin_coordinates is not None:
                assert abs(segment_holder.origin_coordinates.x - global_position_x) <= ZERO_THRESHOLD
                assert abs(segment_holder.origin_coordinates.y - global_position_y) <= ZERO_THRESHOLD
                assert abs(segment_holder.global_angle - global_angle) <= ZERO_THRESHOLD

            segment_holder.origin_coordinates = Point(global_position_x, global_position_y)
            segment_holder.global_angle = global_angle
            for connection in segment_holder.connections:
                if connection[0] is not None and connection[0] not in seen:
                    q.append(connection)
                    seen.append(connection[0])

    def create(self, angle_of_first_segment):
        self.check_map()
        self.set_global_origin_coordinates_and_angles(angle_of_first_segment)


m = Map()

m.add_segment(SymmetricSplitterSegment(3, 3))
m.add_segment(RightTurnSegment(3, 3, 3, 60))
m.add_segment(RightTurnSegment(3, 3, 3, 60))
m.add_segment(SymmetricSplitterSegment(3, 3))
m.add_segment(RightTurnSegment(5, 3, 3, 90))
m.add_segment(RectangleSegment(8, 3))

m.add_connection(0, 1, 2, 1)
m.add_connection(0, 2, 1, 0)
m.add_connection(1, 1, 3, 1)
m.add_connection(2, 0, 3, 2)
m.add_connection(3, 0, 4, 0)
m.add_connection(4, 1, 5, 1)

m.create(270)
m.current_segment_holder_index = 4

print(m.scan_lidar(Car(Point(23, -7), 270, 0, 0, 0, 0)))
