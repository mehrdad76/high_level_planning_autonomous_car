import math


def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_tuple(self):
        return self.x, self.y

    def get_distance_to(self, point):
        return math.sqrt((math.pow(self.y - point.y, 2)) + (math.pow(self.x - point.x, 2)))

    def get_rotated(self, alpha):
        return Point(self.x * math.cos(math.radians(alpha)) - self.y * math.sin(math.radians(alpha)),
                     self.x * math.sin(math.radians(alpha)) + self.y * math.cos(math.radians(alpha)))

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'


class LineSegment:

    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point

    def is_on_segment(self, point):
        return ((point.y - self.start_point.y) * (self.end_point.x - self.start_point.x) ==
                (point.x - self.start_point.x) * (self.end_point.y - self.start_point.y)) and \
               ((self.start_point.x <= point.x <= self.end_point.x) or
                (self.end_point.x <= point.x <= self.start_point.x)) and \
               ((self.start_point.y <= point.y <= self.end_point.y) or
                (self.end_point.y <= point.y <= self.start_point.y))

    def collinear_on_segment(self, point):
        if ((point.x <= max(self.start_point.x, self.end_point.x)) and (
                point.x >= min(self.start_point.x, self.end_point.x)) and
                (point.y <= max(self.start_point.y, self.end_point.y)) and (
                        point.y >= min(self.start_point.y, self.end_point.y))):
            return True
        return False

    def do_intersect_to(self, line_segment):
        o1 = orientation(self.start_point, self.end_point, line_segment.start_point)
        o2 = orientation(self.start_point, self.end_point, line_segment.end_point)
        o3 = orientation(line_segment.start_point, line_segment.end_point, self.start_point)
        o4 = orientation(line_segment.start_point, line_segment.end_point, self.end_point)

        if (o1 != o2) and (o3 != o4):
            return True

        if (o1 == 0) and self.collinear_on_segment(line_segment.start_point):
            return True

        if (o2 == 0) and self.collinear_on_segment(line_segment.end_point):
            return True

        if (o3 == 0) and line_segment.collinear_on_segment(self.start_point):
            return True

        if (o4 == 0) and line_segment.collinear_on_segment(self.end_point):
            return True

        return False

    def get_intersection_to(self, line_segment):
        if not self.do_intersect_to(line_segment):
            return None

        xdiff = (self.start_point.x - self.end_point.x, line_segment.start_point.x - line_segment.end_point.x)
        ydiff = (self.start_point.y - self.end_point.y, line_segment.start_point.y - line_segment.end_point.y)

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(self.start_point.get_tuple(), self.end_point.get_tuple()),
             det(line_segment.start_point.get_tuple(), line_segment.end_point.get_tuple()))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)

    def length(self):
        return abs(self.start_point.get_distance_to(self.end_point))

    def __str__(self):
        return '(' + str(self.start_point) + ' -> ' + str(self.end_point) + ')'

