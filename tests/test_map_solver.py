import unittest

from map_solver import motion_polyline, ordered_shape_distance, resample_polyline


class MapSolverTests(unittest.TestCase):
    def test_motion_polyline_integrates_turns_and_ignores_rejections(self):
        points = motion_polyline([
            {"accepted": True, "yaw_delta_degrees": 0},
            {"accepted": False, "yaw_delta_degrees": 90},
            {"accepted": True, "yaw_delta_degrees": 30},
        ])
        self.assertEqual(len(points), 3)
        self.assertAlmostEqual(points[1][0], 0.0)
        self.assertGreater(points[2][0], 0.0)

    def test_ordered_shape_is_rotation_scale_and_reversal_invariant(self):
        route = [(0, 0), (0, 3), (2, 3), (2, 6), (5, 6)]
        transformed = [(10 - 4*y, -7 + 4*x) for x, y in reversed(route)]
        wrong = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]
        matching = ordered_shape_distance(route, transformed)
        mismatch = ordered_shape_distance(route, wrong)
        self.assertLess(matching["score"], mismatch["score"])
        self.assertIn("reverse", matching["hypothesis"])

    def test_resample_polyline_preserves_endpoints(self):
        sampled = resample_polyline([(1, 2), (1, 4), (4, 4)], count=7)
        self.assertEqual(sampled.shape, (7, 2))
        self.assertTrue((sampled[0] == (1, 2)).all())
        self.assertTrue((sampled[-1] == (4, 4)).all())

    def test_vo_projection_can_treat_reflection_as_free_axis_ambiguity(self):
        route = [(0, 0), (0, 3), (2, 3), (2, 6), (5, 6)]
        reflected = [(-x, y) for x, y in route]
        penalised = ordered_shape_distance(route, reflected, allow_reverse=False)
        free = ordered_shape_distance(
            route, reflected, allow_reverse=False, reflection_penalty=0.0
        )
        self.assertEqual(free["hypothesis"], "forward-reflect")
        self.assertAlmostEqual(free["score"], 0.0, places=8)
        self.assertGreater(penalised["score"], free["score"])


if __name__ == "__main__": unittest.main()
