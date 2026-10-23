import unittest

from kilometer_vision.motion_solver import compass_option_index, compass_prediction, summarize_steps, wrap_degrees


class MotionSolverTests(unittest.TestCase):
    def test_wrap_and_compass_prediction(self):
        self.assertEqual(wrap_degrees(270), -90)
        question={"question":"Given that the camera is pointing towards the North at the start, where is it facing?"}
        self.assertEqual(compass_prediction(question,90),"east")

    def test_nearest_offered_direction_handles_missing_exact_octant(self):
        question={"question":"The camera is pointing towards the North at the start.",
                  "options":["East","South","West","Northwest","North"]}
        self.assertEqual(compass_option_index(question,-148),1)

    def test_summary_ignores_rejected_steps(self):
        result=summarize_steps([{"accepted":True,"yaw_delta_degrees":30,"confidence":.8},{"accepted":False,"yaw_delta_degrees":100,"confidence":1}])
        self.assertEqual(result["net_yaw_degrees"],30)
        self.assertEqual(result["accepted_step_count"],1)


if __name__ == "__main__": unittest.main()
