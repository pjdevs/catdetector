import unittest

from catdetector_model.transforms import eval_transform


class EvalTransformTest(unittest.TestCase):
    def test_eval_transform_is_deterministic_inference_pipeline(self):
        transform = eval_transform()

        step_names = [type(step).__name__ for step in transform.transforms]

        self.assertEqual(step_names, ["Resize", "ToTensor", "Normalize"])


if __name__ == "__main__":
    unittest.main()
