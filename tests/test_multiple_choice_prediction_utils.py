import unittest

from forecasting_tools import PredictedOption

from main import _get_option_name, _get_option_probability


class MultipleChoicePredictionUtilsTests(unittest.TestCase):
    def test_extracts_values_from_pydantic_model_and_dict(self) -> None:
        options = [
            PredictedOption(option_name="A", probability=0.2),
            {"option_name": "B", "probability": 0.8},
        ]

        self.assertEqual([0.2, 0.8], [_get_option_probability(opt) for opt in options])
        self.assertEqual(["A", "B"], [_get_option_name(opt) for opt in options])


if __name__ == "__main__":
    unittest.main()
