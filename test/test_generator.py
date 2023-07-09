import unittest


class GeneratorTest(unittest.TestCase):
    """Test Generator Class. TODO: Before Train model, check data from your generator."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_generator_length_is_not_zero(self):
        raise NotImplementedError()

    def test_generated_batch_shape(self):
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
