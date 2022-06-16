import math
import unittest
import numpy as np

from cftool.misc import timeit
from cftool.array import allclose

from cfc.api import *
from cfc.stat import *


test_dict = {}
types = ["used", "naive"]


class TestCases(unittest.TestCase):
    arr1 = arr2 = arr3 = None

    @classmethod
    def setUpClass(cls) -> None:
        multiple = 10**5
        cls.arr1 = ["1", 2, 3.4, "5.6"] * multiple
        cls.arr2 = ["1", 2, 3.4, "5.6", "7.8.9"] * multiple
        cls.arr3 = [1, 2, 3.4, 5.6, 7.8] * multiple

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.arr1, cls.arr2, cls.arr3

    @staticmethod
    def _print_header(title: str) -> None:
        print("\n".join(["=" * 100, title, "-" * 100]))

    def test_is_all_numeric(self) -> None:
        self._print_header("is_all_numeric")
        methods = [is_all_numeric, naive_is_all_numeric]
        for t, m in zip(types, methods):
            with timeit(t):
                self.assertTrue(m(self.arr1))
                self.assertFalse(m(self.arr2))

    def test_flat_arr_to_float32(self) -> None:
        self._print_header("flat_arr_to_float32")
        results = []
        methods = [flat_arr_to_float32, naive_flat_arr_to_float32]
        for t, m in zip(types, methods):
            with timeit(t):
                results.append(m(self.arr1))
        self.assertTrue(allclose(*results))

    def test_transform_flat_data_with_dict(self) -> None:
        self._print_header("transform_flat_data_with_dict")
        results = []
        arr = np.array(self.arr3, dtype=np.float32)
        transform_dict = {1: 0, 2: 1, 3.4: 2, 5.6: 3, 7.8: 4}
        methods = [transform_flat_data_with_dict, naive_transform_flat_data_with_dict]
        for t, m in zip(types, methods):
            with timeit(t):
                results.append(m(arr, transform_dict, False))
        self.assertTrue(allclose(*results))

    def test_rolling_stat(self):
        n = 10000
        arr = np.arange(n)
        mean_gt = np.arange(1, n - 1)
        std_gt = np.full([n - 2], math.sqrt(2.0 / 3.0))
        self.assertTrue(np.allclose(RollingStat.mean(arr, 3), mean_gt))
        self.assertTrue(np.allclose(RollingStat.std(arr, 3), std_gt))


if __name__ == "__main__":
    unittest.main()
