import os
import shutil
import tempfile
import unittest
from datetime import date

import pandas as pd

from backend.tools.analysis import compute_comparison, normalized_prices
from backend.tools.errors import PlottingError
from backend.tools.plotting import plot_comparison, plot_single
from tests.fixtures import make_price_frame


class PlotComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_a_real_png_file(self):
        a = make_price_frame(date(2026, 1, 1), 60, seed=1)
        b = make_price_frame(date(2026, 1, 1), 60, seed=2)
        combined = compute_comparison(
            {"AAPL": a, "MSFT": b}, metric="normalized_returns"
        )
        path = plot_comparison(
            combined, metric="normalized_returns", output_dir=self.tmp_dir
        )
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        with open(path, "rb") as f:
            header = f.read(8)
        self.assertEqual(header[:4], b"\x89PNG"[:4])

    def test_raises_on_empty_dataframe(self):
        with self.assertRaises(PlottingError):
            plot_comparison(pd.DataFrame(), output_dir=self.tmp_dir)

    def test_custom_filename_is_respected(self):
        a = make_price_frame(date(2026, 1, 1), 30)
        combined = compute_comparison({"AAPL": a}, metric="price")
        path = plot_comparison(
            combined, output_dir=self.tmp_dir, filename="my_chart.png"
        )
        self.assertTrue(path.endswith("my_chart.png"))

    def test_creates_output_dir_if_missing(self):
        nested = os.path.join(self.tmp_dir, "nested", "dirs")
        a = make_price_frame(date(2026, 1, 1), 20)
        combined = compute_comparison({"AAPL": a}, metric="price")
        path = plot_comparison(combined, output_dir=nested)
        self.assertTrue(os.path.exists(path))


class PlotSingleTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_a_real_png_file(self):
        df = make_price_frame(date(2026, 1, 1), 40)
        series = normalized_prices(df)
        path = plot_single(
            series, ticker="TSLA", metric="normalized_returns", output_dir=self.tmp_dir
        )
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_raises_on_empty_series(self):
        with self.assertRaises(PlottingError):
            plot_single(pd.Series(dtype=float), ticker="TSLA", output_dir=self.tmp_dir)


if __name__ == "__main__":
    unittest.main()
