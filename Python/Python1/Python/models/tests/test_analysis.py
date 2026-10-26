import tempfile
import struct
import unittest
from pathlib import Path

import numpy as np

from models.imageprocessing import ImageProcessing
from models.loading_analysis import measure_signals, fit_signals
from models.array_analysis import detect_sites, bootstrap_sites, resample_indices


def write_tga(path, image, top_down):
    height,width=image.shape
    header=struct.pack("<BBBHHBHHHHBB",0,0,3,0,0,0,0,0,width,height,16,
                       32 if top_down else 0)
    pixels=image if top_down else image[::-1]
    path.write_bytes(header+np.asarray(pixels,dtype="<u2").tobytes())


class AnalysisTests(unittest.TestCase):
    def test_pixel_values_and_both_vertical_orientations(self):
        original=np.array([[266,1000,1023],[1024,3146,60000]],dtype=np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            for top_down in (True,False):
                file=Path(folder)/"test.tga"
                write_tga(file,original,top_down)
                np.testing.assert_array_equal(ImageProcessing.read_camera_tga(file),original)

    def test_loader_count_crop_trial_order_and_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            for number in (10,2,1):
                write_tga(Path(folder)/f"P1_{number}_1.tga",
                          np.full((12,4),1000+number),False)
            loader=ImageProcessing()
            images,mean=loader.load_tga_folder(folder)
            self.assertEqual(images.shape,(2,4,3))
            self.assertEqual(loader.image_files,["P1_1_1.tga","P1_2_1.tga","P1_10_1.tga"])
            np.testing.assert_allclose(images[0,0],[1001,1002,1010])
            np.testing.assert_allclose(mean,images.mean(axis=2))

    def test_empty_folder_reports_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                ImageProcessing().load_tga_folder(folder)

    def test_background_subtraction_for_each_trial(self):
        images=np.full((12,12,3),[100,200,300],dtype=float)
        images[2:5,3:6,:]+=np.array([1,10,100])
        signal=measure_signals(images,(2,5),(3,6),(8,12),(8,12))
        np.testing.assert_allclose(signal,[9,90,900])

    def test_two_known_signal_populations_and_bootstrap(self):
        rng=np.random.default_rng(104)
        occupied=rng.random(600)<0.55
        signals=np.where(occupied,rng.normal(1000,110,600),rng.normal(100,30,600))
        result=fit_signals(signals)
        self.assertLess(abs(result["loading_fraction"]-occupied.mean()),0.02)
        self.assertLess(abs(result["p_atom"]-occupied.mean()),0.06)
        self.assertTrue(150<result["threshold"]<900)
        result["site"]=1
        bootstrap_sites([result],repeats=100,seed=44)
        self.assertGreaterEqual(result["bootstrap_successes"],80)
        self.assertLess(result["loading_fraction_ci_low"],occupied.mean())
        self.assertGreater(result["loading_fraction_ci_high"],occupied.mean())
        self.assertTrue(0<=result["fidelity_ci_low"]<=result["fidelity_ci_high"]<=1)

    def test_invalid_signals_rejected(self):
        for values in (np.ones(300),np.array([0,1,np.nan]),np.zeros((20,20))):
            with self.assertRaises(ValueError):
                fit_signals(values)

    def test_peak_detection_keeps_separate_sites(self):
        image=np.full((40,40),400.0)
        image[10,10]=5000;image[25,25]=6000
        positions=detect_sites(image,peak_threshold=550)
        self.assertEqual(positions.tolist(),[[10,10],[25,25]])

    def test_block_bootstrap_preserves_neighbours(self):
        indices=resample_indices(12,np.random.default_rng(2),3)
        self.assertEqual(len(indices),12)
        for block in indices.reshape(-1,3):
            np.testing.assert_array_equal(np.diff(block)%12,[1,1])


if __name__ == "__main__":
    unittest.main()
