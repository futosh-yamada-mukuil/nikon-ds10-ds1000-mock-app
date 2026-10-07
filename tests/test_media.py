"""Generated temporary files verify RGB decoding, seeking and error boundaries."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from app.media import MediaError, MediaSource


class MediaSourceTests(unittest.TestCase):
    def test_image_copy_rgb_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "画像 sample.png"
            Image.new("L", (40, 30), 120).save(path)
            source = MediaSource(path)
            self.assertFalse(source.is_video)
            self.assertEqual((source.frame_count, source.fps, source.duration), (1, 0.0, 0.0))
            image = source.read()
            self.assertEqual((image.mode, image.size), ("RGB", (40, 30)))
            image.putpixel((0, 0), (255, 0, 0))
            self.assertEqual(source.read().getpixel((0, 0)), (120, 120, 120))
            with self.assertRaises(IndexError):
                source.read(1)
            source.close()
            source.close()
            with self.assertRaises(MediaError):
                source.read()

    def test_generated_video_roundtrip_sequential_and_seek(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
            self.assertTrue(writer.isOpened(), "The environment must support MJPG for this test.")
            for color in ((0, 0, 240), (0, 240, 0), (240, 0, 0)):
                frame = np.empty((48, 64, 3), dtype=np.uint8)
                frame[:] = color
                writer.write(frame)
            writer.release()
            with MediaSource(path) as source:
                self.assertTrue(source.is_video)
                self.assertEqual((source.frame_count, source.width, source.height), (3, 64, 48))
                self.assertAlmostEqual(source.fps, 5.0)
                self.assertAlmostEqual(source.duration, 0.6)
                red = source.read(0)
                green = source.read(1)
                blue = source.read(2)
                self.assertGreater(red.getpixel((20, 20))[0], 200)
                self.assertGreater(green.getpixel((20, 20))[1], 200)
                self.assertGreater(blue.getpixel((20, 20))[2], 200)
                self.assertGreater(source.read(0).getpixel((20, 20))[0], 200)
                for index in (-1, 3, 0.2, True):
                    with self.subTest(index=index), self.assertRaises(IndexError):
                        source.read(index)
            with self.assertRaises(MediaError):
                source.read(0)

    def test_missing_nonmedia_and_corrupt_inputs_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with self.assertRaises(MediaError):
                MediaSource(directory / "missing.mp4")
            text = directory / "readme.txt"
            text.write_text("hello", encoding="utf-8")
            with self.assertRaises(MediaError):
                MediaSource(text)
            bad_image = directory / "bad.png"
            bad_image.write_bytes(b"not an image")
            with self.assertRaises(MediaError):
                MediaSource(bad_image)
            bad_video = directory / "bad.mp4"
            bad_video.write_bytes(b"not a video")
            with self.assertRaises(MediaError):
                MediaSource(bad_video)


if __name__ == "__main__":
    unittest.main()
