import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from realsense_camera import RealsenseCamera


class TestRealsenseCameraInit:
    """Test RealsenseCamera initialization"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_creates_pipeline(self, mock_align, mock_config, mock_pipeline):
        """Test that __init__ creates a pipeline"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        camera = RealsenseCamera()

        assert camera.pipeline is not None
        mock_pipeline.assert_called_once()

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_configures_color_stream(self, mock_align, mock_config, mock_pipeline):
        """Test that __init__ configures color stream"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        camera = RealsenseCamera()

        mock_config_instance.enable_stream.assert_any_call(
            mock_config.return_value.enable_stream.__self__.__class__.__bases__[
                0
            ].__dict__.get("color", "color"),
            1280,
            720,
            "bgr8",
            30,
        )

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_starts_pipeline(self, mock_align, mock_config, mock_pipeline):
        """Test that __init__ starts the pipeline"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        camera = RealsenseCamera()

        mock_pipeline_instance.start.assert_called_once()

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_creates_align(self, mock_align, mock_config, mock_pipeline):
        """Test that __init__ creates alignment object"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        camera = RealsenseCamera()

        assert camera.align is not None
        mock_align.assert_called_once()


class TestRealsenseCameraGetFrame:
    """Test get_frame_stream method"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    @patch("realsense_camera.rs.spatial_filter")
    @patch("realsense_camera.rs.hole_filling_filter")
    @patch("realsense_camera.rs.colorizer")
    def test_get_frame_stream_returns_tuple(
        self,
        mock_colorizer,
        mock_hole_fill,
        mock_spatial_filter,
        mock_align,
        mock_config,
        mock_pipeline,
    ):
        """Test that get_frame_stream returns tuple with correct structure"""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_depth_frame = MagicMock()
        mock_color_frame = MagicMock()
        mock_frames = MagicMock()

        mock_aligned_frames = MagicMock()
        mock_aligned_frames.get_depth_frame.return_value = mock_depth_frame
        mock_aligned_frames.get_color_frame.return_value = mock_color_frame

        mock_pipeline_instance.wait_for_frames.return_value = mock_frames
        camera = RealsenseCamera()
        camera.align.process.return_value = mock_aligned_frames

        # Setup depth processing
        mock_spatial_instance = MagicMock()
        mock_spatial_filter.return_value = mock_spatial_instance
        mock_filtered_depth = MagicMock()
        mock_spatial_instance.process.return_value = mock_filtered_depth

        # Setup hole filling
        mock_hole_fill_instance = MagicMock()
        mock_hole_fill.return_value = mock_hole_fill_instance
        mock_filled_depth = MagicMock()
        mock_hole_fill_instance.process.return_value = mock_filled_depth

        # Setup colorizer
        mock_colorizer_instance = MagicMock()
        mock_colorizer.return_value = mock_colorizer_instance
        mock_colorized = MagicMock()
        mock_colorizer_instance.colorize.return_value = mock_colorized
        mock_colorized.get_data.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Setup frame data
        mock_filled_depth.get_data.return_value = np.zeros((720, 1280), dtype=np.uint16)
        mock_color_frame.get_data.return_value = np.zeros(
            (720, 1280, 3), dtype=np.uint8
        )

        ret, color_img, depth_img = camera.get_frame_stream()

        assert isinstance(ret, (bool, np.bool_))
        assert isinstance(color_img, (np.ndarray, type(None)))
        assert isinstance(depth_img, (np.ndarray, type(None)))

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_get_frame_stream_returns_false_on_missing_frames(
        self, mock_align, mock_config, mock_pipeline
    ):
        """Test that get_frame_stream returns False when frames are missing"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_frames = MagicMock()
        mock_aligned_frames = MagicMock()
        mock_aligned_frames.get_depth_frame.return_value = None
        mock_aligned_frames.get_color_frame.return_value = None

        mock_pipeline_instance.wait_for_frames.return_value = mock_frames
        camera = RealsenseCamera()
        camera.align.process.return_value = mock_aligned_frames

        ret, color_img, depth_img = camera.get_frame_stream()

        assert ret is False
        assert color_img is None
        assert depth_img is None


class TestRealsenseCameraRelease:
    """Test release method"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_release_stops_pipeline(self, mock_align, mock_config, mock_pipeline):
        """Test that release stops the pipeline"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        camera = RealsenseCamera()
        camera.release()

        mock_pipeline_instance.stop.assert_called_once()

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_release_does_not_raise_exception(
        self, mock_align, mock_config, mock_pipeline
    ):
        """Test that release doesn't raise an exception"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        camera = RealsenseCamera()

        try:
            camera.release()
        except Exception as e:
            pytest.fail(f"release() raised {e}")
