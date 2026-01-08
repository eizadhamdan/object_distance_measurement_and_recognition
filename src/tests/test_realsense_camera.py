import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys


class TestRealsenseCameraStructure:
    """Test RealsenseCamera class structure without requiring device"""

    def test_realsense_file_exists(self):
        """Test that realsense_camera.py file exists"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        assert os.path.exists(src_path)

    def test_realsense_class_defined(self):
        """Test that RealsenseCamera class is defined"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "class RealsenseCamera" in source

    def test_realsense_init_defined(self):
        """Test that __init__ method is defined"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "def __init__" in source

    def test_realsense_get_frame_stream_defined(self):
        """Test that get_frame_stream method is defined"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "def get_frame_stream" in source

    def test_realsense_release_defined(self):
        """Test that release method is defined"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "def release" in source

    def test_realsense_uses_realsense_sdk(self):
        """Test that realsense_camera uses pyrealsense2"""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "realsense_camera.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "import pyrealsense2" in source or "rs." in source


class TestRealsenseCameraInit:
    """Test RealsenseCamera initialization with mocks only"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_creates_instance(self, mock_align, mock_config, mock_pipeline):
        """Test that RealsenseCamera can be instantiated with mocks"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "realsense_camera" in sys.modules:
            del sys.modules["realsense_camera"]
        from realsense_camera import RealsenseCamera

        camera = RealsenseCamera()
        assert camera is not None

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_init_has_required_attributes(self, mock_align, mock_config, mock_pipeline):
        """Test that __init__ creates required attributes"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "realsense_camera" in sys.modules:
            del sys.modules["realsense_camera"]
        from realsense_camera import RealsenseCamera

        camera = RealsenseCamera()
        assert hasattr(camera, "pipeline")
        assert hasattr(camera, "align")


class TestRealsenseCameraGetFrame:
    """Test get_frame_stream method"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_get_frame_stream_method_exists(
        self, mock_align, mock_config, mock_pipeline
    ):
        """Test that get_frame_stream method exists"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "realsense_camera" in sys.modules:
            del sys.modules["realsense_camera"]
        from realsense_camera import RealsenseCamera

        camera = RealsenseCamera()
        assert callable(camera.get_frame_stream)


class TestRealsenseCameraRelease:
    """Test release method"""

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_release_method_exists(self, mock_align, mock_config, mock_pipeline):
        """Test that release method exists"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "realsense_camera" in sys.modules:
            del sys.modules["realsense_camera"]
        from realsense_camera import RealsenseCamera

        camera = RealsenseCamera()
        assert callable(camera.release)

    @patch("realsense_camera.rs.pipeline")
    @patch("realsense_camera.rs.config")
    @patch("realsense_camera.rs.align")
    def test_release_calls_stop(self, mock_align, mock_config, mock_pipeline):
        """Test that release calls pipeline.stop()"""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "realsense_camera" in sys.modules:
            del sys.modules["realsense_camera"]
        from realsense_camera import RealsenseCamera

        camera = RealsenseCamera()
        camera.release()

        # Verify stop was called on pipeline
        mock_pipeline_instance.stop.assert_called_once()
