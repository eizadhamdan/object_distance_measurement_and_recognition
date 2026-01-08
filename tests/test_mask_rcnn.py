import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open

# ------------------------------------------------------------------
# 🔴 CRITICAL FIX: Mock cv2 BEFORE importing mask_rcnn
# ------------------------------------------------------------------
mock_cv2 = MagicMock()
mock_cv2.dnn = MagicMock()
sys.modules["cv2"] = mock_cv2
sys.modules["cv2.dnn"] = mock_cv2.dnn
# ------------------------------------------------------------------


class TestMaskRCNNStructure:
    """Test MaskRCNN class structure without requiring device"""

    def test_mask_rcnn_file_exists(self):
        """Test that mask_rcnn.py file exists"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        assert os.path.exists(src_path)

    def test_mask_rcnn_class_defined(self):
        """Test that MaskRCNN class is defined in source"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        with open(src_path, "r") as f:
            source = f.read()
        assert "class MaskRCNN" in source

    def test_mask_rcnn_init_method_defined(self):
        """Test that __init__ method is defined"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        with open(src_path, "r") as f:
            source = f.read()
        assert "def __init__" in source

    def test_mask_rcnn_detect_objects_method_defined(self):
        """Test that detect_objects_mask method is defined"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        with open(src_path, "r") as f:
            source = f.read()
        assert "def detect_objects_mask" in source

    def test_mask_rcnn_draw_mask_method_defined(self):
        """Test that draw_object_mask method is defined"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        with open(src_path, "r") as f:
            source = f.read()
        assert "def draw_object_mask" in source

    def test_mask_rcnn_draw_info_method_defined(self):
        """Test that draw_object_info method is defined"""
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "mask_rcnn.py")
        with open(src_path, "r") as f:
            source = f.read()
        assert "def draw_object_info" in source


class TestMaskRCNNInit:
    """Test MaskRCNN initialization"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_init_creates_instance(self, mock_file, mock_read_net):
        """Test that MaskRCNN can be instantiated"""
        mock_read_net.return_value = MagicMock()

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from mask_rcnn import MaskRCNN

        mrcnn = MaskRCNN()
        assert mrcnn is not None

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_init_has_required_attributes(self, mock_file, mock_read_net):
        """Test that __init__ creates required attributes"""
        mock_read_net.return_value = MagicMock()

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        if "mask_rcnn" in sys.modules:
            del sys.modules["mask_rcnn"]

        from mask_rcnn import MaskRCNN

        mrcnn = MaskRCNN()
        assert hasattr(mrcnn, "net")
        assert hasattr(mrcnn, "classes")
        assert hasattr(mrcnn, "colors")
        assert hasattr(mrcnn, "detection_threshold")
        assert hasattr(mrcnn, "mask_threshold")
        assert hasattr(mrcnn, "obj_boxes")
        assert hasattr(mrcnn, "obj_classes")
        assert hasattr(mrcnn, "obj_centers")
        assert hasattr(mrcnn, "obj_contours")


class TestMaskRCNNMethods:
    """Test MaskRCNN methods exist and can be called"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_detect_objects_mask_exists_and_callable(self, mock_file, mock_read_net):
        """Test that detect_objects_mask method exists"""
        mock_read_net.return_value = MagicMock()

        from mask_rcnn import MaskRCNN

        mrcnn = MaskRCNN()

        assert callable(mrcnn.detect_objects_mask)

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_draw_object_mask_exists_and_callable(self, mock_file, mock_read_net):
        """Test that draw_object_mask method exists"""
        mock_read_net.return_value = MagicMock()

        from mask_rcnn import MaskRCNN

        mrcnn = MaskRCNN()

        assert callable(mrcnn.draw_object_mask)

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_draw_object_info_exists_and_callable(self, mock_file, mock_read_net):
        """Test that draw_object_info method exists"""
        mock_read_net.return_value = MagicMock()

        from mask_rcnn import MaskRCNN

        mrcnn = MaskRCNN()

        assert callable(mrcnn.draw_object_info)
