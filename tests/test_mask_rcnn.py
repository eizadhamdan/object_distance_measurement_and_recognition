import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mask_rcnn import MaskRCNN


class TestMaskRCNNInit:
    """Test MaskRCNN initialization"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_init_loads_network(self, mock_open, mock_read_net):
        """Test that __init__ loads the Mask RCNN network"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        mrcnn = MaskRCNN()

        assert mrcnn.net is not None
        mock_read_net.assert_called_once()

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_init_sets_backend_to_cuda(self, mock_open, mock_read_net):
        """Test that __init__ sets the backend to CUDA"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        mrcnn = MaskRCNN()

        mock_net.setPreferableBackend.assert_called_once()

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_init_loads_classes(self, mock_open, mock_read_net):
        """Test that __init__ loads class names"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net

        mock_file_content = ["person\n", "dog\n", "cat\n"]
        mock_open.return_value.__enter__.return_value = mock_file_content

        mrcnn = MaskRCNN()

        assert len(mrcnn.classes) == 3
        assert "person" in mrcnn.classes

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_init_initializes_parameters(self, mock_open, mock_read_net):
        """Test that __init__ initializes all required parameters"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        mrcnn = MaskRCNN()

        assert mrcnn.detection_threshold == 0.7
        assert mrcnn.mask_threshold == 0.3
        assert isinstance(mrcnn.colors, np.ndarray)
        assert len(mrcnn.obj_boxes) == 0


class TestMaskRCNNDetectObjects:
    """Test detect_objects_mask method"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_detect_objects_returns_four_outputs(self, mock_open, mock_read_net):
        """Test that detect_objects_mask returns boxes, classes, contours, centers"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = ["person\n", "dog\n"]

        # Create a test frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Setup forward pass
        boxes = np.array([[[[0, 0, 0.9, 0.1, 0.1, 0.5, 0.5]]]])
        masks = np.zeros((1, 90, 1, 1))
        mock_net.forward.return_value = [boxes, masks]

        mrcnn = MaskRCNN()
        result = mrcnn.detect_objects_mask(frame)

        assert len(result) == 4
        assert isinstance(result[0], list)  # boxes
        assert isinstance(result[1], list)  # classes
        assert isinstance(result[2], list)  # contours
        assert isinstance(result[3], list)  # centers

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_detect_objects_filters_by_threshold(self, mock_open, mock_read_net):
        """Test that detect_objects_mask filters detections by threshold"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = ["person\n"]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create detection with low confidence (below threshold)
        boxes = np.array([[[[0, 0, 0.5, 0.1, 0.1, 0.5, 0.5]]]])  # score = 0.5 < 0.7
        masks = np.zeros((1, 90, 1, 1))
        mock_net.forward.return_value = [boxes, masks]

        mrcnn = MaskRCNN()
        boxes_out, classes_out, contours_out, centers_out = mrcnn.detect_objects_mask(
            frame
        )

        # Should filter out the low confidence detection
        assert len(boxes_out) == 0

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("mask_rcnn.cv2.dnn.blobFromImage")
    @patch("builtins.open", create=True)
    def test_detect_objects_calls_blob_from_image(
        self, mock_open, mock_blob, mock_read_net
    ):
        """Test that detect_objects_mask creates blob from image"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_blob_result = np.zeros((1, 3, 416, 416))
        mock_blob.return_value = mock_blob_result

        boxes = np.array([[]])
        masks = np.array([])
        mock_net.forward.return_value = [boxes, masks]

        mrcnn = MaskRCNN()
        mrcnn.detect_objects_mask(frame)

        mock_blob.assert_called_once()


class TestMaskRCNNDrawObjectMask:
    """Test draw_object_mask method"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_draw_object_mask_returns_frame(self, mock_open, mock_read_net):
        """Test that draw_object_mask returns the modified frame"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        mrcnn = MaskRCNN()
        result = mrcnn.draw_object_mask(frame)

        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("mask_rcnn.cv2.drawContours")
    @patch("mask_rcnn.cv2.fillPoly")
    @patch("mask_rcnn.cv2.addWeighted")
    @patch("builtins.open", create=True)
    def test_draw_object_mask_processes_objects(
        self,
        mock_open,
        mock_add_weighted,
        mock_fill_poly,
        mock_draw_contours,
        mock_read_net,
    ):
        """Test that draw_object_mask processes detected objects"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        mrcnn = MaskRCNN()
        # Manually set object data
        mrcnn.obj_boxes = [[100, 100, 200, 200]]
        mrcnn.obj_classes = [0]
        mrcnn.obj_contours = [
            [np.array([[110, 110], [190, 110], [190, 190], [110, 190]])]
        ]

        mock_add_weighted.return_value = frame.copy()

        result = mrcnn.draw_object_mask(frame)

        assert isinstance(result, np.ndarray)


class TestMaskRCNNDrawObjectInfo:
    """Test draw_object_info method"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("mask_rcnn.cv2.line")
    @patch("mask_rcnn.cv2.rectangle")
    @patch("mask_rcnn.cv2.putText")
    @patch("builtins.open", create=True)
    def test_draw_object_info_returns_frame(
        self, mock_open, mock_put_text, mock_rectangle, mock_line, mock_read_net
    ):
        """Test that draw_object_info returns the modified frame"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = ["person\n"]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        depth_frame = np.zeros((720, 1280), dtype=np.uint16)

        mrcnn = MaskRCNN()
        # Manually set object data
        mrcnn.obj_boxes = [[100, 100, 200, 200]]
        mrcnn.obj_classes = [0]
        mrcnn.obj_centers = [(150, 150)]

        result = mrcnn.draw_object_info(frame, depth_frame)

        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("mask_rcnn.cv2.line")
    @patch("mask_rcnn.cv2.rectangle")
    @patch("mask_rcnn.cv2.putText")
    @patch("builtins.open", create=True)
    def test_draw_object_info_displays_depth(
        self, mock_open, mock_put_text, mock_rectangle, mock_line, mock_read_net
    ):
        """Test that draw_object_info displays depth information"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = ["person\n"]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        depth_frame = np.full((720, 1280), 500, dtype=np.uint16)

        mrcnn = MaskRCNN()
        mrcnn.obj_boxes = [[100, 100, 200, 200]]
        mrcnn.obj_classes = [0]
        mrcnn.obj_centers = [(150, 150)]

        mrcnn.draw_object_info(frame, depth_frame)

        # Verify putText was called with depth information
        mock_put_text.assert_called()

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("mask_rcnn.cv2.line")
    @patch("mask_rcnn.cv2.rectangle")
    @patch("mask_rcnn.cv2.putText")
    @patch("builtins.open", create=True)
    def test_draw_object_info_empty_objects(
        self, mock_open, mock_put_text, mock_rectangle, mock_line, mock_read_net
    ):
        """Test that draw_object_info handles empty object lists"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        depth_frame = np.zeros((720, 1280), dtype=np.uint16)

        mrcnn = MaskRCNN()
        result = mrcnn.draw_object_info(frame, depth_frame)

        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape


class TestMaskRCNNAttributes:
    """Test MaskRCNN attributes and state"""

    @patch("mask_rcnn.cv2.dnn.readNetFromTensorflow")
    @patch("builtins.open", create=True)
    def test_init_creates_empty_object_lists(self, mock_open, mock_read_net):
        """Test that __init__ creates empty lists for objects"""
        mock_net = MagicMock()
        mock_read_net.return_value = mock_net
        mock_open.return_value.__enter__.return_value = []

        mrcnn = MaskRCNN()

        assert mrcnn.obj_boxes == []
        assert mrcnn.obj_classes == []
        assert mrcnn.obj_centers == []
        assert mrcnn.obj_contours == []
