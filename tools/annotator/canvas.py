from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QWheelEvent, QMouseEvent
import numpy as np

class ImageCanvas(QWidget):
    # Signal emitted when annotations change (to update save status etc)
    annotationsChanged = pyqtSignal()
    # Signal emitted when pipette picks a value
    pipettePicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None # QImage (Original)
        self.pixmap = None # QPixmap (Displayed)
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        
        self.pores = [] # List of {'center': (x, y), 'radius': r}
        
        # Interaction state
        self.last_mouse_pos = QPoint()
        self.panning = False
        self.drawing = False
        self.space_pressed = False # Track Space key
        self.current_center = None
        self.current_radius = 0
        
        # Pipette / Threshold state
        self.pipette_active = False
        self.threshold_val = None
        self.original_pixmap = None
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Colors
        self.pore_color = QColor(0, 255, 0, 100) # Green transparent
        self.pore_border = QColor(0, 255, 0)
        self.active_color = QColor(0, 0, 255, 100) # Blue transparent

    def set_image(self, image_array):
        """Sets the image from a numpy array (Grayscale 2D)."""
        # Keep reference to data to prevent GC
        self._data = image_array
        height, width = image_array.shape
        bytes_per_line = width
        self.image = QImage(self._data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        # Force a copy for the pixmap to be safe
        self.pixmap = QPixmap.fromImage(self.image)
        self.original_pixmap = self.pixmap
        
        # Reset view
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.pores = []
        self.threshold_val = None
        self.update()

    def set_pipette_mode(self, active):
        self.pipette_active = active
        if active:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def reset_view(self):
        """Resets threshold view to original grayscale."""
        if self.original_pixmap:
            self.pixmap = self.original_pixmap
            self.threshold_val = None
            self.update()

    def apply_threshold_view(self, threshold):
        """
        Updates display: pixels <= threshold become BLACK, others WHITE.
        This is purely visual.
        """
        if self.image is None:
            return
            
        # Convert QImage to numpy for fast processing
        # Handle potential padding in QImage (bytesPerLine >= width)
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        
        # Reshape to (height, bytesPerLine) then crop to width
        arr_padded = np.array(ptr).reshape(self.image.height(), self.image.bytesPerLine())
        arr = arr_padded[:, :self.image.width()]
        
        # Create binary view
        # Pores (darker than threshold) -> 0 (Black)
        # Background (lighter) -> 255 (White)
        binary_view = np.where(arr <= threshold, 0, 255).astype(np.uint8)
        
        height, width = binary_view.shape
        # We need to keep binary_view alive while QImage uses it, or make QImage copy it.
        # QImage(data, ...) doesn't copy.
        # But QPixmap.fromImage(qimg) DOES copy.
        # So as long as binary_view lives until fromImage returns, we are fine.
        qimg_bin = QImage(binary_view.data, width, height, width, QImage.Format.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg_bin)
        self.threshold_val = threshold
        self.update()

    def get_mask(self):
        """Returns the binary mask as a numpy array."""
        if self.image is None:
            return None
            
        width = self.image.width()
        height = self.image.height()
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw pores on mask
        import cv2
        for pore in self.pores:
            cx, cy = pore['center']
            r = pore['radius']
            cv2.circle(mask, (int(cx), int(cy)), int(r), (255), -1)
            
        return mask

    def wheelEvent(self, event: QWheelEvent):
        if self.image is None:
            return

        # Zoom
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        
        old_scale = self.scale_factor
        self.scale_factor *= factor
        
        # Limit zoom
        self.scale_factor = max(0.1, min(self.scale_factor, 20.0))
        
        # Adjust offset to zoom towards mouse
        if self.scale_factor != old_scale:
            mouse_pos = event.position()
            # Logic to keep mouse pos stable could be added here, 
            # for now simple zoom is fine.
            
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        self.setFocus() # Ensure we have focus for key events
        
        if self.image is None:
            return
            
        # Pipette Logic
        if self.pipette_active and event.button() == Qt.MouseButton.LeftButton:
            img_pos = self.transform_pos(event.pos())
            if self.is_valid_pos(img_pos):
                # Get pixel value
                # QImage.pixel returns ARGB, we need gray value (0-255)
                # Since it's Format_Grayscale8, qRed/Green/Blue are all same
                pixel_val = QColor(self.image.pixel(img_pos)).red()
                
                self.pipettePicked.emit(pixel_val)
                
                # Deactivate pipette after use
                self.set_pipette_mode(False)
                # Emit signal if needed (e.g. to uncheck button in main window)
                # For now main window handles button state manually or we can just let user click again.
            return

        # Pan if Middle Button OR (Left Button AND Space is pressed)
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self.space_pressed):
            self.panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton:
            # Start drawing
            img_pos = self.transform_pos(event.pos())
            if self.is_valid_pos(img_pos):
                self.drawing = True
                self.current_center = img_pos
                self.current_radius = 0

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.panning:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
            
        elif self.drawing:
            img_pos = self.transform_pos(event.pos())
            # Calculate radius
            dx = img_pos.x() - self.current_center.x()
            dy = img_pos.y() - self.current_center.y()
            self.current_radius = (dx**2 + dy**2)**0.5
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self.panning):
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.current_radius > 1: # Minimum size
                self.pores.append({
                    'center': (self.current_center.x(), self.current_center.y()),
                    'radius': self.current_radius
                })
                self.annotationsChanged.emit()
            self.current_center = None
            self.current_radius = 0
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30)) # Dark background
        
        if self.pixmap is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a folder to start annotating")
            return

        painter.save()
        
        # Apply transformations
        painter.translate(self.width() / 2, self.height() / 2)
        painter.translate(self.offset)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.translate(-self.pixmap.width() / 2, -self.pixmap.height() / 2)
        
        # Draw Image
        painter.drawPixmap(0, 0, self.pixmap)
        
        # Draw Pores
        painter.setPen(QPen(self.pore_border, 2 / self.scale_factor))
        painter.setBrush(QBrush(self.pore_color))
        
        for pore in self.pores:
            cx, cy = pore['center']
            r = pore['radius']
            painter.drawEllipse(QPoint(int(cx), int(cy)), int(r), int(r))
            
        # Draw current drawing
        if self.drawing and self.current_center:
            painter.setPen(QPen(self.active_color, 2 / self.scale_factor))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(int(self.current_center.x()), int(self.current_center.y())), int(self.current_radius), int(self.current_radius))

        painter.restore()

    def transform_pos(self, widget_pos):
        """Converts widget coordinates to image coordinates."""
        # Reverse the transformations in paintEvent
        # 1. Translate back to center
        x = widget_pos.x() - self.width() / 2 - self.offset.x()
        y = widget_pos.y() - self.height() / 2 - self.offset.y()
        
        # 2. Scale
        x /= self.scale_factor
        y /= self.scale_factor
        
        # 3. Translate to image origin
        x += self.pixmap.width() / 2
        y += self.pixmap.height() / 2
        
        return QPoint(int(x), int(y))

    def is_valid_pos(self, pos):
        if self.pixmap is None:
            return False
        return 0 <= pos.x() < self.pixmap.width() and 0 <= pos.y() < self.pixmap.height()

    def undo(self):
        if self.pores:
            self.pores.pop()
            self.annotationsChanged.emit()
            self.update()
