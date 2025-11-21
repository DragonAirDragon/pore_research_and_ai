import sys
import os
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QPushButton, QFileDialog, 
                             QLabel, QMessageBox, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QColor

# Import local modules
# Ensure the current directory is in path if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from canvas import ImageCanvas
from utils import load_image, save_results
from styles import DARK_THEME

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pore Annotator Pro")
        self.resize(1200, 800)
        
        # State
        self.current_dir = None
        self.image_files = []
        self.current_image_index = -1
        self.output_dir = None
        
        # UI Setup
        self.setup_ui()
        self.apply_styles()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)
        
        # Buttons
        self.btn_open = QPushButton("Open Folder")
        self.btn_open.clicked.connect(self.open_folder)
        sidebar_layout.addWidget(self.btn_open)
        
        self.lbl_file_list = QLabel("Files:")
        sidebar_layout.addWidget(self.lbl_file_list)
        
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.change_image)
        sidebar_layout.addWidget(self.file_list)
        
        self.btn_save = QPushButton("Save (S)")
        self.btn_save.clicked.connect(self.save_current)
        sidebar_layout.addWidget(self.btn_save)
        
        self.btn_undo = QPushButton("Undo (Z)")
        self.btn_undo.clicked.connect(self.undo_annotation)
        sidebar_layout.addWidget(self.btn_undo)
        
        self.btn_pipette = QPushButton("Pipette (P)")
        self.btn_pipette.setCheckable(True)
        self.btn_pipette.clicked.connect(self.toggle_pipette)
        sidebar_layout.addWidget(self.btn_pipette)
        
        self.btn_reset = QPushButton("Reset View (R)")
        self.btn_reset.clicked.connect(self.reset_view)
        sidebar_layout.addWidget(self.btn_reset)
        
        # Instructions
        instructions = QLabel(
            "Controls:\n"
            "• Left Drag: Draw Pore\n"
            "• Middle Drag: Pan\n"
            "• Wheel: Zoom\n"
            "• S: Save\n"
            "• Z: Undo\n"
            "• A/D: Prev/Next Image"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 12px;")
        sidebar_layout.addWidget(instructions)
        
        # Canvas Area
        self.canvas = ImageCanvas()
        self.canvas.pipettePicked.connect(self.on_pipette_picked)
        
        self.preview_canvas = ImageCanvas()
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.preview_canvas)
        splitter.setStretchFactor(1, 1) # Main Canvas
        splitter.setStretchFactor(2, 1) # Preview Canvas
        
        main_layout.addWidget(splitter)
        
        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def apply_styles(self):
        self.setStyleSheet(DARK_THEME)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if folder:
            self.current_dir = Path(folder)
            self.output_dir = self.current_dir / "dataset_manual"
            self.output_dir.mkdir(exist_ok=True)
            
            # Load images
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
            self.image_files = sorted([
                f for f in self.current_dir.iterdir() 
                if f.suffix.lower() in extensions and "mask" not in f.name and "dist" not in f.name
            ])
            
            self.file_list.clear()
            for f in self.image_files:
                self.file_list.addItem(f.name)
                
            if self.image_files:
                self.file_list.setCurrentRow(0)
                
            self.status_bar.showMessage(f"Loaded {len(self.image_files)} images from {self.current_dir}")

    def change_image(self, index):
        if index < 0 or index >= len(self.image_files):
            return
            
        self.current_image_index = index
        file_path = self.image_files[index]
        
        try:
            img = load_image(str(file_path))
            self.canvas.set_image(img)
            self.preview_canvas.set_image(img)
            self.status_bar.showMessage(f"Opened {file_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")

    def save_current(self):
        if self.current_image_index < 0:
            return
            
        mask = self.canvas.get_mask()
        if mask is None:
            return
            
        # Get original image from canvas (it's stored as QImage, but we have the file path)
        # We reload from disk to ensure quality and avoid QImage format issues.
        
        file_path = self.image_files[self.current_image_index]
        original_img = load_image(str(file_path))
        
        try:
            save_path = save_results(original_img, mask, self.output_dir, file_path.stem)
            self.status_bar.showMessage(f"Saved to {save_path}")
            
            # Mark in list (visual feedback)
            item = self.file_list.item(self.current_image_index)
            item.setForeground(QColor("#00ff00")) # Green text
            item.setText(f"✓ {file_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")

    def undo_annotation(self):
        self.canvas.undo()
        self.status_bar.showMessage("Undo last pore")

    def toggle_pipette(self):
        active = self.btn_pipette.isChecked()
        self.canvas.set_pipette_mode(active)
        if active:
            self.status_bar.showMessage("Pipette Active: Click on MAIN image to set threshold on PREVIEW")
        else:
            self.status_bar.showMessage("Pipette Deactivated")

    def on_pipette_picked(self, threshold):
        self.preview_canvas.apply_threshold_view(threshold)
        self.btn_pipette.setChecked(False)
        self.status_bar.showMessage(f"Threshold applied: {threshold}")

    def reset_view(self):
        self.canvas.reset_view()
        self.preview_canvas.reset_view()
        self.btn_pipette.setChecked(False)
        self.canvas.set_pipette_mode(False)
        self.status_bar.showMessage("View Reset")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.save_current()
        elif event.key() == Qt.Key.Key_Z:
            self.undo_annotation()
        elif event.key() == Qt.Key.Key_P:
            self.btn_pipette.click()
        elif event.key() == Qt.Key.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key.Key_A:
            # Prev
            new_row = max(0, self.file_list.currentRow() - 1)
            self.file_list.setCurrentRow(new_row)
        elif event.key() == Qt.Key.Key_D:
            # Next
            new_row = min(self.file_list.count() - 1, self.file_list.currentRow() + 1)
            self.file_list.setCurrentRow(new_row)
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
