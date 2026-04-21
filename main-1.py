import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ui.app_window import PeacockAgentApp
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("PeacockAgent")
    app.setApplicationVersion("1.0.0")
    window = PeacockAgentApp()
    window.show()
    sys.exit(app.exec())
