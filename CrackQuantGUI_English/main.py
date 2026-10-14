"""CrackQuant - coal CT crack recognition and quantification v1.0."""
import sys, os
os.environ["QT_QPA_PLATFORM"] = "windows:darkmode=0"
os.environ["MPLBACKEND"] = "Agg"

# Fix DLL search paths for PyInstaller bundled app
if getattr(sys, 'frozen', False):
    # Tcl/Tk
    tcl = os.path.join(sys._MEIPASS, 'tcl8.6')
    if os.path.exists(tcl): os.environ['TCL_LIBRARY'] = tcl
    tk  = os.path.join(sys._MEIPASS, 'tk8.6')
    if os.path.exists(tk):  os.environ['TK_LIBRARY']  = tk
    # CUDA DLLs (torch runtime needs these on PATH)
    torch_lib = os.path.join(sys._MEIPASS, 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')

from PyQt5.QtWidgets import QApplication
from gui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec_())
