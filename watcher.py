import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Tentukan folder yang ingin Anda pantau
PATH_TO_WATCH = "." 

# Perintah untuk membangun Jupyter Book
BUILD_COMMAND = ["jupyter-book", "build", PATH_TO_WATCH]

class JupyterBookBuilder(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return None
        
        # Abaikan file yang dibuat oleh proses build itu sendiri
        if "_build" in event.src_path:
            return None

        # Jalankan perintah build saat ada perubahan file
        print(f"Perubahan terdeteksi pada: {event.src_path}. Memulai build...")
        subprocess.run(BUILD_COMMAND)
        print("Build selesai.")

if __name__ == "__main__":
    event_handler = JupyterBookBuilder()
    observer = Observer()
    observer.schedule(event_handler, PATH_TO_WATCH, recursive=True)
    observer.start()

    print(f"Memulai watcher di direktori: {PATH_TO_WATCH}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()