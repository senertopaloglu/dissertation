import re

class StdoutCapture:
    def __init__(self, original_stdout, progress_queue):
        self.original_stdout = original_stdout
        self.progress_queue = progress_queue
        self.buffer = ""
    
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        self.buffer += text
        if "\n" in text:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                self.process_line(line)
            self.buffer = lines[-1]
    
    def process_line(self, line):
        # look for "frame loading" messages
        m = re.search(r"frame loading \(JPEG\):\s*(\d+)%", line)
        if m:
            percent = int(m.group(1))
            self.progress_queue.put(("Loading frames", percent))
        # look for "propagate in video" messages
        m2 = re.search(r"propagate in video:\s*(\d+)%", line)
        if m2:
            percent = int(m2.group(1))
            self.progress_queue.put(("Propagating segmentation", percent))
        # use successfully installed dependencies as a signal that container is ready
        if "Successfully installed" in line:
            self.progress_queue.put(("Preparing model", 100))
    
    def flush(self):
        self.original_stdout.flush()
