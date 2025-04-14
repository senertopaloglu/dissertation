from queue import Queue
import re
from typing import TextIO, Tuple


ProgressState = Tuple[str, int] # (state name, percent)
ProgressQueue = Queue[ProgressState]

class StdoutCapture:
    """
    This class wraps an original stdout stream to intercept and process output lines.
    It looks for specific patterns in the output (such as frame loading and propagation messages)
    and sends corresponding progress updates to a provided progress queue.
    """
    def __init__(self, original_stdout: TextIO, progress_queue: ProgressQueue):
        self.original_stdout = original_stdout
        self.progress_queue = progress_queue
        self.buffer = ""
    
    def write(self, text: str) -> None:
        """
        Write text to the original stdout and process complete lines for progress updates.

        Args:
            text (str): The text to write.
        
        Returns:
            None
        """
        self.original_stdout.write(text)
        self.original_stdout.flush()
        self.buffer += text
        if "\n" in text:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                self.process_line(line)
            self.buffer = lines[-1]
    
    def process_line(self, line: str) -> None:
        """
        Process a single line of text to extract progress information.

        Args:
            line (str): The line of text to process.
        
        Returns:
            None
        """
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
    
    def flush(self) -> None:
        """
        Flush the original stdout stream.

        Returns:
            None
        """
        self.original_stdout.flush()
