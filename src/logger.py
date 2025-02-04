import logging
import os

from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
from typing import Dict, Any, Optional, List

class CustomLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.console = Console()

        self.shell_handler = RichHandler(console=self.console)
        file_loc = "src/data/logs"
        os.makedirs(file_loc, exist_ok=True)
        self.file_handler = logging.FileHandler(file_loc + "/debug.log")

        self.setLevel(logging.DEBUG)
        self.shell_handler.setLevel(logging.DEBUG)
        self.file_handler.setLevel(logging.DEBUG)

        shell_formatter = logging.Formatter('%(message)s')
        file_formatter = logging.Formatter(
            '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
        )

        self.shell_handler.setFormatter(shell_formatter)
        self.file_handler.setFormatter(file_formatter)

        self.addHandler(self.shell_handler)
        self.addHandler(self.file_handler)

    def title(self, title: str, level: int = logging.INFO) -> None:
        """
        Log a title with a given level.
        """
        self.log(level, self.console.rule(f"[bold]{title}"))

    def table(self,
            data: Dict[str, Any],
            title: Optional[str] = None,
            columns: Optional[List[str]] = None,
            level: int = logging.INFO
        ) -> None:
        """
        Log a dictionary as a formatted table.
        """
        table = Table(title=title, show_header=True)

        col_names = columns if columns else ["Key", "Value"]
        for col in col_names:
            table.add_column(col)

        if columns:
            table.add_row(*[str(data.get(col, "")) for col in columns])
        else:
            for key, value in data.items():
                table.add_row(str(key), str(value))

        self.log(level, self.console.print(table, overflow="ignore"))


logging.setLoggerClass(CustomLogger)

logging.getLogger('matplotlib.font_manager').disabled = True
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
