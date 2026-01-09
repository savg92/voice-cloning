
import re
import os
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class BenchmarkResultsManager:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.content = ""
        if self.file_path.exists():
            self.content = self.file_path.read_text()

    def update_result(self, result, audio_duration: float = 0.0, device: str = "MPS"):
        """
        Update or add a result in the markdown tables.
        """
        if not self.content:
            # If file doesn't exist, create a basic structure
            self.content = "# Benchmark Results\n\n## Complete Benchmark Data\n\n### All Models - Raw Data\n\n"
            self.content += "| Model | Type | Latency (ms) | RTF | Memory (MB) | Audio Duration (s) | Speed Multiplier | Device |\n"
            self.content += "|---|---|---|---|---|---|---|---|\n"

        # Update "All Models - Raw Data" table
        self.content = self._update_table_row(
            table_title="### All Models - Raw Data",
            model_name=result.model,
            new_row_data=[
                f"**{result.model}**",
                result.type,
                f"{result.latency_ms:.2f}",
                f"{result.rtf:.4f}",
                f"{result.memory_mb:.2f}",
                f"{audio_duration:.2f}",
                f"{1/result.rtf:.1f}×" if result.rtf > 0 else "0×",
                device
            ]
        )

        # Update TTS specific table if type is TTS
        if result.type == "TTS":
            if "## Text-to-Speech (TTS) Results" not in self.content:
                self.content += "\n## Text-to-Speech (TTS) Results\n\n"
                self.content += "| Model | Latency | RTF | Memory | Speed Multiplier | Notes |\n"
                self.content += "|---|---|---|---|---|---|\n"
            
            self.content = self._update_table_row(
                table_title="## Text-to-Speech (TTS) Results",
                model_name=result.model,
                new_row_data=[
                    f"**{result.model}**",
                    f"{result.latency_ms:.2f}ms",
                    f"{result.rtf:.3f}",
                    f"{result.memory_mb:.2f} MB",
                    f"**{1/result.rtf:.1f}× real-time**" if result.rtf > 0 else "0×",
                    result.notes
                ]
            )

        self.file_path.write_text(self.content)

    def _update_table_row(self, table_title: str, model_name: str, new_row_data: list[str]):
        """
        Find a table by title, then update or add a row for the model.
        """
        lines = self.content.split("\n")
        table_start_idx = -1
        for i, line in enumerate(lines):
            if table_title in line:
                table_start_idx = i
                break
        
        if table_start_idx == -1:
            # Table not found, append it at the end
            self.content += f"\n{table_title}\n\n"
            # We need the header here, but for simplicity let's assume it exists or we skip
            return self.content

        # Find table boundaries
        header_idx = -1
        separator_idx = -1
        row_start_idx = -1
        for i in range(table_start_idx + 1, len(lines)):
            if "|" in lines[i]:
                if header_idx == -1:
                    header_idx = i
                elif separator_idx == -1:
                    separator_idx = i
                    row_start_idx = i + 1
            elif header_idx != -1:
                # Table ended
                table_end_idx = i
                break
        else:
            table_end_idx = len(lines)

        # Find existing row for model
        model_row_idx = -1
        for i in range(row_start_idx, table_end_idx):
            if f"**{model_name}**" in lines[i] or f"| {model_name} |" in lines[i]:
                model_row_idx = i
                break

        new_row_str = "| " + " | ".join(new_row_data) + " |"

        if model_row_idx != -1:
            lines[model_row_idx] = new_row_str
        else:
            # Add new row at the end of the table
            lines.insert(table_end_idx, new_row_str)

        return "\n".join(lines)
