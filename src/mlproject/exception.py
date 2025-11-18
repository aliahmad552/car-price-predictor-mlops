import sys
from src.mlproject.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line [{line_number}] "
        f"with error message: {str(error)}"
    )
    return error_message  # ✅ VERY IMPORTANT
        

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize actual Exception class
        super().__init__(error_message_detail(error_message, error_detail))
        # Store detailed message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
