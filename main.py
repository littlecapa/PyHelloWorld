import logging
from hw_trainer import HW_Trainer
import torch

def setup_logging():
    logging.basicConfig(
        filename='app.log',  # Change this to your desired log file path
        level=logging.DEBUG,  # Change the log level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Torch Device: {device}")
    
def main():
    torch.backends.cudnn.enabled = True

    setup_logging()
    logging.debug('Starting the program')
    check_env()

    trainer = HW_Trainer()
    for _ in range(2):
        trainer.do_training() 
    logging.debug('Program execution completed')

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
