from colorama import Fore, Style


def log_warning(message):
    print(Fore.LIGHTRED_EX + Style.BRIGHT + message + Style.RESET_ALL)
