import subprocess
import sys

PACKAGES = [
    "vk_api",
    "requests",
    "Pillow",
    "numpy",
    "scikit-learn",
    "imagehash",
    "chromedriver-autoinstaller",
    "selenium",
    "beautifulsoup4",
    "PySide6",
    "mistralai",
    "cryptography"
]

def install_packages(packages):
    print("Устанавливаю зависимости...")
    for package in packages:
        print(f"Устанавливаю: {package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при установке {package}: {e}")
            sys.exit(1)
    print("\nВсе зависимости успешно установлены!")

if __name__ == "__main__":
    install_packages(PACKAGES)