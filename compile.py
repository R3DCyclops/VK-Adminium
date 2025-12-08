import subprocess
import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))

script_path = os.path.join(project_dir, 'vkadminium.py')
output_dir = os.path.join(project_dir, 'dist')
icon_path = os.path.join(project_dir, 'ico.ico')

logo_path = os.path.join(project_dir, 'bckg.png')
dog_path = os.path.join(project_dir, 'dog.png')
avatar1_path = os.path.join(project_dir, 'avatar1.png')
pin_path = os.path.join(project_dir, 'pin.png')

# Проверка наличия файлов
required_files = {
    "Скрипт": script_path,
    "Иконка": icon_path,
    "Фон (bckg.png)": logo_path,
    "Изображение (dog.png)": dog_path,
    "Аватар (avatar1.png)": avatar1_path,
    "Пин (pin.png)": pin_path,
}

for name, path in required_files.items():
    if not os.path.exists(path):
        print(f"[Ошибка] {name} не найден: {path}")
        sys.exit(1)

command = [
    'pyinstaller',
    '--onefile',
    '--windowed',
    '--icon', icon_path,
    '--distpath', output_dir,
    '--add-data', f'{logo_path};.',
    '--add-data', f'{dog_path};.',
    '--add-data', f'{icon_path};.',
    '--add-data', f'{avatar1_path};.',
    '--add-data', f'{pin_path};.',
    
    '--hidden-import=PySide6.QtCore',
    '--hidden-import=PySide6.QtGui',
    '--hidden-import=PySide6.QtNetwork',
    '--hidden-import=PySide6.QtSvg',
    '--hidden-import=PySide6.QtWidgets',
    '--hidden-import=vk_api',
    '--hidden-import=requests',
    '--hidden-import=concurrent.futures',
    '--hidden-import=concurrent.futures.thread',
    '--hidden-import=pkg_resources',
    '--hidden-import=random',
    '--hidden-import=datetime',
    '--hidden-import=json',
    '--hidden-import=time',
    '--hidden-import=os',
    '--hidden-import=sys',
    '--hidden-import=threading',
    '--hidden-import=string',
    '--hidden-import=pickle',
    
    '--hidden-import=PIL.Image',
    '--hidden-import=PIL.ImageFilter',
    '--hidden-import=PIL.ImageOps',
    '--hidden-import=PIL.PngImagePlugin',
    '--hidden-import=PIL.JpegImagePlugin',
    
    '--hidden-import=urllib3',
    '--hidden-import=idna',
    '--hidden-import=chardet',
    
    '--hidden-import=imagehash',
    '--hidden-import=numpy.core.multiarray',
    '--hidden-import=numpy.core._asarray',
    '--hidden-import=numpy.core._dtype',
    '--hidden-import=numpy.core._internal',
    '--hidden-import=numpy.fft.helper',
    '--hidden-import=numpy.random.bit_generator',
    '--hidden-import=numpy.random._generator',
    
    '--hidden-import=sklearn.cluster._kmeans',
    '--hidden-import=sklearn.utils._weight_vector',
    '--hidden-import=sklearn.neighbors._partition_nodes',
    '--hidden-import=scipy',
    '--hidden-import=scipy.sparse.csgraph._tools',
    '--hidden-import=scipy.sparse._index',
    '--hidden-import=scipy.spatial.transform.rotation',
    '--hidden-import=scipy.signal',
    '--hidden-import=scipy.linalg.cython_lapack',
    
    '--hidden-import=selenium',
    '--hidden-import=selenium.webdriver',
    '--hidden-import=selenium.webdriver.common.by',
    '--hidden-import=bs4',
    '--hidden-import=chromedriver_autoinstaller',
    
    '--hidden-import=mistralai',
    '--hidden-import=httpx',
    '--hidden-import=httpcore',
    '--hidden-import=h11',
    '--hidden-import=httpcore._async.connection',
    '--hidden-import=httpcore._sync.connection',
    '--hidden-import=httpx._models',
    '--hidden-import=httpx._transports.default',
    '--hidden-import=httpx._api',
    '--hidden-import=anyio',
    '--hidden-import=anyio._backends._asyncio',
    '--hidden-import=sniffio',
    '--hidden-import=certifi',
    '--hidden-import=ssl',
    
    '--collect-all=mistralai',
    '--collect-all=vk_api',
    '--collect-all=requests',
    '--collect-all=PIL',
    '--collect-all=chromedriver_autoinstaller',
    
    script_path
]

print("Начинаю компиляцию...")
try:
    subprocess.run(command, check=True)
    print(f"\nГотово. файл находится в: {output_dir}")
except subprocess.CalledProcessError as e:
    print(f"\nОшибка компиляции: {e}")
    sys.exit(1)