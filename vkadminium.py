import sys
import os
import time
from datetime import datetime
import random
import string
import pickle
import requests
import shutil
from hashlib import md5
from queue import Queue, Empty
from pathlib import Path

import vk_api

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Thread

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import imagehash

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from time import sleep

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QSplitter, QDateTimeEdit,
    QCheckBox, QFileDialog, QSlider, QGridLayout, QRadioButton, QListWidget
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return kdf.derive(password.encode())

def encrypt_data(data: str, password: str) -> str:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
    return base64.b64encode(salt + iv + ciphertext).decode()

def decrypt_data(encrypted_data: str, password: str) -> str:
    data = base64.b64decode(encrypted_data.encode())
    salt, iv, ciphertext = data[:16], data[16:32], data[32:]
    key = derive_key(password, salt)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext.decode()
    
def get_normalized_image_hash(filepath):
    #Возвращает MD5 от нормализованного изображения 128x128, для поиска точных визуальных дубликатов
    try:
        img = Image.open(filepath)
        img = img.resize((128, 128))
        img = img.convert("RGB")
        data = img.tobytes()
        return md5(data).hexdigest()
    except Exception:
        return None

CONFIG_PATH = os.path.join(os.path.dirname(sys.argv[0]), "last_settings.cfg")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            encrypted_content = f.read().strip()

        CONFIG_SECRET = "14VKadminium88"  #Рекомендую заменить на что-то своё
        content = decrypt_data(encrypted_content, CONFIG_SECRET)

        config = {
            "token": "",
            "group_id": "",
            "last_post_time": None,
            "mistral_api_key": "",
            "ai_prompt": ""
        }

        lines = content.split("\n")
        if len(lines) > 0:
            config["token"] = lines[0]
        if len(lines) > 1:
            config["group_id"] = lines[1]
        if len(lines) > 2 and lines[2].strip():
            try:
                config["last_post_time"] = int(lines[2])
            except (ValueError, TypeError):
                pass
        if len(lines) > 3:
            config["mistral_api_key"] = lines[3]

        start_marker = "!n!"
        end_marker = "!n!"
        start_idx = content.find(start_marker)
        if start_idx != -1:
            end_idx = content.rfind(end_marker)
            if end_idx != -1 and end_idx > start_idx + len(start_marker):
                prompt_raw = content[start_idx + len(start_marker):end_idx]
                config["ai_prompt"] = prompt_raw

        return config
    except Exception as e:
        print(f"🧰[ERROR] Не удалось загрузить конфиг: {e}")
        return {}

def save_config(token="", group_id="", mistral_api_key="", last_post_time=None, ai_prompt=""):
    try:
        if last_post_time is None:
            current = load_config()
            last_post_time = current.get("last_post_time")
            if last_post_time is None:
                last_post_time = int(time.time())
        config_str = "\n".join([
            token,
            group_id,
            str(last_post_time),
            mistral_api_key,
        ]) + "\n"
        if ai_prompt:
            config_str += "!n!" + ai_prompt + "!n!\n"

        CONFIG_SECRET = "14VKadminium88"  #Рекомендую заменить на что-то своё
        encrypted = encrypt_data(config_str, CONFIG_SECRET)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(encrypted)
    except Exception as e:
        print(f"🧰[ERROR] Не удалось сохранить конфиг: {e}")

class DuplicateWorker(QThread):
    log_signal = Signal(str)
    result_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.total_files_full = 0
        self.processed = 0

    def run(self):
        exact_hashes = {}  #MD5 > путь
        phash_dict = {}    #phash > список путей

        exact_duplicates = []
        soft_duplicates = []

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

        self.total_files = 0
        self.processed = 0
        report_interval = 100

        for root, _, files in os.walk(self.folder):
            self.total_files += len([f for f in files if f.lower().endswith(image_extensions)])
        self.total_files_full = self.total_files * 2

        self.log_signal.emit(f"🔎[DEBUG] Начинаю поиск дубликатов в папке: {self.folder}")
        self.log_signal.emit(f"😱😱😱[FOUND] Обнаружено изображений: {self.total_files}")

        #Точный поиск
        self.log_signal.emit("[1/2] Поиск точных дубликатов...")
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    path = os.path.join(root, file)
                    img_hash = get_normalized_image_hash(path)
                    if img_hash:
                        if img_hash in exact_hashes:
                            exact_duplicates.append(path)
                            if exact_hashes[img_hash] not in exact_duplicates:
                                exact_duplicates.append(exact_hashes[img_hash])
                                exact_hashes[img_hash] = None
                        else:
                            exact_hashes[img_hash] = path
                    self.processed += 1
                    if self.processed % report_interval == 0:
                        self.send_progress()

        # Мягкий поиск
        self.log_signal.emit("[2/2] Поиск похожих изображений...")
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    path = os.path.join(root, file)
                    ph = self.get_phash(path)
                    if ph:
                        found = False
                        for existing_ph in phash_dict:
                            if ph - existing_ph < 10:
                                phash_dict[existing_ph].append(path)
                                found = True
                                break
                        if not found:
                            phash_dict[ph] = [path]
                    self.processed += 1
                    if self.processed % report_interval == 0:
                        self.send_progress()

        self.send_progress(final=True)

        soft_duplicates = []
        for key in phash_dict:
            for path in phash_dict[key]:
                if len(phash_dict[key]) > 1 and path not in exact_duplicates:
                    soft_duplicates.append(path)
        soft_duplicates = list(set(soft_duplicates))

        self.log_signal.emit("👍[SUCCESS] Поиск завершён.")
        self.log_signal.emit(f"🔎[DEBUG] Точных дубликатов: {len(exact_duplicates)}")
        self.log_signal.emit(f"🔎[DEBUG] Потенциальных дубликатов: {len(soft_duplicates)}")

        result = {
            "exact": list(set(exact_duplicates)),
            "soft": soft_duplicates
        }

        self.result_signal.emit(result)
        self.finished_signal.emit()

    def send_progress(self, final=False):
        percent = (self.processed / self.total_files_full) * 100
        message = f"💾[DEBUG] Обработано: {self.processed} / {self.total_files_full} — {percent:.1f}%"
        if final:
            message = f"💌[DONE] Итог: {self.processed} / {self.total_files_full} — 100%"
        self.log_signal.emit(message)

    def get_phash(self, filepath):
        #Perceptual hash изображения для мягкого сравнения
        try:
            return imagehash.phash(Image.open(filepath))
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Ошибка при получении phash для {filepath}: {e}")
            return None 

class PosterWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    update_last_post_time = Signal(int)
    def __init__(self, token, group_id, interval_hours, folder_path, start_timestamp,
                 photos_per_post, caption="", use_random_emoji=False, random_photos=False, emoji_list=None, use_carousel=False, cluster_mode=False,
                 use_ai_caption=False, mistral_api_key="", ai_custom_prompt=""):
        super().__init__()
        self.token = token
        self.group_id = group_id
        self.interval_hours = interval_hours
        self.folder_path = folder_path
        self.start_timestamp = start_timestamp
        self.photos_per_post = photos_per_post
        self.posts_saved = 0
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.caption = caption
        self.use_random_emoji = use_random_emoji
        self.random_photos = random_photos
        self.emoji_list = emoji_list or []
        self.use_carousel = use_carousel
        self.cluster_mode = cluster_mode
        self.use_ai_caption = use_ai_caption
        self.mistral_api_key = mistral_api_key
        self.last_quotes = []
        self.ai_custom_prompt = ai_custom_prompt or (
            "Мне нужны цитаты в стиле подростковых депрессивных пабликов ВКонтакте — короткие, острые, меланхоличные, "
            "сырого и немного наивного отчаяния. Они должны звучать как обрывки мыслей, вырванные из контекста, будто "
            "кто-то записал их в блокноте или на полях учебника.\n\n"
            "Ключевые требования:\n"
            "1. Эмоциональная насыщенность — боль, пустота, одиночество — но без диагнозов.\n"
            "2. Краткость — 1–2 предложения, максимум 3.\n"
            "3. Стиль: обрывки монолога, без логики между ними.\n"
            "4. Подростковая наивность: крик души, а не философия.\n"
            "5. Никаких имён, мест, событий. Только чувства и абстракции.\n"
            "6. Подходит под ч/б фото: дождь, улицы, силуэты, пустые комнаты."
        )
    
    def toggle_pause(self):
        with self.pause_cond:
            self.paused = not self.paused
            if not self.paused:
                self.pause_cond.notify()

    def run(self):
        try:
            self.log_signal.emit("[📶] Подключение к API ВКонтакте...")
            vk_session = vk_api.VkApi(token=self.token)
            vk = vk_session.get_api()
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Не удалось подключиться к API ВК: {e}")
            self.finished_signal.emit()
            return

        try:
            server_time = vk.utils.getServerTime()
            current_time = server_time
            self.log_signal.emit(
                f"[⏰] Точное время сервера: {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M')}"
            )
        except:
            current_time = int(time.time())
            self.log_signal.emit(
                f"🤬[WARN] Не удалось получить время сервера. Используется локальное время."
            )

        delay_between_posts = 5
        post_delay_seconds = self.interval_hours * 3600
        current_post_time = self.start_timestamp

        photos = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        self.log_signal.emit(f"[🔎] Найдено {len(photos)} изображений для публикации.")

        if self.cluster_mode:
            #группировка по кластерам, имена типа "123_4.jpg"
            from collections import defaultdict
            clusters = defaultdict(list)
            valid_photos = []
            for photo in photos:
                stem = Path(photo).stem
                if '_' in stem and stem.split('_')[0].isdigit():
                    cluster_id = stem.split('_')[0]
                    clusters[cluster_id].append(photo)
                    valid_photos.append(photo)
                else:
                    self.log_signal.emit(f"⚠️🧰[SKIP] Пропущено (не по шаблону кластера): {photo}")
            #сортировка кластеров по номерам
            sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]))
            batches = [batch for _, batch in sorted_clusters]
            self.log_signal.emit(f"[🧩] Режим кластеров: найдено {len(batches)} кластеров.")
        elif self.random_photos:
            self.log_signal.emit("[🔀] Рандомизация кол-ва фото для каждого поста.")
            current_index = 0
            batches = []
            while current_index < len(photos):
                remaining = len(photos) - current_index
                actual_batch_size = min(random.randint(1, 9), remaining)
                batches.append(photos[current_index:current_index + actual_batch_size])
                current_index += actual_batch_size
        else:
            batch_size = int(self.photos_per_post)
            batches = [photos[i:i + batch_size] for i in range(0, len(photos), batch_size)]

        from concurrent.futures import ThreadPoolExecutor

        for batch_number, photo_batch in enumerate(batches):
            while self.paused:
                with self.pause_cond:
                    self.pause_cond.wait(timeout=1.0)

            try:
                media_ids = []

                def upload_single_photo(photo_file):
                    try:
                        self.log_signal.emit(f"[📩] Загружаю {photo_file}")
                        full_path = os.path.join(self.folder_path, photo_file)

                        upload_server = vk.photos.getWallUploadServer(group_id=abs(int(self.group_id)))
                        server, photo_data, photo_hash = self.upload_photo(upload_server, full_path)
                        media_id = self.save_wall_photo(vk, self.group_id, server, photo_data, photo_hash)

                        posted_folder = os.path.join(self.folder_path, "posted")
                        new_path = os.path.join(posted_folder, photo_file)

                        if os.path.exists(full_path):
                            try:
                                os.rename(full_path, new_path)
                                self.log_signal.emit(f"[📂] Фото '{photo_file}' перемещено в папку 'posted'.")
                            except Exception as move_error:
                                self.log_signal.emit(f"🧰[ERROR] Не удалось переместить '{photo_file}': {move_error}")

                        return media_id
                    except Exception as e:
                        error_str = str(e)
                        self.log_signal.emit(f"🧰[ERROR] Ошибка при обработке файла '{photo_file}': {error_str}")

                        #перемещение при ошибке 100
                        if "[100]" in error_str and "photo is undefined" in error_str:
                            try:
                                failed_folder = os.path.join(self.folder_path, "failed")
                                os.makedirs(failed_folder, exist_ok=True)
                                failed_path = os.path.join(failed_folder, photo_file)
                                if os.path.exists(full_path):
                                    shutil.move(full_path, failed_path)
                                    self.log_signal.emit(f"[⚠️] Файл '{photo_file}' перемещён в 'failed' (photo is undefined).")
                            except Exception as move_err:
                                self.log_signal.emit(f"🧰[ERROR] Не удалось переместить в 'failed': {move_err}")

                        return None

                with ThreadPoolExecutor(max_workers=1) as executor:
                    results = list(executor.map(upload_single_photo, photo_batch))
                    media_ids = [result for result in results if result is not None]
                    
                if not media_ids:
                    self.log_signal.emit("😢[SKIP] Нет файлов для поста, пропускаю.")
                    continue
                    
                attachment_str = ",".join(media_ids)
                if not attachment_str.strip():
                    self.log_signal.emit("😢[SKIP] Вложения отсутствуют, пропускаю пост.")
                    continue
                    
                post_time = current_post_time + batch_number * post_delay_seconds
                if post_time < int(time.time()):
                    post_time = int(time.time()) + 60 * (batch_number + 1)
                    self.log_signal.emit(
                        f"🤬[WARN] Скорректировано время для поста #{batch_number} на {datetime.fromtimestamp(post_time).strftime('%Y-%m-%d %H:%M')}"
                    )
                else:
                    self.log_signal.emit(
                        f"[📅] Пост #{batch_number} запланирован на {datetime.fromtimestamp(post_time).strftime('%Y-%m-%d %H:%M')}"
                    )

                post_kwargs = {
                    'owner_id': int(self.group_id),
                    'from_group': 1,
                    'attachments': ",".join(media_ids),
                    'publish_date': post_time,
                    'primary_attachments_mode': 'carousel' if self.use_carousel else 'grid'
                }

                post_text = self.caption
                if self.use_ai_caption:
                    ai_quote = self.generate_ai_caption()
                    if ai_quote:
                        if post_text:
                            post_text += f"\n\n{ai_quote}"
                        else:
                            post_text = ai_quote
                if self.use_random_emoji and self.emoji_list:
                    emoji = random.choice(self.emoji_list)
                    if post_text:
                        post_text += f"\n{emoji}"
                    else:
                        post_text = emoji
                if post_text.strip():
                    post_kwargs['message'] = post_text

                vk.wall.post(**post_kwargs)

                self.posts_saved += 1
                self.update_last_post_time.emit(post_time)
                save_config(
                    token=self.token,
                    group_id=self.group_id,
                    mistral_api_key=self.mistral_api_key,
                    last_post_time=post_time,
                    ai_prompt=self.ai_custom_prompt
                )
                time.sleep(delay_between_posts)

            except Exception as e:
                self.log_signal.emit(f"🧰[ERROR] Ошибка при обработке пакета #{batch_number}: {e}")

        self.log_signal.emit("[📝] 🧃 Все посты добавлены в отложку. Можешь пойти пить пиво.🍺")
        self.finished_signal.emit()
        
    def generate_ai_caption(self):
        if not self.use_ai_caption or not self.mistral_api_key.strip():
            return ""
        from mistralai import Mistral
        client = Mistral(api_key=self.mistral_api_key.strip())

        banned_section = ""
        if self.last_quotes:
            banned_list = "\n".join(f"  - {q}" for q in self.last_quotes)
            banned_section = (
                "\n\nНЕ используй и не повторяй следующие цитаты (даже по смыслу или структуре):\n"
                f"{banned_list}\n"
            )

        #Получаем пользовательский промпт, он должен быть передан в worker
        technical_suffix = "\n\nЭкранируй её в три восклицательных знака: !!! текст !!!"
        full_prompt = self.ai_custom_prompt + technical_suffix + banned_section
        messages = [{"role": "user", "content": full_prompt}]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.log_signal.emit("[🧠] Запрос к Mistral AI за подписью...")
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.85
                )
                raw_text = response.choices[0].message.content.strip()

                if "!!!" in raw_text:
                    parts = raw_text.split("!!!")
                    if len(parts) >= 3:
                        quote = parts[1].strip()
                        if quote:
                            if quote not in self.last_quotes:
                                self.last_quotes.append(quote)
                                if len(self.last_quotes) > 6:
                                    self.last_quotes.pop(0)
                            self.log_signal.emit(f"[🧠] Получена подпись: {quote}")
                            return quote

                self.log_signal.emit("[🧠] Подпись не найдена в нужном формате (Жаль).")
                return ""
            except Exception as e:
                self.log_signal.emit(f"[🔄] Ошибка Mistral (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    self.log_signal.emit("[❌] Все попытки исчерпаны. Пропускаю ИИ подпись.")
                    return ""
        return ""
    
    def upload_photo(self, server, photo_path):
        import requests
        import json
        import time

        for attempt in range(3):
            try:
                with open(photo_path, 'rb') as f:
                    files = {'photo': f}
                    response = requests.post(server['upload_url'], files=files, timeout=15)
                
                text = response.text.strip()
                if not text:
                    raise Exception("Получен пустой ответ от сервера")

                #Проверка: пришёл ли HTML вместо JSON
                if text.startswith('<!DOCTYPE') or '<html' in text.lower():
                    raise Exception("Ошибка HTML: сервер временно недоступен")

                result = response.json()

                if "error" in result:
                    raise Exception(f"Ошибка от ВК: {result['error']}")

                return result['server'], result['photo'], result['hash']

            except Exception as e:
                self.log_signal.emit(f"[🔄] Ошибка загрузки {photo_path} (попытка {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 + attempt * 2)  #прогрессивная задержка: 2, 4, и тд
                else:
                    raise

        
    def save_wall_photo(self, vk, group_id, server, photo_data, photo_hash):
        photos = vk.photos.saveWallPhoto(
            group_id=abs(int(group_id)),
            server=server,
            photo=photo_data,
            hash=photo_hash
        )
        return f"photo{photos[0]['owner_id']}_{photos[0]['id']}"

    def upload_single_photo(self, vk, group_id, folder_path, photo_file):
        try:
            self.log_signal.emit(f"[📩] Загружаю {photo_file}")
            full_path = os.path.join(folder_path, photo_file)
            upload_server = vk.photos.getWallUploadServer(group_id=abs(int(group_id)))
            server, photo_data, photo_hash = self.upload_photo(upload_server, full_path)
            media_id = self.save_wall_photo(vk, group_id, server, photo_data, photo_hash)
            return media_id
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Ошибка при загрузке {photo_file}: {e}")
            return None
            
class AutobotWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    def __init__(self, token, group_id, interval_hours, main_folder, pin_email, pin_password,
                 caption, use_random_emoji, use_carousel, use_ai_caption, mistral_api_key, ai_prompt,
                 wm_path, wm_opacity, wm_size, wm_position, wm_bw, emoji_list):
        super().__init__()
        self.token = token
        self.group_id = group_id
        self.interval_hours = interval_hours
        self.main_folder = main_folder
        self.pin_email = pin_email
        self.pin_password = pin_password
        self.caption = caption
        self.use_random_emoji = use_random_emoji
        self.use_carousel = use_carousel
        self.use_ai_caption = use_ai_caption
        self.mistral_api_key = mistral_api_key
        self.ai_prompt = ai_prompt
        self.wm_path = wm_path
        self.wm_opacity = wm_opacity
        self.wm_size = wm_size
        self.wm_position = wm_position
        self.wm_bw = wm_bw
        self.emoji_list = emoji_list
        self.running = True

    def run(self):
        self.log_signal.emit("🤖[Autobot] Запущен. Проверяю отложку каждые 5 часов...")
        while self.running:
            try:
                delayed_count = self.check_delayed_count()
                self.log_signal.emit(f"📨[Autobot] Текущая отложка: {delayed_count} постов")
                if delayed_count < 50:
                    self.run_autobot_cycle()
                else:
                    self.log_signal.emit("💤[Autobot] Отложка полная. Жду 5 часов...")
                if self.running:
                    time.sleep(18000)  #5 часов
            except Exception as e:
                self.log_signal.emit(f"🧰[ERROR] Ошибка в основном цикле: {e}")
        self.log_signal.emit("🛑[Autobot] Остановлен.")
        
    def run_autobot_cycle(self):
        self.log_signal.emit("🔄[Autobot] Запуск цикла обработки...")
        main_folder = self.main_folder
        posted_dir = os.path.join(main_folder, "posted")
        autobot_pin_dir = os.path.join(main_folder, "autobotPin")
        dupes_dir = os.path.join(main_folder, "dupes")

        #1 Очистка папки posted/
        if os.path.exists(posted_dir):
            try:
                shutil.rmtree(posted_dir)
                os.makedirs(posted_dir)
                self.log_signal.emit("🧹[Autobot] Папка 'posted' очищена.")
            except Exception as e:
                self.log_signal.emit(f"🧰[ERROR] Не удалось очистить 'posted': {e}")
        else:
            os.makedirs(posted_dir)

        #2 Проверка количества изображений в main_folder
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        existing_files = [
            f for f in os.listdir(main_folder)
            if os.path.isfile(os.path.join(main_folder, f)) and f.lower().endswith(image_extensions)
        ]
        current_count = len(existing_files)
        self.log_signal.emit(f"📊[Autobot] В папке {current_count} изображений.")

        #3 Если <2500 - запускаем Pinterest в подпапку autobotPin/
        if current_count < 2500:
            need = 2500 - current_count
            self.log_signal.emit(f"📥[Autobot] Требуется скачать ещё {need} изображений.")

            #проверка подпапки
            os.makedirs(autobot_pin_dir, exist_ok=True)

            #Запуск Пинтерест через существующий метод
            try:
                self.log_signal.emit("🔑[Pinterest] Попытка входа через cookies...")
                #Используем логику из класса Pinterest
                pinterest = Pinterest(
                    login=self.pin_email,
                    pw=self.pin_password,
                    headless=True,
                    log_callback=lambda msg: self.log_signal.emit(f"[Pinterest] {msg}")
                )

                self.log_signal.emit("🌐[Pinterest] Начинаю загрузку с https://pinterest.com/...")
                #Модифицируем single_download: останавливаем при достижении need
                pinterest.driver.get("https://pinterest.com/")
                time.sleep(3)

                downloaded_count = 0
                page = 0
                while downloaded_count < need and not pinterest._stop_requested:
                    try:
                        pinterest.crawl(autobot_pin_dir)
                        new_downloaded = len(pinterest.piclist) - downloaded_count
                        if new_downloaded > 0:
                            downloaded_count = len(pinterest.piclist)
                        self.log_signal.emit(f"🖼️[Pinterest] Страница {page + 1}, всего скачано: {downloaded_count}")
                        page += 1
                        if downloaded_count >= need:
                            break
                        time.sleep(2)
                    except Exception as e:
                        self.log_signal.emit(f"🧰[Pinterest ERROR] {e}")
                        break

                pinterest.driver.quit()
                self.log_signal.emit(f"🤙[Pinterest] Загружено {downloaded_count} изображений в 'autobotPin/'.")

            except Exception as e:
                self.log_signal.emit(f"💔[Autobot] Ошибка при загрузке с Pinterest: {e}")
                if 'pinterest' in locals() and hasattr(pinterest, 'driver'):
                    try:
                        pinterest.driver.quit()
                    except:
                        pass

        #4 Watermark для всех файлов в autobotPin/ (если указан путь к водяному знаку)
        if os.path.exists(autobot_pin_dir) and os.listdir(autobot_pin_dir):
            if self.wm_path and os.path.isfile(self.wm_path):
                self.log_signal.emit("🖋️[Watermark] Накладываю водяной знак на новые изображения...")
                wm_worker = WatermarkWorker(
                    folder=autobot_pin_dir,
                    watermark_path=self.wm_path,
                    opacity=self.wm_opacity,
                    size=self.wm_size,
                    position=self.wm_position,
                    bw=self.wm_bw
                )
                wm_worker.run()
                self.log_signal.emit("🖋️[Watermark] Завершено.")
            else:
                self.log_signal.emit("⏭️[Watermark] Водяной знак не указан — пропускаю...")

        #5 Перенос из autobotPin/ в main_folder/
        moved_count = 0
        if os.path.exists(autobot_pin_dir):
            for filename in os.listdir(autobot_pin_dir):
                src = os.path.join(autobot_pin_dir, filename)
                dst = os.path.join(main_folder, filename)
                if os.path.isfile(src):
                    if not os.path.exists(dst):  #Не перезаписываем
                        try:
                            shutil.move(src, dst)
                            moved_count += 1
                        except Exception as e:
                            self.log_signal.emit(f"🧰[ERROR] Не удалось переместить {filename}: {e}")
                    else:
                        #Если файл уже есть - удаляем дубликат из подпапки
                        os.remove(src)
            shutil.rmtree(autobot_pin_dir, ignore_errors=True)
        self.log_signal.emit(f"🚚[Autobot] Перемещено {moved_count} файлов в основную папку.")

        #6 рандомайзер имён в main_folder
        self.log_signal.emit("🔀[Autobot] Рандомизация имён всех файлов...")
        randomizer = RandomizerWorker(main_folder)
        randomizer.run()  #синхронно
        self.log_signal.emit("🔀[Autobot] Рандомизация завершена.")

        #7 Anti-Dupe: только точные визуальные дубликаты - dupes/ - удаление dupes/
        self.log_signal.emit("🔍[Autobot] Поиска визуальных дубликатов...")
        exact_hashes = {}
        duplicates_to_move = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

        for filename in os.listdir(main_folder):
            if filename.lower().endswith(image_extensions):
                path = os.path.join(main_folder, filename)
                h = get_normalized_image_hash(path)
                if h:
                    if h in exact_hashes:
                        duplicates_to_move.append(path)
                    else:
                        exact_hashes[h] = path
                else:
                    self.log_signal.emit(f"🧰[WARN] Пропущен файл (ошибка хэширования): {filename}")

        if duplicates_to_move:
            dupes_dir = os.path.join(main_folder, "dupes")
            os.makedirs(dupes_dir, exist_ok=True)
            for dup_path in duplicates_to_move:
                if os.path.exists(dup_path):
                    basename = os.path.basename(dup_path)
                    dest = os.path.join(dupes_dir, basename)
                    counter = 1
                    while os.path.exists(dest):
                        name, ext = os.path.splitext(basename)
                        dest = os.path.join(dupes_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    try:
                        shutil.move(dup_path, dest)
                    except Exception as e:
                        self.log_signal.emit(f"🧰[ERROR] Не удалось переместить дубликат: {e}")
            self.log_signal.emit(f"🗂️[Autobot] Перемещено {len(duplicates_to_move)} дубликатов в 'dupes'.")

            shutil.rmtree(dupes_dir, ignore_errors=True)
            self.log_signal.emit("🧹[Autobot] Папка 'dupes' удалена.")
        else:
            self.log_signal.emit("👍[Autobot] Точных дубликатов не найдено.")

        if duplicates_to_move:
            os.makedirs(dupes_dir, exist_ok=True)
            for dup_path in duplicates_to_move:
                if os.path.exists(dup_path):
                    basename = os.path.basename(dup_path)
                    dest = os.path.join(dupes_dir, basename)
                    counter = 1
                    while os.path.exists(dest):
                        name, ext = os.path.splitext(basename)
                        dest = os.path.join(dupes_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    try:
                        shutil.move(dup_path, dest)
                    except Exception as e:
                        self.log_signal.emit(f"🧰[ERROR] Не удалось переместить дубликат: {e}")
            self.log_signal.emit(f"🗂️[Autobot] Перемещено {len(duplicates_to_move)} дубликатов в 'dupes'.")

            #Сразу удаляем папку dupes
            shutil.rmtree(dupes_dir, ignore_errors=True)
            self.log_signal.emit("🧹[Autobot] Папка 'dupes' удалена.")

        #8 Bimbo Sorter в режиме автораспределения
        self.log_signal.emit("💖[Bimbo] Запуск сортировки по цвету...")
        bimbo_worker = BimboSorterWorker(
            folder_path=main_folder,
            auto_distribute=True
        )
        bimbo_worker.run()
        self.log_signal.emit("💖[Bimbo] Сортировка завершена.")

        #9 Автопостинг до 100 в отложке, каждые 10 постов проверка
        self.log_signal.emit("📮[Autobot] Начинаю автопостинг...")

        #Получаем список кластеров
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        images = [f for f in os.listdir(main_folder) if f.lower().endswith(image_extensions)]

        from collections import defaultdict
        clusters = defaultdict(list)
        for img in images:
            stem = Path(img).stem
            if '_' in stem and stem.split('_')[0].isdigit():
                cluster_id = stem.split('_')[0]
                clusters[cluster_id].append(img)
            else:
                self.log_signal.emit(f"⚠️[SKIP] Пропущено (не формат кластера): {img}")

        sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]))
        cluster_list = [batch for _, batch in sorted_clusters]

        if not cluster_list:
            self.log_signal.emit("📭[Autobot] Нет кластеров для публикации.")
        else:
            #Подключаемся к VK один раз
            try:
                vk_session = vk_api.VkApi(token=self.token)
                vk = vk_session.get_api()
            except Exception as e:
                self.log_signal.emit(f"🧰[ERROR] Не удалось подключиться к ВК: {e}")
                return

            posted_count = 0
            total_clusters = len(cluster_list)

            for idx, photo_batch in enumerate(cluster_list, start=1):
                if not self.running:
                    self.log_signal.emit("🛑[Autobot] Остановлен пользователем.")
                    break

                #Публикуем один кластер
                self.log_signal.emit(f"📬[Autobot] Пост №{idx} из {total_clusters}...")

                media_ids = []
                posted_folder = os.path.join(main_folder, "posted")
                os.makedirs(posted_folder, exist_ok=True)

                for photo_file in photo_batch:
                    full_path = os.path.join(main_folder, photo_file)
                    try:
                        self.log_signal.emit(f"[📩] Загружаю {photo_file}")
                        upload_server = vk.photos.getWallUploadServer(group_id=abs(int(self.group_id)))
                        with open(full_path, 'rb') as f:
                            response = requests.post(upload_server['upload_url'], files={'photo': f})
                        result = response.json()
                        saved = vk.photos.saveWallPhoto(
                            group_id=abs(int(self.group_id)),
                            server=result['server'],
                            photo=result['photo'],
                            hash=result['hash']
                        )
                        media_ids.append(f"photo{saved[0]['owner_id']}_{saved[0]['id']}")
                        shutil.move(full_path, os.path.join(posted_folder, photo_file))
                    except Exception as e:
                        self.log_signal.emit(f"🧰[ERROR] Ошибка загрузки {photo_file}: {e}")

                if not media_ids:
                    self.log_signal.emit(f"😢[SKIP] Пост №{idx} пропущен (нет вложений).")
                    continue

                #Подпись
                post_text = self.caption
                if self.use_ai_caption:
                    ai_quote = self.generate_ai_caption_simple()
                    if ai_quote:
                        post_text = f"{post_text}\n{ai_quote}" if post_text else ai_quote
                if self.use_random_emoji and self.emoji_list:
                    emoji = random.choice(self.emoji_list)
                    post_text = f"{post_text}\n{emoji}" if post_text else emoji

                #Получаем актуальное время
                current_time = int(time.time())
                post_time = current_time + (idx - 1) * self.interval_hours * 3600

                #publish_date должен быть > текущего времени ВК
                min_allowed_time = current_time + 60
                if post_time < min_allowed_time:
                    post_time = min_allowed_time
                    self.log_signal.emit(f"[⚠️] Время поста №{idx} скорректировано на {datetime.fromtimestamp(post_time).strftime('%Y-%m-%d %H:%M')}")

                try:
                    vk.wall.post(
                        owner_id=int(self.group_id),
                        from_group=1,
                        attachments=','.join(media_ids),
                        publish_date=post_time,
                        message=post_text if post_text.strip() else None,
                        primary_attachments_mode='carousel' if self.use_carousel else 'grid'
                    )
                    self.log_signal.emit(f"[📅] Пост №{idx} запланирован.")
                    posted_count += 1
                    time.sleep(3)
                except Exception as e:
                    self.log_signal.emit(f"🧰[ERROR] Ошибка публикации поста №{idx}: {e}")
                    continue

                #🔁 Проверка каждые 10 постов
                if posted_count % 10 == 0:
                    delayed = self.check_delayed_count()
                    self.log_signal.emit(f"📊[Autobot] После {posted_count} постов: отложка = {delayed}")
                    if delayed >= 100: #порог отложки
                        self.log_signal.emit("🛑[Autobot] Отложка достигла 100. Остановка.")
                        break

            self.log_signal.emit(f"👍[Autobot] Завершено. Всего опубликовано: {posted_count} постов. Следующая проверка через 5 часов.")

    def stop(self):
        self.running = False

    def check_delayed_count(self):
        try:
            vk_session = vk_api.VkApi(token=self.token)
            vk = vk_session.get_api()
            offset = 0
            count = 100
            total = 0
            while True:
                response = vk.wall.get(owner_id=int(self.group_id), filter='postponed', count=count, offset=offset)
                items = response.get('items', [])
                if not items:
                    break
                total += len(items)
                offset += count
                time.sleep(0.3)
            return total
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Не удалось проверить отложку: {e}")
            return 0

class BimboSorterWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    
    def __init__(self, folder_path, max_per_cluster=9, auto_distribute=False):
        super().__init__()
        self.folder_path = folder_path
        self.max_per_cluster = max_per_cluster
        self.auto_distribute = auto_distribute

    def run(self):
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in os.listdir(self.folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

            if not images:
                self.log_signal.emit("😭[ERROR] В папке нет подходящих изображений.")
                return

            self.log_signal.emit(f"🔍 [FOUND] Найдено изображений: {len(images)}")

            image_paths = [os.path.join(self.folder_path, img) for img in images]
            colors = []

            self.log_signal.emit(f"💅[PROGRESS] Обработано 0/{len(image_paths)} изображений")
            for i, path in enumerate(image_paths):
                color = self.get_average_color(path)
                colors.append(color)
                self.log_signal.emit(f"💅[PROGRESS] Обработано {i + 1}/{len(image_paths)} изображений")

            #Кластеризация по цвету
            if self.auto_distribute:
                n_clusters = max(1, (len(images) + 8) // 9)  #округление вверх
            else:
                n_clusters = max(1, (len(images) + self.max_per_cluster - 1) // self.max_per_cluster)

            self.log_signal.emit(f"🧠[GROUP] Группирую на {n_clusters} кластеров по цвету...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = kmeans.fit_predict(colors)
            grouped = [[] for _ in range(n_clusters)]
            for img, label in zip(images, labels):
                grouped[label].append(img)

            #Формирование final_groups
            final_groups = []
            if self.auto_distribute:
                for group in grouped:
                    for i in range(0, len(group), 9):
                        chunk = group[i:i + 9]
                        final_groups.append(chunk)  #даже если <9, добавляем
                self.log_signal.emit(f"🧩[AUTO] Разбито на {len(final_groups)} подкластеров (макс. 9 на кластер).")
            else:
                sorted_images = []
                for group in grouped:
                    sorted_images.extend(group)
                incomplete_group = []
                for i in range(0, len(sorted_images), self.max_per_cluster):
                    chunk = sorted_images[i:i + self.max_per_cluster]
                    if len(chunk) == self.max_per_cluster:
                        final_groups.append(chunk)
                    else:
                        incomplete_group = chunk

            #Перемещение в 'check', только если НЕ автораспределение
            if not self.auto_distribute:
                if len(incomplete_group) > 0:
                    check_folder = os.path.join(self.folder_path, 'check')
                    if not os.path.exists(check_folder):
                        os.makedirs(check_folder)
                        self.log_signal.emit(f"📁[INFO] Создана папка: {check_folder}")
                    moved_count = 0
                    for filename in incomplete_group:
                        old_path = os.path.join(self.folder_path, filename)
                        new_path = os.path.join(check_folder, filename)
                        if os.path.exists(new_path):
                            self.log_signal.emit(f"😢[SKIP] Пропущено {filename}: файл уже существует в папке check")
                            continue
                        try:
                            shutil.move(old_path, new_path)
                            self.log_signal.emit(f"🚚[MOVE] Перемещён: {filename} → check/")
                            moved_count += 1
                        except Exception as e:
                            self.log_signal.emit(f"😭[ERROR] Не удалось переместить {filename}: {e}")
                    self.log_signal.emit(f"👍[CHECK] Перемещено {moved_count} файлов в папку 'check'")
            else:
                self.log_signal.emit("🧩[AUTO] Режим автораспределения: пропускаю перемещение в 'check'.")

            #Переименование всех файлов из final_groups
            renamed_count = 0
            new_names_log = []
            for group_idx, group in enumerate(final_groups):
                for item_idx, filename in enumerate(group, start=1):
                    old_path = os.path.join(self.folder_path, filename)
                    ext = os.path.splitext(filename)[1]
                    new_name = f"{group_idx + 1}_{item_idx}{ext}"
                    new_path = os.path.join(self.folder_path, new_name)
                    if os.path.exists(new_path):
                        self.log_signal.emit(f"😢[SKIP] Пропущено {filename}: файл {new_name} уже существует")
                        continue
                    try:
                        os.replace(old_path, new_path)
                        new_names_log.append(f"{filename} → {new_name}")
                        renamed_count += 1
                    except Exception as e:
                        self.log_signal.emit(f"😭[ERROR] Не удалось переименовать {filename}: {e}")

            self.log_signal.emit(f"💅[NAME] Переименовано {renamed_count} файлов:")
            for line in new_names_log:
                self.log_signal.emit(f" → {line}")

            self.log_signal.emit("👍[SUCCESS] Обработка завершена.")

        except Exception as e:
            self.log_signal.emit(f"😭[ERROR] Ошибка: {e}")

        self.finished_signal.emit()

    def get_average_color(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize((16, 16))
                pixels = list(img.getdata())
                avg_color = np.mean(pixels, axis=0).astype(int)
                return tuple(avg_color)
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Ошибка при обработке {image_path}: {e}")
            return (0, 0, 0)

class WatermarkWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    def __init__(self, folder, watermark_path, opacity, size, position, bw):
        super().__init__()
        self.folder = folder
        self.watermark_path = watermark_path
        self.opacity = opacity
        self.size = size
        self.position = position
        self.bw = bw

    def run(self):
        try:
            from PIL import Image
            self.log_signal.emit("[🖼️] Загрузка водяного знака...")
            watermark = Image.open(self.watermark_path).convert("RGBA")
            opacity_factor = self.opacity / 100.0
            watermark = watermark.resize((self.size, self.size))
            original_alpha = watermark.getchannel('A')

            if self.bw:
                self.log_signal.emit("[🖤] Преобразование водяного знака в ЧБ...")
                watermark = watermark.convert("L").convert("RGBA")
                watermark.putalpha(original_alpha)

            alpha = watermark.getchannel('A')
            alpha = alpha.point(lambda p: p * opacity_factor)
            watermark.putalpha(alpha)

            margin = int(self.size * 0.2) #20 проц от текущего размера вотермарки
            #Ограничение отступа, если изображение меньше вотермарки
            pos_map = {
                "top-left": lambda img_w, img_h, wm_w, wm_h: (
                    max(0, margin),
                    max(0, margin)
                ),
                "top-right": lambda img_w, img_h, wm_w, wm_h: (
                    max(0, img_w - wm_w - margin),
                    max(0, margin)
                ),
                "bottom-left": lambda img_w, img_h, wm_w, wm_h: (
                    max(0, margin),
                    max(0, img_h - wm_h - margin)
                ),
                "bottom-right": lambda img_w, img_h, wm_w, wm_h: (
                    max(0, img_w - wm_w - margin),
                    max(0, img_h - wm_h - margin)
                ),
            }

            supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            files = [f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f)) and f.lower().endswith(supported_extensions)]
            total_files = len(files)
            processed = 0
            skipped = []

            self.log_signal.emit(f"🔎[FOUND] Найдено файлов для обработки: {total_files}")

            for filename in files:
                full_path = os.path.join(self.folder, filename)
                try:
                    self.log_signal.emit(f"[🖋️] Обработка файла: {filename}")
                    base_image = Image.open(full_path).convert("RGBA")
                    position_coords = pos_map[self.position](
                        base_image.width,
                        base_image.height,
                        watermark.width,
                        watermark.height
                    )

                    base_image.paste(watermark, position_coords, watermark)
                    base_image = base_image.convert("RGB") if filename.lower().endswith(('.jpg', '.jpeg')) else base_image
                    base_image.save(full_path)
                    processed += 1
                except Exception as e:
                    skipped.append((filename, str(e)))
                    self.log_signal.emit(f"🧰[ERROR] Ошибка при обработке файла '{filename}': {e}")

            self.log_signal.emit(f"[💖💕] Обработано: {processed} файлов")
            if skipped:
                self.log_signal.emit(f"😢[SKIP] Пропущено: {len(skipped)} файлов:")
                for fname, err in skipped:
                    self.log_signal.emit(f" - {fname}: {err}")

        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Критическая ошибка: {e}")
        finally:
            self.finished_signal.emit()

class CheckAndClearWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    count_ready = Signal(int)

    def __init__(self, token, group_id, action="check", new_date=None, interval_hours=1):
        super().__init__()
        self.token = token
        self.group_id = group_id
        self.action = action  # "check", "clear", *"reschedule"
        self.new_date = new_date
        self.interval_hours = interval_hours

    def run(self):
        try:
            self.log_signal.emit("[📶] Подключение к API ВКонтакте...")
            vk_session = vk_api.VkApi(token=self.token)
            vk = vk_session.get_api()
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Не удалось подключиться к API ВК: {e}")
            self.finished_signal.emit()
            return

        try:
            self.log_signal.emit("[📝⏰] Получаем список отложенных записей...")
            offset = 0
            count = 100
            all_posts = []
            while True:
                response = vk.wall.get(owner_id=int(self.group_id), filter='postponed', count=count, offset=offset)
                items = response.get('items', [])
                if not items:
                    break
                all_posts.extend(items)
                offset += count
                time.sleep(0.3)

            count_posts = len(all_posts)
            self.count_ready.emit(count_posts)

            if self.action == "check":
                self.log_signal.emit(f"[🔎] Найдено {count_posts} отложенных записей.")

            elif self.action == "clear":
                self.log_signal.emit(f"[🧼🧼🧼] Начинаю удаление {count_posts} отложенных записей.")
                for post in all_posts:
                    try:
                        vk.wall.delete(owner_id=int(self.group_id), post_id=post['id'])
                        self.log_signal.emit(f"[🧼] Удалён пост ID={post['id']}")
                        time.sleep(0.2)
                    except Exception as e:
                        self.log_signal.emit(f"🧰[ERROR] Ошибка при удалении поста ID={post['id']}: {e}")
                self.log_signal.emit(f"[👍] Все {count_posts} отложенных записей удалены.")
                
                """

            elif self.action == "reschedule":
                if not isinstance(self.new_date, datetime):
                    raise ValueError("Не указана начальная дата")

                self.log_signal.emit(f"[🔄] Начинаю перепланирование {count_posts} постов...")

                current_time = self.new_date
                for i, post in enumerate(all_posts):
                    try:
                        self.log_signal.emit(
                            f"[{i+1}/{count_posts}] Обновление поста ID={post['id']} на {current_time.strftime('%d.%m.%Y %H:%M')}"
                        )
                        vk.wall.editScheduledPost(
                            owner_id=int(self.group_id),
                            post_id=post['id'],
                            publish_date=int(current_time.timestamp())
                        )
                        current_time += datetime.timedelta(hours=self.interval_hours)
                        time.sleep(0.3)
                    except Exception as e:
                        self.log_signal.emit(f"🧰[ERROR] Ошибка при обновлении поста ID={post['id']}: {e}")

                self.log_signal.emit(f"[⏰] Все {count_posts} записей перепланированы.")
                
                """

        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Ошибка при работе с API: {e}")

        self.finished_signal.emit()
        
class PinterestWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, email, password, link, save_dir, pages):
        super().__init__()
        self.email = email
        self.password = password
        self.link = link
        self.save_dir = save_dir
        self.pages = pages
        self.pinterest = None

    def run(self):
        try:
            self.log_signal.emit("Запуск браузера...")
            self.pinterest = Pinterest(self.email, self.password, headless=True, log_callback=self.log_signal.emit)
            self.log_signal.emit("Авторизация завершена.")
            self.pinterest.single_download(pages=self.pages, url=self.link, dir=self.save_dir)
            self.log_signal.emit("Загрузка завершена!")
        except Exception as e:
            self.log_signal.emit(f"Ошибка: {str(e)}")
        finally:
            self.finished_signal.emit()

    def stop_gracefully(self):
        if self.pinterest:
            self.pinterest.stop()


class Pinterest:
    def __init__(self, login, pw, headless=True, log_callback=None):
        self._stop_requested = False
        self.log = log_callback or print
        self.domains = [".pinterest.com", "www.pinterest.com", ".www.pinterest.co.kr", "www.pinterest.co.kr"]
        self.piclist = []
        import chromedriver_autoinstaller
        chromedriver_autoinstaller.install()
        from selenium import webdriver
        from time import sleep

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--log-level=3')
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        self.driver = webdriver.Chrome(options=options)

        #попытка входа через куки
        if os.path.exists("cookies.pkl"):
            self.log("🍪[LOADING] Загружаю cookies...")
            self.driver.get("https://pinterest.com")
            sleep(2)
            try:
                import pickle
                cookies = pickle.load(open("cookies.pkl", "rb"))
                for cookie in cookies:
                    for domain in self.domains:
                        try:
                            self.driver.add_cookie({
                                "domain": domain,
                                "name": cookie["name"],
                                "value": cookie["value"],
                                "path": "/"
                            })
                        except:
                            pass
                self.driver.get("https://pinterest.com")
                sleep(2)
                if self._is_logged_in():
                    self.log("🍪[SUCCESS] Успешный вход через cookies.")
                    return  #выходим, если cookies сработали
            except Exception as e:
                self.log(f"💔[ERROR] Ошибка загрузки cookies: {e}")

        #если куки не сработали, требуем логин/пароль
        if not login or not pw:
            raise Exception("💔[ERROR] Требуется email и пароль (cookies недействительны или отсутствуют).")

        #ручной вход
        self.log("Выполняется вход в аккаунт...")
        self.driver.get("https://pinterest.com/login")
        sleep(2)
        try:
            from selenium.webdriver.common.by import By
            emailelem = self.driver.find_element(By.ID, "email")
            passelem = self.driver.find_element(By.ID, "password")
            emailelem.send_keys(login)
            passelem.send_keys(pw)
            self.driver.find_element(By.XPATH, "//button[@type='submit']").click()
            sleep(5)
        except Exception as e:
            raise Exception(f"Ошибка входа: {e}")

        for _ in range(20):
            if self._is_logged_in():
                self.log("Вход выполнен успешно.")
                self._dump_cookies()
                return
            sleep(1)
        raise Exception("💔[ERROR]Не удалось войти. Проверь логин/пароль.")

    def _is_logged_in(self):
        try:
            from selenium.webdriver.common.by import By
            self.driver.find_element(By.XPATH, '//*[@id="HeaderContent"]')
            return True
        except:
            return False

    def _dump_cookies(self):
        import pickle
        cookies = self.driver.get_cookies()
        pickle.dump(cookies, open("cookies.pkl", "wb"))
        
    def stop(self):
        #Запрос на остановку загрузки
        self._stop_requested = True

    def crawl(self, dir_path):
        from time import sleep
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        timeout = 0
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height != last_height:
                self.download_image(dir_path)
                last_height = new_height
                break
            else:
                timeout += 1
                if timeout >= 5:
                    raise Exception("🏁[END] Конец страницы.")

    def single_download(self, pages=10, url="https://pinterest.com/", dir="./download"):
        import os
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.driver.get(url)
        from time import sleep
        sleep(3)
        for i in range(pages):
            if self._stop_requested:
                self.log("🛑[STOP] Загрузка прервана по запросу.")
                break
            try:
                self.crawl(dir)
                self.log(f"📃[DWNLD] Страница {i + 1} загружена. Всего изображений: {len(self.piclist)}")
            except Exception as e:
                self.log(f"Завершение: {e}")
                break
        try:
            self.driver.quit()
        except:
            pass

    def download_image(self, dir_path):
        from bs4 import BeautifulSoup
        req = self.driver.page_source
        soup = BeautifulSoup(req, 'html.parser')
        pics = soup.find_all("img")
        if not pics:
            return
        for pic in pics:
            src = pic.get("src")
            if not src or "75x75_RS" in src:
                continue

            #замена превью на оригинал
            original_src = (
                src
                .replace("/236x/", "/originals/")
                .replace("/474x/", "/originals/")
                .replace("/736x/", "/originals/")
                .replace("/564x/", "/originals/")
            )

            if original_src not in self.piclist:
                self.piclist.append(original_src)
                self._save_image_async(original_src, dir_path)

    def _save_image_async(self, url, dir_path):
        from concurrent.futures import ThreadPoolExecutor
        def save():
            try:
                import requests
                from urllib.parse import urlparse
                filename = os.path.basename(urlparse(url).path)
                if not filename or '.' not in filename:
                    filename = f"img_{hash(url) % 1000000}.jpg"
                filepath = os.path.join(dir_path, filename)
                if os.path.exists(filepath):
                    return
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if resp.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(resp.content)
            except Exception:
                pass
        executor = ThreadPoolExecutor(max_workers=10)
        executor.submit(save)
        

class AlbumDownloaderWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, token, album_input, download_folder):
        super().__init__()
        self.token = token
        self.album_input = album_input.strip()
        self.download_folder = download_folder.strip()
        self.photo_queue = Queue()
        self.downloader_threads = []
        self.scanner_thread = None
        self.stop_flag = False

    def run(self):
        try:
            self.log_signal.emit("🔍[DEBUG] Начинаю загрузку альбома...")
            session = vk_api.VkApi(token=self.token)
            vk = session.get_api()
        except Exception as e:
            self.log_signal.emit(f"💔[ERROR] Ошибка подключения к ВК: {e}")
            self.finished_signal.emit()
            return

        try:
            owner_id, album_id = self.parse_album_input(self.album_input)

            if str(album_id) == '00':
                album_id = 'wall'

            if album_id == 'wall':
                today = datetime.now().strftime("%Y-%m-%d")
                base_title = "Фото со стены"
                folder_path = None
                counter = 1
                while True:
                    potential_title = f"{base_title} {counter} {today}"
                    potential_path = os.path.join(self.download_folder, potential_title)
                    if not os.path.exists(potential_path):
                        folder_path = potential_path
                        break
                    counter += 1
                album_title = potential_title
            else:
                try:
                    album_title = self.get_album_title(vk, owner_id, album_id)
                    safe_title = "".join(c for c in album_title if c.isalnum() or c in (" ", "_", "-")).strip()
                    folder_path = os.path.join(self.download_folder, f"{safe_title} ({owner_id}_{album_id})")
                except Exception as e:
                    self.log_signal.emit(f"🤬[WARN] Не удалось получить название альбома: {e}")
                    album_title = f"альбом_{owner_id}_{album_id}"
                    folder_path = os.path.join(self.download_folder, album_title)

            self.log_signal.emit(f"📥[INFO] Загрузка альбома '{album_title}' ({owner_id}_{album_id})...")

            os.makedirs(folder_path, exist_ok=True)
            self.log_signal.emit(f"📁[INFO] Сохраняю в папку: {folder_path}")

            #Запуск сканера в отдельном потоке
            self.stop_flag = False
            self.scanner_thread = Thread(
                target=self.scanner_task,
                args=(vk, owner_id, album_id),
                daemon=True
            )
            self.scanner_thread.start()

            #Запуск нескольких загрузчиков
            for _ in range(3):  #3 потока для скачивания
                downloader = Thread(
                    target=self.downloader_task,
                    args=(folder_path,),
                    daemon=True
                )
                downloader.start()
                self.downloader_threads.append(downloader)

            self.scanner_thread.join()

            self.photo_queue.join()
            self.stop_flag = True

            for t in self.downloader_threads:
                t.join(timeout=1)

            self.log_signal.emit("🍺[INFO] Все фото успешно загружены. Можно попить пиво.")
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Фатальная ошибка: {e}")
        self.finished_signal.emit()

    def scanner_task(self, vk, owner_id, album_id):
        offset = 0
        count = 100
        while not self.stop_flag:
            try:
                self.log_signal.emit(f"📡[DEBUG] Получаю фото (offset={offset})...")
                response = vk.photos.get(owner_id=owner_id, album_id=album_id, count=count, offset=offset)
                items = response.get('items', [])
                if not items:
                    self.log_signal.emit("💔[DEBUG] Больше нет фото.")
                    break
                self.log_signal.emit(f"🖼️[DEBUG] Получено {len(items)} фото (offset={offset})")
                for photo in items:
                    self.photo_queue.put(photo)
                offset += count
                time.sleep(0.5)
            except vk_api.ApiError as e:
                self.log_signal.emit(f"🧰[VK ERROR] {e.error_code}: {e.error_msg}")
                if e.error_code == 6:
                    time.sleep(1)
                else:
                    break
            except Exception as e:
                self.log_signal.emit(f"🌐[NETWORK ERROR] {e}")
                break

        for _ in range(len(self.downloader_threads)):
            self.photo_queue.put(None)

    def downloader_task(self, folder_path):
        while not self.stop_flag:
            try:
                photo = self.photo_queue.get(timeout=2)
                if photo is None:
                    self.photo_queue.task_done()
                    break
                try:
                    max_size_url = max(photo['sizes'], key=lambda x: x['width'])['url']
                    filename = f"{photo['id']}.jpg"
                    filepath = os.path.join(folder_path, filename)
                    if os.path.exists(filepath):
                        self.log_signal.emit(f"🔁[SKIP] Файл уже существует: {filename}")
                        self.photo_queue.task_done()
                        continue
                    response = requests.get(max_size_url, stream=True)
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024 * 1024):  # 1MB chunks
                            f.write(chunk)
                    self.log_signal.emit(f"🖤[SUCCESS] Скачано фото: {filename}")
                    self.photo_queue.task_done()
                    time.sleep(0.2)
                except Exception as e:
                    self.log_signal.emit(f"❤[ERROR] Ошибка при скачивании фото: {e}")
                    self.photo_queue.task_done()
            except Empty:
                continue

    def parse_album_input(self, input_str):
        input_str = input_str.strip()
        if not input_str:
            raise ValueError("Пустой ввод")
        owner_id = None
        album_id = None
        if input_str.startswith("http"):
            if "album" not in input_str:
                raise ValueError("Ссылка не содержит информации об альбоме")
            parts = input_str.split("album")[-1].split("_")
            if len(parts) < 2:
                raise ValueError("Неверная ссылка на альбом")
            owner_id = parts[0]
            album_id = parts[1].split("?")[0]
        elif input_str.startswith("album"):
            try:
                _, owner_id, album_id = input_str.split("_", maxsplit=2)
            except ValueError:
                raise ValueError(f"Неверный формат album_id: {input_str}")
        elif "_" in input_str:
            owner_id, album_id = input_str.split("_", maxsplit=1)
        else:
            raise ValueError("Некорректный формат ID или ссылки")

        if str(album_id) == '00':
            album_id = 'wall'

        return int(owner_id), album_id

    def get_album_title(self, vk, owner_id, album_id):
        try:
            albums = vk.photos.getAlbums(owner_id=owner_id, album_ids=[album_id])
            return albums['items'][0]['title']
        except Exception as e:
            self.log_signal.emit(f"🤬[WARN] Название альбома не найдено: {e}")
            return f"альбом_{owner_id}_{album_id}"
            
class VKWorker(QThread):
    log_signal = Signal(str)
    result_signal = Signal(list)
    finished_signal = Signal()

    def __init__(self, token, group_id, action="scan"):
        super().__init__()
        self.token = token
        self.group_id = group_id
        self.action = action  # "scan" или "clear"
        self.blocked_users = []

    def run(self):
        try:
            vk_session = vk_api.VkApi(token=self.token)
            vk = vk_session.get_api()
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Не удалось подключиться к API: {e}")
            self.finished_signal.emit()
            return

        try:
            self.log_signal.emit("🔍[DEBUG] Получаю список участников...")
            members = []
            offset = 0
            count = 1000
            while True:
                response = vk.groups.getMembers(group_id=abs(int(self.group_id)), offset=offset, count=count)
                items = response.get('items', [])
                if not items:
                    break
                members.extend(items)
                offset += count
                self.msleep(300)  #анти-спам задержка

            self.log_signal.emit(f"🔍👽[SUCCESS] Найдено {len(members)} участников.")

            #Проверка статусов
            for i in range(0, len(members), 200):
                batch = members[i:i + 200]
                user_info = vk.users.get(user_ids=",".join(map(str, batch)))
                for u in user_info:
                    if 'deactivated' in u:
                        self.blocked_users.append(u)
                        self.log_signal.emit(f"☠[DEBUG] Заблокирован/удалён: {u['id']} | {u.get('deactivated', 'unknown')}")
                self.msleep(300)

            #Действие от режима
            if self.action == "scan":
                self.result_signal.emit(self.blocked_users)

            elif self.action == "clear":
                if not self.blocked_users:
                    self.log_signal.emit("📭[INFO] Нет заблокированных пользователей для удаления.")
                else:
                    self.log_signal.emit(f"[🗑️] Начинаю удаление {len(self.blocked_users)} пользователей...")
                    for user in self.blocked_users:
                        try:
                            vk.groups.removeUser(group_id=abs(int(self.group_id)), user_id=user['id'])
                            self.log_signal.emit(f"🧼[SUCCESS] Удалён: {user['id']}")
                            self.msleep(350)  # задержка между удалениями (обход лимитов ВК)
                        except vk_api.ApiError as e:
                            if e.code == 15:
                                self.log_signal.emit(f"🧰[ERROR] Нет прав на удаление {user['id']} (возможно, не админ).")
                            else:
                                self.log_signal.emit(f"🧰[ERROR] Ошибка API при удалении {user['id']}: {e}")
                        except Exception as e:
                            self.log_signal.emit(f"🧰[ERROR] Неизвестная ошибка при удалении {user['id']}: {e}")
                self.result_signal.emit(self.blocked_users)

        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Ошибка: {e}")

        self.finished_signal.emit()
            
class RandomizerWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            script_name = os.path.basename(sys.argv[0])  #Имя самого запущенного скрипта
            files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]

            self.log_signal.emit(f"[🔎] Найдено {len(files)} файлов для переименования.")

            for filename in files:
                file_path = os.path.join(self.folder_path, filename)

                if filename == script_name:
                    self.log_signal.emit(f"😢[SKIP] Пропущен: {filename} (это сам скрипт)")
                    continue

                name, ext = os.path.splitext(filename)
                new_name = self.random_string() + ext
                new_path = os.path.join(self.folder_path, new_name)

                try:
                    os.rename(file_path, new_path)
                    self.log_signal.emit(f"📝[SUCCESS] Переименован: {filename} -> {new_name}")
                except Exception as e:
                    self.log_signal.emit(f"🧰[ERROR] Ошибка при переименовании '{filename}': {e}")

            self.log_signal.emit("👍[SUCCESS] Все файлы обработаны.")
        except Exception as e:
            self.log_signal.emit(f"🧰[ERROR] Критическая ошибка: {e}")
        finally:
            self.finished_signal.emit()

    def random_string(self, length=15):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))

class VKAutoPosterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VK Adminium")
        self.resize(1350, 650)
        icon_path = resource_path("ico.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
                font-family: Arial;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: none;
                background-color: #2e2e2e;
            }
            QTabBar::tab {
                background-color: #2e2e2e;
                color: white;
                padding: 8px 16px;
                border: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3c3c3c;
                border-bottom: 2px solid #668eff;
            }
            QLineEdit {
                background-color: #444;
                border: 1px solid #555;
                padding: 5px;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #444;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #668eff;
                border: none;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #008ecc;
            }
            QPushButton#clear_button {
                background-color: #ff4444;
            }
            QPushButton#clear_button:hover {
                background-color: #cc3333;
            }
            QPushButton#pause_button {
                background-color: #ffa500;
            }
            QPushButton#pause_button:hover {
                background-color: #dd8800;
            }
            QDateTimeEdit {
                background-color: #444;
                border: 1px solid #555;
                padding: 5px;
                color: white;
            }
            QPushButton#pause_button {
                background-color: #668eff;
            }
            QPushButton#pause_button:hover {
                background-color: #008ecc;
            }
            QPushButton#pause_button[paused="true"] {
                background-color: #ff4444;
            }
            QPushButton#pause_button[paused="true"]:hover {
                background-color: #cc3333;
            }
            QPushButton#eyeButton {
                background-color: transparent;
                border: none;
                padding: 0;
                margin: 0;
                font-size: 16px;
                color: white;
            }
        """)
        self.init_ui()
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        
        self.album_downloader_worker = AlbumDownloaderWorker("", "", "")
        
        self.emoji_list = [
            "💋", "💄", "🧴", "🧼", "🧖‍♀️", "✨", "🌟", "💫", "💅", "💎",
            "🌸", "👠", "👡", "👢", "👜", "👛", "👒", "🎀", "🧥", "🩱",
            "👗", "👚", "🕶️", "💘", "💗", "💓", "💞", "❤️", "💌", "🌹",
            "💋", "😏", "😍", "😘", "🥰", "🎉", "✨", "🍾", "🥂", "🍷",
            "🍸", "🍹", "🧁", "🍰", "🍭", "🍬", "🍫", "🍩", "🍪", "🍧",
            "🍨", "🍦", "🧁", "🧚", "🦄", "🧸", "🎀", "🔮", "🌌", "🪐",
            "💫", "🌠", "😈", "👅", "🍑", "🍒", "🍓", "🥵", "👙", "🩳",
            "💦", "🩸", "😳", "😍", "🤤", "😜", "😏", "😒", "😌", "🥰",
            "😱", "🤯", "😵‍💫", "🐾", "🌷", "🌼", "🌻", "🌿", "🍀", "🍁",
            "🥀", "🌺", "🌌", "🪐", "🌕", "🌑", "🛸", "👽", "👾", "🛰️",
            "☕", "🍵", "🥛", "🍯", "🧁", "🍰", "🍩", "🍪", "🍧", "🍨",
            "🍦", "🎵", "🎶", "🎧", "📻", "🎹", "🎼", "🎤", "🎙️", "🎚️",
            "📼", "💝", "💖", "💕", "🖤", "👀", "👄", "🌒", "🌓", "🌔",
            "🌖", "🌗", "🌘", "🌙", "🌚", "🌛", "🌜", "☀", "⭐", "☁",
            "⛅", "⛈", "🌤", "🌥", "🌦", "🌧", "🌨", "🌩", "🌪", "⛱",
            "☄", "🔥", "🍉", "🍺", "🗾", "🏘", "🏯", "🏰", "💒", "🗼",
            "🗽", "⛪", "🌉", "🌇", "🌆", "🌅", "🌄", "🏙", "🌃", "🌁",
            "⛺", "⛲", "🕋", "⛩", "🎠", "🎡", "🎇", "🎆", "🎃", "🎴",
            "🎛", "🚬", "🛒", "🚿", "🎱",
        ]
        
        config = load_config()
        self.current_ai_prompt = config.get("ai_prompt", "").strip()
        if not self.current_ai_prompt:
            self.current_ai_prompt = (
                "Мне нужны цитаты в стиле подростковых депрессивных пабликов ВКонтакте — короткие, острые, меланхоличные, "
                "сырого и немного наивного отчаяния. Они должны звучать как обрывки мыслей, вырванные из контекста, будто "
                "кто-то записал их в блокноте или на полях учебника.\n\n"
                "Ключевые требования:\n"
                "1. Эмоциональная насыщенность — боль, пустота, одиночество — но без диагнозов.\n"
                "2. Краткость — 1–2 предложения, максимум 3.\n"
                "3. Стиль: обрывки монолога, без логики между ними.\n"
                "4. Подростковая наивность: крик души, а не философия.\n"
                "5. Никаких имён, мест, событий. Только чувства и абстракции.\n"
                "6. Подходит под ч/б фото: дождь, улицы, силуэты, пустые комнаты."
            )
            
    def open_prompt_editor(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Редактор промпта ИИ")
        dialog.setModal(True)
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)

        label = QLabel("Введите промпт для генерации подписей:")
        layout.addWidget(label)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.current_ai_prompt)
        self.prompt_edit.setStyleSheet("""
            background-color: #2e2e2e;
            color: white;
            border: 1px solid #555;
            font-family: Consolas, monospace;
        """)
        layout.addWidget(self.prompt_edit)

        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(lambda: self.save_custom_prompt(dialog))
        layout.addWidget(apply_btn)

        dialog.exec()
        
    def save_custom_prompt(self, dialog):
        new_prompt = self.prompt_edit.toPlainText().strip()
        if not new_prompt:
            QMessageBox.warning(self, "Ошибка", "Промпт не может быть пустым.")
            return

        self.current_ai_prompt = new_prompt

        token = self.token_input.text().strip()
        group_id = self.group_input.text().strip()
        mistral_api_key = self.mistral_token_input.text().strip()

        save_config(
            token=token,
            group_id=group_id,
            mistral_api_key=mistral_api_key,
            last_post_time=None,
            ai_prompt=self.current_ai_prompt
        )
        dialog.accept()
            

    def init_ui(self):
        self.tabs = QTabWidget()

        adminium_tab = QWidget()
        adminium_layout = QHBoxLayout()
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        config = load_config()

        token_label = QLabel()
        token_label.setText('<a href="https://vkhost.github.io/"  style="color: #668eff; text-decoration: none;">Токен API:</a>')
        token_label.setOpenExternalLinks(False)
        token_label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(link))
        left_layout.addWidget(token_label)

        self.token_input = QLineEdit(config.get("token", ""))
        self.token_input.setEchoMode(QLineEdit.Password)
        token_eye_btn = QPushButton("👁️")
        token_eye_btn.setObjectName("eyeButton")
        token_eye_btn.setFixedSize(30, 30)
        #token_eye_btn.setCursor(Qt.PointingHandCursor)
        token_eye_btn.enterEvent = lambda e: self.token_input.setEchoMode(QLineEdit.Normal)
        token_eye_btn.leaveEvent = lambda e: self.token_input.setEchoMode(QLineEdit.Password)
        token_layout = QHBoxLayout()
        token_layout.addWidget(self.token_input)
        token_layout.addWidget(token_eye_btn)
        left_layout.addLayout(token_layout)

        self.group_input = QLineEdit(config.get("group_id", ""))
        left_layout.addWidget(QLabel("Числовой ID сообщества|паблика:"))
        left_layout.addWidget(self.group_input)

        self.photos_per_post_input = QLineEdit(config.get("photos_per_post", "9"))
        self.original_style = self.photos_per_post_input.styleSheet()
        left_layout.addWidget(QLabel("Кол-во фото на один пост (1-9):"))
        left_layout.addWidget(self.photos_per_post_input)

        self.interval_input = QLineEdit("2")
        left_layout.addWidget(QLabel("Интервал постов (в часах):"))
        left_layout.addWidget(self.interval_input)

        last_post_time = config.get("last_post_time")
        default_start = datetime.now().replace(second=0, microsecond=0)
        if last_post_time:
            default_start = datetime.fromtimestamp(last_post_time + 7200)
        self.datetime_edit = QDateTimeEdit(default_start)
        self.datetime_edit.setDisplayFormat("dd.MM.yyyy HH:mm")
        self.datetime_edit.setCalendarPopup(True)
        left_layout.addWidget(QLabel("Дата и время первого поста:"))
        left_layout.addWidget(self.datetime_edit)

        self.caption_input = QLineEdit("")
        left_layout.addWidget(QLabel("Подпись к постам (Необязательно):"))
        left_layout.addWidget(self.caption_input)

        self.random_emoji_checkbox = QCheckBox("Рандомизировать эмодзи")
        left_layout.addWidget(self.random_emoji_checkbox)

        self.random_photos_checkbox = QCheckBox("Рандомизировать кол-во фото на пост")
        left_layout.addWidget(self.random_photos_checkbox)
        
        self.carousel_checkbox = QCheckBox("Карусель")
        left_layout.addWidget(self.carousel_checkbox)
        
        self.cluster_mode_checkbox = QCheckBox("Кол-во фото по соответствию кластеру")
        left_layout.addWidget(self.cluster_mode_checkbox)
        
        ai_caption_layout = QHBoxLayout()
        self.ai_caption_checkbox = QCheckBox()
        ai_caption_label = QLabel("ИИ подписи <b>(BETA)</b>")
        ai_caption_label.setTextFormat(Qt.RichText)
        ai_caption_layout.addWidget(self.ai_caption_checkbox)
        ai_caption_layout.addWidget(ai_caption_label)

        self.edit_prompt_label = QLabel('<a href="#" style="font-size: 70%; color: #668eff;">Редактировать промпт</a>')
        self.edit_prompt_label.setOpenExternalLinks(False)
        self.edit_prompt_label.linkActivated.connect(self.open_prompt_editor)
        self.edit_prompt_label.setToolTip("Настроить промпт для генерации подписей")
        ai_caption_layout.addWidget(self.edit_prompt_label)
        ai_caption_layout.addStretch()
        left_layout.addLayout(ai_caption_layout)

        mistral_label = QLabel()
        mistral_label.setText(
            '<a href="https://console.mistral.ai/home" style="color: #668eff; text-decoration: none;">Mistral API</a> ключ (для ИИ подписей):'
        )
        mistral_label.setOpenExternalLinks(False)
        mistral_label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(QtCore.QUrl(link)))
        left_layout.addWidget(mistral_label)

        self.mistral_token_input = QLineEdit(config.get("mistral_api_key", ""))
        self.mistral_token_input.setEchoMode(QLineEdit.Password)
        mistral_eye_btn = QPushButton("👁️")
        mistral_eye_btn.setObjectName("eyeButton")
        mistral_eye_btn.setFixedSize(30, 30)
        #mistral_eye_btn.setCursor(Qt.PointingHandCursor)
        mistral_eye_btn.enterEvent = lambda e: self.mistral_token_input.setEchoMode(QLineEdit.Normal)
        mistral_eye_btn.leaveEvent = lambda e: self.mistral_token_input.setEchoMode(QLineEdit.Password)
        mistral_layout = QHBoxLayout()
        mistral_layout.addWidget(self.mistral_token_input)
        mistral_layout.addWidget(mistral_eye_btn)
        left_layout.addLayout(mistral_layout)
        

        def toggle_photos_input():
            enabled = not (self.random_photos_checkbox.isChecked() or self.cluster_mode_checkbox.isChecked())
            self.photos_per_post_input.setEnabled(enabled)
            if enabled:
                self.photos_per_post_input.setStyleSheet(self.original_style)
            else:
                self.photos_per_post_input.setStyleSheet("""
                    background-color: #333;
                    color: #777;
                    border: 1px solid #444;
                """)

        self.random_photos_checkbox.stateChanged.connect(toggle_photos_input)
        self.cluster_mode_checkbox.stateChanged.connect(toggle_photos_input)
        toggle_photos_input()
        
        self.photos_folder_input = QLineEdit()
        default_photos_folder = os.path.join(os.path.dirname(sys.argv[0]), "photos")
        self.photos_folder_input.setText(default_photos_folder)

        self.photos_folder_btn = QPushButton("📁 Выбрать папку с фото")
        self.photos_folder_btn.clicked.connect(self.select_photos_folder)

        left_layout.addWidget(QLabel("Папка с изображениями для постов:"))
        left_layout.addWidget(self.photos_folder_input)
        left_layout.addWidget(self.photos_folder_btn)

        self.run_button = QPushButton("GO POSTAL!")
        self.run_button.clicked.connect(self.start_posting)
        left_layout.addWidget(self.run_button)

        check_clear_layout = QHBoxLayout()
        self.check_button = QPushButton("Проверить кол-во отложки")
        self.check_button.setStyleSheet("font-size: 10px;")
        self.check_button.clicked.connect(self.check_delayed)
        self.clear_button = QPushButton("Очистить отложку")
        self.clear_button.setObjectName("clear_button")
        self.clear_button.clicked.connect(self.clear_delayed)
        check_clear_layout.addWidget(self.check_button)
        check_clear_layout.addWidget(self.clear_button)
        left_layout.addLayout(check_clear_layout)

        self.pause_button = QPushButton("⏸️Пауза")
        self.pause_button.setObjectName("pause_button")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        left_layout.addWidget(self.pause_button)
        
        """
        self.logo_label = QLabel()
        logo_path = resource_path("bckg.png")
        if os.path.exists(logo_path):
            logo_pixmap = QtGui.QPixmap(logo_path)
            self.logo_label.setPixmap(logo_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.logo_label.setText("Лого не найдено")
        self.logo_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.logo_label)
        
        """

        left_layout.addStretch()
        left_widget.setLayout(left_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)

        adminium_layout.addWidget(left_widget, stretch=0)
        left_widget.setFixedWidth(360)

        self.log_area.setFixedWidth(450)
        adminium_layout.addWidget(self.log_area)
        adminium_tab.setLayout(adminium_layout)

        #Вкладка Watermark
        watermark_tab = QWidget()
        wm_main_layout = QHBoxLayout(watermark_tab)
        wm_form_widget = QWidget()
        wm_form_widget.setFixedWidth(360)
        wm_form_layout = QVBoxLayout(wm_form_widget)
        wm_form_layout.setSpacing(10)
        wm_form_layout.setContentsMargins(10, 10, 10, 10)

        self.wm_folder = ""
        self.wm_label_folder = QLabel("Папка с изображениями: не выбрана")
        wm_form_layout.addWidget(self.wm_label_folder)
        self.wm_btn_folder = QPushButton("Выбрать папку")
        self.wm_btn_folder.clicked.connect(self.select_wm_folder)
        wm_form_layout.addWidget(self.wm_btn_folder)

        self.wm_path = ""
        self.wm_label_watermark = QLabel("Изображение водяного знака: не выбрано")
        wm_form_layout.addWidget(self.wm_label_watermark)
        self.wm_btn_watermark = QPushButton("Выбрать водяной знак")
        self.wm_btn_watermark.clicked.connect(self.select_wm_image)
        wm_form_layout.addWidget(self.wm_btn_watermark)

        #Непрозрачность
        ##wm_form_layout.addWidget(QLabel("Непрозрачность (%)"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(32)
        self.opacity_label = QLabel("Непрозрачность: 32%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"Непрозрачность: {v}%"))
        wm_form_layout.addWidget(self.opacity_slider)
        wm_form_layout.addWidget(self.opacity_label)

        #Размер водяного знака
        wm_form_layout.addWidget(QLabel("Размер водяного знака (px):"))
        self.size_input = QLineEdit("100")
        wm_form_layout.addWidget(self.size_input)


        position_group_box = QWidget()
        position_group_layout = QVBoxLayout(position_group_box)
        position_group_layout.addWidget(QLabel("Расположение вотермарки:"))

        square_container = QWidget()
        square_container.setFixedSize(100, 100)
        square_container.setStyleSheet("background-color: #3c3c3c; border: 1px solid #555;")
        square_container_layout = QGridLayout(square_container)
        square_container_layout.setContentsMargins(5, 5, 5, 5)
        square_container_layout.setSpacing(0)
        square_container_layout.setColumnMinimumWidth(0, 45)
        square_container_layout.setColumnMinimumWidth(1, 45)
        square_container_layout.setRowMinimumHeight(0, 45)
        square_container_layout.setRowMinimumHeight(1, 45)

        positions = {
            "top-left":     (0, 0, Qt.AlignLeft | Qt.AlignTop),
            "top-right":    (0, 1, Qt.AlignRight | Qt.AlignTop),
            "bottom-left":  (1, 0, Qt.AlignLeft | Qt.AlignBottom),
            "bottom-right": (1, 1, Qt.AlignRight | Qt.AlignBottom),
        }
        self.position_buttons = {}
        for name, (row, col, align) in positions.items():
            rb = QRadioButton()
            rb.setFixedSize(17, 17)
            rb.setStyleSheet(f"""
                QRadioButton::indicator {{
                    width: 14px;
                    height: 14px;
                    border-radius: 0px;
                    border: 1px solid #668eff;
                    background: #2e2e2e;
                }}
                QRadioButton::indicator:checked {{
                    background: #668eff;
                    border: 1px solid #668eff;
                }}
            """)
            self.position_buttons[name] = rb
            square_container_layout.addWidget(rb, row, col, align)
        self.position_buttons["top-right"].setChecked(True)

        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.addStretch()
        center_layout.addWidget(square_container)
        center_layout.addStretch()
        position_group_layout.addWidget(center_widget)

        wm_form_layout.addWidget(position_group_box)

        self.wm_bw_checkbox = QCheckBox("Черно-белый водяной знак")
        wm_form_layout.addWidget(self.wm_bw_checkbox)

        self.wm_apply_button = QPushButton("Применить водяной знак")
        self.wm_apply_button.clicked.connect(self.apply_watermark)
        wm_form_layout.addWidget(self.wm_apply_button)
        wm_form_layout.addStretch()
        wm_main_layout.addWidget(wm_form_widget, stretch=1)

        #лог с фиксированной шириной
        self.wm_log_area = QTextEdit()
        self.wm_log_area.setReadOnly(True)
        self.wm_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.wm_log_area.setFixedWidth(450)
        wm_main_layout.addWidget(self.wm_log_area)

        watermark_tab.setLayout(wm_main_layout)

        downloader_tab = QWidget()
        self.downloader_ui(downloader_tab)

        self.tabs.addTab(adminium_tab, "Going-Postal")
        self.tabs.addTab(watermark_tab, "Watermark")
        self.tabs.addTab(downloader_tab, "Downloader")
        
        pin_downloader_tab = QWidget()
        self.pin_downloader_ui(pin_downloader_tab)
        self.tabs.addTab(pin_downloader_tab, "PinDownloader")
        
        randomizer_tab = QWidget()
        self.randomizer_ui(randomizer_tab)
        self.tabs.addTab(randomizer_tab, "Randomizer")
        
        cleaner_tab = QWidget()
        self.cleaner_ui(cleaner_tab)
        self.tabs.addTab(cleaner_tab, "Dog Cleaner")
        
        bimbo_tab = QWidget()
        self.bimbo_sorter_ui(bimbo_tab)
        self.tabs.addTab(bimbo_tab, "Bimbo sorter")
        
        duplicates_tab = QWidget()
        self.duplicates_ui(duplicates_tab)
        self.tabs.addTab(duplicates_tab, "Anti-Dupe")
        
        autobot_tab = QWidget()
        self.autobot_ui(autobot_tab)
        self.tabs.addTab(autobot_tab, "Autobot")
        
        credits_tab = QWidget()
        self.credits_ui(credits_tab)
        self.tabs.addTab(credits_tab, "Credits")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # запрещаем гориз. скролл
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        #Вкладки становятся содержимым скролла
        scroll.setWidget(self.tabs)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        #Фиксируем ширину окна — как у вкладок
        total_width = 360 + 450 + 70  #левая панель + лог + отступы
        self.setFixedWidth(total_width)
        
        self.setMinimumHeight(300)
        
    def pin_downloader_ui(self, tab):
        layout = QHBoxLayout(tab)
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 10, 10, 10)

        self.pin_email_input = QLineEdit()
        form_layout.addWidget(QLabel("Email:"))
        form_layout.addWidget(self.pin_email_input)

        form_layout.addWidget(QLabel("Пароль:"))

        self.pin_password_input = QLineEdit()
        self.pin_password_input.setEchoMode(QLineEdit.Password)
        pin_eye_btn = QPushButton("👁️")
        pin_eye_btn.setObjectName("eyeButton")
        pin_eye_btn.setFixedSize(30, 30)
        #pin_eye_btn.setCursor(Qt.PointingHandCursor)
        pin_eye_btn.enterEvent = lambda e: self.pin_password_input.setEchoMode(QLineEdit.Normal)
        pin_eye_btn.leaveEvent = lambda e: self.pin_password_input.setEchoMode(QLineEdit.Password)
        pin_pass_layout = QHBoxLayout()
        pin_pass_layout.addWidget(self.pin_password_input)
        pin_pass_layout.addWidget(pin_eye_btn)
        form_layout.addLayout(pin_pass_layout)
        
        hint_label = QLabel("Логин/пароль требуется только при первом входе")
        hint_label.setStyleSheet("""
            color: rgba(255, 255, 255, 160);  /* Белый с прозрачностью */
            font-size: 11px;                  /* На 1–2 пункта меньше основного */
            padding: 4px;
        """)
        hint_label.setAlignment(Qt.AlignLeft)
        form_layout.addWidget(hint_label)

        self.pin_link_input = QLineEdit("https://pinterest.com/")
        form_layout.addWidget(QLabel("Ссылка на доску/поиск/ленту:"))
        form_layout.addWidget(self.pin_link_input)

        folder_layout = QHBoxLayout()
        self.pin_folder_input = QLineEdit(os.path.expanduser("~"))
        self.pin_folder_btn = QPushButton("📁 Выбрать папку")
        self.pin_folder_btn.clicked.connect(self.select_pin_folder)
        folder_layout.addWidget(self.pin_folder_input)
        folder_layout.addWidget(self.pin_folder_btn)
        form_layout.addLayout(folder_layout)

        self.pin_pages_input = QLineEdit("10")
        form_layout.addWidget(QLabel("Кол-во страниц для загрузки (0 - бесконечно):"))
        form_layout.addWidget(self.pin_pages_input)

        btn_layout = QHBoxLayout()
        self.pin_start_btn = QPushButton("💾 Скачать с Pinterest")
        self.pin_start_btn.clicked.connect(self.start_pin_download)
        self.pin_stop_btn = QPushButton("🛑СТОП")
        self.pin_stop_btn.setEnabled(False)
        self.pin_stop_btn.clicked.connect(self.stop_pin_download)
        
        pin_button_style = """
        QPushButton {
            background-color: #ff4444;
            border: none;
            padding: 8px 16px;
            color: white;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #cc3333;
        }
        """

        self.pin_start_btn.setStyleSheet(pin_button_style)
        self.pin_stop_btn.setStyleSheet(pin_button_style)       
        self.pin_folder_btn.setStyleSheet(pin_button_style)
        
        btn_layout.addWidget(self.pin_start_btn)
        btn_layout.addWidget(self.pin_stop_btn)
        form_layout.addLayout(btn_layout)

        #Логотип внизу
        spacer = QLabel()
        spacer.setFixedHeight(150)
        form_layout.addWidget(spacer)

        self.pin_image_label = QLabel()
        self.pin_image_label.setFixedSize(150, 150)
        self.pin_image_label.setAlignment(Qt.AlignCenter)
        pin_path = resource_path("pin.png")
        if os.path.exists(pin_path):
            pixmap = QtGui.QPixmap(pin_path)
            scaled_pixmap = pixmap.scaled(
                self.pin_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.pin_image_label.setPixmap(scaled_pixmap)
        else:
            self.pin_image_label.setText("pin.png не найден")
        form_layout.addWidget(self.pin_image_label, alignment=Qt.AlignCenter)

        form_widget.setFixedWidth(360)

        self.pin_log_area = QTextEdit()
        self.pin_log_area.setReadOnly(True)
        self.pin_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.pin_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.pin_log_area)
        
    def select_pin_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if folder:
            self.pin_folder_input.setText(folder)

    @Slot(str)
    def append_pin_log(self, text):
        self.pin_log_area.append(text)
        self.pin_log_area.verticalScrollBar().setValue(self.pin_log_area.verticalScrollBar().maximum())

    def start_pin_download(self):
        email = self.pin_email_input.text().strip()
        password = self.pin_password_input.text().strip()
        link = self.pin_link_input.text().strip() or "https://pinterest.com/"
        folder = self.pin_folder_input.text().strip()
        pages_text = self.pin_pages_input.text().strip()

        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Ошибка", "Укажи корректную папку загрузки.")
            return

        #Проверка cookies
        cookies_exist = os.path.exists("cookies.pkl")

        #Если cookies есть, пропускаем логин\пароль
        if not cookies_exist and (not email or not password):
            QMessageBox.warning(self, "Ошибка", "Введи email и пароль (cookies не найдены).")
            return

        try:
            pages = int(pages_text)
            if pages <= 0:
                pages = 999999
        except ValueError:
            pages = 10

        self.pin_start_btn.setEnabled(False)
        self.pin_stop_btn.setEnabled(True)
        self.append_pin_log("[🚀] Начинаю загрузку с Pinterest...")

        self.pin_worker = PinterestWorker(email, password, link, folder, pages)
        self.pin_worker.log_signal.connect(self.append_pin_log)
        self.pin_worker.finished_signal.connect(self.on_pin_finished)
        self.pin_worker.start()

    def stop_pin_download(self):
        if hasattr(self, 'pin_worker') and self.pin_worker.isRunning():
            self.append_pin_log("🛑[STOP] Принудительная остановка...")
            self.pin_worker.stop_gracefully()

    def on_pin_finished(self):
        self.pin_start_btn.setEnabled(True)
        self.pin_stop_btn.setEnabled(False)
        self.append_pin_log("🪐[SUCCESS] Загрузка завершена.")
        
    def start_cleaner_scan(self):
        token = self.cleaner_token_input.text().strip()
        group_id = self.cleaner_group_input.text().strip()
        if not token or not group_id:
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные из настроек.")
            return

        try:
            group_id_int = int(group_id)
            if group_id_int > 0:
                group_id_int = -group_id_int
            group_id = str(group_id_int)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "ID группы должно быть числом.")
            return

        self.scan_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        self.cleaner_worker = VKWorker(token, group_id, action="scan")
        self.cleaner_worker.log_signal.connect(self.append_cleaner_log)
        self.cleaner_worker.result_signal.connect(self.handle_cleaner_results)
        self.cleaner_worker.finished_signal.connect(lambda: self.scan_button.setEnabled(True))
        self.cleaner_worker.start()

    def start_cleaner_clear(self):
        token = self.cleaner_token_input.text().strip()
        group_id = self.cleaner_group_input.text().strip()
        if not token or not group_id:
            self.append_cleaner_log("🤬[WARN] Нет данных для очистки.")
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Вы уверены, что хотите удалить {len(self.blocked_users_list)} пользователей?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.clear_button.setEnabled(False)

        self.cleaner_worker = VKWorker(token, group_id, action="clear")
        self.cleaner_worker.log_signal.connect(self.append_cleaner_log)
        self.cleaner_worker.result_signal.connect(lambda _: self.clear_button.setEnabled(True))
        self.cleaner_worker.finished_signal.connect(lambda: self.clear_button.setEnabled(True))
        self.cleaner_worker.start()

    def handle_cleaner_results(self, blocked_list):
        self.blocked_users_list = blocked_list
        self.append_cleaner_log(f"☠[DEBUG] Найдено {len(blocked_list)} заблокированных/удалённых аккаунтов.")
        self.clear_button.setEnabled(True)

    @Slot(str)
    def append_cleaner_log(self, text):
        self.cleaner_log_area.append(text)
        
    def bimbo_sorter_ui(self, tab):
        layout = QHBoxLayout(tab)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 10, 10, 10)

        hint_label = QLabel("Распределение изображений на кластеры по цветам </3")
        hint_label.setStyleSheet("""
            color: rgba(255, 255, 255, 180);  /* Серый цвет с прозрачностью */
            font-size: 11px;                  /* На 1 пункт меньше основного */
            padding: 4px;
        """)
        hint_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(hint_label)

        self.bimbo_folder = ""
        self.bimbo_label_folder = QLabel("Папка с изображениями: не выбрана")
        form_layout.addWidget(self.bimbo_label_folder)

        self.bimbo_btn_folder = QPushButton("📁 Выбрать папку")
        self.bimbo_btn_folder.clicked.connect(self.select_bimbo_folder)
        form_layout.addWidget(self.bimbo_btn_folder)

        self.cluster_size_input = QLineEdit("9")
        form_layout.addWidget(QLabel("Кол-во фото на один кластер:"))
        form_layout.addWidget(self.cluster_size_input)

        self.auto_distribute_checkbox = QCheckBox("Автораспределение (макс. 9 на кластер)")
        form_layout.addWidget(self.auto_distribute_checkbox)

        def toggle_cluster_input():
            enabled = not self.auto_distribute_checkbox.isChecked()
            self.cluster_size_input.setEnabled(enabled)
            if enabled:
                self.cluster_size_input.setStyleSheet(self.original_style)
            else:
                self.cluster_size_input.setStyleSheet("background-color: #333; color: #777; border: 1px solid #444;")

        self.auto_distribute_checkbox.stateChanged.connect(toggle_cluster_input)
        toggle_cluster_input()

        self.bimbo_run_button = QPushButton("💖 Запустить сортировку")
        self.bimbo_run_button.clicked.connect(self.start_bimbo_processing)
        form_layout.addWidget(self.bimbo_run_button)
        
        spacer = QLabel()
        spacer.setFixedHeight(275) 
        form_layout.addWidget(spacer)

        self.bimbo_image_label = QLabel()
        self.bimbo_image_label.setFixedSize(150, 150)
        self.bimbo_image_label.setStyleSheet("background-color: transparent;")
        self.bimbo_image_label.setAlignment(Qt.AlignCenter)

        logo_path = resource_path("bckg.png")
        if os.path.exists(logo_path):
            pixmap = QtGui.QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(
                self.bimbo_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.bimbo_image_label.setPixmap(scaled_pixmap)
        else:
            self.bimbo_image_label.setText("Изображение не найдено")

        image_container = QWidget()
        image_container_layout = QHBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.addWidget(self.bimbo_image_label, alignment=Qt.AlignCenter)

        form_layout.addStretch()
        form_layout.addWidget(image_container)

        image_container = QWidget()
        image_container_layout = QHBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.addWidget(self.bimbo_image_label, alignment=Qt.AlignCenter)

        form_layout.addStretch()
        form_layout.addWidget(image_container)

        form_layout.addStretch()

        form_widget.setFixedWidth(360)

        #лог справа
        self.bimbo_log_area = QTextEdit()
        self.bimbo_log_area.setReadOnly(True)
        self.bimbo_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.bimbo_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.bimbo_log_area)
        
    def select_bimbo_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку с изображениями")
        if folder:
            self.bimbo_folder = folder
            self.bimbo_label_folder.setText(f"Папка с изображениями: {folder}")
            
    def select_photos_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями для постов")
        if folder:
            self.photos_folder_input.setText(folder)
            

    def start_bimbo_processing(self):
        if not self.bimbo_folder:
            self.bimbo_log_area.append("💉Сначала выбери папку.")
            return

        auto_distribute = self.auto_distribute_checkbox.isChecked()
        cluster_size = None
        if not auto_distribute:
            try:
                cluster_size = int(self.cluster_size_input.text())
                if cluster_size < 1 or cluster_size > 20:
                    raise ValueError("Допустимо от 1 до 20 фото на кластер")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Некорректное значение: {e}")
                return

        self.bimbo_run_button.setEnabled(False)
        self.bimbo_worker = BimboSorterWorker(self.bimbo_folder, cluster_size, auto_distribute=auto_distribute)
        self.bimbo_worker.log_signal.connect(self.append_bimbo_log)
        self.bimbo_worker.finished_signal.connect(lambda: self.bimbo_run_button.setEnabled(True))
        self.bimbo_worker.start()
        
    @Slot(str)
    def append_bimbo_log(self, text):
        self.bimbo_log_area.append(text)
        
    def duplicates_ui(self, tab):
        layout = QHBoxLayout(tab)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 10, 10, 10)

        self.dup_folder_input = QLineEdit("")
        self.dup_select_button = QPushButton("📁 Выбрать папку")
        self.dup_select_button.clicked.connect(self.select_duplicates_folder)
        form_layout.addWidget(QLabel("Папка для анализа:"))
        form_layout.addWidget(self.dup_folder_input)
        form_layout.addWidget(self.dup_select_button)

        self.dup_run_button = QPushButton("🔍 Начать анализ")
        self.dup_run_button.clicked.connect(self.start_duplicates_analysis)
        self.dup_move_button = QPushButton("🗂️ Переместить выбранные")
        self.dup_move_button.clicked.connect(self.move_selected_duplicates)
        form_layout.addWidget(self.dup_run_button)
        form_layout.addWidget(self.dup_move_button)

        self.dup_tabs = QTabWidget()
        self.dup_exact_list = QListWidget()
        self.dup_soft_list = QListWidget()
        self.dup_tabs.addTab(self.dup_exact_list, "Точные дубликаты")
        self.dup_tabs.addTab(self.dup_soft_list, "Сомнения (похожие)")
        form_layout.addWidget(self.dup_tabs)

        form_layout.addStretch()
        form_widget.setFixedWidth(360)

        #лог\консоль
        self.dup_log_area = QTextEdit()
        self.dup_log_area.setReadOnly(True)
        self.dup_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.dup_log_area.setFixedWidth(450)

        #Добавляем левую и правую части в общий layout
        layout.addWidget(form_widget)
        layout.addWidget(self.dup_log_area)
        
            
    def select_duplicates_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для анализа")
        if folder:
            self.dup_folder_input.setText(folder)

    def start_duplicates_analysis(self):
        folder = self.dup_folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.critical(self, "Ошибка", "Укажите корректную папку.")
            return
        self.dup_run_button.setEnabled(False)
        self.dup_worker = DuplicateWorker(folder)
        self.dup_worker.log_signal.connect(self.append_duplicates_log)
        self.dup_worker.result_signal.connect(self.show_duplicates_results)
        self.dup_worker.finished_signal.connect(lambda: self.dup_run_button.setEnabled(True))
        self.dup_worker.start()

    def show_duplicates_results(self, results):
        self.dup_exact_list.clear()
        self.dup_soft_list.clear()
        for path in results["exact"]:
            self.dup_exact_list.addItem(path)
        for path in results["soft"]:
            self.dup_soft_list.addItem(path)
        QMessageBox.information(
            self, "Готово",
            f"Найдено:\n- Точных дубликатов: {len(results['exact'])}\n"
            f"- Потенциальных дубликатов: {len(results['soft'])}"
        )

    def move_selected_duplicates(self):
        current_tab = self.dup_tabs.currentIndex()
        target_dir = "dupes" if current_tab == 0 else "somnenie"
        if current_tab == 0:
            to_move = [self.dup_exact_list.item(i).text() for i in range(self.dup_exact_list.count())]
        elif current_tab == 1:
            to_move = [self.dup_soft_list.item(i).text() for i in range(self.dup_soft_list.count())]
        else:
            return

        if not to_move:
            QMessageBox.warning(self, "Ошибка", "Нет файлов для перемещения.")
            return

        script_dir = Path(sys.argv[0]).resolve().parent
        dupe_dir = script_dir / target_dir
        dupe_dir.mkdir(exist_ok=True)

        moved_count = 0
        for src in to_move:
            if os.path.exists(src):
                filename = os.path.basename(src)
                dest = dupe_dir / filename
                counter = 1
                while dest.exists():
                    name, ext = os.path.splitext(filename)
                    dest = dupe_dir / f"{name}_{counter}{ext}"
                    counter += 1
                try:
                    shutil.move(src, dest)
                    moved_count += 1
                except Exception as e:
                    self.append_duplicates_log(f"💉💉💉[ERROR] Ошибка при перемещении {src}: {e}")

        QMessageBox.information(self, "Готово", f"👍[SUCCESS] Перемещено {moved_count} файлов в '{target_dir}'")
        if current_tab == 0:
            self.dup_exact_list.clear()
        else:
            self.dup_soft_list.clear()

    @Slot(str)
    def append_duplicates_log(self, text):
        self.dup_log_area.append(text)
        
    def autobot_ui(self, tab):
        layout = QHBoxLayout(tab)
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 10, 10, 10)

        #Подгрузка конфига
        config = load_config()

        #Токен ВК
        token_label = QLabel('<a href="https://vkhost.github.io/" style="color: #668eff; text-decoration: none;">Токен API:</a>')
        token_label.setOpenExternalLinks(False)
        token_label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(QtCore.QUrl(link)))
        form_layout.addWidget(token_label)
        self.autobot_token_input = QLineEdit(config.get("token", ""))
        self.autobot_token_input.setEchoMode(QLineEdit.Password)
        token_eye_btn = QPushButton("👁️")
        token_eye_btn.setObjectName("eyeButton")
        token_eye_btn.setFixedSize(30, 30)
        token_eye_btn.enterEvent = lambda e: self.autobot_token_input.setEchoMode(QLineEdit.Normal)
        token_eye_btn.leaveEvent = lambda e: self.autobot_token_input.setEchoMode(QLineEdit.Password)
        token_layout = QHBoxLayout()
        token_layout.addWidget(self.autobot_token_input)
        token_layout.addWidget(token_eye_btn)
        form_layout.addLayout(token_layout)

        #ID сообщества
        form_layout.addWidget(QLabel("ID сообщества:"))
        self.autobot_group_input = QLineEdit(config.get("group_id", ""))
        form_layout.addWidget(self.autobot_group_input)

        #Интервал (часы)
        form_layout.addWidget(QLabel("Интервал постов (в часах):"))
        self.autobot_interval_input = QLineEdit("2")
        form_layout.addWidget(self.autobot_interval_input)

        #Pinterest
        form_layout.addWidget(QLabel("Pinterest логин:"))
        self.autobot_pin_email = QLineEdit()
        form_layout.addWidget(self.autobot_pin_email)
        form_layout.addWidget(QLabel("Pinterest пароль:"))
        self.autobot_pin_password = QLineEdit()
        self.autobot_pin_password.setEchoMode(QLineEdit.Password)
        pin_eye_btn = QPushButton("👁️")
        pin_eye_btn.setObjectName("eyeButton")
        pin_eye_btn.setFixedSize(30, 30)
        pin_eye_btn.enterEvent = lambda e: self.autobot_pin_password.setEchoMode(QLineEdit.Normal)
        pin_eye_btn.leaveEvent = lambda e: self.autobot_pin_password.setEchoMode(QLineEdit.Password)
        pin_pass_layout = QHBoxLayout()
        pin_pass_layout.addWidget(self.autobot_pin_password)
        pin_pass_layout.addWidget(pin_eye_btn)
        form_layout.addLayout(pin_pass_layout)

        hint_label = QLabel("Логин/пароль требуется только при первом входе")
        hint_label.setStyleSheet("color: rgba(255, 255, 255, 160); font-size: 11px;")
        form_layout.addWidget(hint_label)

        #Подпись
        form_layout.addWidget(QLabel("Подпись к постам:"))
        self.autobot_caption = QLineEdit()
        form_layout.addWidget(self.autobot_caption)

        #Чекбоксы
        self.autobot_random_emoji = QCheckBox("Рандомизировать эмодзи")
        form_layout.addWidget(self.autobot_random_emoji)
        self.autobot_carousel = QCheckBox("Карусель")
        form_layout.addWidget(self.autobot_carousel)
        self.autobot_ai_caption = QCheckBox("ИИ подписи")

        #ИИ + промпт
        ai_layout = QHBoxLayout()
        ai_layout.addWidget(self.autobot_ai_caption)
        ai_prompt_label = QLabel('<a href="#" style="font-size: 70%; color: #668eff;">Редактировать промпт</a>')
        ai_prompt_label.setOpenExternalLinks(False)
        ai_prompt_label.linkActivated.connect(self.open_prompt_editor)
        ai_layout.addWidget(ai_prompt_label)
        ai_layout.addStretch()
        form_layout.addLayout(ai_layout)

        #Mistral API
        mistral_label = QLabel('<a href="https://console.mistral.ai/home" style="color: #668eff; text-decoration: none;">Mistral API ключ:</a>')
        mistral_label.setOpenExternalLinks(False)
        mistral_label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(QtCore.QUrl(link)))
        form_layout.addWidget(mistral_label)
        self.autobot_mistral_key = QLineEdit(config.get("mistral_api_key", ""))
        self.autobot_mistral_key.setEchoMode(QLineEdit.Password)
        mistral_eye_btn = QPushButton("👁️")
        mistral_eye_btn.setObjectName("eyeButton")
        mistral_eye_btn.setFixedSize(30, 30)
        mistral_eye_btn.enterEvent = lambda e: self.autobot_mistral_key.setEchoMode(QLineEdit.Normal)
        mistral_eye_btn.leaveEvent = lambda e: self.autobot_mistral_key.setEchoMode(QLineEdit.Password)
        mistral_layout = QHBoxLayout()
        mistral_layout.addWidget(self.autobot_mistral_key)
        mistral_layout.addWidget(mistral_eye_btn)
        form_layout.addLayout(mistral_layout)

        #Папка с фото
        form_layout.addWidget(QLabel("Папка с изображениями для постов:"))
        self.autobot_photos_folder = QLineEdit()
        default_photos = os.path.join(os.path.dirname(sys.argv[0]), "photos")
        self.autobot_photos_folder.setText(default_photos)
        self.autobot_folder_btn = QPushButton("📁 Выбрать папку")
        self.autobot_folder_btn.clicked.connect(self.select_autobot_photos_folder)
        form_layout.addWidget(self.autobot_photos_folder)
        form_layout.addWidget(self.autobot_folder_btn)

        #Watermark группа
        form_layout.addWidget(QLabel("Водяной знак:"))

        self.autobot_wm_path = ""
        self.autobot_wm_label = QLabel("Изображение водяного знака: не выбрано")
        form_layout.addWidget(self.autobot_wm_label)
        self.autobot_wm_btn = QPushButton("Выбрать водяной знак")
        self.autobot_wm_btn.clicked.connect(self.select_autobot_wm)
        form_layout.addWidget(self.autobot_wm_btn)

        #Непрозрачность
        self.autobot_opacity_slider = QSlider(Qt.Horizontal)
        self.autobot_opacity_slider.setRange(0, 100)
        self.autobot_opacity_slider.setValue(32)
        self.autobot_opacity_label = QLabel("Непрозрачность: 32%")
        self.autobot_opacity_slider.valueChanged.connect(lambda v: self.autobot_opacity_label.setText(f"Непрозрачность: {v}%"))
        form_layout.addWidget(self.autobot_opacity_slider)
        form_layout.addWidget(self.autobot_opacity_label)

        #Размер
        form_layout.addWidget(QLabel("Размер водяного знака (px):"))
        self.autobot_wm_size = QLineEdit("100")
        form_layout.addWidget(self.autobot_wm_size)

        #Позиция
        pos_group = QWidget()
        pos_layout = QVBoxLayout(pos_group)
        pos_layout.addWidget(QLabel("Расположение:"))
        pos_grid = QGridLayout()
        self.autobot_wm_pos = {}
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        for i, pos in enumerate(positions):
            rb = QRadioButton()
            rb.setFixedSize(17, 17)
            rb.setStyleSheet("""
                QRadioButton::indicator {
                    width: 14px;
                    height: 14px;
                    border: 1px solid #668eff;
                    background: #2e2e2e;
                }
                QRadioButton::indicator:checked {
                    background: #668eff;
                }
            """)
            self.autobot_wm_pos[pos] = rb
            row, col = divmod(i, 2)
            pos_grid.addWidget(rb, row, col, Qt.AlignCenter)
        self.autobot_wm_pos["top-right"].setChecked(True)
        pos_layout.addLayout(pos_grid)
        form_layout.addWidget(pos_group)

        #ЧБ
        self.autobot_wm_bw = QCheckBox("Черно-белый водяной знак")
        form_layout.addWidget(self.autobot_wm_bw)

        #Кнопки управления
        self.autobot_start_btn = QPushButton("▶️ Запустить Autobot")
        self.autobot_start_btn.clicked.connect(self.start_autobot)
        form_layout.addWidget(self.autobot_start_btn)

        self.autobot_stop_btn = QPushButton("⏹️ Остановить")
        self.autobot_stop_btn.clicked.connect(self.stop_autobot)
        self.autobot_stop_btn.setEnabled(False)
        form_layout.addWidget(self.autobot_stop_btn)

        form_layout.addStretch()
        form_widget.setFixedWidth(360)

        #Лог-область справа
        self.autobot_log_area = QTextEdit()
        self.autobot_log_area.setReadOnly(True)
        self.autobot_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.autobot_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.autobot_log_area)
        
    def credits_ui(self, tab):
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)

        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(300, 300)
        self.avatar_label.setCursor(Qt.PointingHandCursor)
        avatar_path = resource_path("avatar1.png")
        if os.path.exists(avatar_path):
            pixmap = QtGui.QPixmap(avatar_path).scaled(
                self.avatar_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.avatar_label.setPixmap(pixmap)
        else:
            self.avatar_label.setText("Фото не найдено")
            self.avatar_label.setStyleSheet("color: red;")
        self.avatar_label.mousePressEvent = lambda e: QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://github.com/R3DCyclops")) 
        layout.addWidget(self.avatar_label, alignment=Qt.AlignCenter)

        text = (
            "<div style='white-space: pre-line; text-align: center;'>"
            "\n💞💖💘 Спасибо, что воспользовались VK Adminium! 💘💖💞"
            "\n\nЯ, Мореслав (R3DCyclops), сделал это приложение в одиночку."
            "\nБуду благодарен Вам, если оставите где-нибудь отзыв, свяжитесь со мной и предложите улучшения."
            "\n\n💗 Я всегда открыт к новым предложениям! 💗"
            "\n\n<a href='https://github.com/R3DCyclops/VK-Adminium'  style='color: #668eff;'>💉 GitHub 💉</a><br>"
            "\n<a href='https://vk.com/id1053382341'  style='color: #668eff;'>ВК для связи</a>"
            "\n\n---🔴---"
            "</div>"
        )

        label = QLabel(text)
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(QtCore.QUrl(link)))
        label.setStyleSheet("""
            font-size: 16px;
            padding: 10px;
            color: white;
            background-color: transparent;
        """)
        layout.addWidget(label, alignment=Qt.AlignCenter)
        
        
    def randomizer_ui(self, tab):
        layout = QHBoxLayout(tab)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        self.randomizer_folder = ""
        self.randomizer_label_folder = QLabel("Папка для рандомизации имён: не выбрана")
        form_layout.addWidget(self.randomizer_label_folder)

        self.randomizer_btn_folder = QPushButton("📁 Выбрать папку")
        self.randomizer_btn_folder.clicked.connect(self.select_randomizer_folder)
        form_layout.addWidget(self.randomizer_btn_folder)

        self.randomizer_run_button = QPushButton("🔀 Запустить рандомизацию имён")
        self.randomizer_run_button.clicked.connect(self.start_randomizer)
        form_layout.addWidget(self.randomizer_run_button)

        form_layout.addStretch()
        form_widget.setFixedWidth(360)

        self.randomizer_log_area = QTextEdit()
        self.randomizer_log_area.setReadOnly(True)
        self.randomizer_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.randomizer_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.randomizer_log_area)
        
    def cleaner_ui(self, tab):
        layout = QHBoxLayout(tab)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        config = load_config()

        token_label = QLabel()
        token_label.setText('<a href="https://vkhost.github.io/"  style="color: #668eff; text-decoration: none;">Токен API:</a>')
        token_label.setOpenExternalLinks(False)
        token_label.linkActivated.connect(lambda link: QtGui.QDesktopServices.openUrl(link))
        form_layout.addWidget(token_label)

        self.cleaner_token_input = QLineEdit(config.get("token", ""))
        self.cleaner_token_input.setEchoMode(QLineEdit.Password)
        self.cleaner_token_input.setStyleSheet("""
            background-color: #444;
            border: 1px solid #555;
            padding: 5px;
            color: white;
        """)
        cleaner_eye_btn = QPushButton("👁️")
        cleaner_eye_btn.setObjectName("eyeButton")
        cleaner_eye_btn.setFixedSize(30, 30)
        #cleaner_eye_btn.setCursor(Qt.PointingHandCursor)
        cleaner_eye_btn.enterEvent = lambda e: self.cleaner_token_input.setEchoMode(QLineEdit.Normal)
        cleaner_eye_btn.leaveEvent = lambda e: self.cleaner_token_input.setEchoMode(QLineEdit.Password)
        cleaner_layout = QHBoxLayout()
        cleaner_layout.addWidget(self.cleaner_token_input)
        cleaner_layout.addWidget(cleaner_eye_btn)
        form_layout.addLayout(cleaner_layout)
        
        self.cleaner_group_input = QLineEdit(config.get("group_id", ""))
        self.cleaner_group_input.setStyleSheet("""
            background-color: #444;
            border: 1px solid #555;
            padding: 5px;
            color: white;
        """)
        form_layout.addWidget(QLabel("ID сообщества:"))
        form_layout.addWidget(self.cleaner_group_input)

        self.scan_button = QPushButton("🔍 Сканировать участников")
        self.clear_button = QPushButton("🗑 Очистить список")
        self.clear_button.setEnabled(False)

        form_layout.addWidget(self.scan_button)
        form_layout.addWidget(self.clear_button)

        form_layout.addStretch()  #Заберёт всё свободное место

        dog_label = QLabel()
        dog_path = resource_path("dog.png")
        if os.path.exists(dog_path):
            pixmap = QtGui.QPixmap(dog_path).scaled(
                256, 256,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            dog_label.setPixmap(pixmap)
            dog_label.setAlignment(Qt.AlignBottom)
        else:
            dog_label.setText("dog.png не найден")

        form_layout.addWidget(dog_label)


        form_widget.setFixedWidth(360)

        self.cleaner_log_area = QTextEdit()
        self.cleaner_log_area.setReadOnly(True)
        self.cleaner_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)
        self.cleaner_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.cleaner_log_area)

        self.scan_button.clicked.connect(self.start_cleaner_scan)
        self.clear_button.clicked.connect(self.start_cleaner_clear)
        
    def downloader_ui(self, tab):
        layout = QHBoxLayout(tab)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        self.downloader_album_input = QLineEdit()
        form_layout.addWidget(QLabel("ID или ссылка на альбом:"))
        form_layout.addWidget(self.downloader_album_input)

        form_layout.addWidget(QLabel("Папка для загрузки:"))

        folder_layout = QHBoxLayout()
        self.downloader_folder_input = QLineEdit(os.path.expanduser("~"))
        self.downloader_select_button = QPushButton("📁 Выбрать папку")
        self.downloader_select_button.clicked.connect(self.select_downloader_folder)
        folder_layout.addWidget(self.downloader_folder_input)
        folder_layout.addWidget(self.downloader_select_button)
        form_layout.addLayout(folder_layout)

        self.downloader_run_button = QPushButton("💾 Скачать альбом")
        self.downloader_run_button.clicked.connect(self.start_downloader_album)
        form_layout.addWidget(self.downloader_run_button)

        form_layout.addStretch()

        self.downloader_log_area = QTextEdit()
        self.downloader_log_area.setReadOnly(True)
        self.downloader_log_area.setStyleSheet("""
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #444;
            font-family: Consolas, monospace;
            font-size: 12px;
        """)

        form_widget.setFixedWidth(360)
        self.downloader_log_area.setFixedWidth(450)

        layout.addWidget(form_widget)
        layout.addWidget(self.downloader_log_area)
        
    def select_downloader_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if folder:
            self.downloader_folder_input.setText(folder)

    def start_downloader_album(self):
        token = self.token_input.text().strip()
        album_input = self.downloader_album_input.text().strip()
        download_folder = self.downloader_folder_input.text().strip()

        if not token:
            QMessageBox.critical(self, "Ошибка", "Введи токен API.")
            return
        if not album_input:
            QMessageBox.critical(self, "Ошибка", "Введи ID или ссылку на альбом.")
            return
        if not os.path.isdir(download_folder):
            QMessageBox.critical(self, "Ошибка", "Выбери корректную папку загрузки.")
            return

        try:
            owner_id, album_id = self.album_downloader_worker.parse_album_input(album_input)
            if str(album_id) == 'wall':
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Предупреждение")
                msg_box.setText("Ты же понимаешь, что воровать - плохо?")
                msg_box.setIcon(QMessageBox.Question)
                yes_button = msg_box.addButton("Да, я беру ответственность на себя.", QMessageBox.YesRole)
                no_button = msg_box.addButton("Я передумал.", QMessageBox.NoRole)
                msg_box.setDefaultButton(no_button)
                msg_box.exec()
                if msg_box.clickedButton() == no_button:
                    self.append_downloader_log("😘[MOLODETS] Загрузка отменена пользователем.")
                    return
        except Exception as e:
            self.append_downloader_log(f"🧰[ERROR] Не удалось распарсить ввод альбома: {e}")
            return

        self.downloader_run_button.setEnabled(False)
        self.downloader_worker = AlbumDownloaderWorker(token, album_input, download_folder)
        self.downloader_worker.log_signal.connect(self.append_downloader_log)
        self.downloader_worker.finished_signal.connect(lambda: self.downloader_run_button.setEnabled(True))
        self.downloader_worker.start()

    @Slot(str)
    def append_downloader_log(self, text):
        self.downloader_log_area.append(text)
        
    def select_wm_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбери папку с изображениями")
        if folder:
            self.wm_folder = folder
            self.wm_label_folder.setText(f"Папка с изображениями: {folder}")

    def select_wm_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбери изображение водяного знака",
                                              "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.wm_path = path
            self.wm_label_watermark.setText(f"Изображение водяного знака: {path}")
            
    def select_autobot_photos_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями для постов")
        if folder:
            self.autobot_photos_folder.setText(folder)

    def select_autobot_wm(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбери изображение водяного знака",
                                              "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.autobot_wm_path = path
            self.autobot_wm_label.setText(f"Изображение водяного знака: {path}")

    def select_randomizer_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбери папку для рандомизации имён")
        if folder:
            self.randomizer_folder = folder
            self.randomizer_label_folder.setText(f"Папка для рандомизации имён: {folder}")

    def start_randomizer(self):
        if not self.randomizer_folder:
            self.randomizer_log_area.append("🤬[WARN] Сначала выбери папку")
            return

        self.randomizer_run_button.setEnabled(False)
        self.randomizer_worker = RandomizerWorker(self.randomizer_folder)
        self.randomizer_worker.log_signal.connect(self.append_randomizer_log)
        self.randomizer_worker.finished_signal.connect(lambda: self.randomizer_run_button.setEnabled(True))
        self.randomizer_worker.start()

    @Slot(str)
    def append_randomizer_log(self, text):
        self.randomizer_log_area.append(text)

    def apply_watermark(self):
        if not self.wm_folder or not self.wm_path:
            self.wm_log_area.append("🤬[WARN] Сначала выбери папку и водяной знак")
            return

        try:
            opacity = self.opacity_slider.value()
            size = int(self.size_input.text())
            position = next(k for k, v in self.position_buttons.items() if v.isChecked())
            bw = self.wm_bw_checkbox.isChecked()

            self.wm_apply_button.setEnabled(False)
            self.worker = WatermarkWorker(
                self.wm_folder,
                self.wm_path,
                opacity,
                size,
                position,
                bw
            )
            self.worker.log_signal.connect(self.append_watermark_log)
            self.worker.finished_signal.connect(lambda: self.wm_apply_button.setEnabled(True))
            self.worker.start()
        except Exception as e:
            self.append_watermark_log(f"🧰[ERROR] Ошибка при подготовке: {e}")
        
    @Slot(str)
    def append_watermark_log(self, text):
        self.wm_log_area.append(text)
        
    def start_posting(self):
        token = self.token_input.text().strip()
        group_id = self.group_input.text().strip()
        photos_per_post = self.photos_per_post_input.text().strip()

        try:
            interval_hours = float(self.interval_input.text().strip())
            if interval_hours < 0.0167:
                raise ValueError("Интервал не может быть меньше 0.0167 часа (1 минуты)")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Некорректный интервал: {e}")
            return

        try:
            photos_per_post_int = int(photos_per_post)
            if photos_per_post_int < 1 or photos_per_post_int > 9:
                raise ValueError("Кол-во фото должно быть от 1 до 9")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Некорректное кол-во фото на пост: {e}")
            return

        if not token or not group_id:
            QMessageBox.critical(self, "Ошибка", "Заполни все поля.")
            return

        try:
            group_id_int = int(group_id)
            if group_id_int > 0:
                group_id_int = -group_id_int
            group_id = str(group_id_int)
            self.group_input.setText(group_id)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "ID должен быть числом.")
            return

        folder_path = self.photos_folder_input.text().strip()
        if not os.path.isdir(folder_path):
            QMessageBox.critical(self, "Ошибка", "Укажите корректную папку с изображениями.")
            return

        #Создаём папку posted внутри выбранной папки
        posted_folder = os.path.join(folder_path, "posted")
        if not os.path.exists(posted_folder):
            try:
                os.makedirs(posted_folder)
                self.append_log(f"📁[DEBUG] Создана папка 'posted' в: {posted_folder}")
            except Exception as e:
                self.append_log(f"🧰[ERROR] Не удалось создать папку 'posted': {e}")

        start_datetime = self.datetime_edit.dateTime()
        start_timestamp = start_datetime.toSecsSinceEpoch()
        if start_timestamp < int(time.time()):
            reply = QMessageBox.question(
                self,
                "Время в прошлом",
                "Выбранная дата прошла. Поставить текущее время?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                return
            else:
                start_timestamp = int(time.time())

        caption = self.caption_input.text().strip()
        use_random_emoji = self.random_emoji_checkbox.isChecked()
        random_photos = self.random_photos_checkbox.isChecked()
        use_carousel = self.carousel_checkbox.isChecked()
        cluster_mode = self.cluster_mode_checkbox.isChecked()
        use_ai_caption = self.ai_caption_checkbox.isChecked()
        mistral_api_key = self.mistral_token_input.text().strip()

        save_config(
            token=token,
            group_id=group_id,
            mistral_api_key=mistral_api_key,
            last_post_time=None,
            ai_prompt=self.current_ai_prompt
        )

        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)

        self.worker = PosterWorker(
            token, group_id, interval_hours, folder_path, start_timestamp,
            photos_per_post, caption, use_random_emoji, random_photos, self.emoji_list,
            use_carousel=use_carousel, cluster_mode=cluster_mode,
            use_ai_caption=use_ai_caption, mistral_api_key=mistral_api_key,
            ai_custom_prompt=self.current_ai_prompt
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(lambda: self.run_button.setEnabled(True))
        self.worker.finished_signal.connect(lambda: self.pause_button.setEnabled(False))
        self.worker.update_last_post_time.connect(lambda t: self.datetime_edit.setDateTime(
            datetime.fromtimestamp(t + 7200)
        ))
        self.worker.start()
        
    def start_autobot(self):
        #Валидация
        token = self.autobot_token_input.text().strip()
        group_id = self.autobot_group_input.text().strip()
        main_folder = self.autobot_photos_folder.text().strip()
        wm_path = self.autobot_wm_path
        if not token or not group_id:
            QMessageBox.critical(self, "Ошибка", "Заполните токен и ID группы.")
            return
        if not os.path.isdir(main_folder):
            QMessageBox.critical(self, "Ошибка", "Укажите корректную папку с изображениями.")
            return
        # Водяной знак опционален, если не указан, wm_path останется пустым

        #Подготовка параметров
        try:
            interval_hours = float(self.autobot_interval_input.text().strip())
        except ValueError:
            interval_hours = 2.0

        #Положение вотермарки
        wm_pos = "top-right"
        for pos, rb in self.autobot_wm_pos.items():
            if rb.isChecked():
                wm_pos = pos
                break

        self.autobot_start_btn.setEnabled(False)
        self.autobot_stop_btn.setEnabled(True)

        self.autobot_worker = AutobotWorker(
            token=token,
            group_id=group_id,
            interval_hours=interval_hours,
            main_folder=main_folder,
            pin_email=self.autobot_pin_email.text().strip(),
            pin_password=self.autobot_pin_password.text().strip(),
            caption=self.autobot_caption.text().strip(),
            use_random_emoji=self.autobot_random_emoji.isChecked(),
            use_carousel=self.autobot_carousel.isChecked(),
            use_ai_caption=self.autobot_ai_caption.isChecked(),
            mistral_api_key=self.autobot_mistral_key.text().strip(),
            ai_prompt=self.current_ai_prompt,
            wm_path=wm_path,
            wm_opacity=self.autobot_opacity_slider.value(),
            wm_size=int(self.autobot_wm_size.text()),
            wm_position=wm_pos,
            wm_bw=self.autobot_wm_bw.isChecked(),
            emoji_list=self.emoji_list
        )
        self.autobot_worker.log_signal.connect(self.append_autobot_log)
        self.autobot_worker.finished_signal.connect(self.on_autobot_finished)
        self.autobot_worker.start()

    def stop_autobot(self):
        if hasattr(self, 'autobot_worker') and self.autobot_worker.isRunning():
            self.autobot_worker.stop()
            self.autobot_stop_btn.setEnabled(False)

    def on_autobot_finished(self):
        self.autobot_start_btn.setEnabled(True)
        self.autobot_stop_btn.setEnabled(False)

    @Slot(str)
    def append_autobot_log(self, text):
        self.autobot_log_area.append(text)
        self.autobot_log_area.verticalScrollBar().setValue(self.autobot_log_area.verticalScrollBar().maximum())

    def check_delayed(self):
        token = self.token_input.text().strip()
        group_id = self.group_input.text().strip()
        if not token or not group_id:
            QMessageBox.critical(self, "Ошибка", "Заполни оба поля.")
            return
        try:
            group_id_int = int(group_id)
            if group_id_int > 0:
                group_id_int = -group_id_int
            group_id = str(group_id_int)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "ID должно быть числом.")
            return
        self.check_button.setEnabled(False)
        self.check_worker = CheckAndClearWorker(token, group_id, action="check")
        self.check_worker.log_signal.connect(self.append_log)
        self.check_worker.finished_signal.connect(lambda: self.check_button.setEnabled(True))
        self.check_worker.start()

    def clear_delayed(self):
        token = self.token_input.text().strip()
        group_id = self.group_input.text().strip()
        if not token or not group_id:
            QMessageBox.critical(self, "Ошибка", "Заполни оба поля.")
            return
        try:
            group_id_int = int(group_id)
            if group_id_int > 0:
                group_id_int = -group_id_int
            group_id = str(group_id_int)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "ID должен быть числом.")
            return

        confirm_dialog = QtWidgets.QDialog(self)
        confirm_dialog.setWindowTitle("Подтверждение")
        confirm_dialog.setFixedSize(300, 120)  #размер окна

        layout = QVBoxLayout(confirm_dialog)

        label = QLabel("Ты уверен?")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px;")

        button_box = QHBoxLayout()
        yes_btn = QPushButton("Да")
        no_btn = QPushButton("Нет")

        yes_btn.setStyleSheet("background-color: #ff4444; color: white; padding: 5px;")
        no_btn.setStyleSheet("background-color: #444444; color: white; padding: 5px;")

        yes_btn.clicked.connect(confirm_dialog.accept)
        no_btn.clicked.connect(confirm_dialog.reject)

        button_box.addWidget(yes_btn)
        button_box.addWidget(no_btn)

        layout.addWidget(label)
        layout.addLayout(button_box)

        if confirm_dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        self.clear_button.setEnabled(False)
        self.clear_worker = CheckAndClearWorker(token, group_id, action="clear")
        self.clear_worker.log_signal.connect(self.append_log)
        self.clear_worker.finished_signal.connect(lambda: self.clear_button.setEnabled(True))
        self.clear_worker.start()

    def toggle_pause(self):
        if hasattr(self, 'worker'):
            self.worker.toggle_pause()
            is_paused = self.worker.paused
            self.pause_button.setText("▶️Пуск" if is_paused else "⏸️Пауза")

            
            self.pause_button.setProperty("paused", is_paused)
            self.pause_button.style().unpolish(self.pause_button)
            self.pause_button.style().polish(self.pause_button)

            if is_paused:
                self.append_log("[⏸️] Работа остановлена.")
            else:
                self.append_log("[▶️] Продолжаю работу...")
                
                
    def select_download_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if folder:
            self.folder_input.setText(folder)
            
    def start_album_download(self):
        token = self.token_input.text().strip()
        album_input = self.album_input.text().strip()
        download_folder = self.folder_input.text().strip()

        if not token:
            QMessageBox.critical(self, "Ошибка", "Введи токен API.")
            return
        if not album_input:
            QMessageBox.critical(self, "Ошибка", "Введи ID или ссылку на альбом.")
            return
        if not download_folder or not os.path.isdir(download_folder):
            QMessageBox.critical(self, "Ошибка", "Выбери корректную папку загрузки.")
            return

        try:
            owner_id, album_id = self.album_downloader_worker.parse_album_input(album_input)
            if str(album_id) == 'wall':
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Предупреждение")
                msg_box.setText("Ты же понимаешь, что воровать - плохо?")
                msg_box.setIcon(QMessageBox.Question)

                yes_button = msg_box.addButton("Да, я беру ответственность на себя.", QMessageBox.YesRole)
                no_button = msg_box.addButton("Я передумал.", QMessageBox.NoRole)
                msg_box.setDefaultButton(no_button)

                msg_box.exec()

                if msg_box.clickedButton() == no_button:
                    self.append_log("[😘] Загрузка отменена пользователем.")
                    return
        except Exception as e:
            self.append_log(f"[💀] Не удалось распарсить ввод альбома: {e}")
            return

        self.download_album_button.setEnabled(False)
        self.album_downloader = AlbumDownloaderWorker(token, album_input, download_folder)
        self.album_downloader.log_signal.connect(self.append_log)
        self.album_downloader.finished_signal.connect(lambda: self.download_album_button.setEnabled(True))
        self.album_downloader.start()
        
    @Slot(str)
    def append_log(self, text):
        self.log_area.append(text)


if __name__ == "__main__":
    import traceback
    try:
        app = QApplication(sys.argv)
        window = VKAutoPosterApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write("Критическая ошибка:\n")
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print("Произошла ошибка:")
        print(traceback.format_exc())
        input("Нажми Enter для выхода...")
