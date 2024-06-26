FROM python:3.8-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Установка зависимостей Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Создание рабочей директории
WORKDIR /app
COPY . /app

# Создание директорий для загрузки файлов и установки прав доступа
RUN mkdir -p /uploads && chmod -R 777 /uploads
RUN mkdir -p /app && chmod -R 777 /app
RUN mkdir -p /app/runs && chmod -R 777 /app/runs

# Установка переменной окружения для Flask
ENV FLASK_APP=__init__.py

# Запуск приложения
CMD ["flask", "run", "--host=0.0.0.0"]
