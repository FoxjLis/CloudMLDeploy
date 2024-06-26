from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import cv2
from ultralytics import YOLO

model = YOLO('C:/Users/Karma/PycharmProjects/svarka/runs/detect/train4/weights/best.pt')

# Словарь с описанием дефектов
defect_descriptions = {
    0: '3M ESPE Implant: Имплант от компании 3M ESPE.',
    33: 'AGS Medikal Implant: Имплант от компании AGS Medikal.',
    34: 'AMerOss Implant: Имплант от компании AMerOss.',
    35: 'Amalgam filling: Амальгамовая пломба.',
    36: 'Anthogyr Implant: Имплант от компании Anthogyr.',
    37: 'Bicon Implant: Имплант от компании Bicon.',
    38: 'BioHorizons Implant: Имплант от компании BioHorizons.',
    39: 'BioLife Implant: Имплант от компании BioLife.',
    40: 'Biomet 3i Implant: Имплант от компании Biomet 3i.',
    41: 'Blue Sky Bio Implant: Имплант от компании Blue Sky Bio.',
    42: 'Camlog Implant: Имплант от компании Camlog.',
    43: 'Caries: Кариес.',
    44: 'Composite filling: Композитная пломба.',
    45: 'Cowellmedi Implant: Имплант от компании Cowellmedi.',
    46: 'Crown: Коронка.',
    47: 'DENTSPLY Implant: Имплант от компании DENTSPLY.',
    48: 'Dentatus Implant: Имплант от компании Dentatus.',
    49: 'Dentis Implant: Имплант от компании Dentis.',
    50: 'Dentium Implant: Имплант от компании Dentium.',
    51: 'Euroteknika Implant: Имплант от компании Euroteknika.',
    52: 'Filling: Пломба.',
    53: 'Frontier Implant: Имплант от компании Frontier.',
    54: 'Hiossen Implant: Имплант от компании Hiossen.',
    55: 'Implant: Имплант.',
    56: 'Implant Direct: Имплант от компании Implant Direct.',
    57: 'Keystone Dental Implant: Имплант от компании Keystone Dental.',
    58: 'Leone Implant: Имплант от компании Leone.',
    59: 'MIS Implant: Имплант от компании MIS.',
    60: 'Mandible: Нижняя челюсть.',
    61: 'Maxilla: Верхняя челюсть.',
    62: 'Megagen Implant: Имплант от компании Megagen.',
    63: 'Neodent Implant: Имплант от компании Neodent.',
    64: 'Neoss Implant: Имплант от компании Neoss.',
    65: 'Nobel Biocare Implant: Имплант от компании Nobel Biocare.',
    66: 'Novodent Implant: Имплант от компании Novodent.',
    67: 'NucleOSS Implant: Имплант от компании NucleOSS.',
    68: 'OCO Biomedical Implant: Имплант от компании OCO Biomedical.',
    69: 'OsseoLink Implant: Имплант от компании OsseoLink.',
    70: 'Osstem Implant: Имплант от компании Osstem.',
    71: 'Prefabricated metal post: Готовый металлический штифт.',
    72: 'Retained root: Сохранившийся корень.',
    73: 'Root canal filling: Пломба корневого канала.',
    74: 'Root canal obturation: Обтурация корневого канала.',
    75: 'Sterngold Implant: Имплант от компании Sterngold.',
    76: 'Straumann Implant: Имплант от компании Straumann.',
    77: 'Titan Implant: Имплант от компании Titan.',
    78: 'Zimmer Implant: Имплант от компании Zimmer.'
}

# Порог уверенности
CONFIDENCE_THRESHOLD = 0.3

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Отправь мне фотографию сварного шва, и я классифицирую её.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Отправь мне фотографию сварного шва, и я скажу тебе, есть ли на ней дефекты.')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    photo_path = 'photo.jpg'
    await photo_file.download_to_drive(photo_path)

    # Загрузка и предобработка изображения
    image = cv2.imread(photo_path)
    results = model(image)

    # Обработка результатов и рисование ректов
    response = "Классификация завершена. Результаты:\n"
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)  # Преобразование Tensor к float
            if 1 <= class_id <= 32 or confidence < CONFIDENCE_THRESHOLD:
                continue  # Пропуск классов с 1 по 32 и низкой уверенности
            label = model.names[class_id]
            defect_description = defect_descriptions.get(class_id, label)
            response += f'{defect_description}: {confidence:.2f}\n'

            # Рисование ректов на изображении
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Сохранение изображения с ректами
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)

    # Отправка результата пользователю
    await update.message.reply_text(response)
    await context.bot.send_photo(chat_id=update.message.chat_id, photo=open(output_path, 'rb'))

def main() -> None:
    TOKEN = ''

    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))

    application.run_polling()

if __name__ == '__main__':
    main()
