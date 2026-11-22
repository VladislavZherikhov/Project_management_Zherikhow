import asyncio
from io import BytesIO
import json
import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from pipeline import DocumentPipeline

MAX_BYTES = 10 * 1024 * 1024


async def start(update, context):
    await update.effective_message.reply_text(
        "Send a photo or a JPG/PNG file of an INN certificate, a passport, or a SNILS card. "
        "Use a synthetic sample for testing. Send it as a file to preserve the original image."
    )


async def on_upload(update, context):
    message = update.effective_message
    item = message.photo[-1] if message.photo else message.document
    filename = getattr(item, "file_name", None) or "telegram-photo.jpg"
    pipe = context.application.bot_data["pipeline"]
    if item.file_size and item.file_size > MAX_BYTES:
        result = await asyncio.to_thread(pipe.run, BytesIO(), filename=filename)
        if result["reason"] == "unsupported_file":
            result["reason"] = "file_too_large"
    else:
        try:
            remote = await item.get_file()
            content = await remote.download_as_bytearray()
        except Exception:
            result = await asyncio.to_thread(pipe.run, BytesIO(), filename=filename)
            if result["reason"] == "unsupported_file":
                result["reason"] = "download_failed"
        else:
            if len(content) > MAX_BYTES:
                result = await asyncio.to_thread(pipe.run, BytesIO(), filename=filename)
                if result["reason"] == "unsupported_file":
                    result["reason"] = "file_too_large"
            else:
                result = await asyncio.to_thread(pipe.run, BytesIO(content), filename=filename)
    await message.reply_text(json.dumps(result, ensure_ascii=False, indent=2))


async def announce(application):
    print(f"Telegram bot ready: @{application.bot.username}", flush=True)


def main():
    logging.basicConfig(level=logging.WARNING)
    # HTTP request URLs include the bot token.
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("telegram").setLevel(logging.CRITICAL)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in the environment.")
    app = Application.builder().token(token).post_init(announce).build()
    app.bot_data["pipeline"] = DocumentPipeline()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, on_upload))
    try:
        print("Telegram polling is starting.", flush=True)
        app.run_polling(allowed_updates=[Update.MESSAGE])
    except Exception as exc:
        raise SystemExit(f"Telegram startup failed ({type(exc).__name__}).") from None


if __name__ == "__main__":
    main()
