from click import Command
from telegram.ext import *
from predict import *
import os
import emoji
from keys import *

# Commands

async def start_command(update, context):
    text = 'Hey, my name is Mimo! Send me a song and I\'ll tell you the genre of it.'
    await update.message.reply_text(text)

async def info_command(update, context):
    text = 'I\'ve been created by @kacperaleksander as a side project.\nGithub: https://github.com/kacperfin/music-genre-prediction'
    await update.message.reply_text(text)

def handle_response(text: str) -> str:
    # Response logic
    response = 'Please send a valid mp3 or preferrably wav file.'
    return response

async def handle_message(update, context):
    message_type = update.message.chat.type
    text = update.message.text
    username = update.message.from_user.username

    print(f"{username}: {text}")

    response = handle_response(text)
    await update.message.reply_text(response)

    print(f"B: {response}")

async def handle_audio(update, context):
    audio = update.message.audio or update.message.voice
    chat_id = str(update.message.chat.id)
    song_path = 'songs_predict/song' + chat_id + '.wav'
    song_features_path = 'data_predict/data' + chat_id + '.json'
    user_id = update.message.chat.id
    username = update.message.from_user.username

    if audio:
        print(f"{username} ({user_id}) sent a song")
        await update.message.reply_text('I\'m downloading the song...')
        file_id = audio.file_id
        new_file = await context.bot.get_file(file_id)
        await new_file.download_to_drive(song_path)

        # Now file has been downloaded. Let's get the genre
        await update.message.reply_text('I\'m analyzing the song...')
        genre = predict(song_path, song_features_path)
        text = f'It\'s {genre}!'
        text += emoji.emojize(' :musical_notes:')
        await update.message.reply_text(text)

async def error(update, context):
    print(f"Update {update} caused error: {context.error}")

def main():
    print('Starting the bot...')
    app = Application.builder().token(API_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('info', info_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))

    # Errors
    app.add_error_handler(error)

    # Run the bot
    print('Polling...')
    app.run_polling(poll_interval=2)

if __name__ == '__main__':
    os.system('clear')
    main()