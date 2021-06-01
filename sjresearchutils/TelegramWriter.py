import configparser
import telegram
import sys


class TelegramWriter:
  def __init__(self, token, chat_id):
    self.token = token
    self.chat_id = chat_id
    self.bot = telegram.Bot(token=self.token)

  def write(self, msg):
     self.bot.sendMessage(chat_id=self.chat_id, text=msg)


if __name__ == '__main__':

  config = configparser.ConfigParser()
  config.read('TelegramBot.ini')

  token = config['telegram']['token']
  chat_id = config['telegram']['chat_id']

  w = TelegramWriter(token, chat_id)
  if len(sys.argv) > 1:
    w.write(sys.argv[1])
  else:
    w.write('Test Message')
