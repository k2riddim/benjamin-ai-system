import json
import types
import pytest


class DummyUpdate:
    class Msg:
        def __init__(self):
            self.sent = []

        async def reply_text(self, text, parse_mode=None, reply_markup=None):
            self.sent.append((text, parse_mode))

    def __init__(self):
        self.message = self.Msg()


class DummyContext:
    pass


class DummyResp:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self._text = text
        self._json = json_data

    def json(self):
        if self._json is None:
            raise ValueError("Invalid JSON")
        return self._json

    @property
    def text(self):
        return self._text


@pytest.mark.asyncio
async def test_discussion_non_json_response(monkeypatch):
    import os
    os.environ["TELEGRAM_APP_TELEGRAM_BOT_TOKEN"] = "TEST_TOKEN"
    from telegram_app.src.telegram_bot import BenjaminTelegramBot
    bot = BenjaminTelegramBot()  # do not call setup_application; we unit test handler only

    # Case 1: HTTP 200 but non-JSON body -> should send fallback error message
    def fake_post(*args, **kwargs):
        return DummyResp(status_code=200, text="<html>OK</html>", json_data=None)

    monkeypatch.setattr("telegram_app.src.telegram_bot.requests.post", fake_post)

    update = DummyUpdate()
    context = DummyContext()

    await bot.discussion_command(update, context)

    assert update.message.sent, "No message was sent to user"
    last_msg, parse_mode = update.message.sent[-1]
    assert "Agentic API" in last_msg or "Discussion failed" in last_msg


@pytest.mark.asyncio
async def test_discussion_missing_expected_keys(monkeypatch):
    import os
    os.environ["TELEGRAM_APP_TELEGRAM_BOT_TOKEN"] = "TEST_TOKEN"
    from telegram_app.src.telegram_bot import BenjaminTelegramBot
    bot = BenjaminTelegramBot()  # do not call setup_application

    # Case 2: JSON but missing 'telegram_message' key
    def fake_post(*args, **kwargs):
        return DummyResp(status_code=200, json_data={"foo": "bar"})

    monkeypatch.setattr("telegram_app.src.telegram_bot.requests.post", fake_post)

    update = DummyUpdate()
    context = DummyContext()

    await bot.discussion_command(update, context)

    assert update.message.sent, "No message was sent to user"
    last_msg, parse_mode = update.message.sent[-1]
    # Should still handle gracefully and send something (HTML or plain fallback)
    assert "Discussion" in last_msg


